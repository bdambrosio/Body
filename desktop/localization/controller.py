"""LocalizationController — MCL against a frozen ReferenceMap.

Replaces FuserController for navigation sessions. Does not fuse
``body/map/local_2p5d`` into a world grid.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from desktop.localization.config import LocalizationConfig
from desktop.localization.mcl_pose_source import MCLPoseSource, MCLPoseSourceConfig
from desktop.reference_map.legacy_convert import load_map_auto
from desktop.reference_map.reference_map import ReferenceMap
from desktop.world_map.particle_filter_pose import ParticleFilterConfig
from desktop.world_map.pose_source import PoseSource
from desktop.world_map.transport import open_session
from desktop.world_map.world_grid import encode_for_publish

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _decode_json(payload: bytes) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None


def _odom_xyt(msg: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    try:
        ts = float(msg.get("ts") or _now())
        x = float(msg.get("x") or 0.0)
        y = float(msg.get("y") or 0.0)
        if "theta" in msg:
            th = float(msg["theta"])
        elif "theta_rad" in msg:
            th = float(msg["theta_rad"])
        elif "yaw" in msg:
            th = float(msg["yaw"])
        else:
            th = 0.0
        return ts, x, y, th
    except (TypeError, ValueError):
        return None


class LocalizationController:
    """Owns Zenoh I/O, frozen map, and MCL pose source."""

    def __init__(self, config: LocalizationConfig, reference_map: ReferenceMap):
        self.config = config
        self.reference_map = reference_map
        self.pose_source: PoseSource = MCLPoseSource(
            reference_map,
            pf_config=ParticleFilterConfig(
                device=config.pf_device,
                n_particles=config.pf_n_particles,
                defensive_resample_fraction=0.0,
                scan_temperature_log_ratio=config.pf_scan_temperature_log_ratio,
                odom_process_blur_xy_m=config.pf_odom_blur_xy_m,
                odom_process_blur_theta_rad=config.pf_odom_blur_theta_rad,
            ),
            config=MCLPoseSourceConfig(
                scan_hz=config.scan_hz,
                imu_obs_hz=config.pf_imu_obs_hz,
            ),
        )

        self._session: Optional[Any] = None
        self._subscribers: List[Any] = []
        self._pub_world_driveable: Optional[Any] = None
        self._pub_world_status: Optional[Any] = None
        self._stop_event = threading.Event()
        self._publish_thread: Optional[threading.Thread] = None

        self._lock = threading.RLock()
        self._connected = False
        self._notes: Optional[str] = None
        self._arr_odom: Deque[float] = deque(maxlen=64)
        self._arr_lidar: Deque[float] = deque(maxlen=32)
        self._last_odom_ts = 0.0
        self._last_lidar_ts = 0.0
        self._pose_trail: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=int(max(16, config.pose_trail_max_points)),
        )
        self._on_session_change: Optional[Callable[[str], None]] = None
        self._on_grid_update: Optional[Callable[[], None]] = None

    # Compatibility shim: nav code reads ``controller.grid.session_id``.
    @property
    def grid(self) -> "_GridShim":
        return _GridShim(self.reference_map)

    @property
    def session(self) -> Optional[Any]:
        return self._session

    @property
    def connected(self) -> bool:
        return self._session is not None

    def set_on_session_change(self, cb: Optional[Callable[[str], None]]) -> None:
        self._on_session_change = cb

    def set_on_grid_update(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_grid_update = cb

    def connect(self) -> Tuple[bool, Optional[str]]:
        if self._session is not None:
            return True, None
        try:
            self._session = open_session(self.config.router)
        except Exception as e:
            logger.exception("zenoh open failed")
            return False, f"{type(e).__name__}: {e}"
        try:
            self._declare()
        except Exception as e:
            logger.exception("declare failed")
            self._teardown_zenoh()
            return False, f"{type(e).__name__}: {e}"

        connect_pose = getattr(self.pose_source, "connect", None)
        if callable(connect_pose):
            try:
                connect_pose(self._session)
            except Exception:
                logger.exception("pose_source.connect raised")

        self._stop_event.clear()
        self._publish_thread = threading.Thread(
            target=self._publish_loop, name="loc-publish", daemon=True,
        )
        self._publish_thread.start()
        with self._lock:
            self._connected = True
        return True, None

    def disconnect(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._publish_thread is not None and self._publish_thread.is_alive():
            self._publish_thread.join(timeout=1.0)
        self._publish_thread = None
        disconnect_pose = getattr(self.pose_source, "disconnect", None)
        if callable(disconnect_pose):
            try:
                disconnect_pose()
            except Exception:
                pass
        self._teardown_zenoh()
        with self._lock:
            self._connected = False

    def _teardown_zenoh(self) -> None:
        for sub in self._subscribers:
            try:
                sub.undeclare()
            except Exception:
                pass
        self._subscribers.clear()
        for pub in (self._pub_world_driveable, self._pub_world_status):
            if pub is not None:
                try:
                    pub.undeclare()
                except Exception:
                    pass
        self._pub_world_driveable = None
        self._pub_world_status = None
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def _declare(self) -> None:
        assert self._session is not None
        t = self.config.topics
        self._subscribers.append(
            self._session.declare_subscriber(t.odom, self._on_odom),
        )
        self._subscribers.append(
            self._session.declare_subscriber(t.world_cmd, self._on_world_cmd),
        )
        self._pub_world_driveable = self._session.declare_publisher(
            t.world_driveable,
        )
        self._pub_world_status = self._session.declare_publisher(t.world_status)

    def _payload_bytes(self, sample: Any) -> bytes:
        try:
            return bytes(sample.payload.to_bytes())
        except AttributeError:
            return bytes(sample.payload)

    def _on_odom(self, sample: Any) -> None:
        msg = _decode_json(self._payload_bytes(sample))
        if msg is None:
            return
        triple = _odom_xyt(msg)
        if triple is None:
            return
        ts, x, y, theta = triple
        self.pose_source.update(ts, x, y, theta)
        with self._lock:
            self._arr_odom.append(_now())
            self._last_odom_ts = _now()
            latest = self.pose_source.latest_pose()
            if latest is not None:
                self._maybe_record_pose(latest[0], latest[1])

    def _on_world_cmd(self, sample: Any) -> None:
        msg = _decode_json(self._payload_bytes(sample))
        if not isinstance(msg, dict):
            return
        action = msg.get("action")
        if action == "reset":
            self.request_reset(reason=str(msg.get("reason") or "cmd_reset"))
        elif action == "relocate":
            self.request_relocate(reason=str(msg.get("reason") or "cmd_relocate"))

    def request_reset(self, *, reason: str = "ui_reset") -> None:
        self.pose_source.rebind_world_to_current()
        with self._lock:
            self._notes = f"reset:{reason}"
            self._pose_trail.clear()
        logger.info("localization reset reason=%s", reason)

    def request_relocate(self, *, reason: str = "ui_relocate") -> dict:
        result = self.pose_source.relocate()
        if result.get("success"):
            logger.info("localization relocate ok reason=%s %s", reason, result)
        else:
            logger.warning("localization relocate failed reason=%s %s", reason, result)
        return result

    def request_relocate_at(
        self, x: float, y: float, *, reason: str = "ui_locate",
    ) -> dict:
        """Operator override: trust world (x, y), recover yaw only."""
        fn = getattr(self.pose_source, "relocate_at", None)
        if fn is None:
            return {"success": False, "reason": "not_supported"}
        result = fn(x, y)
        if result.get("success"):
            logger.info(
                "localization relocate_at ok reason=%s (%.2f,%.2f) %s",
                reason, x, y, result,
            )
        else:
            logger.warning(
                "localization relocate_at failed reason=%s (%.2f,%.2f) %s",
                reason, x, y, result,
            )
        return result

    def _maybe_record_pose(
        self, pose: Tuple[float, float, float], sample_ts: float,
    ) -> None:
        cfg = self.config
        with self._lock:
            now = _now()
            cutoff = now - cfg.pose_trail_seconds
            while self._pose_trail and self._pose_trail[0][3] < cutoff:
                self._pose_trail.popleft()
            if not self._pose_trail:
                self._pose_trail.append((pose[0], pose[1], pose[2], now))
                return
            x_prev, y_prev, th_prev, t_prev = self._pose_trail[-1]
            dxy = math.hypot(pose[0] - x_prev, pose[1] - y_prev)
            dth = abs(_wrap_pi(pose[2] - th_prev))
            dt = now - t_prev
            if (dxy >= cfg.pose_trail_min_dxy_m
                    or dth >= cfg.pose_trail_min_dtheta_rad
                    or dt >= cfg.pose_trail_min_period_s):
                self._pose_trail.append((pose[0], pose[1], pose[2], now))

    def _publish_loop(self) -> None:
        status_period = 1.0 / max(0.1, self.config.status_hz)
        publish_period = 1.0 / max(0.1, self.config.publish_hz)
        next_status = time.monotonic()
        next_publish = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= next_status:
                try:
                    self._publish_status()
                except Exception:
                    logger.exception("status publish failed")
                next_status = now + status_period
            if now >= next_publish:
                try:
                    self._publish_world_driveable()
                except Exception:
                    logger.exception("world_driveable publish failed")
                next_publish = now + publish_period
            self._stop_event.wait(
                min(max(0.0, next_status - time.monotonic()),
                    max(0.0, next_publish - time.monotonic()), 0.2),
            )

    def _publish_world_driveable(self) -> None:
        if self._pub_world_driveable is None:
            return
        snap = self.reference_map.snapshot_for_ui()
        bounds = snap.get("bounds_ij")
        if bounds is None:
            return
        i0, i1, j0, j1 = bounds
        margin = 4
        i0 = max(0, i0 - margin)
        j0 = max(0, j0 - margin)
        i1 = min(self.reference_map.nx - 1, i1 + margin)
        j1 = min(self.reference_map.ny - 1, j1 + margin)
        sl = (slice(i0, i1 + 1), slice(j0, j1 + 1))
        drive = snap["driveable"][sl]
        meta = snap["meta"]
        import numpy as np
        crop = {
            "max_height_m": np.full(drive.shape, np.nan, dtype=np.float32),
            "driveable": drive,
            "observation_count": np.zeros(drive.shape, dtype=np.int32),
            "last_observed_ts": np.full(drive.shape, np.nan, dtype=np.float32),
            "traversed_ts": np.full(drive.shape, np.nan, dtype=np.float32),
            "origin_x_m": meta["origin_x_m"] + i0 * meta["resolution_m"],
            "origin_y_m": meta["origin_y_m"] + j0 * meta["resolution_m"],
            "nx": drive.shape[0],
            "ny": drive.shape[1],
            "resolution_m": meta["resolution_m"],
            "session_id": self.reference_map.session_id,
            "world_anchor_pose": (0.0, 0.0, 0.0),
            "bounds_m": None,
        }
        payload = encode_for_publish(
            crop, pose_source_name=self.pose_source.source_name(),
        )
        self._pub_world_driveable.put(json.dumps(payload).encode("utf-8"))

    def _publish_status(self) -> None:
        if self._pub_world_status is None:
            return
        latest = self.pose_source.latest_pose()
        pose_world = None
        if latest is not None:
            pose_world = {
                "x_m": float(latest[0][0]),
                "y_m": float(latest[0][1]),
                "theta_rad": float(latest[0][2]),
            }
        with self._lock:
            arr_od = list(self._arr_odom)
            last_od = self._last_odom_ts
            notes = self._notes
        payload = {
            "ts": _now(),
            "session_id": self.reference_map.session_id,
            "pose_source": self.pose_source.source_name(),
            "pose_world": pose_world,
            "input_rates_hz": {"odom": _rate_from_arrivals(arr_od)},
            "input_age_s": {"odom": _age(last_od)},
            "notes": notes,
        }
        self._pub_world_status.put(json.dumps(payload).encode("utf-8"))

    def snapshot_for_ui(self) -> Optional[Dict[str, Any]]:
        return self.reference_map.snapshot_for_ui()

    def pose_trail(self) -> List[Tuple[float, float, float]]:
        with self._lock:
            return [(x, y, th) for (x, y, th, _t) in self._pose_trail]

    def status_summary(self) -> Dict[str, Any]:
        with self._lock:
            arr_od = list(self._arr_odom)
            last_od = self._last_odom_ts
            notes = self._notes
        latest = self.pose_source.latest_pose()
        pose = latest[0] if latest is not None else None
        drive = self.reference_map.driveable_int8()
        cells_observed = int(np.count_nonzero(drive >= 0))
        return {
            "session_id": self.reference_map.session_id,
            "pose": pose,
            "rates": {"odom": _rate_from_arrivals(arr_od)},
            "ages": {"odom": _age(last_od), "local_map": None},
            "cells_observed": cells_observed,
            "cells_traversed": 0,
            "notes": notes,
            "pose_source": self.pose_source.source_name(),
            "correction_summary": self.pose_source.correction_summary(),
            "pose_unavail_streak": 0,
            "scan_match": (
                self.pose_source.scan_match_summary()
                if hasattr(self.pose_source, "scan_match_summary")
                else {}
            ),
        }

    def save_snapshot_bundle(self, base_dir: Optional[str] = None) -> str:
        from desktop.mapping.export import export_reference_map_bundle
        return export_reference_map_bundle(self, base_dir=base_dir)

    def load_snapshot(self, npz_path: str) -> Dict[str, Any]:
        """Restore a saved map into the live localizer."""
        self.reference_map = load_map_auto(npz_path)
        if isinstance(self.pose_source, MCLPoseSource):
            self.pose_source._refresh_map(self.reference_map)
            self.pose_source._mcl.set_reference_map(self.reference_map)
        drive = self.reference_map.driveable_int8()
        cells = int(np.count_nonzero(drive >= 0))
        return {
            "loaded_session_id": self.reference_map.session_id,
            "current_session_id": self.reference_map.session_id,
            "cells_observed": cells,
        }


class _GridShim:
    """Minimal compatibility for nav code expecting ``fuser.grid``."""

    def __init__(self, ref: ReferenceMap) -> None:
        self._ref = ref

    @property
    def session_id(self) -> str:
        return self._ref.session_id

    @property
    def resolution_m(self) -> float:
        return self._ref.resolution_m

    @property
    def n_cells(self) -> int:
        return self._ref.nx


def _rate_from_arrivals(ts_list: List[float]) -> Optional[float]:
    if len(ts_list) < 2:
        return None
    span = ts_list[-1] - ts_list[0]
    if span <= 0:
        return None
    return (len(ts_list) - 1) / span


def _age(ts: float) -> Optional[float]:
    if ts <= 0:
        return None
    return _now() - ts

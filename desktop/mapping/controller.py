"""Mapping session controller — EKF + pose-graph SLAM."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from desktop.fusion.ekf_pose_tracker import EkfPoseTracker
from desktop.fusion.load_slam_config import load_slam_config
from desktop.mapping.export import export_mapping_session
from desktop.mapping.pose_graph_mapper import PoseGraphMapper
from desktop.nav.slam.types import ImuReading
from desktop.reference_map.reference_map import ReferenceMap
from desktop.world_map.transport import open_session

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


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
        th = float(msg.get("theta", msg.get("theta_rad", msg.get("yaw", 0.0))))
        return ts, x, y, th
    except (TypeError, ValueError):
        return None


class MappingConfig:
    def __init__(
        self,
        *,
        router: str = "tcp/127.0.0.1:7447",
        extent_m: float = 33.0,
        resolution_m: float = 0.05,
        scan_match_hz: float = 2.0,
        map_stale_s: float = 2.0,
        ui_redraw_hz: float = 5.0,
    ):
        self.router = router
        self.extent_m = extent_m
        self.resolution_m = resolution_m
        self.scan_match_hz = scan_match_hz
        self.map_stale_s = map_stale_s
        self.ui_redraw_hz = ui_redraw_hz


class MappingController:
    """Builds a pose-graph map during a teleop mapping drive."""

    def __init__(self, config: MappingConfig):
        self.config = config
        slam_cfg = load_slam_config()
        self._slam_cfg = slam_cfg
        self.ekf = EkfPoseTracker(noise=slam_cfg.fusion)
        self.mapper = PoseGraphMapper(
            slam_config=slam_cfg.slam,
            fusion_config=slam_cfg.fusion,
            ekf=self.ekf,
        )
        self._session_id = uuid.uuid4().hex[:12]
        self.reference_map: Optional[ReferenceMap] = None
        self._lock = threading.RLock()
        self._session: Optional[Any] = None
        self._subs: List[Any] = []
        self._on_update: Optional[Callable[[], None]] = None

    @property
    def grid(self) -> MappingController:
        return self

    @property
    def session_id(self) -> str:
        return self._session_id

    def set_on_grid_update(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_update = cb

    def connect(self) -> Tuple[bool, Optional[str]]:
        if self._session is not None:
            return True, None
        try:
            self._session = open_session(self.config.router)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        self._subs.append(
            self._session.declare_subscriber("body/odom", self._on_odom),
        )
        self._subs.append(
            self._session.declare_subscriber("body/imu", self._on_imu),
        )
        self._subs.append(
            self._session.declare_subscriber("body/lidar/scan", self._on_scan),
        )
        return True, None

    def shutdown(self) -> None:
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                pass
        self._subs.clear()
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    disconnect = shutdown

    @property
    def connected(self) -> bool:
        return self._session is not None

    @property
    def session(self) -> Optional[Any]:
        return self._session

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
        self.ekf.update_odom(ts, x, y, theta)

    def _on_imu(self, sample: Any) -> None:
        msg = _decode_json(self._payload_bytes(sample))
        if msg is None:
            return
        reading = ImuReading.from_payload(msg)
        if reading is not None:
            self.ekf.update_imu(reading)

    def _on_scan(self, sample: Any) -> None:
        msg = _decode_json(self._payload_bytes(sample))
        if msg is None:
            return
        ts = float(msg.get("ts") or _now())
        ranges = msg.get("ranges")
        angle_min = float(msg.get("angle_min", 0.0))
        angle_inc = float(msg.get("angle_increment", 0.0))
        if not isinstance(ranges, list) or angle_inc <= 0:
            return
        angles = np.arange(len(ranges), dtype=np.float64) * angle_inc + angle_min
        ranges_arr = np.asarray(
            [r if isinstance(r, (int, float)) else np.nan for r in ranges],
            dtype=np.float64,
        )
        added = self.mapper.add_scan(ts, ranges_arr, angles)
        if added:
            cb = self._on_update
            if cb is not None:
                try:
                    cb()
                except Exception:
                    logger.exception("on_grid_update failed")

    def snapshot_for_ui(self) -> Optional[Dict[str, Any]]:
        snap = self.mapper.snapshot_for_ui()
        if snap is None:
            return None
        snap["session_id"] = self._session_id
        drive = snap["driveable"]
        snap["grid"] = np.full(drive.shape, np.nan, dtype=np.float32)
        return snap

    def pose_trail(self) -> List[Tuple[float, float, float]]:
        return self.mapper.trajectory()

    def status_summary(self) -> Dict[str, Any]:
        pose = self.mapper.display_pose()
        ekf_diag = self.ekf.diagnostics()
        slam_diag = self.mapper.diagnostics()
        cov = self.ekf.cov_at(_now())
        cov_xy = None
        if cov is not None:
            cov_xy = float(cov[0, 0] + cov[1, 1])
        return {
            "session_id": self._session_id,
            "pose": pose,
            "pose_source": "pose_graph",
            "imu_settled": ekf_diag.get("imu_settled"),
            "heading_source": ekf_diag.get("heading_source"),
            "yaw_at_misses": ekf_diag.get("yaw_at_misses"),
            "ekf_cov_trace_xy": cov_xy,
            "graph_nodes": slam_diag.get("node_count"),
            "graph_edges": slam_diag.get("edge_count"),
            "last_match_improvement": slam_diag.get("last_match_improvement"),
            "loop_closures": slam_diag.get("loop_closure_count"),
        }

    def finalize_map(self) -> ReferenceMap:
        ref = self.mapper.to_reference_map(
            session_id=self._session_id,
            metadata={"mapping_version": 2},
        )
        self.reference_map = ref
        return ref

    def save_snapshot_bundle(self, base_dir: Optional[str] = None) -> str:
        self.finalize_map()
        return export_mapping_session(self, base_dir=base_dir)

    def request_reset(self, *, reason: str = "ui_reset") -> None:
        self.ekf.rebind_world_to_current()
        self.mapper.reset()
        self._session_id = uuid.uuid4().hex[:12]
        self.reference_map = None

    def request_relocate(self, *, reason: str = "ui_relocate") -> dict:
        return {"success": False, "reason": "not_supported_in_mapping"}

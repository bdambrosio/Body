"""Mapping session controller — builds ReferenceMap from lidar."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from desktop.mapping.export import export_mapping_session
from desktop.mapping.mapping_pose_tracker import MappingPoseTracker
from desktop.mapping.occupancy_builder import OccupancyBuilder
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
        extent_m: float = 40.0,
        resolution_m: float = 0.05,
        scan_match_hz: float = 2.0,
    ):
        self.router = router
        self.extent_m = extent_m
        self.resolution_m = resolution_m
        self.scan_match_hz = scan_match_hz


class MappingController:
    """Builds occupancy during a teleop mapping drive."""

    def __init__(self, config: MappingConfig):
        self.config = config
        self.builder = OccupancyBuilder(
            extent_m=config.extent_m,
            resolution_m=config.resolution_m,
        )
        self.pose_tracker = MappingPoseTracker()
        self._session_id = uuid.uuid4().hex[:12]
        self.reference_map: Optional[ReferenceMap] = None
        self._trajectory: Deque[Tuple[float, float, float, float]] = deque(maxlen=4096)
        self._lock = threading.RLock()
        self._session: Optional[Any] = None
        self._subs: List[Any] = []
        self._last_scan_mono = 0.0
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
        self.pose_tracker.update_odom(ts, x, y, theta)
        pose = self.pose_tracker.pose()
        self._trajectory.append((ts, pose[0], pose[1], pose[2]))

    def _on_imu(self, sample: Any) -> None:
        msg = _decode_json(self._payload_bytes(sample))
        if msg is None:
            return
        reading = ImuReading.from_payload(msg)
        if reading is not None:
            self.pose_tracker.update_imu(reading)

    def _on_scan(self, sample: Any) -> None:
        now_mono = time.monotonic()
        if now_mono - self._last_scan_mono < 1.0 / max(0.5, self.config.scan_match_hz):
            return
        self._last_scan_mono = now_mono
        msg = _decode_json(self._payload_bytes(sample))
        if msg is None:
            return
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
        self.pose_tracker.try_scan_match(ranges_arr, angles, self.builder)
        pose = self.pose_tracker.pose()
        self.builder.integrate_scan(ranges_arr, angles, pose)
        cb = self._on_update
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.exception("on_grid_update failed")

    def snapshot_for_ui(self) -> Optional[Dict[str, Any]]:
        snap = self.builder.snapshot_for_ui()
        snap["session_id"] = self._session_id
        drive = snap["driveable"]
        snap["grid"] = np.full(drive.shape, np.nan, dtype=np.float32)
        return snap

    def pose_trail(self) -> List[Tuple[float, float, float]]:
        with self._lock:
            return [(x, y, th) for (_t, x, y, th) in self._trajectory]

    def status_summary(self) -> Dict[str, Any]:
        pose = self.pose_tracker.pose()
        return {
            "session_id": self._session_id,
            "pose": pose,
            "pose_source": "mapping",
        }

    def finalize_map(self) -> ReferenceMap:
        traj = None
        if self._trajectory:
            traj = np.array(list(self._trajectory), dtype=np.float64)
        ref = self.builder.to_reference_map(
            session_id=self._session_id,
            trajectory=traj,
            metadata={"mapping_version": 1},
        )
        self.reference_map = ref
        return ref

    def save_snapshot_bundle(self, base_dir: Optional[str] = None) -> str:
        self.finalize_map()
        return export_mapping_session(self, base_dir=base_dir)

    def request_reset(self, *, reason: str = "ui_reset") -> None:
        self.builder = OccupancyBuilder(
            extent_m=self.config.extent_m,
            resolution_m=self.config.resolution_m,
        )
        self.pose_tracker = MappingPoseTracker()
        self._session_id = uuid.uuid4().hex[:12]
        self._trajectory.clear()

    def request_relocate(self, *, reason: str = "ui_relocate") -> dict:
        return {"success": False, "reason": "not_supported_in_mapping"}

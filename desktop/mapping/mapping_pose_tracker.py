"""Pose tracking during mapping sessions (odom translation + IMU yaw)."""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, Optional, Tuple

from desktop.nav.slam.imu_yaw import ImuYawTracker
from desktop.nav.slam.types import ImuReading
from desktop.world_map.pose_source import OdomPose, Pose

PoseTuple = Tuple[float, float, float]


class MappingPoseTracker:
    """IMU yaw + encoder translation for mapping ray casts.

    Mirrors the pose half of ImuPlusScanMatchPose without online scan
    matching — scan match against a map being built fights rotation.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._odom = OdomPose()
        self._imu = ImuYawTracker()
        self._yaw_offset = 0.0
        self._seeded = False
        self._yaw_at_misses = 0
        self._last_heading_source = "none"

    def update_imu(self, reading: ImuReading) -> None:
        self._imu.update(reading)

    def update_odom(self, ts: float, x: float, y: float, theta: float) -> None:
        self._odom.update(ts, x, y, theta)
        if not self._seeded and self._imu.is_settled() and self._odom.latest_pose() is not None:
            self.rebind_world_to_current()

    def rebind_world_to_current(self) -> Optional[Pose]:
        """Anchor world frame at the current robot pose."""
        with self._lock:
            self._odom.rebind_world_to_current()
            latest_imu = self._imu.latest()
            if latest_imu is not None:
                _ts, yaw, _sigma = latest_imu
                self._yaw_offset = yaw
            else:
                self._yaw_offset = 0.0
            self._seeded = True
            self._yaw_at_misses = 0
            self._last_heading_source = "none"
        latest = self._odom.latest_pose()
        return latest[0] if latest is not None else None

    def is_ready(self) -> bool:
        with self._lock:
            return self._seeded and self._imu.is_settled()

    def pose_at(self, ts: float) -> Optional[Pose]:
        odom_pose = self._odom.pose_at(ts)
        if odom_pose is None:
            return None
        x_w, y_w, theta_enc = odom_pose
        yaw = self._yaw_at_world(ts)
        if yaw is None:
            with self._lock:
                self._yaw_at_misses += 1
                self._last_heading_source = "encoder"
            return (x_w, y_w, theta_enc)
        with self._lock:
            self._last_heading_source = "imu"
        return (x_w, y_w, yaw)

    def pose(self) -> PoseTuple:
        latest = self._odom.latest_pose()
        if latest is None:
            return (0.0, 0.0, 0.0)
        (_x, _y, _th), ts = latest
        at = self.pose_at(ts)
        return at if at is not None else (_x, _y, _th)

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "imu_settled": self._imu.is_settled(),
                "seeded": self._seeded,
                "heading_source": self._last_heading_source,
                "yaw_at_misses": self._yaw_at_misses,
            }

    def _yaw_at_world(self, ts: float) -> Optional[float]:
        if not self._imu.is_settled():
            return None
        result = self._imu.yaw_at(ts)
        if result is None:
            return None
        yaw_imu, _sigma = result
        return _wrap(yaw_imu - self._yaw_offset)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

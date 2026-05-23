"""Pose tracking during mapping sessions (IMU heading + IMU-projected translation)."""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from desktop.nav.slam.imu_yaw import ImuYawTracker
from desktop.nav.slam.types import ImuReading
from desktop.world_map.pose_source import OdomPose, Pose

PoseTuple = Tuple[float, float, float]
_OdomSample = Tuple[float, float, float, float]


class MappingPoseTracker:
    """Mapping pose: IMU yaw drives heading and world-frame translation.

    Encoder odom contributes forward distance (ds) only; wheel-reported
    (x, y) in the odom frame is not used for world position. Each ds
    step is projected with IMU heading at that timestamp so slip or
    lateral encoder error does not drag the map off the IMU bearing.
    """

    def __init__(self, *, buffer_seconds: float = 2.0) -> None:
        self._lock = threading.RLock()
        self._odom = OdomPose(buffer_seconds=buffer_seconds)
        self._imu = ImuYawTracker()
        self._yaw_offset = 0.0
        self._seeded = False
        self._yaw_at_misses = 0
        self._last_heading_source = "none"
        self._buf_seconds = buffer_seconds
        self._world_buf: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=256,
        )
        self._last_raw_odom: Optional[_OdomSample] = None

    def update_imu(self, reading: ImuReading) -> None:
        self._imu.update(reading)

    def update_odom(self, ts: float, x: float, y: float, theta: float) -> None:
        self._odom.update(ts, x, y, theta)
        if not self._seeded and self._imu.is_settled() and self._odom.latest_pose() is not None:
            self.rebind_world_to_current()
            return
        if not self._seeded:
            return
        self._integrate_raw_odom_sample(ts, x, y, theta)

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
            self._world_buf.clear()
            self._last_raw_odom = None

        latest = self._odom.latest_pose()
        if latest is None:
            return None
        _pose, ts = latest
        raw = self._odom.pose_at_in_odom_frame(ts)
        if raw is not None:
            with self._lock:
                self._last_raw_odom = (ts, raw[0], raw[1], raw[2])
                self._world_buf.append((ts, 0.0, 0.0, 0.0))
        return (0.0, 0.0, 0.0)

    def is_ready(self) -> bool:
        with self._lock:
            return self._seeded and self._imu.is_settled()

    def pose_at(self, ts: float) -> Optional[Pose]:
        if not self._seeded:
            return None
        yaw = self._yaw_at_world(ts)
        if yaw is None:
            with self._lock:
                self._yaw_at_misses += 1
                self._last_heading_source = "none"
            return None
        xy = self._interp_world_xy(ts)
        if xy is None:
            return None
        with self._lock:
            self._last_heading_source = "imu"
        return (xy[0], xy[1], yaw)

    def pose(self) -> PoseTuple:
        latest = self._odom.latest_pose()
        if latest is None:
            return (0.0, 0.0, 0.0)
        (_x, _y, _th), ts = latest
        at = self.pose_at(ts)
        return at if at is not None else (0.0, 0.0, 0.0)

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "imu_settled": self._imu.is_settled(),
                "seeded": self._seeded,
                "heading_source": self._last_heading_source,
                "yaw_at_misses": self._yaw_at_misses,
                "translation_source": "imu_projected_odom_ds",
            }

    def _integrate_raw_odom_sample(
        self, ts: float, x: float, y: float, theta: float,
    ) -> None:
        yaw = self._yaw_at_world(ts)
        with self._lock:
            if self._last_raw_odom is None:
                self._last_raw_odom = (ts, x, y, theta)
                if yaw is not None:
                    self._world_buf.append((ts, 0.0, 0.0, yaw))
                return

            _pts, lx, ly, lth = self._last_raw_odom
            dx = x - lx
            dy = y - ly
            dth = _wrap(theta - lth)
            th_mid = lth + 0.5 * dth
            ds = dx * math.cos(th_mid) + dy * math.sin(th_mid)

            if yaw is None:
                self._yaw_at_misses += 1
                self._last_heading_source = "none"
                self._last_raw_odom = (ts, x, y, theta)
                return

            if self._world_buf:
                x_w, y_w, _ = self._world_buf[-1][1:]
            else:
                x_w, y_w = 0.0, 0.0
            x_w += ds * math.cos(yaw)
            y_w += ds * math.sin(yaw)
            self._world_buf.append((ts, x_w, y_w, yaw))
            self._last_raw_odom = (ts, x, y, theta)
            self._last_heading_source = "imu"
            self._trim_world_buf()

    def _trim_world_buf(self) -> None:
        if len(self._world_buf) < 2:
            return
        cutoff = self._world_buf[-1][0] - self._buf_seconds
        while len(self._world_buf) > 2 and self._world_buf[0][0] < cutoff:
            self._world_buf.popleft()

    def _interp_world_xy(self, ts: float) -> Optional[Tuple[float, float]]:
        with self._lock:
            n = len(self._world_buf)
            if n == 0:
                return None
            if n == 1:
                return (self._world_buf[0][1], self._world_buf[0][2])

            first_ts = self._world_buf[0][0]
            last_ts = self._world_buf[-1][0]
            if ts <= first_ts:
                grace = (last_ts - first_ts) / max(1, n - 1) * 0.5
                if first_ts - ts <= grace:
                    return (self._world_buf[0][1], self._world_buf[0][2])
                return None
            if ts >= last_ts:
                grace = (last_ts - first_ts) / max(1, n - 1) * 0.5
                if ts - last_ts <= grace:
                    return (self._world_buf[-1][1], self._world_buf[-1][2])
                return None

            lo, hi = 0, n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if self._world_buf[mid][0] <= ts:
                    lo = mid
                else:
                    hi = mid
            t0, x0, y0, _ = self._world_buf[lo]
            t1, x1, y1, _ = self._world_buf[hi]
            if t1 == t0:
                return (x1, y1)
            alpha = (ts - t0) / (t1 - t0)
            return (x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0))

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

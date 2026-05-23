"""2D EKF pose tracker — IMU predict, encoder odom + IMU yaw update."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from desktop.fusion.load_slam_config import FusionNoiseConfig
from desktop.nav.slam.imu_yaw import ImuYawTracker
from desktop.nav.slam.types import ImuReading
from desktop.world_map.pose_source import OdomPose, Pose

PoseTuple = Tuple[float, float, float]
_OdomSample = Tuple[float, float, float, float]
_StateSample = Tuple[float, float, float, float, float, float, float, float, float, float]
# ts, x, y, theta, P00, P01, P02, P11, P12, P22


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class EkfPoseTrackerConfig:
    noise: FusionNoiseConfig
    buffer_seconds: float = 2.0


class EkfPoseTracker:
    """Fuse IMU yaw with encoder forward motion (Thrun diff-drive noise).

    The IMU owns heading; encoder contributes forward ``ds`` only, projected
    with IMU world yaw at each odom tick. Maintains a 3×3 covariance on
    (x, y, θ) for diagnostics and SLAM edge information.
    """

    def __init__(
        self,
        *,
        config: Optional[EkfPoseTrackerConfig] = None,
        noise: Optional[FusionNoiseConfig] = None,
        buffer_seconds: float = 2.0,
    ) -> None:
        if config is not None:
            self._noise = config.noise
            self._buf_seconds = config.buffer_seconds
        else:
            self._noise = noise or FusionNoiseConfig()
            self._buf_seconds = buffer_seconds

        self._lock = threading.RLock()
        self._odom = OdomPose(buffer_seconds=buffer_seconds)
        self._imu = ImuYawTracker(
            min_settle_samples=max(20, int(self._noise.imu_settle_time_s * 100)),
        )
        self._yaw_offset = 0.0
        self._seeded = False
        self._yaw_at_misses = 0
        self._last_heading_source = "none"
        self._last_raw_odom: Optional[_OdomSample] = None
        self._state_buf: Deque[_StateSample] = deque(maxlen=512)

        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._P = np.eye(3, dtype=np.float64) * 1e-4

        self._predict_count = 0
        self._imu_update_count = 0

    def update_imu(self, reading: ImuReading) -> None:
        self._imu.update(reading)

    def update_odom(self, ts: float, x: float, y: float, theta: float) -> None:
        self._odom.update(ts, x, y, theta)
        if not self._seeded and self._imu.is_settled() and self._odom.latest_pose() is not None:
            self.rebind_world_to_current()
            return
        if not self._seeded:
            return
        self._integrate_odom(ts, x, y, theta)

    def rebind_world_to_current(self) -> Optional[Pose]:
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
            self._state_buf.clear()
            self._last_raw_odom = None
            self._x = 0.0
            self._y = 0.0
            self._theta = 0.0
            self._P = np.eye(3, dtype=np.float64) * 1e-4

        latest = self._odom.latest_pose()
        if latest is None:
            return None
        _pose, ots = latest
        raw = self._odom.pose_at_in_odom_frame(ots)
        if raw is not None:
            yaw = self._yaw_at_world(ots)
            if yaw is not None:
                with self._lock:
                    self._theta = yaw
                    self._last_raw_odom = (ots, raw[0], raw[1], raw[2])
                    self._append_state(ots)
        return (0.0, 0.0, 0.0)

    def is_ready(self) -> bool:
        with self._lock:
            return self._seeded and self._imu.is_settled()

    def pose_at(self, ts: float) -> Optional[Pose]:
        if not self._seeded:
            return None
        with self._lock:
            sample = self._interp_state(ts)
            if sample is None:
                return None
            return (sample[1], sample[2], sample[3])

    def cov_at(self, ts: float) -> Optional[np.ndarray]:
        if not self._seeded:
            return None
        with self._lock:
            sample = self._interp_state(ts)
            if sample is None:
                return None
            return np.array(
                [
                    [sample[4], sample[5], sample[6]],
                    [sample[5], sample[7], sample[8]],
                    [sample[6], sample[8], sample[9]],
                ],
                dtype=np.float64,
            )

    def pose(self) -> PoseTuple:
        latest = self._odom.latest_pose()
        if latest is None:
            return (0.0, 0.0, 0.0)
        (_x, _y, _th), ts = latest
        at = self.pose_at(ts)
        return at if at is not None else (self._x, self._y, self._theta)

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            p = self._P.copy()
            return {
                "imu_settled": self._imu.is_settled(),
                "seeded": self._seeded,
                "heading_source": self._last_heading_source,
                "yaw_at_misses": self._yaw_at_misses,
                "translation_source": "ekf_imu_projected_odom_ds",
                "cov_trace_xy": float(p[0, 0] + p[1, 1]),
                "cov_trace": float(np.trace(p)),
                "predict_count": self._predict_count,
                "imu_update_count": self._imu_update_count,
            }

    def _integrate_odom(
        self, ts: float, x: float, y: float, theta: float,
    ) -> None:
        yaw = self._yaw_at_world(ts)
        with self._lock:
            if self._last_raw_odom is None:
                self._last_raw_odom = (ts, x, y, theta)
                if yaw is not None:
                    self._theta = yaw
                    self._append_state(ts)
                return

            _pts, lx, ly, lth = self._last_raw_odom
            dx = x - lx
            dy = y - ly
            dth_enc = _wrap(theta - lth)
            th_mid = lth + 0.5 * dth_enc
            ds = dx * math.cos(th_mid) + dy * math.sin(th_mid)

            if yaw is None:
                self._yaw_at_misses += 1
                self._last_heading_source = "none"
                self._last_raw_odom = (ts, x, y, theta)
                return

            self._predict_motion(ds, dth_enc, yaw)
            self._update_imu_yaw(yaw)
            self._append_state(ts)
            self._last_raw_odom = (ts, x, y, theta)
            self._last_heading_source = "imu"
            self._trim_state_buf()

    def _predict_motion(self, ds: float, dth_enc: float, yaw: float) -> None:
        """Propagate (x,y) with ds along IMU yaw; grow covariance."""
        c = math.cos(yaw)
        s = math.sin(yaw)
        self._x += ds * c
        self._y += ds * s
        self._theta = yaw

        n = self._noise
        sigma_trans = n.alpha_trans_per_m * abs(ds) + n.alpha_rot_per_m * abs(dth_enc)
        sigma_rot = n.alpha_rot_per_m * abs(ds) + n.alpha_rot_per_rad * abs(dth_enc)
        q_xy = max(sigma_trans ** 2, 1e-8)
        q_th = max(sigma_rot ** 2, 1e-8)

        G = np.array([[1.0, 0.0, -ds * s], [0.0, 1.0, ds * c], [0.0, 0.0, 1.0]])
        Q = np.diag([q_xy, q_xy, q_th])
        self._P = G @ self._P @ G.T + Q
        self._predict_count += 1

    def _update_imu_yaw(self, yaw: float) -> None:
        """Kalman update: θ observation from IMU (tight R)."""
        z = yaw
        H = np.array([[0.0, 0.0, 1.0]])
        R = np.array([[self._noise.imu_sigma_rad ** 2]])
        innov = _wrap(z - self._theta)
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        delta = K @ np.array([[innov]])
        self._x += float(delta[0, 0])
        self._y += float(delta[1, 0])
        self._theta = _wrap(self._theta + float(delta[2, 0]))
        I = np.eye(3)
        self._P = (I - K @ H) @ self._P
        self._imu_update_count += 1

    def _append_state(self, ts: float) -> None:
        p = self._P
        self._state_buf.append(
            (
                ts,
                self._x,
                self._y,
                self._theta,
                float(p[0, 0]),
                float(p[0, 1]),
                float(p[0, 2]),
                float(p[1, 1]),
                float(p[1, 2]),
                float(p[2, 2]),
            ),
        )

    def _trim_state_buf(self) -> None:
        if len(self._state_buf) < 2:
            return
        cutoff = self._state_buf[-1][0] - self._buf_seconds
        while len(self._state_buf) > 2 and self._state_buf[0][0] < cutoff:
            self._state_buf.popleft()

    def _interp_state(self, ts: float) -> Optional[_StateSample]:
        n = len(self._state_buf)
        if n == 0:
            return None
        if n == 1:
            return self._state_buf[0]

        first_ts = self._state_buf[0][0]
        last_ts = self._state_buf[-1][0]
        if ts <= first_ts:
            grace = (last_ts - first_ts) / max(1, n - 1) * 0.5
            if first_ts - ts <= grace:
                return self._state_buf[0]
            return None
        if ts >= last_ts:
            grace = (last_ts - first_ts) / max(1, n - 1) * 0.5
            if ts - last_ts <= grace:
                return self._state_buf[-1]
            return None

        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._state_buf[mid][0] <= ts:
                lo = mid
            else:
                hi = mid
        s0 = self._state_buf[lo]
        s1 = self._state_buf[hi]
        t0, t1 = s0[0], s1[0]
        if t1 == t0:
            return s1
        alpha = (ts - t0) / (t1 - t0)
        out = list(s0)
        out[0] = ts
        for i in range(1, 10):
            out[i] = s0[i] + alpha * (s1[i] - s0[i])
        out[3] = _wrap(out[3])
        return tuple(out)  # type: ignore[return-value]

    def _yaw_at_world(self, ts: float) -> Optional[float]:
        if not self._imu.is_settled():
            return None
        result = self._imu.yaw_at(ts)
        if result is None:
            return None
        yaw_imu, _sigma = result
        return _wrap(yaw_imu - self._yaw_offset)

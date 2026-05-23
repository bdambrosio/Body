"""MCLPoseSource — PoseSource backed by MCLLocalizer + frozen ReferenceMap."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from desktop.fusion.ekf_pose_tracker import EkfPoseTracker
from desktop.fusion.load_slam_config import load_slam_config
from desktop.localization.mcl_localizer import MCLConfig, MCLLocalizer
from desktop.localization.pose_buffer import PoseBuffer
from desktop.nav.slam.imu_yaw import ImuYawTracker
from desktop.nav.slam.types import ImuReading
from desktop.reference_map.reference_map import ReferenceMap
from desktop.world_map.particle_filter_pose import ParticleFilterConfig
from desktop.world_map.pose_source import Pose, PoseSource

logger = logging.getLogger(__name__)


@dataclass
class MCLPoseSourceConfig:
    scan_hz: float = 5.0
    imu_obs_hz: float = 5.0
    teleport_distance_m: float = 0.5
    teleport_rotation_rad: float = math.radians(45.0)
    relocate_max_scan_age_s: float = 2.0


class MCLPoseSource(PoseSource):
    IMU_TOPIC = "body/imu"
    SCAN_TOPIC = "body/lidar/scan"

    def __init__(
        self,
        reference_map: ReferenceMap,
        *,
        pf_config: Optional[ParticleFilterConfig] = None,
        mcl_config: Optional[MCLConfig] = None,
        config: Optional[MCLPoseSourceConfig] = None,
    ) -> None:
        self._map = reference_map
        self._mcl = MCLLocalizer(reference_map, pf_config=pf_config, config=mcl_config)
        self._config = config or MCLPoseSourceConfig()
        self._ekf = EkfPoseTracker(noise=load_slam_config().fusion)
        self._imu_tracker = ImuYawTracker()
        self._pose_buffer = PoseBuffer()
        self._lock = threading.RLock()
        self._scan_rate_lock = threading.Lock()

        self._seeded = False
        self._last_odom: Optional[Tuple[float, float, float, float]] = None
        self._last_ekf_pose: Optional[Pose] = None
        self._off_x = 0.0
        self._off_y = 0.0
        self._off_theta = 0.0
        self._yaw_offset = 0.0

        self._session: Optional[Any] = None
        self._subs: List[Any] = []
        self._last_scan_mono = 0.0
        self._last_imu_obs_mono = 0.0
        self._last_scan_ts = 0.0
        self._last_ranges: Optional[np.ndarray] = None
        self._last_angles: Optional[np.ndarray] = None

        self._counters: Dict[str, int] = {
            "odom_seen": 0,
            "predicts_run": 0,
            "teleports": 0,
            "imu_received": 0,
            "imu_obs_applied": 0,
            "scan_received": 0,
            "scan_obs_run": 0,
            "resamples_fired": 0,
        }
        self._correction_total_m = 0.0
        self._correction_n_applied = 0

    def update(self, ts: float, x: float, y: float, theta: float) -> None:
        self._counters["odom_seen"] += 1
        self._ekf.update_odom(ts, x, y, theta)
        with self._lock:
            if not self._seeded:
                if not self._ekf.is_ready():
                    return
                ekf_pose = self._ekf.pose_at(ts)
                if ekf_pose is None:
                    return
                self._mcl.seed_at(0.0, 0.0, 0.0)
                self._off_x, self._off_y, self._off_theta = x, y, theta
                imu = self._imu_tracker.yaw_at(ts)
                if imu is not None:
                    self._yaw_offset = _wrap(imu[0])
                self._last_odom = (ts, x, y, theta)
                self._last_ekf_pose = ekf_pose
                self._seeded = True
                self._record_pose(ts)
                logger.info("mcl_pose: seeded at odom (%+.2f, %+.2f)", x, y)
                return

            assert self._last_odom is not None
            _, last_xo, last_yo, last_tho = self._last_odom
            dx = x - last_xo
            dy = y - last_yo
            dth = _wrap(theta - last_tho)
            if (math.hypot(dx, dy) > self._config.teleport_distance_m
                    or abs(dth) > self._config.teleport_rotation_rad):
                self._counters["teleports"] += 1
                self._last_odom = (ts, x, y, theta)
                self._last_ekf_pose = self._ekf.pose_at(ts)
                return

            ekf_pose = self._ekf.pose_at(ts)
            if ekf_pose is None or self._last_ekf_pose is None:
                self._last_odom = (ts, x, y, theta)
                return

            edx = ekf_pose[0] - self._last_ekf_pose[0]
            edy = ekf_pose[1] - self._last_ekf_pose[1]
            edth = _wrap(ekf_pose[2] - self._last_ekf_pose[2])
            th_mid = self._last_ekf_pose[2] + 0.5 * edth
            ds = edx * math.cos(th_mid) + edy * math.sin(th_mid)
            self._mcl.predict(ds, edth)

            imu = self._imu_tracker.yaw_at(ts)
            if imu is not None:
                hz = float(self._config.imu_obs_hz)
                now_mono = time.monotonic()
                if hz > 0 and now_mono - self._last_imu_obs_mono >= 1.0 / hz:
                    world_yaw = _wrap(imu[0] - self._yaw_offset)
                    self._mcl.observe_imu_yaw(world_yaw)
                    self._last_imu_obs_mono = now_mono
                    self._counters["imu_obs_applied"] += 1

            self._counters["predicts_run"] += 1
            self._last_odom = (ts, x, y, theta)
            self._last_ekf_pose = ekf_pose
            self._record_pose(ts)

    def _record_pose(self, ts: float) -> None:
        pose = self._mcl.posterior_mean()
        cov = self._mcl.posterior_cov()
        self._pose_buffer.append(ts, pose, cov)

    def latest_pose(self) -> Optional[Tuple[Pose, float]]:
        with self._lock:
            if not self._seeded or self._last_odom is None:
                return None
            return self._mcl.posterior_mean(), self._last_odom[0]

    def pose_at(self, ts: float) -> Optional[Pose]:
        buf_pose = self._pose_buffer.pose_at(ts)
        if buf_pose is not None:
            return buf_pose
        latest = self.latest_pose()
        return latest[0] if latest else None

    def best_pose_at(self, ts: float) -> Optional[Pose]:
        return self.pose_at(ts)

    def cov_at(self, ts: float) -> Optional[np.ndarray]:
        cov = self._pose_buffer.cov_at(ts)
        if cov is not None:
            return cov
        with self._lock:
            if not self._seeded:
                return None
            return self._mcl.posterior_cov()

    def rebind_world_to_current(self) -> Optional[Pose]:
        with self._lock:
            if not self._seeded or self._last_odom is None:
                return None
            ts, x, y, theta = self._last_odom
            imu = self._imu_tracker.yaw_at(ts)
            self._ekf.rebind_world_to_current()
            self._mcl.seed_at(0.0, 0.0, 0.0)
            self._off_x, self._off_y, self._off_theta = x, y, theta
            if imu is not None:
                self._yaw_offset = _wrap(imu[0])
            self._last_ekf_pose = self._ekf.pose_at(ts)
            self._pose_buffer.clear()
            self._record_pose(ts)
            self._correction_total_m = 0.0
            self._correction_n_applied = 0
            return (x, y, theta)

    def to_world(self, x_o: float, y_o: float, th_o: float) -> Pose:
        dx = x_o - self._off_x
        dy = y_o - self._off_y
        c, s = math.cos(-self._off_theta), math.sin(-self._off_theta)
        x_w = c * dx - s * dy
        y_w = s * dx + c * dy
        th_w = _wrap(th_o - self._off_theta)
        return (x_w, y_w, th_w)

    def source_name(self) -> str:
        return "mcl"

    def match_summary(self) -> dict:
        return dict(self._counters)

    def correction_summary(self) -> dict:
        with self._lock:
            return {
                "total_m": float(self._correction_total_m),
                "total_rad": 0.0,
                "n_applied": int(self._correction_n_applied),
            }

    def relocate(self) -> dict:
        with self._lock:
            if not self._seeded:
                return {"success": False, "reason": "not_seeded"}
            ranges = self._last_ranges
            angles = self._last_angles
            scan_ts = self._last_scan_ts
            prior = self._mcl.posterior_mean()
        if ranges is None or angles is None or scan_ts <= 0:
            return {"success": False, "reason": "no_scan_cached"}
        if time.time() - scan_ts > self._config.relocate_max_scan_age_s:
            return {"success": False, "reason": "scan_too_stale"}

        self._mcl.spray_particles(
            prior[0], prior[1], prior[2],
            sigma_xy_m=self._mcl._config.relocate_seed_sigma_xy_m,
            sigma_theta_rad=self._mcl._config.relocate_seed_sigma_theta_rad,
        )
        with self._lock:
            self._mcl.observe_scan_ranges(ranges, angles)
            resampled = self._mcl.maybe_resample()
            if resampled:
                self._counters["resamples_fired"] += 1
            new_pose = self._mcl.posterior_mean()
            dx = new_pose[0] - prior[0]
            dy = new_pose[1] - prior[1]
            self._correction_total_m += math.hypot(dx, dy)
            self._correction_n_applied += 1
            if self._last_odom is not None:
                self._record_pose(self._last_odom[0])

        logger.info(
            "mcl_pose.relocate: dx=%+.2f dy=%+.2f",
            new_pose[0] - prior[0], new_pose[1] - prior[1],
        )
        return {
            "success": True,
            "method": "mcl",
            "dx": float(new_pose[0] - prior[0]),
            "dy": float(new_pose[1] - prior[1]),
            "dtheta": float(_wrap(new_pose[2] - prior[2])),
            "prior_pose": list(prior),
            "best_pose": list(new_pose),
            "particle_count": int(self._mcl.filter.n_particles()),
        }

    def connect(self, session: Any) -> None:
        if self._subs:
            return
        self._session = session
        self._subs.append(session.declare_subscriber(self.IMU_TOPIC, self._on_imu))
        self._subs.append(session.declare_subscriber(self.SCAN_TOPIC, self._on_scan))

    def disconnect(self) -> None:
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                pass
        self._subs.clear()
        self._session = None

    def _payload_bytes(self, sample: Any) -> bytes:
        try:
            return bytes(sample.payload.to_bytes())
        except AttributeError:
            return bytes(sample.payload)

    def _on_imu(self, sample: Any) -> None:
        self._counters["imu_received"] += 1
        try:
            msg = json.loads(self._payload_bytes(sample).decode("utf-8"))
        except Exception:
            return
        reading = ImuReading.from_payload(msg)
        if reading is not None:
            self._imu_tracker.update(reading)
            self._ekf.update_imu(reading)

    def _on_scan(self, sample: Any) -> None:
        self._counters["scan_received"] += 1
        with self._scan_rate_lock:
            now_mono = time.monotonic()
            if now_mono - self._last_scan_mono < 1.0 / max(0.1, self._config.scan_hz):
                return
            self._last_scan_mono = now_mono

        try:
            msg = json.loads(self._payload_bytes(sample).decode("utf-8"))
        except Exception:
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
        scan_ts = float(msg.get("ts") or time.time())

        with self._lock:
            if not self._seeded:
                return
            self._last_scan_ts = scan_ts
            self._last_ranges = ranges_arr
            self._last_angles = angles
            self._mcl.observe_scan_ranges(ranges_arr, angles)
            if self._mcl.maybe_resample():
                self._counters["resamples_fired"] += 1
            if self._last_odom is not None:
                self._record_pose(self._last_odom[0])
            self._counters["scan_obs_run"] += 1


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

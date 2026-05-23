"""Pose tracking during mapping sessions (odom + IMU + scan vs building map)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from desktop.mapping.occupancy_builder import OccupancyBuilder
from desktop.nav.slam.imu_yaw import ImuYawTracker
from desktop.nav.slam.scan_matcher import ScanMatcher, ScanMatcherConfig, lidar_scan_to_xy
from desktop.nav.slam.types import Pose2D


class MappingPoseTracker:
    """Point-estimate pose for mapping-only ray casting."""

    def __init__(self) -> None:
        self._imu = ImuYawTracker()
        self._matcher = ScanMatcher(ScanMatcherConfig(
            xy_half_m=0.25,
            theta_half_rad=math.radians(10.0),
            min_improvement=3.0,
        ))
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._yaw_offset = 0.0
        self._off_x = 0.0
        self._off_y = 0.0
        self._off_theta = 0.0
        self._seeded = False
        self._last_odom: Optional[Tuple[float, float, float, float]] = None

    def update_imu(self, reading) -> None:
        self._imu.update(reading)

    def update_odom(self, ts: float, x: float, y: float, theta: float) -> None:
        if not self._seeded:
            imu = self._imu.yaw_at(ts)
            if imu is None:
                return
            self._off_x, self._off_y, self._off_theta = x, y, theta
            self._yaw_offset = imu[0]
            self._x, self._y, self._theta = 0.0, 0.0, 0.0
            self._seeded = True
            self._last_odom = (ts, x, y, theta)
            return
        assert self._last_odom is not None
        _, lx, ly, lth = self._last_odom
        dx = x - lx
        dy = y - ly
        dth = _wrap(theta - lth)
        th_mid = lth + 0.5 * dth
        ds = dx * math.cos(th_mid) + dy * math.sin(th_mid)
        c, s = math.cos(self._theta), math.sin(self._theta)
        self._x += ds * c
        self._y += ds * s
        imu = self._imu.yaw_at(ts)
        if imu is not None:
            self._theta = _wrap(imu[0] - self._yaw_offset)
        else:
            self._theta = _wrap(self._theta + dth)
        self._last_odom = (ts, x, y, theta)

    def pose(self) -> Tuple[float, float, float]:
        return (self._x, self._y, self._theta)

    def try_scan_match(
        self,
        ranges_m: np.ndarray,
        angles_rad: np.ndarray,
        builder: OccupancyBuilder,
    ) -> None:
        if not self._seeded:
            return
        occ = builder.occupied_mask()
        if int(occ.sum()) < 50:
            return
        evidence = occ.astype(np.float32)
        points = lidar_scan_to_xy(ranges_m, angles_rad)
        if points.shape[0] < 10:
            return
        prior = Pose2D(*self.pose())
        result = self._matcher.search(
            points, prior, evidence,
            builder.origin_x_m, builder.origin_y_m, builder.resolution_m,
        )
        if result.accepted and not result.search_exhausted:
            self._x = result.pose.x
            self._y = result.pose.y
            self._theta = result.pose.theta


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

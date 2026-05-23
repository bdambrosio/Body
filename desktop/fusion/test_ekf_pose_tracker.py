"""Tests for EkfPoseTracker."""

from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.fusion.ekf_pose_tracker import EkfPoseTracker
from desktop.fusion.load_slam_config import FusionNoiseConfig
from desktop.nav.slam.types import FusionMode, ImuReading


def _quat_for_yaw(yaw_rad: float):
    return (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))


def _imu(
    ts: float,
    yaw: float,
    *,
    mag_yaw: float | None = None,
    mag_valid: bool = False,
    mag_accuracy_rad: float = 0.035,
) -> ImuReading:
    mag_quat = None
    if mag_yaw is not None:
        mag_quat = _quat_for_yaw(mag_yaw)
    return ImuReading(
        ts=ts,
        gyro_z=0.0,
        quat_wxyz=_quat_for_yaw(yaw),
        fusion_mode=FusionMode.GAME_ROTATION_VECTOR,
        accuracy_rad=0.175,
        mag_valid=mag_valid,
        mag_quat_wxyz=mag_quat,
        mag_accuracy_rad=mag_accuracy_rad if mag_valid else None,
    )


class TestEkfPoseTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.ekf = EkfPoseTracker(noise=FusionNoiseConfig())

    def _settle(self, t0: float = 100.0) -> None:
        for i in range(250):
            ts = t0 + i * 0.01
            self.ekf.update_imu(_imu(ts, 0.0))
        self.ekf.update_odom(t0 + 2.5, 0.0, 0.0, 0.0)

    def test_rotation_in_place(self) -> None:
        self._settle()
        t = 103.0
        for i in range(50):
            yaw = math.radians(i * 2.0)
            ts = t + i * 0.01
            self.ekf.update_imu(_imu(ts, yaw))
            self.ekf.update_odom(ts, 0.0, 0.0, 0.0)
        pose = self.ekf.pose_at(t + 0.49)
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose[0], 0.0, places=2)
        self.assertAlmostEqual(pose[1], 0.0, places=2)
        self.assertAlmostEqual(pose[2], math.radians(98.0), places=1)

    def test_straight_line_grows_covariance(self) -> None:
        self._settle()
        t = 103.0
        x = 0.0
        for i in range(20):
            ts = t + i * 0.02
            x += 0.05
            self.ekf.update_imu(_imu(ts, 0.0))
            self.ekf.update_odom(ts, x, 0.0, 0.0)
        cov0 = self.ekf.cov_at(t)
        cov1 = self.ekf.cov_at(t + 0.38)
        self.assertIsNotNone(cov0)
        self.assertIsNotNone(cov1)
        assert cov0 is not None and cov1 is not None
        self.assertGreater(float(np.trace(cov1)), float(np.trace(cov0)))

    def test_encoder_slip_does_not_dominate_theta(self) -> None:
        self._settle()
        t = 103.0
        for i in range(30):
            ts = t + i * 0.02
            self.ekf.update_imu(_imu(ts, 0.0))
            enc_th = math.radians(30.0)
            self.ekf.update_odom(ts, 0.1 * i, 0.0, enc_th)
        pose = self.ekf.pose_at(t + 0.58)
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertLess(abs(pose[2]), math.radians(5.0))

    def test_pose_at_interpolation(self) -> None:
        self._settle()
        t = 103.0
        for i, x in enumerate([0.0, 0.5, 1.0]):
            ts = t + i * 0.1
            self.ekf.update_imu(_imu(ts, 0.0))
            self.ekf.update_odom(ts, x, 0.0, 0.0)
        mid = self.ekf.pose_at(t + 0.05)
        self.assertIsNotNone(mid)
        assert mid is not None
        self.assertAlmostEqual(mid[0], 0.25, places=1)

    def test_mag_idle_corrects_game_yaw_drift(self) -> None:
        noise = FusionNoiseConfig(mag_update_min_interval_s=0.0)
        ekf = EkfPoseTracker(noise=noise)
        for i in range(250):
            ts = 100.0 + i * 0.01
            ekf.update_imu(_imu(ts, 0.0))
        ekf.update_odom(102.5, 0.0, 0.0, 0.0)

        t = 103.0
        for i in range(40):
            ts = t + i * 0.02
            game_yaw = math.radians(i * 0.5)
            ekf.update_imu(
                _imu(ts, game_yaw, mag_yaw=0.0, mag_valid=True, mag_accuracy_rad=0.035),
            )
            ekf.update_odom(ts, 0.0, 0.0, 0.0)

        pose = ekf.pose_at(t + 0.78)
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertLess(abs(pose[2]), math.radians(8.0))
        self.assertGreater(ekf.diagnostics()["mag_update_count"], 0)

    def test_mag_invalid_is_ignored(self) -> None:
        noise = FusionNoiseConfig(mag_update_min_interval_s=0.0)
        ekf = EkfPoseTracker(noise=noise)
        for i in range(250):
            ts = 100.0 + i * 0.01
            ekf.update_imu(_imu(ts, 0.0))
        ekf.update_odom(102.5, 0.0, 0.0, 0.0)

        t = 103.0
        for i in range(20):
            ts = t + i * 0.02
            game_yaw = math.radians(i * 2.0)
            ekf.update_imu(_imu(ts, game_yaw, mag_yaw=0.0, mag_valid=False))
            ekf.update_odom(ts, 0.0, 0.0, 0.0)

        pose = ekf.pose_at(t + 0.38)
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertGreater(abs(pose[2]), math.radians(20.0))
        self.assertEqual(ekf.diagnostics()["mag_update_count"], 0)


if __name__ == "__main__":
    unittest.main()

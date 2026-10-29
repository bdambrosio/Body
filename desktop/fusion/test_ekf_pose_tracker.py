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
    session_id: str | None = None,
    seq: int | None = None,
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
        session_id=session_id,
        seq=seq,
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
        noise = FusionNoiseConfig(mag_update_min_interval_s=0.0,
                                  mag_idle_fusion_enabled=True)
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

    def test_mag_idle_disabled_by_default(self) -> None:
        self.assertFalse(FusionNoiseConfig().mag_idle_fusion_enabled)
        ekf = EkfPoseTracker(noise=FusionNoiseConfig(mag_update_min_interval_s=0.0))
        for i in range(250):
            ekf.update_imu(_imu(100.0 + i * 0.01, 0.0))
        ekf.update_odom(102.5, 0.0, 0.0, 0.0)
        for i in range(20):
            ts = 103.0 + i * 0.02
            ekf.update_imu(_imu(ts, 0.0, mag_yaw=0.5, mag_valid=True))
            ekf.update_odom(ts, 0.0, 0.0, 0.0)
        self.assertEqual(ekf.diagnostics()["mag_update_count"], 0)

    def test_ds_not_dropped_without_imu_yaw(self) -> None:
        # S1: odom ticks with no IMU sample covering their ts must still
        # contribute translation (wheel heading), not be discarded.
        self._settle()
        t = 103.0
        self.ekf.update_imu(_imu(t, 0.0))
        self.ekf.update_odom(t, 0.0, 0.0, 0.0)
        for i in range(1, 21):                     # IMU silent for 0.4 s
            self.ekf.update_odom(t + i * 0.02, 0.05 * i, 0.0, 0.0)
        pose = self.ekf.pose_at(t + 0.40)
        assert pose is not None
        self.assertAlmostEqual(pose[0], 1.0, places=3)
        self.assertGreater(self.ekf.diagnostics()["wheel_heading_steps"], 0)

    def test_wheel_heading_used_when_imu_missing(self) -> None:
        self._settle()
        t = 103.0
        self.ekf.update_imu(_imu(t, 0.0))
        self.ekf.update_odom(t, 0.0, 0.0, 0.0)
        for i in range(1, 11):                     # arc: wheels rotate 0.5 rad
            self.ekf.update_odom(t + i * 0.02, 0.0, 0.0, 0.05 * i)
        pose = self.ekf.pose_at(t + 0.20)
        assert pose is not None
        self.assertAlmostEqual(pose[2], 0.5, places=3)

    def test_odom_session_change_rebaselines(self) -> None:
        # X4: motor_controller restart → odom back to (0,0,0). The EKF must
        # not integrate the -3 m jump.
        self._settle()
        t = 103.0
        for i in range(31):
            ts = t + i * 0.02
            self.ekf.update_imu(_imu(ts, 0.0))
            self.ekf.update_odom(ts, 0.1 * i, 0.0, 0.0, session_id="aaaa", seq=i)
        before = self.ekf.pose_at(t + 0.60)
        assert before is not None
        ts = t + 0.62
        self.ekf.update_imu(_imu(ts, 0.0))
        reset = self.ekf.update_odom(ts, 0.0, 0.0, 0.0, session_id="bbbb", seq=0)
        self.assertTrue(reset)
        after = self.ekf.pose_at(ts)
        assert after is not None
        self.assertAlmostEqual(after[0], before[0], places=3)
        # Integration continues from the new baseline.
        ts2 = ts + 0.02
        self.ekf.update_imu(_imu(ts2, 0.0))
        self.ekf.update_odom(ts2, 0.1, 0.0, 0.0, session_id="bbbb", seq=1)
        p2 = self.ekf.pose_at(ts2)
        assert p2 is not None
        self.assertAlmostEqual(p2[0], before[0] + 0.1, places=3)
        self.assertEqual(self.ekf.diagnostics()["odom_resets"], 1)

    def test_odom_seq_decrease_rebaselines(self) -> None:
        self._settle()
        t = 103.0
        for i in range(10):
            ts = t + i * 0.02
            self.ekf.update_imu(_imu(ts, 0.0))
            self.ekf.update_odom(ts, 0.1 * i, 0.0, 0.0, seq=100 + i)
        ts = t + 0.2
        self.ekf.update_imu(_imu(ts, 0.0))
        self.assertTrue(self.ekf.update_odom(ts, -5.0, 0.0, 0.0, seq=0))
        p = self.ekf.pose_at(ts)
        assert p is not None
        self.assertAlmostEqual(p[0], 0.9, places=3)

    def test_absent_epoch_fields_never_reset(self) -> None:
        self._settle()
        self.assertFalse(self.ekf.update_odom(103.0, 0.0, 0.0, 0.0))
        self.assertFalse(self.ekf.update_odom(103.02, 0.01, 0.0, 0.0))

    def test_imu_session_change_rebinds_offset(self) -> None:
        # I4: imu_driver restart → GRV yaw origin jumps 1 rad. Heading must
        # not step.
        ekf = EkfPoseTracker(noise=FusionNoiseConfig())
        for i in range(250):
            ekf.update_imu(_imu(100.0 + i * 0.01, 0.2, session_id="s1", seq=i))
        ekf.update_odom(102.5, 0.0, 0.0, 0.0)
        t = 103.0
        for i in range(5):
            ts = t + i * 0.02
            ekf.update_imu(_imu(ts, 0.2, session_id="s1", seq=300 + i))
            ekf.update_odom(ts, 0.0, 0.0, 0.0)
        th0 = ekf.pose_at(t + 0.08)[2]
        for i in range(5, 15):
            ts = t + i * 0.02
            ekf.update_imu(_imu(ts, 1.2, session_id="s2", seq=i))
            ekf.update_odom(ts, 0.0, 0.0, 0.0)
        th1 = ekf.pose_at(t + 0.28)[2]
        self.assertAlmostEqual(th1, th0, places=3)

    def test_frozen_imu_guard_falls_back_to_wheels(self) -> None:
        # I2: IMU keeps publishing a constant yaw while the wheels rotate the
        # chassis 0.6 rad/s for 2 s → guard trips; heading follows wheels,
        # including the rotation hidden before the trip.
        self._settle()
        t = 103.0
        for i in range(101):
            ts = t + i * 0.02
            self.ekf.update_imu(_imu(ts, 0.0))
            self.ekf.update_odom(ts, 0.0, 0.0, 0.012 * i)
        pose = self.ekf.pose_at(t + 2.0)
        assert pose is not None
        self.assertAlmostEqual(pose[2], 1.2, places=2)
        d = self.ekf.diagnostics()
        self.assertTrue(d["imu_distrusted"])
        self.assertEqual(d["imu_guard_trips"], 1)
        # IMU comes alive and moves with the wheels → trusted again, re-bound
        # to the current heading (no snap back to the frozen value).
        for i in range(101, 161):
            ts = t + i * 0.02
            self.ekf.update_imu(_imu(ts, 0.012 * (i - 100)))
            self.ekf.update_odom(ts, 0.0, 0.0, 0.012 * i)
        self.assertFalse(self.ekf.diagnostics()["imu_distrusted"])
        pose = self.ekf.pose_at(t + 3.2)
        assert pose is not None
        self.assertAlmostEqual(pose[2], _wrap(0.012 * 160), places=1)

    def test_pose_at_wrap_safe_interpolation(self) -> None:
        # S6: interpolating between θ=3.13 and θ=-3.13 must stay near ±π.
        self._settle()
        t = 103.0
        self.ekf.update_imu(_imu(t, 3.13))
        self.ekf.update_odom(t, 0.0, 0.0, 0.0)
        self.ekf.update_imu(_imu(t + 0.02, -3.13))
        self.ekf.update_odom(t + 0.02, 0.0, 0.0, 0.0)
        mid = self.ekf.pose_at(t + 0.01)
        assert mid is not None
        self.assertGreater(abs(mid[2]), 3.1)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


if __name__ == "__main__":
    unittest.main()

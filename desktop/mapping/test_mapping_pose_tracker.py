"""Unit tests for MappingPoseTracker.

Run:
    PYTHONPATH=. desktop/.venv/bin/python -m desktop.mapping.test_mapping_pose_tracker -v
"""

from __future__ import annotations

import math
import unittest

from desktop.mapping.mapping_pose_tracker import MappingPoseTracker
from desktop.nav.slam.types import FusionMode, ImuReading


def _quat_for_yaw(yaw_rad: float):
    return (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))


def _imu_reading(
    ts: float, yaw_rad: float,
    *, mode: FusionMode = FusionMode.GAME_ROTATION_VECTOR,
    accuracy_rad: float = 0.175,
) -> ImuReading:
    return ImuReading(
        ts=ts, gyro_z=0.0, quat_wxyz=_quat_for_yaw(yaw_rad),
        fusion_mode=mode, accuracy_rad=accuracy_rad,
    )


def _settle_imu(tracker: MappingPoseTracker, base_ts: float, yaw_rad: float,
                n: int = 25) -> None:
    for i in range(n):
        tracker.update_imu(_imu_reading(base_ts + i * 0.01, yaw_rad))


class TestMappingPoseTracker(unittest.TestCase):
    def test_pose_at_uses_imu_yaw_after_settle(self):
        t = MappingPoseTracker()
        _settle_imu(t, base_ts=-1.0, yaw_rad=0.0)
        t.update_odom(0.0, 0.0, 0.0, 0.0)
        for i in range(25):
            t.update_imu(_imu_reading(0.0 + i * 0.01, math.pi / 2))
        t.update_odom(0.1, 1.0, 0.0, 0.5)
        out = t.pose_at(0.1)
        self.assertIsNotNone(out)
        x, y, theta = out
        self.assertAlmostEqual(x, 1.0, places=5)
        self.assertAlmostEqual(theta, math.pi / 2, places=2)

    def test_pose_at_scan_timestamp_interpolates_yaw(self):
        t = MappingPoseTracker()
        _settle_imu(t, base_ts=-1.0, yaw_rad=0.0)
        t.update_odom(0.0, 0.0, 0.0, 0.0)
        t.update_odom(0.1, 0.0, 0.0, 0.0)
        for ts, yaw in ((0.0, 0.0), (0.05, 0.5), (0.10, 1.0)):
            t.update_imu(_imu_reading(ts, yaw))
        out = t.pose_at(0.05)
        self.assertIsNotNone(out)
        _x, _y, theta = out
        self.assertAlmostEqual(theta, 0.5, places=2)

    def test_rebind_zeros_world_at_reset(self):
        t = MappingPoseTracker()
        _settle_imu(t, base_ts=-1.0, yaw_rad=2.0)
        t.update_odom(0.0, 0.0, 0.0, 0.0)
        t.update_odom(0.1, 0.5, 0.0, 0.0)
        t.rebind_world_to_current()
        out = t.pose_at(0.1)
        self.assertIsNotNone(out)
        x, y, theta = out
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)
        self.assertAlmostEqual(theta, 0.0, places=3)

    def test_rotation_in_place_zero_translation(self):
        t = MappingPoseTracker()
        _settle_imu(t, base_ts=-1.0, yaw_rad=0.0)
        t.update_odom(0.0, 0.0, 0.0, 0.0)
        for i in range(26):
            yaw = i * (math.pi / 2) / 25
            t.update_imu(_imu_reading(0.0 + i * 0.02, yaw))
        t.update_odom(0.5, 0.0, 0.0, 0.0)
        out = t.pose_at(0.5)
        self.assertIsNotNone(out)
        x, y, theta = out
        self.assertAlmostEqual(x, 0.0, places=4)
        self.assertAlmostEqual(y, 0.0, places=4)
        self.assertAlmostEqual(theta, math.pi / 2, places=2)
        self.assertEqual(t.diagnostics()["heading_source"], "imu")

    def test_is_ready_requires_settle_and_seed(self):
        t = MappingPoseTracker()
        self.assertFalse(t.is_ready())
        _settle_imu(t, base_ts=-1.0, yaw_rad=0.0)
        self.assertFalse(t.is_ready())
        t.update_odom(0.0, 0.0, 0.0, 0.0)
        self.assertTrue(t.is_ready())

    def test_auto_rebind_on_first_odom_after_settle(self):
        t = MappingPoseTracker()
        _settle_imu(t, base_ts=-1.0, yaw_rad=1.5)
        t.update_odom(0.0, 0.0, 0.0, 0.0)
        out = t.pose_at(0.0)
        self.assertIsNotNone(out)
        _x, _y, theta = out
        self.assertAlmostEqual(theta, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()

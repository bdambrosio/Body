"""Unit tests for ImuPlusScanMatchPose.

Run:
    PYTHONPATH=. python3 -m unittest desktop.world_map.test_imu_scan_pose -v
"""
from __future__ import annotations

import math
import unittest

from desktop.nav.slam.types import FusionMode, ImuReading

from .imu_scan_pose import ImuPlusScanMatchPose, ImuScanPoseConfig


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


def _settle_imu(pose: ImuPlusScanMatchPose, base_ts: float, yaw_rad: float,
                n: int = 25) -> None:
    """Push enough GAME_RV samples to flip the tracker to settled."""
    for i in range(n):
        pose._imu_yaw.update(_imu_reading(base_ts + i * 0.01, yaw_rad))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class TestImuPlusScanMatchPose(unittest.TestCase):
    def test_pose_at_uses_encoder_yaw_until_imu_settles(self):
        p = ImuPlusScanMatchPose()
        p.update(0.0, 0.0, 0.0, 0.0)
        p.update(0.1, 1.0, 0.0, 0.5)  # encoder θ = 0.5 rad
        # IMU not yet settled — pose_at falls back to encoder θ.
        out = p.pose_at(0.1)
        self.assertIsNotNone(out)
        x, y, theta = out
        self.assertAlmostEqual(x, 1.0, places=5)
        self.assertAlmostEqual(theta, 0.5, places=5)

    def test_pose_at_substitutes_imu_yaw_after_settle(self):
        p = ImuPlusScanMatchPose()
        p.update(0.0, 0.0, 0.0, 0.0)
        p.update(0.1, 1.0, 0.0, 0.5)  # encoder θ = 0.5
        _settle_imu(p, base_ts=0.0, yaw_rad=1.2)  # IMU yaw = 1.2 (different)
        # After settle and with default yaw_offset=0, pose_at returns IMU yaw.
        out = p.pose_at(0.1)
        x, y, theta = out
        self.assertAlmostEqual(x, 1.0, places=5)
        self.assertAlmostEqual(theta, 1.2, places=3)

    def test_rebind_captures_imu_yaw_offset(self):
        # World yaw should be 0 at the moment of Reset world, regardless
        # of GAME_RV's arbitrary boot heading.
        p = ImuPlusScanMatchPose()
        p.update(0.0, 0.0, 0.0, 0.0)
        p.update(0.1, 0.5, 0.0, 0.0)
        _settle_imu(p, base_ts=0.0, yaw_rad=2.0)  # boot yaw 2.0 rad
        p.rebind_world_to_current()
        out = p.pose_at(0.1)
        # World x,y at reset → (0,0). World θ at reset → 0.
        x, y, theta = out
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)
        self.assertAlmostEqual(theta, 0.0, places=3)

    def test_rebind_pre_settle_yaw_offset_zero(self):
        # If user resets before IMU settles, yaw_offset is 0; pre-settle
        # queries fall back to encoder θ. After settle, queries return
        # raw IMU yaw (offset=0) — meaningful only relative to itself.
        p = ImuPlusScanMatchPose()
        p.update(0.0, 0.0, 0.0, 0.0)
        p.rebind_world_to_current()
        self.assertEqual(p._yaw_offset, 0.0)

    def test_apply_correction_rewrites_offsets(self):
        # Drive: at t=0, encoder pose (0,0,0), IMU yaw 1.0 rad.
        # Reset world → world is anchored here (translation + yaw).
        # Then drive to encoder (1, 0, 0), IMU yaw 1.0 (no rotation).
        # Without correction, pose_at(0.5) → roughly (1, 0, 0).
        # Suppose scan-match says actual world pose at t=0.5 is
        # (1.05, 0.10, 0.02). Apply correction.
        # After: pose_at(0.5) ≈ (1.05, 0.10, 0.02).
        p = ImuPlusScanMatchPose()
        p.update(0.0, 0.0, 0.0, 0.0)
        _settle_imu(p, base_ts=-0.3, yaw_rad=1.0)
        p.rebind_world_to_current()
        # Drive forward 1 m in encoder frame.
        p.update(0.5, 1.0, 0.0, 0.0)
        # IMU stayed constant at 1.0; with offset captured at reset,
        # world yaw stays 0.
        before = p.pose_at(0.5)
        self.assertIsNotNone(before)
        self.assertAlmostEqual(before[0], 1.0, places=4)
        self.assertAlmostEqual(before[1], 0.0, places=4)
        self.assertAlmostEqual(before[2], 0.0, places=3)

        target = (1.05, 0.10, 0.02)
        # Need an IMU sample at t=0.5 so _apply_correction can re-pin
        # yaw_offset; push one with the same IMU yaw.
        p._imu_yaw.update(_imu_reading(0.5, 1.0))
        p._apply_correction(target, ts=0.5)

        after = p.pose_at(0.5)
        self.assertIsNotNone(after)
        self.assertAlmostEqual(after[0], target[0], places=4)
        self.assertAlmostEqual(after[1], target[1], places=4)
        self.assertAlmostEqual(after[2], target[2], places=3)

    def test_to_world_translation_uses_odom_offset(self):
        # to_world() must agree with OdomPose's translation transform
        # so Pi-stamped anchor poses still resolve correctly.
        p = ImuPlusScanMatchPose()
        p.update(0.0, 1.5, -0.5, 0.0)
        p.rebind_world_to_current()
        # An anchor at the same odom pose should map to (0,0,_) in world.
        x_w, y_w, _ = p.to_world(1.5, -0.5, 0.0)
        self.assertAlmostEqual(x_w, 0.0, places=5)
        self.assertAlmostEqual(y_w, 0.0, places=5)

    def test_source_name_distinct_from_odom(self):
        self.assertEqual(ImuPlusScanMatchPose().source_name(), "imu+scan_match")

    def test_match_summary_keys_present(self):
        p = ImuPlusScanMatchPose()
        s = p.match_summary()
        for k in (
            "attempted", "accepted", "exhausted",
            "rejected_low_imp", "rejected_too_large",
            "skipped_sparse_grid", "skipped_no_prior",
            "skipped_imu_unsettled",
        ):
            self.assertIn(k, s)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for ImuYawTracker.

Run:
    PYTHONPATH=. python3 -m unittest desktop.nav.slam.test_imu_yaw -v
"""
from __future__ import annotations

import math
import unittest

from .imu_yaw import ImuYawTracker
from .types import FusionMode, ImuReading


def _quat_for_yaw(yaw_rad: float):
    """Pure yaw rotation around z-axis → (w, x, y, z)."""
    return (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))


def _reading(
    ts: float, yaw: float, *,
    accuracy_rad: float = 0.02,
    mode: FusionMode = FusionMode.GAME_ROTATION_VECTOR,
    gyro_z: float = 0.0,
    session_id=None,
    seq=None,
) -> ImuReading:
    return ImuReading(
        ts=ts, gyro_z=gyro_z, quat_wxyz=_quat_for_yaw(yaw),
        fusion_mode=mode, accuracy_rad=accuracy_rad,
        session_id=session_id, seq=seq,
    )


class TestImuYawTracker(unittest.TestCase):
    def test_unsettled_until_enough_good_samples(self):
        t = ImuYawTracker(min_settle_samples=5, settle_accuracy_rad=0.05)
        # 4 good samples: still not settled
        for i in range(4):
            t.update(_reading(ts=i * 0.01, yaw=0.0, accuracy_rad=0.02))
        self.assertFalse(t.is_settled())
        self.assertIsNone(t.yaw_at(0.02))
        # 5th good sample crosses the gate
        t.update(_reading(ts=4 * 0.01, yaw=0.0, accuracy_rad=0.02))
        self.assertTrue(t.is_settled())

    def test_settle_run_resets_on_bad_sample(self):
        # Reset-on-bad-accuracy is RV-only behavior; GAME_RV bypasses
        # the accuracy gate entirely (constant accuracy_rad in that
        # mode would otherwise lock the gate forever).
        t = ImuYawTracker(min_settle_samples=3, settle_accuracy_rad=0.05)
        rv = FusionMode.ROTATION_VECTOR
        t.update(_reading(ts=0.00, yaw=0.0, mode=rv, accuracy_rad=0.02))
        t.update(_reading(ts=0.01, yaw=0.0, mode=rv, accuracy_rad=0.02))
        # Bad sample: resets the run.
        t.update(_reading(ts=0.02, yaw=0.0, mode=rv, accuracy_rad=0.20))
        t.update(_reading(ts=0.03, yaw=0.0, mode=rv, accuracy_rad=0.02))
        t.update(_reading(ts=0.04, yaw=0.0, mode=rv, accuracy_rad=0.02))
        # Only 2 good since last bad → still unsettled.
        self.assertFalse(t.is_settled())
        t.update(_reading(ts=0.05, yaw=0.0, mode=rv, accuracy_rad=0.02))
        self.assertTrue(t.is_settled())

    def test_constant_rotation_tracks(self):
        t = ImuYawTracker(min_settle_samples=1, settle_accuracy_rad=0.05)
        # Rotate at 1 rad/s for 2 seconds, sampling at 100 Hz.
        rate = 1.0
        for i in range(200):
            ts = i * 0.01
            yaw = rate * ts
            t.update(_reading(ts=ts, yaw=yaw))
        result = t.yaw_at(1.00)
        self.assertIsNotNone(result)
        yaw, sigma = result
        self.assertAlmostEqual(yaw, 1.0, places=5)
        self.assertLess(sigma, 0.05)

    def test_unwrap_across_pi(self):
        t = ImuYawTracker(min_settle_samples=1, settle_accuracy_rad=0.05)
        # Rotate from 0 to 4 rad (past +π ~ 3.14). Quaternion-derived
        # yaw wraps back to ~-2.28; tracker must unwrap to continue
        # increasing past π.
        for i in range(401):
            ts = i * 0.01
            yaw = 0.01 * i  # monotonic 0 → 4 rad over 4 s
            t.update(_reading(ts=ts, yaw=yaw))
        yaw_now, _ = t.latest()[1:]  # (yaw, sigma)
        self.assertAlmostEqual(yaw_now, 4.0, places=5)
        # Cross-check: yaw_at mid-trajectory (past π).
        r = t.yaw_at(3.50)
        self.assertIsNotNone(r)
        yaw, _ = r
        self.assertAlmostEqual(yaw, 3.5, places=5)

    def test_interpolation_between_samples(self):
        t = ImuYawTracker(min_settle_samples=1, settle_accuracy_rad=0.05)
        # Sparse samples at 0 s (yaw=0) and 1 s (yaw=2). Query at 0.5 s.
        t.update(_reading(ts=0.0, yaw=0.0))
        t.update(_reading(ts=1.0, yaw=2.0))
        r = t.yaw_at(0.5)
        self.assertIsNotNone(r)
        yaw, _ = r
        self.assertAlmostEqual(yaw, 1.0, places=5)

    def test_out_of_buffer_returns_none(self):
        t = ImuYawTracker(min_settle_samples=1, settle_accuracy_rad=0.05,
                          buffer_seconds=0.5)
        for i in range(100):
            t.update(_reading(ts=i * 0.01, yaw=0.01 * i))
        # Far-past ts should be rejected; near-latest should work.
        self.assertIsNotNone(t.yaw_at(0.99))
        self.assertIsNone(t.yaw_at(-1.0))
        self.assertIsNone(t.yaw_at(10.0))

    def test_out_of_order_sample_is_dropped(self):
        t = ImuYawTracker(min_settle_samples=1, settle_accuracy_rad=0.05)
        t.update(_reading(ts=1.00, yaw=0.5))
        t.update(_reading(ts=0.99, yaw=-10.0))  # older ts, bogus yaw
        latest_ts, latest_yaw, _ = t.latest()
        self.assertAlmostEqual(latest_ts, 1.00, places=5)
        self.assertAlmostEqual(latest_yaw, 0.5, places=5)

    def test_fusion_mode_reflected(self):
        t = ImuYawTracker(min_settle_samples=1, settle_accuracy_rad=0.05)
        t.update(_reading(ts=0.0, yaw=0.0, mode=FusionMode.ROTATION_VECTOR))
        self.assertEqual(t.fusion_mode(), FusionMode.ROTATION_VECTOR)
        t.update(_reading(ts=0.01, yaw=0.0, mode=FusionMode.GAME_ROTATION_VECTOR))
        self.assertEqual(t.fusion_mode(), FusionMode.GAME_ROTATION_VECTOR)

    def test_game_rv_settles_by_count_despite_high_accuracy(self):
        # GAME_RV reports a constant accuracy_rad (= imu.game_rotation_vector_accuracy_rad,
        # 0.175 by default) that would never pass the 0.06 default
        # gate. Settle must succeed by sample count alone.
        t = ImuYawTracker(min_settle_samples=5, settle_accuracy_rad=0.06)
        for i in range(4):
            t.update(_reading(
                ts=i * 0.01, yaw=0.0,
                mode=FusionMode.GAME_ROTATION_VECTOR,
                accuracy_rad=0.175,
            ))
        self.assertFalse(t.is_settled())
        t.update(_reading(
            ts=4 * 0.01, yaw=0.0,
            mode=FusionMode.GAME_ROTATION_VECTOR,
            accuracy_rad=0.175,
        ))
        self.assertTrue(t.is_settled())
        # Once settled, yaw_at returns a usable value.
        result = t.yaw_at(0.04)
        self.assertIsNotNone(result)

    def test_rv_still_uses_accuracy_gate(self):
        # Sanity: changing GAME_RV settle behavior must not loosen
        # the RV gate. Bad-accuracy RV samples still reset the run.
        t = ImuYawTracker(min_settle_samples=3, settle_accuracy_rad=0.05)
        t.update(_reading(ts=0.00, yaw=0.0,
                          mode=FusionMode.ROTATION_VECTOR, accuracy_rad=0.20))
        t.update(_reading(ts=0.01, yaw=0.0,
                          mode=FusionMode.ROTATION_VECTOR, accuracy_rad=0.20))
        t.update(_reading(ts=0.02, yaw=0.0,
                          mode=FusionMode.ROTATION_VECTOR, accuracy_rad=0.20))
        self.assertFalse(t.is_settled())


class TestImuYawTrackerStreamResets(unittest.TestCase):
    def _settled(self):
        t = ImuYawTracker(min_settle_samples=1, settle_accuracy_rad=0.05)
        for i in range(10):
            t.update(_reading(ts=i * 0.01, yaw=0.3, session_id="a", seq=i))
        return t

    def test_session_change_continues_yaw(self):
        # I4: driver restart → new GRV origin (yaw reads 1.5 instead of 0.3).
        # The tracker must continue from 0.3, not step by 1.2 rad.
        t = self._settled()
        t.update(_reading(ts=0.10, yaw=1.5, session_id="b", seq=0))
        t.update(_reading(ts=0.11, yaw=1.6, session_id="b", seq=1))
        self.assertEqual(t.epoch(), 1)
        y, _ = t.yaw_at(0.10)
        self.assertAlmostEqual(y, 0.3, places=6)
        y, _ = t.yaw_at(0.11)
        self.assertAlmostEqual(y, 0.4, places=6)    # real motion still tracked

    def test_duplicate_seq_dropped(self):
        t = self._settled()
        t.update(_reading(ts=0.20, yaw=0.3, session_id="a", seq=9))  # re-publish
        self.assertAlmostEqual(t.latest()[0], 0.09, places=6)
        self.assertTrue(t.is_stale(0.5))
        self.assertFalse(t.is_stale(0.2))

    def test_backward_clock_step_rebaselines(self):
        t = self._settled()
        t.update(_reading(ts=-50.0, yaw=0.3))      # Pi clock stepped back
        self.assertEqual(t.epoch(), 1)
        self.assertAlmostEqual(t.latest()[0], -50.0)

    def test_absent_fields_never_reset(self):
        t = ImuYawTracker(min_settle_samples=1, settle_accuracy_rad=0.05)
        for i in range(5):
            t.update(_reading(ts=i * 0.01, yaw=0.1 * i))
        self.assertEqual(t.epoch(), 0)


if __name__ == "__main__":
    unittest.main()

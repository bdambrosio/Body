"""Tests for OdomPose odom-epoch re-baselining (X4) and clock steps (S7)."""
from __future__ import annotations

import math
import unittest

from desktop.world_map.pose_source import OdomPose


class TestOdomPoseEpoch(unittest.TestCase):
    def test_session_change_keeps_world_pose_continuous(self):
        p = OdomPose()
        p.update(1.0, 0.0, 0.0, 0.0, session_id="a", seq=0)
        p.update(1.1, 3.0, 1.0, 0.5, session_id="a", seq=1)
        before = p.latest_pose()[0]
        # motor_controller restart: odom back at origin.
        p.update(1.2, 0.0, 0.0, 0.0, session_id="b", seq=0)
        after = p.latest_pose()[0]
        for a, b in zip(before, after):
            self.assertAlmostEqual(a, b, places=6)
        # Motion in the new frame continues from there.
        p.update(1.3, 1.0, 0.0, 0.0, session_id="b", seq=1)
        x, y, th = p.latest_pose()[0]
        self.assertAlmostEqual(x, before[0] + math.cos(0.5), places=6)
        self.assertAlmostEqual(y, before[1] + math.sin(0.5), places=6)
        self.assertEqual(p.odom_resets, 1)

    def test_seq_decrease_resets(self):
        p = OdomPose()
        p.update(1.0, 2.0, 0.0, 0.0, seq=50)
        p.update(1.1, 0.0, 0.0, 0.0, seq=0)
        self.assertAlmostEqual(p.latest_pose()[0][0], 2.0, places=6)

    def test_backward_clock_step_accepted(self):
        p = OdomPose()
        p.update(1000.0, 1.0, 0.0, 0.0)
        p.update(10.0, 1.0, 0.0, 0.0)          # Pi clock stepped back
        self.assertEqual(p.latest_pose()[1], 10.0)

    def test_small_out_of_order_still_ignored(self):
        p = OdomPose()
        p.update(1.0, 1.0, 0.0, 0.0)
        p.update(0.99, 5.0, 0.0, 0.0)
        self.assertEqual(p.latest_pose()[1], 1.0)
        self.assertEqual(p.odom_resets, 0)


if __name__ == "__main__":
    unittest.main()

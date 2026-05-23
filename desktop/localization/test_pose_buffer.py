"""Pose buffer interpolation tests."""

from __future__ import annotations

import unittest

from desktop.localization.pose_buffer import PoseBuffer


class TestPoseBuffer(unittest.TestCase):
    def test_interpolate_pose(self):
        buf = PoseBuffer()
        buf.append(0.0, (0.0, 0.0, 0.0))
        buf.append(1.0, (1.0, 0.0, 0.0))
        mid = buf.pose_at(0.5)
        self.assertIsNotNone(mid)
        assert mid is not None
        self.assertAlmostEqual(mid[0], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the live-overlay pure transform.

Run: PYTHONPATH=. python3 -m unittest desktop.map_editor.test_live_overlay -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.map_editor.live_overlay import scan_to_world


class TestScanToWorld(unittest.TestCase):
    def test_identity_pose(self):
        # One beam straight ahead (angle 0), r=2 → body (2,0) → world (2,0).
        w = scan_to_world([2.0], angle_min=0.0, angle_increment=0.1,
                          pose=(0.0, 0.0, 0.0))
        self.assertEqual(w.shape, (1, 2))
        np.testing.assert_allclose(w[0], [2.0, 0.0], atol=1e-9)

    def test_translation_and_yaw(self):
        # Pose at (1,2) facing +y (yaw=pi/2). Beam ahead (angle 0, r=1)
        # is body (1,0); rotated by +90° → world +y → (1, 3).
        w = scan_to_world([1.0], angle_min=0.0, angle_increment=0.1,
                          pose=(1.0, 2.0, math.pi / 2))
        np.testing.assert_allclose(w[0], [1.0, 3.0], atol=1e-9)

    def test_beam_angles(self):
        # Two beams: angle 0 (ahead) and angle +pi/2 (left), r=1 each,
        # identity pose → (1,0) and (0,1).
        w = scan_to_world([1.0, 1.0], angle_min=0.0,
                          angle_increment=math.pi / 2, pose=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(w[0], [1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(w[1], [0.0, 1.0], atol=1e-9)

    def test_filters_invalid(self):
        ranges = [float("inf"), float("nan"), 0.0, 0.01, 50.0, 3.0]
        w = scan_to_world(ranges, angle_min=0.0, angle_increment=0.1,
                          pose=(0.0, 0.0, 0.0),
                          min_range_m=0.05, max_range_m=12.0)
        # Only the r=3.0 beam (index 5) survives.
        self.assertEqual(w.shape, (1, 2))

    def test_empty(self):
        w = scan_to_world([], 0.0, 0.1, (0.0, 0.0, 0.0))
        self.assertEqual(w.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()

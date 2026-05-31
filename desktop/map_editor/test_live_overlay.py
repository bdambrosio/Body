"""Unit tests for the live-overlay pure transform.

Run: PYTHONPATH=. python3 -m unittest desktop.map_editor.test_live_overlay -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.map_editor.live_overlay import (
    body_xy_to_world, pose_compose, pose_relative, scan_to_world,
)


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


class TestPoseMath(unittest.TestCase):
    def test_compose_relative_roundtrip(self):
        # compose(a, relative(a, b)) == b, for arbitrary poses.
        a = (1.0, 2.0, 0.3)
        b = (3.5, -1.0, -0.8)
        d = pose_relative(a, b)
        got = pose_compose(a, d)
        for g, e in zip(got, b):
            self.assertAlmostEqual(g, e, places=9)

    def test_relative_zero_when_same(self):
        a = (2.0, -1.0, 1.1)
        dx, dy, dth = pose_relative(a, a)
        self.assertAlmostEqual(dx, 0.0, places=12)
        self.assertAlmostEqual(dy, 0.0, places=12)
        self.assertAlmostEqual(dth, 0.0, places=12)

    def test_deadreckon_pure_forward(self):
        # Robot facing +y (θ=π/2). Odom moves +1 in its local x (forward).
        # In world that's +y. Overlay pose should advance +y by 1.
        anchor = (0.0, 0.0, 0.0)        # odom frame anchor
        now = (1.0, 0.0, 0.0)           # odom moved +1 forward
        overlay = (5.0, 5.0, math.pi / 2)
        d = pose_relative(anchor, now)  # (1, 0, 0) local
        moved = pose_compose(overlay, d)
        np.testing.assert_allclose(moved[:2], [5.0, 6.0], atol=1e-9)

    def test_body_xy_to_world(self):
        body = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = body_xy_to_world(body, (1.0, 2.0, math.pi / 2))
        np.testing.assert_allclose(w[0], [1.0, 3.0], atol=1e-9)  # fwd→+y
        np.testing.assert_allclose(w[1], [0.0, 2.0], atol=1e-9)  # left→-x

    def test_body_xy_to_world_empty(self):
        self.assertEqual(
            body_xy_to_world(np.empty((0, 2)), (0, 0, 0)).shape, (0, 2))


if __name__ == "__main__":
    unittest.main()

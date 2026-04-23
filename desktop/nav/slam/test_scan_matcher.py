"""Unit tests for ScanMatcher.

Synthetic setup: a rectangular "room" expressed as a 2D evidence grid
with high-value cells along the perimeter. "Lidar scans" are just the
wall-cell positions expressed in the robot's body frame. Running the
matcher with a perturbed prior should recover the ground-truth pose
within grid resolution.

Run:
    PYTHONPATH=. python3 -m unittest desktop.nav.slam.test_scan_matcher -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from .scan_matcher import ScanMatcher, ScanMatcherConfig, lidar_scan_to_xy
from .types import Pose2D


# Room: 8 m × 6 m, 0.04 m cells → 200 × 150 grid, origin at (-4, -3).
ROOM_EXTENT_X_M = 8.0
ROOM_EXTENT_Y_M = 6.0
RESOLUTION_M = 0.04
ORIGIN_X_M = -ROOM_EXTENT_X_M / 2.0
ORIGIN_Y_M = -ROOM_EXTENT_Y_M / 2.0


def _build_room_evidence() -> np.ndarray:
    nx = int(round(ROOM_EXTENT_X_M / RESOLUTION_M))
    ny = int(round(ROOM_EXTENT_Y_M / RESOLUTION_M))
    ev = np.zeros((nx, ny), dtype=np.float32)
    # Perimeter cells as obstacle evidence (weight 10 each).
    ev[0, :] = 10.0
    ev[-1, :] = 10.0
    ev[:, 0] = 10.0
    ev[:, -1] = 10.0
    return ev


def _wall_world_points(step_m: float = RESOLUTION_M) -> np.ndarray:
    """Sample (x, y) world-frame points along the room perimeter."""
    x_min = ORIGIN_X_M + RESOLUTION_M / 2.0
    x_max = ORIGIN_X_M + ROOM_EXTENT_X_M - RESOLUTION_M / 2.0
    y_min = ORIGIN_Y_M + RESOLUTION_M / 2.0
    y_max = ORIGIN_Y_M + ROOM_EXTENT_Y_M - RESOLUTION_M / 2.0
    xs = np.arange(x_min, x_max + step_m / 2.0, step_m)
    ys = np.arange(y_min, y_max + step_m / 2.0, step_m)
    top = np.stack([xs, np.full_like(xs, y_max)], axis=-1)
    bot = np.stack([xs, np.full_like(xs, y_min)], axis=-1)
    left = np.stack([np.full_like(ys, x_min), ys], axis=-1)
    right = np.stack([np.full_like(ys, x_max), ys], axis=-1)
    return np.concatenate([top, bot, left, right], axis=0)


def _world_points_to_body(
    points_world: np.ndarray, truth_pose: Pose2D,
) -> np.ndarray:
    """Transform world-frame points into the robot's body frame."""
    c = math.cos(-truth_pose.theta)
    s = math.sin(-truth_pose.theta)
    dx = points_world[:, 0] - truth_pose.x
    dy = points_world[:, 1] - truth_pose.y
    return np.stack([c * dx - s * dy, s * dx + c * dy], axis=-1)


class TestScanMatcher(unittest.TestCase):
    def setUp(self):
        self.evidence = _build_room_evidence()
        self.matcher = ScanMatcher(ScanMatcherConfig(
            xy_half_m=0.30,
            theta_half_rad=math.radians(8.0),
            xy_step_m=RESOLUTION_M,      # = 4 cm, matches grid
            theta_step_rad=math.radians(1.0),
            min_improvement=5.0,
        ))
        self.world_points = _wall_world_points()

    def _scan_for_truth(self, truth_pose: Pose2D) -> np.ndarray:
        return _world_points_to_body(self.world_points, truth_pose)

    def _search(self, truth_pose: Pose2D, prior_pose: Pose2D):
        return self.matcher.search(
            self._scan_for_truth(truth_pose),
            prior_pose,
            self.evidence,
            ORIGIN_X_M, ORIGIN_Y_M, RESOLUTION_M,
        )

    # ── Tests ───────────────────────────────────────────────────────

    def test_zero_offset_no_correction(self):
        # Prior equals truth → matcher should not drift away from prior.
        truth = Pose2D(x=0.5, y=-0.3, theta=0.1)
        r = self._search(truth, truth)
        # Either not accepted (no improvement beyond tie), or accepted
        # at the prior itself. Either way, pose == prior within cell.
        self.assertLessEqual(abs(r.pose.x - truth.x), RESOLUTION_M)
        self.assertLessEqual(abs(r.pose.y - truth.y), RESOLUTION_M)
        self.assertLessEqual(
            abs(r.pose.theta - truth.theta), math.radians(1.0),
        )

    def test_recovers_xy_offset(self):
        truth = Pose2D(x=0.0, y=0.0, theta=0.0)
        prior = Pose2D(x=0.10, y=-0.08, theta=0.0)  # off by 10 cm, 8 cm
        r = self._search(truth, prior)
        self.assertTrue(r.accepted, f"not accepted; result={r}")
        self.assertLessEqual(abs(r.pose.x - truth.x), RESOLUTION_M)
        self.assertLessEqual(abs(r.pose.y - truth.y), RESOLUTION_M)
        self.assertFalse(r.search_exhausted)

    def test_recovers_theta_offset(self):
        truth = Pose2D(x=0.0, y=0.0, theta=0.0)
        prior = Pose2D(x=0.0, y=0.0, theta=math.radians(3.0))
        r = self._search(truth, prior)
        self.assertTrue(r.accepted)
        self.assertLessEqual(
            abs(r.pose.theta - truth.theta), math.radians(1.0),
        )

    def test_recovers_combined_offset(self):
        truth = Pose2D(x=0.3, y=-0.2, theta=math.radians(5.0))
        prior = Pose2D(
            x=truth.x + 0.06, y=truth.y - 0.08,
            theta=truth.theta + math.radians(2.0),
        )
        r = self._search(truth, prior)
        self.assertTrue(r.accepted)
        self.assertLessEqual(abs(r.pose.x - truth.x), RESOLUTION_M * 1.5)
        self.assertLessEqual(abs(r.pose.y - truth.y), RESOLUTION_M * 1.5)
        self.assertLessEqual(
            abs(r.pose.theta - truth.theta), math.radians(1.5),
        )

    def test_prior_too_far_flags_exhausted(self):
        truth = Pose2D(x=0.0, y=0.0, theta=0.0)
        prior = Pose2D(x=1.0, y=0.0, theta=0.0)  # way beyond window
        r = self._search(truth, prior)
        self.assertTrue(r.search_exhausted)

    def test_empty_scan_returns_prior(self):
        prior = Pose2D(x=0.0, y=0.0, theta=0.0)
        r = self.matcher.search(
            np.empty((0, 2), dtype=np.float64),
            prior, self.evidence,
            ORIGIN_X_M, ORIGIN_Y_M, RESOLUTION_M,
        )
        self.assertEqual(r.pose, prior)
        self.assertFalse(r.accepted)

    def test_lidar_scan_to_xy_filters_invalid(self):
        ranges = np.array([1.0, np.nan, 2.0, -0.5, 3.0, np.inf], dtype=np.float64)
        angles = np.linspace(0, math.pi, 6)
        xy = lidar_scan_to_xy(ranges, angles)
        # nan, negative, inf dropped → 3 points.
        self.assertEqual(xy.shape, (3, 2))
        # First one at angle 0, range 1.0 → (1.0, 0.0)
        self.assertAlmostEqual(xy[0, 0], 1.0, places=6)
        self.assertAlmostEqual(xy[0, 1], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

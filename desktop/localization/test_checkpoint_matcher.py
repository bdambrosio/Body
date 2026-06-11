"""Tests for the radius-limited checkpoint matcher."""
from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.localization.checkpoints import Checkpoint
from desktop.localization.checkpoint_matcher import (
    CheckpointMatchConfig,
    CheckpointMatcher,
    crop_disk,
)
from desktop.localization import raycast_match as rc

RES = 0.05
OX = OY = -3.0
N = 120


def _cell(x, y):
    return int(math.floor((x - OX) / RES)), int(math.floor((y - OY) / RES))


def _box_room(half_m=1.5):
    occ = np.zeros((N, N), dtype=bool)
    _box_at(occ, 0.0, 0.0, half_m)
    return occ


def _box_at(occ, cx, cy, half_m):
    i0, j0 = _cell(cx - half_m, cy - half_m)
    i1, j1 = _cell(cx + half_m, cy + half_m)
    occ[i0:i1 + 1, j0] = True
    occ[i0:i1 + 1, j1] = True
    occ[i0, j0:j1 + 1] = True
    occ[i1, j0:j1 + 1] = True


def _synth(occ, pose, n=360, max_range=2.0):
    angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
    bearings = pose[2] + angles
    ranges = rc.predicted_ranges(
        occ, OX, OY, RES, (pose[0], pose[1]), bearings,
        max_range_m=max_range, step_m=0.025)
    return angles, ranges


class TestCropDisk(unittest.TestCase):
    def test_zeros_outside_disk(self):
        occ = np.ones((N, N), dtype=bool)
        sub, sox, soy = crop_disk(occ, OX, OY, RES, (0.0, 0.0), 1.0)
        # Corner of the cropped bbox is outside the 1 m disk → cleared.
        self.assertFalse(sub[0, 0])
        # Center stays occupied.
        ci = int((0.0 - sox) / RES)
        cj = int((0.0 - soy) / RES)
        self.assertTrue(sub[ci, cj])


class TestCheckpointMatcher(unittest.TestCase):
    def setUp(self):
        self.occ = _box_room(1.5)
        self.cps = [Checkpoint("cp_000", 0.0, 0.0, 0.0, 2.0, 0.0)]
        self.cfg = CheckpointMatchConfig(min_inlier_frac=0.6)
        self.matcher = CheckpointMatcher(
            self.occ, OX, OY, RES, self.cps, self.cfg)

    def test_recovers_pose_and_id_from_drifted_prior(self):
        true = (0.1, -0.05, math.radians(5))
        angles, ranges = _synth(self.occ, true)
        prior = (0.3, 0.1, math.radians(-3))     # odom drift
        m = self.matcher.match(prior, angles, ranges)
        self.assertIsNotNone(m)
        self.assertEqual(m.checkpoint_id, "cp_000")
        self.assertLess(math.hypot(m.pose[0] - 0.1, m.pose[1] + 0.05), 0.1)
        self.assertLess(abs(m.pose[2] - math.radians(5)), math.radians(4))
        self.assertGreater(m.inlier_frac, 0.8)

    def test_no_candidate_when_prior_far(self):
        true = (0.0, 0.0, 0.0)
        angles, ranges = _synth(self.occ, true)
        far = (5.0, 5.0, 0.0)                     # > select_radius from cp
        self.assertIsNone(self.matcher.match(far, angles, ranges))

    def test_relocalize_recovers_with_no_odom_prior(self):
        # Cold start: the search is centred on the checkpoint's own pose, not
        # an odom prior. A wrong-by-meters prior would defeat match(), but
        # relocalize finds it (yaw IMU-hinted).
        true = (0.2, -0.1, math.radians(30))
        angles, ranges = _synth(self.occ, true)
        m = self.matcher.relocalize(
            angles, ranges, yaw_hint=math.radians(25),
            xy_half_m=0.5, theta_half_rad=math.radians(45))
        self.assertIsNotNone(m)
        self.assertEqual(m.checkpoint_id, "cp_000")
        self.assertLess(math.hypot(m.pose[0] - 0.2, m.pose[1] + 0.1), 0.12)
        self.assertLess(abs(m.pose[2] - math.radians(30)), math.radians(5))

    def test_relocalize_rejects_contradicted_scan(self):
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        ranges = np.full(360, 1.9)
        self.assertIsNone(self.matcher.relocalize(ranges=ranges, angles=angles))

    def test_gate_rejects_contradicted_scan(self):
        # Measured ranges (1.9 m) are well beyond the 1.5 m walls → every beam
        # is blocked early (predicted < measured = contradiction) → rejected.
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        ranges = np.full(360, 1.9)
        prior = (0.0, 0.0, 0.0)
        self.assertIsNone(self.matcher.match(prior, angles, ranges))


class TestRelocalizeAmbiguity(unittest.TestCase):
    """Perceptual aliasing: identical rooms must not be guessed between."""

    def setUp(self):
        # Two identical box rooms, 6 m apart, each with its own checkpoint.
        self.occ = np.zeros((240, 240), dtype=bool)
        _box_at(self.occ, 0.0, 0.0, 1.5)
        _box_at(self.occ, 0.0, 6.0, 1.5)
        self.cps = [
            Checkpoint("cp_A", 0.0, 0.0, 0.0, 2.0, 0.0),
            Checkpoint("cp_B", 0.0, 6.0, 0.0, 2.0, 0.0),
        ]
        self.matcher = CheckpointMatcher(
            self.occ, OX, OY, RES, self.cps,
            CheckpointMatchConfig(min_inlier_frac=0.6))
        self.true = (0.1, -0.05, 0.0)            # physically in room A
        self.angles, self.ranges = _synth(self.occ, self.true)

    def test_aliased_rooms_rejected_without_prior(self):
        # Both rooms score (near-)identically → ambiguous → None, not a guess.
        self.assertIsNone(self.matcher.relocalize(
            self.angles, self.ranges, yaw_hint=0.0))

    def test_prior_breaks_the_alias(self):
        # A believed position near room A demotes room B's twin score.
        m = self.matcher.relocalize(
            self.angles, self.ranges, yaw_hint=0.0, prior_xy=(0.3, 0.2))
        self.assertIsNotNone(m)
        self.assertEqual(m.checkpoint_id, "cp_A")
        # Loose tolerance: this asserts the *room* choice, not grid precision
        # (the 0.10 m search step quantizes the pose; precision is covered by
        # TestCheckpointMatcher).
        self.assertLess(math.hypot(m.pose[0] - 0.1, m.pose[1] + 0.05), 0.2)

    def test_agreeing_overlapping_checkpoints_not_ambiguous(self):
        # Two checkpoints 0.3 m apart over the SAME room both match the same
        # physical pose — that is agreement, not ambiguity → seat the best.
        occ = _box_room(1.5)
        cps = [
            Checkpoint("cp_000", 0.0, 0.0, 0.0, 2.0, 0.0),
            Checkpoint("cp_001", 0.3, 0.0, 0.0, 2.0, 0.0),
        ]
        matcher = CheckpointMatcher(
            occ, OX, OY, RES, cps, CheckpointMatchConfig(min_inlier_frac=0.6))
        true = (0.1, -0.05, math.radians(5))
        angles, ranges = _synth(occ, true)
        m = matcher.relocalize(angles, ranges, yaw_hint=math.radians(5))
        self.assertIsNotNone(m)
        self.assertLess(math.hypot(m.pose[0] - true[0], m.pose[1] - true[1]), 0.2)


if __name__ == "__main__":
    unittest.main()

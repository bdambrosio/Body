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
    i0, j0 = _cell(-half_m, -half_m)
    i1, j1 = _cell(half_m, half_m)
    occ[i0:i1 + 1, j0] = True
    occ[i0:i1 + 1, j1] = True
    occ[i0, j0:j1 + 1] = True
    occ[i1, j0:j1 + 1] = True
    return occ


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

    def test_gate_rejects_contradicted_scan(self):
        # Measured ranges (1.9 m) are well beyond the 1.5 m walls → every beam
        # is blocked early (predicted < measured = contradiction) → rejected.
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        ranges = np.full(360, 1.9)
        prior = (0.0, 0.0, 0.0)
        self.assertIsNone(self.matcher.match(prior, angles, ranges))


if __name__ == "__main__":
    unittest.main()

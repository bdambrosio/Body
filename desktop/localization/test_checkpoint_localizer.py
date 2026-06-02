"""Tests for the checkpoint localizer (dead-reckon + re-anchor) and provider."""
from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.localization import raycast_match as rc
from desktop.localization.checkpoint_localizer import (
    CheckpointLocalizer,
    CheckpointPoseProvider,
    pose_compose,
    pose_relative,
)
from desktop.localization.checkpoint_matcher import (
    CheckpointMatchConfig,
    CheckpointMatcher,
)
from desktop.localization.checkpoints import Checkpoint

RES = 0.05
OX = OY = -3.0
N = 120


def _cell(x, y):
    return int(math.floor((x - OX) / RES)), int(math.floor((y - OY) / RES))


def _box_room(half=1.5):
    occ = np.zeros((N, N), dtype=bool)
    i0, j0 = _cell(-half, -half)
    i1, j1 = _cell(half, half)
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


def _matcher():
    occ = _box_room(1.5)
    cps = [Checkpoint("cp_000", 0.0, 0.0, 0.0, 2.0, 0.0)]
    return CheckpointMatcher(occ, OX, OY, RES, cps, CheckpointMatchConfig()), occ


class TestPoseMath(unittest.TestCase):
    def test_compose_inverts_relative(self):
        a = (1.0, 2.0, 0.7)
        b = (3.0, -1.0, -2.0)
        got = pose_compose(a, pose_relative(a, b))
        self.assertAlmostEqual(got[0], b[0])
        self.assertAlmostEqual(got[1], b[1])
        self.assertAlmostEqual(got[2], math.atan2(math.sin(b[2]), math.cos(b[2])))


class TestCheckpointLocalizer(unittest.TestCase):
    def test_pose_none_before_seed(self):
        m, _ = _matcher()
        loc = CheckpointLocalizer(m)
        self.assertIsNone(loc.pose())
        self.assertFalse(loc.seeded)

    def test_dead_reckon_in_map_frame(self):
        m, _ = _matcher()
        loc = CheckpointLocalizer(m)
        # Map heading +90°: a forward odom step moves the map pose along +y.
        loc.seed((10.0, 10.0, math.pi / 2), (0.0, 0.0, 0.0))
        loc.on_odom((0.3, 0.0, 0.0))
        x, y, th = loc.pose()
        self.assertAlmostEqual(x, 10.0, places=6)
        self.assertAlmostEqual(y, 10.3, places=6)
        self.assertAlmostEqual(th, math.pi / 2, places=6)

    def test_reanchor_corrects_drift(self):
        m, occ = _matcher()
        loc = CheckpointLocalizer(m, reanchor_min_interval_s=0.0)
        true = (0.1, -0.05, math.radians(5))
        angles, ranges = _synth(occ, true)
        loc.seed((0.3, 0.1, math.radians(-3)), (0.0, 0.0, 0.0))   # drifted
        match = loc.try_reanchor(1.0, angles, ranges)
        self.assertIsNotNone(match)
        p = loc.pose()
        self.assertLess(math.hypot(p[0] - 0.1, p[1] + 0.05), 0.1)
        self.assertLess(abs(p[2] - math.radians(5)), math.radians(4))

    def test_reanchor_throttled(self):
        m, occ = _matcher()
        loc = CheckpointLocalizer(m, reanchor_min_interval_s=0.5)
        angles, ranges = _synth(occ, (0.0, 0.0, 0.0))
        loc.seed((0.15, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertIsNotNone(loc.try_reanchor(1.0, angles, ranges))
        self.assertIsNone(loc.try_reanchor(1.2, angles, ranges))   # within interval
        self.assertIsNotNone(loc.try_reanchor(1.6, angles, ranges))


class TestCheckpointPoseProvider(unittest.TestCase):
    def setUp(self):
        self.m, self.occ = _matcher()
        self.loc = CheckpointLocalizer(self.m, reanchor_min_interval_s=0.0)
        self.src = {"odom": None, "scan": None, "seed": None, "age": None}
        self.t = [100.0]
        self.p = CheckpointPoseProvider(
            self.loc,
            odom_fn=lambda: self.src["odom"],
            scan_fn=lambda: self.src["scan"],
            seed_fn=lambda: self.src["seed"],
            age_fn=lambda: self.src["age"],
            clock=lambda: self.t[0],
        )

    def test_none_without_odom(self):
        self.assertIsNone(self.p.world_pose())

    def test_none_when_cannot_seed(self):
        self.src["odom"] = (0.0, 0.0, 0.0)
        self.assertIsNone(self.p.world_pose())   # seed_fn None

    def test_seeds_then_dead_reckons(self):
        self.src["seed"] = (5.0, 5.0, 0.0)
        self.src["odom"] = (0.0, 0.0, 0.0)
        first = self.p.world_pose()
        self.assertEqual(first, (5.0, 5.0, 0.0))
        self.src["odom"] = (0.4, 0.0, 0.0)        # drove forward 0.4 m
        second = self.p.world_pose()
        self.assertAlmostEqual(second[0], 5.4, places=6)

    def test_stale_age_returns_none(self):
        self.src["seed"] = (0.0, 0.0, 0.0)
        self.src["odom"] = (0.0, 0.0, 0.0)
        self.src["age"] = 2.0                     # > max_pose_age_s
        self.assertIsNone(self.p.world_pose())


if __name__ == "__main__":
    unittest.main()

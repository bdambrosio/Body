"""Tests for the occlusion-aware ray-cast scan-match scorer.

Run: PYTHONPATH=. python3 -m unittest desktop.localization.test_raycast_match -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.localization import raycast_match as rc

# 6 m × 6 m grid at 0.05 m, origin (-3, -3) → world spans [-3, 3].
RES = 0.05
OX = OY = -3.0
N = 120
CFG = rc.RaycastConfig(max_range_m=4.0, step_m=0.025)


def _cell(x, y):
    return int(math.floor((x - OX) / RES)), int(math.floor((y - OY) / RES))


def _box_room(half_m: float = 2.0) -> np.ndarray:
    """Occupied perimeter of a square room centered at the origin."""
    occ = np.zeros((N, N), dtype=bool)
    i0, j0 = _cell(-half_m, -half_m)
    i1, j1 = _cell(half_m, half_m)
    occ[i0:i1 + 1, j0] = True
    occ[i0:i1 + 1, j1] = True
    occ[i0, j0:j1 + 1] = True
    occ[i1, j0:j1 + 1] = True
    return occ


def _synth_scan(occ, pose, n_beams=360):
    """Perfect lidar: ray-cast the (true) map at `pose` → (angles, ranges)."""
    angles = np.linspace(-math.pi, math.pi, n_beams, endpoint=False)
    bearings = pose[2] + angles
    ranges = rc.predicted_ranges(
        occ, OX, OY, RES, (pose[0], pose[1]), bearings,
        max_range_m=CFG.max_range_m, step_m=CFG.step_m)
    return angles, ranges


class TestRaycastMatch(unittest.TestCase):
    def setUp(self):
        self.occ = _box_room(2.0)
        self.pose = (0.0, 0.0, 0.0)
        self.angles, self.ranges = _synth_scan(self.occ, self.pose)

    def _score(self, occ, pose):
        return rc.score_pose(occ, OX, OY, RES, pose, self.angles, self.ranges, CFG)

    def test_true_pose_clean_map_high_inlier(self):
        s = self._score(self.occ, self.pose)
        self.assertGreater(s.inlier_frac, 0.95)
        self.assertGreater(s.score, 0.9)
        self.assertEqual(s.short_frac, 0.0)

    def test_offset_pose_scores_lower(self):
        good = self._score(self.occ, self.pose).score
        off = self._score(self.occ, (0.4, 0.0, 0.0)).score
        self.assertLess(off, good)

    def test_all_occupied_map_is_degenerate_low(self):
        # The key property: a solid-red map does NOT match — every beam is
        # blocked immediately (predicted << measured), so it scores low.
        solid = np.ones((N, N), dtype=bool)
        s = self._score(solid, self.pose)
        self.assertGreater(s.short_frac, 0.9)
        self.assertLess(s.score, -0.5)

    def test_behind_wall_smear_is_ignored(self):
        # Occupied cells *behind* the true walls (from the sensor) must not
        # change the score — the first hit is still the true wall.
        smear = self._box_room_with_behind_smear()
        clean = self._score(self.occ, self.pose).score
        smeared = self._score(smear, self.pose).score
        self.assertAlmostEqual(clean, smeared, places=6)

    def _box_room_with_behind_smear(self):
        occ = self.occ.copy()
        # A second wall ring just outside the true one (behind, from origin).
        i0, j0 = _cell(-2.3, -2.3)
        i1, j1 = _cell(2.3, 2.3)
        occ[i0:i1 + 1, j0] = True
        occ[i0:i1 + 1, j1] = True
        occ[i0, j0:j1 + 1] = True
        occ[i1, j0:j1 + 1] = True
        return occ

    def test_phantom_in_front_penalized(self):
        # A blob between the sensor and the +x wall blocks those beams early
        # → contradiction (short), so the score drops below clean.
        occ = self.occ.copy()
        bi0, bj0 = _cell(0.9, -0.3)
        bi1, bj1 = _cell(1.1, 0.3)
        occ[bi0:bi1 + 1, bj0:bj1 + 1] = True
        s = self._score(occ, self.pose)
        self.assertGreater(s.short_frac, 0.0)
        self.assertLess(s.score, self._score(self.occ, self.pose).score)

    def test_missing_wall_tolerated_not_punished(self):
        # Remove the +x wall: those beams now read max-range (predicted long)
        # → neutral, NOT a hard penalty. Score stays well above the all-red
        # degenerate case and stays positive.
        occ = self.occ.copy()
        i1, _ = _cell(2.0, 0.0)
        occ[i1, :] = False                      # delete the +x wall
        s = self._score(occ, self.pose)
        self.assertEqual(s.short_frac, 0.0)     # no contradiction
        self.assertLess(s.inlier_frac, 0.95)    # some beams now miss
        self.assertGreater(s.score, 0.5)        # but still a good match

    def test_search_recovers_offset(self):
        # Scan taken at the true pose; search from an offset prior recovers it.
        prior = (0.2, -0.15, math.radians(6.0))
        best, s = rc.best_pose_in_window(
            self.occ, OX, OY, RES, prior, self.angles, self.ranges,
            xy_half_m=0.3, xy_step_m=0.05,
            theta_half_rad=math.radians(12.0), theta_step_rad=math.radians(3.0),
            cfg=CFG)
        self.assertLess(math.hypot(best[0], best[1]), 0.06)     # ~back to (0,0)
        self.assertLess(abs(best[2]), math.radians(4.0))
        self.assertGreater(s.inlier_frac, 0.95)


if __name__ == "__main__":
    unittest.main()

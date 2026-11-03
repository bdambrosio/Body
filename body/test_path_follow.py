"""Unit tests for body/lib/path_follow.py (Tier-3 path mode geometry)."""
import math
import unittest

import numpy as np

from body.lib.path_follow import (
    cumulative, first_blocked, max_deviation, point_at, project, raster_lookup,
    rejoin_index, resample,
)

L_PATH = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]


class TestGeometry(unittest.TestCase):
    def test_project_and_point_at(self):
        cum = cumulative(L_PATH)
        self.assertAlmostEqual(cum[-1], 4.0)
        s, lat = project(L_PATH, cum, (1.0, 0.3))
        self.assertAlmostEqual(s, 1.0)
        self.assertAlmostEqual(lat, 0.3)
        self.assertEqual(point_at(L_PATH, cum, 3.0), (2.0, 1.0))

    def test_hint_stops_recapture_on_earlier_leg(self):
        # A path that doubles back: without a hint the point projects onto
        # the far (later) leg only on a tie; with a hint it may not go back.
        path = [(0.0, 0.0), (2.0, 0.0), (2.0, 0.2), (0.0, 0.2)]
        cum = cumulative(path)
        s, _ = project(path, cum, (1.0, 0.05), s_hint=3.0, back_m=0.3)
        self.assertGreater(s, 2.5)

    def test_resample_spacing(self):
        cum = cumulative(L_PATH)
        smp = resample(L_PATH, cum, 0.5, 2.5, 0.04)
        self.assertAlmostEqual(smp[0][0], 0.5)
        self.assertAlmostEqual(smp[-1][0], 2.5)
        self.assertEqual(smp[-1][1], (2.0, 0.5))

    def test_blocked_and_rejoin(self):
        meta = {"resolution_m": 0.1, "origin_x_m": -1.0, "origin_y_m": -1.0}
        mask = np.zeros((40, 40), bool)
        mask[15:18, 10:12] = True                 # x 0.5..0.8, y 0..0.2
        f = raster_lookup(mask, meta)
        path = [(0.0, 0.05), (2.5, 0.05)]
        cum = cumulative(path)
        smp = resample(path, cum, 0.0, 2.5, 0.04)
        i0, n_on = first_blocked(smp, f)
        self.assertAlmostEqual(smp[i0][1][0], 0.52, delta=0.05)
        j = rejoin_index(smp, i0, f, 0.12)
        self.assertGreaterEqual(smp[j][1][0], 0.8)
        self.assertLess(smp[j][1][0], 0.9)
        self.assertEqual(n_on, len(smp))         # all on the 4 m raster

    def test_off_raster_stops_the_check(self):
        meta = {"resolution_m": 0.1, "origin_x_m": -1.0, "origin_y_m": -1.0}
        f = raster_lookup(np.zeros((20, 20), bool), meta)
        path = [(0.0, 0.0), (3.0, 0.0)]
        smp = resample(path, cumulative(path), 0.0, 3.0, 0.1)
        i0, n_on = first_blocked(smp, f)
        self.assertIsNone(i0)
        self.assertEqual(n_on, 10)               # x < 1.0 only

    def test_max_deviation(self):
        cum = cumulative(L_PATH)
        self.assertAlmostEqual(max_deviation([(1.0, 0.4), (1.5, -0.1)], L_PATH, cum), 0.4)


if __name__ == "__main__":
    unittest.main()

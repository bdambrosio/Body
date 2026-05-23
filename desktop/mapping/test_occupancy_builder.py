"""Occupancy builder tests."""

from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.mapping.occupancy_builder import OccupancyBuilder


class TestOccupancyBuilder(unittest.TestCase):
    def test_ray_marks_free_and_occupied(self):
        b = OccupancyBuilder(extent_m=4.0, resolution_m=0.1)
        ranges = np.array([1.0], dtype=np.float64)
        angles = np.array([0.0], dtype=np.float64)
        b.integrate_scan(ranges, angles, (0.0, 0.0, 0.0))
        occ = b.occupied_mask()
        self.assertGreater(int(occ.sum()), 0)
        self.assertGreater(int((b.log_odds < 0).sum()), 0)

    def test_export_reference_map(self):
        b = OccupancyBuilder(extent_m=2.0, resolution_m=0.1)
        ref = b.to_reference_map(session_id="t1")
        self.assertEqual(ref.session_id, "t1")
        self.assertEqual(ref.nx, b.n_cells)


if __name__ == "__main__":
    unittest.main()

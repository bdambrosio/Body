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


class TestStage0Gates(unittest.TestCase):
    """Mapping plan Stage 0: far returns clear, near returns are ignored,
    export despeckles the planner layer."""

    def test_return_beyond_max_range_clears_without_hit(self):
        b = OccupancyBuilder(extent_m=12.0, resolution_m=0.1, max_range_m=5.0, far_clear_m=4.5)
        b.integrate_scan(np.array([8.0]), np.array([0.0]), (0.0, 0.0, 0.0))
        n = b.n_cells // 2
        self.assertLess(b.log_odds[n + 30, n], 0.0)      # 3 m: cleared
        self.assertEqual(b.log_odds[n + 50, n], 0.0)     # 5 m: untouched
        self.assertEqual(int(b.occupied_mask().sum()), 0)

    def test_near_return_ignored(self):
        b = OccupancyBuilder(extent_m=4.0, resolution_m=0.1)
        b.integrate_scan(np.array([0.08]), np.array([0.0]), (0.0, 0.0, 0.0))
        self.assertEqual(int((b.log_odds != 0).sum()), 0)

    def test_export_despeckle_drops_lone_cell_keeps_wall(self):
        b = OccupancyBuilder(extent_m=4.0, resolution_m=0.1)
        n = b.n_cells // 2
        b.log_odds[n + 10, n - 5:n + 6] = 3.0      # a wall line
        b.log_odds[n - 5, n - 5] = 3.0             # one speck
        ref = b.to_reference_map(session_id="t", despeckle=True)
        drive = ref.driveable_int8()
        self.assertEqual(drive[n + 10, n], 0)
        self.assertEqual(drive[n - 5, n - 5], -1)
        raw = b.to_reference_map(session_id="t").driveable_int8()
        self.assertEqual(raw[n - 5, n - 5], 0)

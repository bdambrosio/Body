"""Tests for the body-frame local costmap."""
import math
import unittest

import numpy as np

from body.lib.local_costmap import (
    LocalCostmapConfig, build_local_costmap, dilate_bool, drop_speckle,
    wavefront_distance,
)

RES, HALF = 0.08, 2.5
N = 2 * int(math.ceil(HALF / RES))
META = {"resolution_m": RES, "origin_x_m": -HALF, "origin_y_m": -HALF,
        "nx": N, "ny": N, "frame": "body"}


def _clear():
    return np.ones((N, N), dtype=np.int8)


def _cell(x, y):
    return (int((x + HALF) / RES), int((y + HALF) / RES))


def _block(g, x, y):
    i, j = _cell(x, y)
    g[i, j] = 0


class TestLocalCostmap(unittest.TestCase):
    def setUp(self):
        self.cfg = LocalCostmapConfig(footprint_radius_m=0.11)

    def test_footprint_dilates_lethal(self):
        grid = _clear()
        # a small cluster so denoise keeps it
        for dx in (-RES, 0, RES):
            _block(grid, 0.5 + dx, 0.0)
        cm = build_local_costmap(grid, META, self.cfg)
        i0, j0 = _cell(0.5, 0.0)
        self.assertTrue(cm.lethal[i0, j0])                       # blocked cell lethal
        # a cell ~0.08 m away (< footprint 0.11) is lethal by dilation
        self.assertTrue(cm.lethal[i0, j0 + 1])
        # a cell ~0.25 m away is not lethal
        i2, j2 = _cell(0.5, 0.25)
        self.assertFalse(cm.lethal[i2, j2])

    def test_halo_decays_with_distance(self):
        grid = _clear()
        for dx in (-RES, 0, RES):
            _block(grid, 0.5 + dx, 0.0)
        cm = build_local_costmap(grid, META, self.cfg)
        near = cm.cost[_cell(0.5, 0.22)]      # just outside lethal
        far = cm.cost[_cell(0.5, 0.6)]        # well clear
        self.assertGreater(near, far)
        self.assertGreaterEqual(far, 0.0)

    def test_denoise_drops_isolated(self):
        grid = _clear()
        _block(grid, 0.5, 0.0)                # single isolated cell
        cm = build_local_costmap(grid, META, LocalCostmapConfig(denoise=True))
        self.assertFalse(cm.lethal[_cell(0.5, 0.0)])             # speckle removed

    def test_gap_sealing(self):
        # Two walls leaving a ~0.20 m gap → sealed by footprint 0.11 (lethal
        # radius 0.11 from each wall closes a < ~0.22 m gap). A ~0.5 m gap stays open.
        def wall_pair(gap_half):
            g = _clear()
            for x in np.arange(0.3, 0.7, RES / 2):
                for y in np.arange(gap_half, gap_half + 0.3, RES / 2):
                    _block(g, float(x), float(y))
                for y in np.arange(-gap_half - 0.3, -gap_half, RES / 2):
                    _block(g, float(x), float(y))
            return g
        narrow = build_local_costmap(wall_pair(0.10), META, self.cfg)  # 0.20 m gap
        wide = build_local_costmap(wall_pair(0.25), META, self.cfg)    # 0.50 m gap
        self.assertTrue(narrow.lethal[_cell(0.5, 0.0)])          # sealed
        self.assertFalse(wide.lethal[_cell(0.5, 0.0)])           # open

    def test_unknown_not_lethal_by_default(self):
        grid = _clear()
        grid[_cell(0.5, 0.0)] = -1
        cm = build_local_costmap(grid, META, self.cfg)
        self.assertFalse(cm.lethal[_cell(0.5, 0.0)])
        self.assertGreater(cm.cost[_cell(0.5, 0.0)], 0.0)        # unknown_cost

    def test_helpers(self):
        m = np.zeros((5, 5), dtype=bool); m[2, 2] = True
        self.assertEqual(int(dilate_bool(m, iters=1).sum()), 9)
        d = wavefront_distance(m == False, max_cells=0)  # noqa: E712
        self.assertEqual(d.shape, (5, 5))


if __name__ == "__main__":
    unittest.main()

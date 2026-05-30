"""Tests for the local A* planner façade."""
import math
import unittest

import numpy as np

from body.lib.local_planner import (
    LocalPlanConfig, lookahead_on_path, plan_local,
)
from body.lib.local_costmap import LocalCostmapConfig

RES, HALF = 0.08, 2.5
N = 2 * int(math.ceil(HALF / RES))
META = {"resolution_m": RES, "origin_x_m": -HALF, "origin_y_m": -HALF,
        "nx": N, "ny": N, "frame": "body"}


def _clear():
    return np.ones((N, N), dtype=np.int8)


def _block(g, x, y):
    g[int((x + HALF) / RES), int((y + HALF) / RES)] = 0


def _cfg():
    return LocalPlanConfig(costmap=LocalCostmapConfig(footprint_radius_m=0.11))


class TestPlanLocal(unittest.TestCase):
    def test_clear_straight_path(self):
        p = plan_local(_clear(), META, (1.0, 0.0), _cfg())
        self.assertTrue(p.ok)
        self.assertEqual(p.reason, "ok")
        self.assertGreaterEqual(len(p.path_body), 2)
        # ends near the goal; stays roughly on the x-axis (no detour needed)
        self.assertAlmostEqual(p.path_body[-1][0], 1.0, delta=3 * RES)
        self.assertLess(max(abs(y) for _x, y in p.path_body), 0.2)

    def test_routes_around_obstacle(self):
        grid = _clear()
        for y in np.arange(-0.3, 0.31, RES / 2):
            _block(grid, 0.6, float(y))          # wall on the direct line
        p = plan_local(grid, META, (1.2, 0.0), _cfg())
        self.assertTrue(p.ok)
        # the path must bow off the x-axis to get around the wall
        self.assertGreater(max(abs(y) for _x, y in p.path_body), 0.3)

    def test_sealed_corridor_no_path(self):
        grid = _clear()
        for x in np.arange(0.5, 0.71, RES / 2):  # thick band, spans whole grid
            for y in np.arange(-HALF, HALF + RES, RES / 2):
                _block(grid, float(x), float(y))
        p = plan_local(grid, META, (1.2, 0.0), _cfg())
        self.assertFalse(p.ok)
        self.assertIn(p.reason, ("no_path", "goal_unreachable"))

    def test_goal_in_obstacle_snapped(self):
        grid = _clear()
        for dx in (-RES, 0, RES):
            for dy in (-RES, 0, RES):
                _block(grid, 1.0 + dx, dy)        # blocked blob at the goal
        p = plan_local(grid, META, (1.0, 0.0), _cfg())
        self.assertTrue(p.ok)                     # snapped to a nearby free cell
        self.assertIsNotNone(p.goal_body_snapped)

    def test_goal_out_of_map(self):
        p = plan_local(_clear(), META, (10.0, 0.0), _cfg())
        self.assertFalse(p.ok)
        self.assertEqual(p.reason, "goal_out_of_map")

    def test_lookahead_on_path(self):
        path = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
        la = lookahead_on_path(path, 0.3)
        self.assertAlmostEqual(la[0], 0.3, delta=1e-6)
        self.assertAlmostEqual(la[1], 0.0, delta=1e-6)
        # beyond the path → last point
        self.assertEqual(lookahead_on_path(path, 5.0), (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()

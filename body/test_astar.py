"""Tests for the grid-agnostic A* core."""
import unittest

import numpy as np

from body.lib.astar import astar_8c, astar_toward, nearest_non_lethal


def _octile_d(a, b):
    di, dj = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (di + dj) + (2 ** 0.5 - 2) * min(di, dj)


def _grids(n=20):
    return np.zeros((n, n), dtype=np.float32), np.zeros((n, n), dtype=bool)


class TestAStar(unittest.TestCase):
    def test_straight_path_clear(self):
        cost, lethal = _grids()
        cells, n, msg = astar_8c(cost=cost, lethal=lethal, start=(2, 2), goal=(2, 10))
        self.assertIsNotNone(cells)
        self.assertEqual(cells[0], (2, 2))
        self.assertEqual(cells[-1], (2, 10))
        self.assertEqual(msg, "ok")

    def test_routes_around_wall_with_gap(self):
        cost, lethal = _grids()
        lethal[:, 6] = True            # wall across column 6...
        lethal[15:, 6] = False         # ...with a gap at the bottom
        cells, n, msg = astar_8c(cost=cost, lethal=lethal, start=(2, 2), goal=(2, 12))
        self.assertIsNotNone(cells)
        # Path must pass through the gap rows (i >= 15) at column 6.
        self.assertTrue(any(c[1] == 6 and c[0] >= 15 for c in cells))


class TestAStarToward(unittest.TestCase):
    def test_reachable_goal_reaches_it(self):
        cost, lethal = _grids()
        cells, _n, msg = astar_toward(cost=cost, lethal=lethal, start=(2, 2), goal=(2, 10))
        self.assertEqual(msg, "ok")
        self.assertEqual(cells[-1], (2, 10))

    def test_unreachable_goal_heads_to_closest_reachable(self):
        cost, lethal = _grids()
        lethal[:, 6] = True            # full wall — goal at col 12 is sealed off
        goal = (2, 12)
        cells, _n, msg = astar_toward(cost=cost, lethal=lethal, start=(2, 2), goal=goal)
        self.assertEqual(msg, "frontier")
        self.assertEqual(cells[0], (2, 2))
        # Endpoint is reachable (col < 6) and closer to the goal than the start.
        self.assertLess(cells[-1][1], 6)
        self.assertLess(_octile_d(cells[-1], goal), _octile_d((2, 2), goal))

    def test_boxed_in_returns_start_only(self):
        cost, lethal = _grids()
        lethal[1:4, 1:4] = True
        lethal[2, 2] = False           # robot's cell free, walled on all sides
        cells, _n, msg = astar_toward(cost=cost, lethal=lethal, start=(2, 2), goal=(2, 10))
        self.assertEqual(msg, "no path")
        self.assertEqual(cells, [(2, 2)])

    def test_no_path_when_sealed(self):
        cost, lethal = _grids()
        lethal[:, 6] = True            # full wall, no gap
        cells, n, msg = astar_8c(cost=cost, lethal=lethal, start=(2, 2), goal=(2, 12))
        self.assertIsNone(cells)
        self.assertEqual(msg, "no path")

    def test_max_expansions(self):
        cost, lethal = _grids(40)
        cells, n, msg = astar_8c(cost=cost, lethal=lethal, start=(0, 0),
                                 goal=(39, 39), max_expansions=5)
        self.assertIsNone(cells)
        self.assertEqual(msg, "max_expansions exceeded")

    def test_prefers_low_cost(self):
        # One costly cell on the straight line; a 1-cell diagonal detour is far
        # cheaper than the halo penalty → A* steps around it.
        cost, lethal = _grids()
        cost[5, 6] = 100.0
        cells, _n, _m = astar_8c(cost=cost, lethal=lethal, start=(5, 5),
                                 goal=(5, 7), cost_per_unit=0.10)
        self.assertIsNotNone(cells)
        self.assertNotIn((5, 6), cells)        # routed around the costly cell

    def test_nearest_non_lethal(self):
        _cost, lethal = _grids()
        lethal[5, 5] = True
        self.assertEqual(nearest_non_lethal(lethal, 0, 0, radius=3), (0, 0))
        got = nearest_non_lethal(lethal, 5, 5, radius=3)
        self.assertIsNotNone(got)
        self.assertFalse(lethal[got])

    def test_start_equals_goal(self):
        cost, lethal = _grids()
        cells, n, msg = astar_8c(cost=cost, lethal=lethal, start=(3, 3), goal=(3, 3))
        self.assertEqual(cells, [(3, 3)])


if __name__ == "__main__":
    unittest.main()

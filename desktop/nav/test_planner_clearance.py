"""Tests for the planner's min_clearance_cells requirement.

After two doorjamb collisions during autonomous nav, the planner must
keep at least one cell of clearance between any planned waypoint and
any lethal cell. The halo cost already biases away but doesn't forbid
0-clearance paths; this dilation makes 0-clearance plans impossible.

Run:
    PYTHONPATH=. python3 -m unittest desktop.nav.test_planner_clearance -v
"""
from __future__ import annotations

import unittest
from dataclasses import field

import numpy as np

from desktop.nav.planner import AStarConfig, plan_path
from desktop.world_map.costmap import Costmap, CostmapConfig


def _make_corridor_costmap(
    *,
    nx: int = 40,
    ny: int = 40,
    res: float = 0.08,
    corridor_y_min: int = 18,
    corridor_y_max: int = 21,
) -> Costmap:
    """A vertical corridor through a sea of obstacles. The corridor
    is `corridor_y_max - corridor_y_min + 1` cells wide. Cells inside
    the corridor are clear, cells outside are lethal."""
    lethal = np.ones((nx, ny), dtype=bool)
    # Punch a horizontal "corridor" of clear cells.
    lethal[:, corridor_y_min : corridor_y_max + 1] = False
    cost = np.where(lethal, np.inf, 0.0).astype(np.float32)
    unknown = np.zeros((nx, ny), dtype=bool)
    distance_m = np.zeros((nx, ny), dtype=np.float32)
    return Costmap(
        cost=cost,
        lethal=lethal,
        unknown=unknown,
        distance_m=distance_m,
        meta={
            "resolution_m": res,
            "origin_x_m": 0.0,
            "origin_y_m": 0.0,
            "nx": nx,
            "ny": ny,
        },
        bounds_ij=None,
        config=CostmapConfig(),
    )


class TestPlannerClearance(unittest.TestCase):

    def test_default_clearance_keeps_one_cell_off_lethal(self):
        # 4-cell-wide corridor (rows 18..21). With min_clearance_cells=1,
        # the planner must stay at least one cell off the corridor's
        # walls — i.e., only rows 19, 20 should be used.
        cm = _make_corridor_costmap(corridor_y_min=18, corridor_y_max=21)
        res = float(cm.meta["resolution_m"])
        # Start near the left edge of the corridor, goal near the right.
        start_w = (1.0 * res, 19.5 * res)
        goal_w = (38.0 * res, 19.5 * res)
        result = plan_path(cm, start_w, goal_w)
        self.assertTrue(result.ok, f"plan failed: {result.msg}")
        # Every waypoint must have ≥1 cell clearance from a lethal cell.
        for (i, j) in result.waypoints_cells:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < cm.lethal.shape[0] and 0 <= jj < cm.lethal.shape[1]:
                        self.assertFalse(
                            cm.lethal[ii, jj],
                            msg=(
                                f"waypoint ({i}, {j}) is adjacent to "
                                f"lethal ({ii}, {jj}); planner violated "
                                f"min_clearance_cells=1"
                            ),
                        )

    def test_clearance_zero_allows_hugging_lethal(self):
        # min_clearance_cells=0 → legacy behavior (planner can touch
        # cells adjacent to lethal). Confirm at least one waypoint sits
        # next to a lethal cell.
        cm = _make_corridor_costmap(corridor_y_min=18, corridor_y_max=21)
        res = float(cm.meta["resolution_m"])
        start_w = (1.0 * res, 18.0 * res)  # right against the wall
        goal_w = (38.0 * res, 18.0 * res)
        result = plan_path(
            cm, start_w, goal_w, config=AStarConfig(min_clearance_cells=0),
        )
        self.assertTrue(result.ok)
        # Some waypoint must be adjacent to a lethal cell.
        any_adjacent = False
        for (i, j) in result.waypoints_cells:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < cm.lethal.shape[0] and 0 <= jj < cm.lethal.shape[1]:
                        if cm.lethal[ii, jj]:
                            any_adjacent = True
        self.assertTrue(any_adjacent, "clearance=0 should allow hugging lethal")

    def test_too_narrow_corridor_fails_at_clearance_1(self):
        # 1-cell-wide corridor — there's no cell anywhere with ≥1 cell
        # clearance from lethal. Plan must fail.
        cm = _make_corridor_costmap(corridor_y_min=20, corridor_y_max=20)
        res = float(cm.meta["resolution_m"])
        start_w = (1.0 * res, 20.0 * res)
        goal_w = (38.0 * res, 20.0 * res)
        result = plan_path(cm, start_w, goal_w)
        self.assertFalse(result.ok)
        # Either start was unrelaxable or A* couldn't find a path.
        self.assertIn("clearance", result.msg.lower() + " no path")

    def test_3_cell_corridor_works_at_clearance_1(self):
        # 3-cell corridor (rows 19, 20, 21). With clearance=1, only
        # row 20 is allowed (rows 19 and 21 touch lethal). Single
        # row 1-cell-wide path must succeed.
        cm = _make_corridor_costmap(corridor_y_min=19, corridor_y_max=21)
        res = float(cm.meta["resolution_m"])
        start_w = (1.0 * res, 20.0 * res)
        goal_w = (38.0 * res, 20.0 * res)
        result = plan_path(cm, start_w, goal_w)
        self.assertTrue(result.ok, f"plan failed: {result.msg}")
        # All waypoints must be in row 20 (the only clearance-1 row).
        for (i, j) in result.waypoints_cells:
            self.assertEqual(j, 20, f"waypoint at ({i}, {j}) outside row 20")


if __name__ == "__main__":
    unittest.main()

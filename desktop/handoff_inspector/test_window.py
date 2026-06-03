"""Unit tests for the inspector's pure decode/format helpers (no Qt loop)."""
import unittest

import numpy as np

from body.lib import schemas
from desktop.handoff_inspector.window import grid_and_meta, format_record


class TestHandoffHelpers(unittest.TestCase):
    def test_grid_decode_roundtrip(self):
        g = np.array([[1, 0, -1], [-1, 1, 0]], dtype=np.int8)
        rec = schemas.handoff_t2(
            bearing_rad=0.0, src="clear", free_dist_m=1.0, subgoal_body=(1.0, 0.0),
            arrival_tol_m=0.15, grid_rows=g.tolist(),
            meta={"resolution_m": 0.08, "origin_x_m": -0.5, "origin_y_m": -0.5,
                  "nx": 2, "ny": 3})
        grid, meta = grid_and_meta(rec)
        self.assertEqual(grid.dtype, np.int8)
        self.assertTrue(np.array_equal(grid, g))
        self.assertEqual(meta["nx"], 2)

    def test_grid_placeholder_when_absent(self):
        rec = schemas.handoff_t1(
            pose=(0, 0, 0), wp=(1, 1), wp_index=0, wp_total=2, terminal=False,
            arrival_tol_m=0.3, bearing_rad=0.1, wp_dist_m=1.4)
        grid, meta = grid_and_meta(rec)
        self.assertEqual(grid.shape, (meta["nx"], meta["ny"]))
        self.assertTrue((grid == -1).all())          # all-unknown placeholder

    def test_format_each_tier_nonempty(self):
        recs = [
            schemas.handoff_t1(pose=(0, 0, 0), wp=(1, 1), wp_index=0, wp_total=2,
                               terminal=True, arrival_tol_m=0.3, bearing_rad=0.1,
                               wp_dist_m=1.4),
            schemas.handoff_t2(bearing_rad=0.1, src="clear", free_dist_m=1.3,
                               subgoal_body=(1.3, 0.1), arrival_tol_m=0.15, cmd_id=7),
            schemas.handoff_t3(cmd_id=7, goal_body=(1, 0), plan_reason="ok",
                               v_mps=0.1, omega_radps=-0.2, swept_blocked=False),
        ]
        for rec in recs:
            s = format_record(rec)
            self.assertIsInstance(s, str)
            self.assertTrue(s)


if __name__ == "__main__":
    unittest.main()

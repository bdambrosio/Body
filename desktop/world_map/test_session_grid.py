"""SessionGrid (explore plan A1): ray-casting, hit/free precedence,
prior-wall protection, reset."""
import math
import unittest

import numpy as np

from desktop.world_map.session_grid import SessionGrid, SessionGridConfig

META = {"nx": 100, "ny": 100, "resolution_m": 0.05, "origin_x_m": -2.5, "origin_y_m": -2.5}


def cell(g, x, y):
    return int(math.floor((x - g.ox) / g.res)), int(math.floor((y - g.oy) / g.res))


def one_beam(r, a=0.0):
    return np.array([r]), np.array([a])


class TestIntegrate(unittest.TestCase):
    def test_hit_and_free_along_beam(self):
        g = SessionGrid(META)
        # One hit marks blocked; free needs two passes (errs toward blocked).
        g.integrate(*one_beam(1.0), (0.0, 0.0, 0.0), ts=4.0)
        self.assertEqual(g.driveable_int8()[cell(g, 0.5, 0.01)], -1)
        g.integrate(*one_beam(1.0), (0.0, 0.0, 0.0), ts=5.0)
        d = g.driveable_int8()
        self.assertEqual(d[cell(g, 1.01, 0.01)], 0)          # hit → blocked
        for x in (0.1, 0.5, 0.9):
            self.assertEqual(d[cell(g, x, 0.01)], 1, x)        # passed → free
        self.assertEqual(d[cell(g, 1.5, 0.01)], -1)            # beyond → unknown
        self.assertEqual(g.last_hit_ts[cell(g, 1.01, 0.01)], 5.0)
        self.assertEqual(g.last_free_ts[cell(g, 0.5, 0.01)], 5.0)

    def test_seen_and_occupied_near(self):
        g = SessionGrid(META)
        self.assertIsNone(g.occupied_near(1.0, 0.0, 0.15))
        for ts in (1.0, 2.0):
            g.integrate(*one_beam(1.0), (0.0, 0.0, 0.0), ts=ts)
        self.assertTrue(g.seen_mask()[cell(g, 0.5, 0.01)])
        self.assertFalse(g.seen_mask()[cell(g, 0.5, 1.0)])
        self.assertTrue(g.occupied_near(1.0, 0.0, 0.15))
        self.assertFalse(g.occupied_near(0.5, 0.0, 0.10))
        self.assertIsNone(g.occupied_near(0.5, 1.0, 0.10))

    def test_pose_rotation(self):
        g = SessionGrid(META)
        g.integrate(*one_beam(1.0), (0.0, 0.0, math.pi / 2), ts=1.0)
        self.assertEqual(g.driveable_int8()[cell(g, 0.01, 1.01)], 0)

    def test_nan_and_too_close_skipped(self):
        g = SessionGrid(META, SessionGridConfig(max_range_m=2.0))
        n = g.integrate(np.array([np.nan, 0.0, 0.05]), np.zeros(3), (0, 0, 0), 1.0)
        self.assertEqual(n, 0)
        self.assertEqual((g.driveable_int8() != -1).sum(), 0)

    def test_return_beyond_max_range_clears_without_hit(self):
        g = SessionGrid(META, SessionGridConfig(max_range_m=2.0))
        for t in (1.0, 2.0):
            g.integrate(*one_beam(3.0), (0, 0, 0), t)
        d = g.driveable_int8()
        self.assertEqual(d[cell(g, 1.9, 0.01)], 1)
        self.assertEqual(d[cell(g, 2.2, 0.01)], -1)
        self.assertEqual((d == 0).sum(), 0)

    def test_moved_object_clears(self):
        """A box seen for a while, then removed: rays through it clear it."""
        g = SessionGrid(META)
        for t in range(10):
            g.integrate(*one_beam(1.0), (0, 0, 0), float(t))
        self.assertEqual(g.driveable_int8()[cell(g, 1.01, 0.01)], 0)
        for t in range(10, 30):
            g.integrate(*one_beam(2.0), (0, 0, 0), float(t))
        self.assertEqual(g.driveable_int8()[cell(g, 1.01, 0.01)], 1)

    def test_edge_beams_out_of_grid_ignored(self):
        g = SessionGrid(META)
        for t in (1.0, 2.0):                               # runs off the grid
            g.integrate(*one_beam(4.0), (2.0, 0.0, 0.0), t)
        self.assertEqual(g.driveable_int8()[cell(g, 2.3, 0.01)], 1)

    def test_reset(self):
        g = SessionGrid(META)
        g.integrate(*one_beam(1.0), (0, 0, 0), 1.0)
        v = g.version
        g.reset()
        self.assertGreater(g.version, v)
        self.assertEqual((g.driveable_int8() != -1).sum(), 0)
        self.assertEqual(g.last_hit_ts.max(), 0.0)


class TestPlanningLayer(unittest.TestCase):
    def test_new_object_blocks_prior_free(self):
        g = SessionGrid(META)
        prior = np.ones((100, 100), dtype=np.int8)
        g.integrate(*one_beam(1.0), (0, 0, 0), 1.0)
        self.assertEqual(g.planning_layer(prior)[cell(g, 1.01, 0.01)], 0)

    def test_prior_wall_needs_strong_free_evidence(self):
        g = SessionGrid(META)
        prior = np.full((100, 100), -1, dtype=np.int8)
        wc = cell(g, 1.01, 0.01)
        prior[wc] = 0
        g.integrate(*one_beam(2.0), (0, 0, 0), 1.0)     # one pass through
        self.assertEqual(g.planning_layer(prior)[wc], 0)
        for t in range(2, 10):                           # door opened
            g.integrate(*one_beam(2.0), (0, 0, 0), float(t))
        self.assertEqual(g.planning_layer(prior)[wc], 1)

    def test_session_hits_next_to_robot_left_out(self):
        g = SessionGrid(META)
        prior = np.ones((100, 100), dtype=np.int8)
        g.integrate(*one_beam(0.2), (0, 0, 0), 1.0)      # 20 cm off the robot
        g.integrate(*one_beam(1.0, 0.5), (0, 0, 0), 1.0)  # 1 m away
        out = g.planning_layer(prior, exclude_xy=(0.0, 0.0), exclude_radius_m=0.3)
        self.assertEqual(out[cell(g, 0.21, 0.01)], 1)
        self.assertEqual(out[cell(g, math.cos(0.5) + 0.01, math.sin(0.5) + 0.01)], 0)
        # Prior walls near the robot stay walls.
        prior[cell(g, 0.21, 0.01)] = 0
        out = g.planning_layer(prior, exclude_xy=(0.0, 0.0), exclude_radius_m=0.3)
        self.assertEqual(out[cell(g, 0.21, 0.01)], 0)

    def test_unseen_cells_keep_prior(self):
        g = SessionGrid(META)
        prior = np.zeros((100, 100), dtype=np.int8)
        prior[:50] = 1
        out = g.planning_layer(prior)
        np.testing.assert_array_equal(out, prior)


if __name__ == "__main__":
    unittest.main()


class TestFirstHit(unittest.TestCase):
    def test_episode_start_and_end(self):
        g = SessionGrid(META)
        c = cell(g, 1.01, 0.01)
        g.integrate(*one_beam(1.0), (0, 0, 0), 3.0)
        g.integrate(*one_beam(1.0), (0, 0, 0), 4.0)
        self.assertEqual(g.first_hit_ts[c], 3.0)
        for t in range(5, 15):
            g.integrate(*one_beam(2.0), (0, 0, 0), float(t))
        self.assertEqual(g.first_hit_ts[c], 0.0)
        g.integrate(*one_beam(1.0), (0, 0, 0), 20.0)
        self.assertEqual(g.first_hit_ts[c], 20.0)


class TestSeeThrough(unittest.TestCase):
    """A reference wall counts as seen-through only when the beam crosses
    it and carries on (11-02: 65 'missing' on wall faces)."""

    def _grid(self, wall_cells):
        g = SessionGrid(META)
        wall = np.zeros((100, 100), dtype=bool)
        for c in wall_cells:
            wall[c] = True
        g.set_wall_mask(wall)
        return g

    def test_beam_crossing_thin_wall_counts(self):
        g = SessionGrid(META)
        wc = cell(g, 1.01, 0.01)
        g = self._grid([wc])
        g.integrate(*one_beam(2.0), (0, 0, 0), 1.0)
        self.assertEqual(g.through[wc], 1)

    def test_beam_ending_just_past_wall_is_not_through(self):
        g = SessionGrid(META)
        wc = cell(g, 1.01, 0.01)
        g = self._grid([wc])
        g.integrate(*one_beam(1.15), (0, 0, 0), 1.0)    # real wall one cell further
        self.assertEqual(g.through[wc], 0)

    def test_beam_along_a_wall_line_is_not_through(self):
        g = SessionGrid(META)
        i0, j0 = cell(g, 0.51, 0.01)
        cells = [(i0 + k, j0) for k in range(20)]   # 1 m of wall along the beam (by index: float steps leave gaps)
        g = self._grid(cells)
        g.integrate(*one_beam(2.0), (0, 0, 0), 1.0)
        self.assertEqual(int(g.through.sum()), 0)

    def _wall_line_x(self, g, x=1.01, half_len_m=1.0):
        """A straight reference wall along y at x (a 'missing' wall: no hits)."""
        i, _ = cell(g, x, 0.0)
        _, j0 = cell(g, x, -half_len_m)
        _, j1 = cell(g, x, half_len_m)
        return [(i, j) for j in range(j0, j1 + 1)]

    def test_steep_beam_through_missing_wall_counts(self):
        g = SessionGrid(META)
        g = self._grid(self._wall_line_x(g))
        for a in (0.0, math.radians(40)):            # 90° and 50° to the wall
            g.integrate(*one_beam(2.5, a), (0, 0, 0), 1.0)
        self.assertGreaterEqual(int((g.through > 0).sum()), 2)   # the 50° beam may cross two cells

    def test_glancing_beam_does_not_count(self):
        # 11-03: every "missing" on four patrols sat on walls swept at a
        # glancing angle. 20° to the wall: not evidence it is gone.
        g = SessionGrid(META)
        g = self._grid(self._wall_line_x(g))
        g.integrate(*one_beam(2.4, math.radians(70)), (0.6, -1.0, 0.0), 1.0)
        self.assertEqual(int(g.through.sum()), 0)
        g2 = SessionGrid(META, SessionGridConfig(through_min_incidence_deg=0.0))
        g2.set_wall_mask(g._wall)
        g2.integrate(*one_beam(2.4, math.radians(70)), (0.6, -1.0, 0.0), 1.0)
        self.assertGreater(int(g2.through.sum()), 0)   # the old test counted it

    def test_hit_resets_through(self):
        g = SessionGrid(META)
        wc = cell(g, 1.01, 0.01)
        g = self._grid([wc])
        g.integrate(*one_beam(2.0), (0, 0, 0), 1.0)
        g.integrate(*one_beam(1.0), (0, 0, 0), 2.0)
        self.assertEqual(g.through[wc], 0)

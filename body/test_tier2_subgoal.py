"""Tests for Tier-2 visibility sub-goal selection (pure)."""
import math
import unittest

import numpy as np

from body.lib.drive_safety import FootprintConfig, swept_path_blocked
from body.lib.tier2_subgoal import (
    Tier2Config,
    bearing_to_waypoint,
    furthest_free_point,
    plan_tier2,
)

RES = 0.08
HALF = 2.5
N = 2 * int(math.ceil(HALF / RES))   # 64, matches ScanRasterConfig defaults
META = {
    "resolution_m": RES,
    "origin_x_m": -HALF,
    "origin_y_m": -HALF,
    "nx": N,
    "ny": N,
    "frame": "body",
}


def _clear_grid():
    return np.ones((N, N), dtype=np.int8)


def _unknown_grid():
    return np.full((N, N), -1, dtype=np.int8)


def _cell(x, y):
    i = int(math.floor((x - META["origin_x_m"]) / RES))
    j = int(math.floor((y - META["origin_y_m"]) / RES))
    return i, j


def _block(grid, x, y):
    i, j = _cell(x, y)
    grid[i, j] = 0


class TestFurthestFreePoint(unittest.TestCase):
    def setUp(self):
        # Pin the knobs the assertions below depend on, independent of the
        # production defaults (which get tuned during bring-up).
        self.cfg = Tier2Config(horizon_m=2.0, backoff_m=0.30, min_subgoal_m=0.20)

    def test_clear_lane_ahead(self):
        r = furthest_free_point(_clear_grid(), META, 0.0, self.cfg)
        self.assertTrue(r.ok)
        self.assertEqual(r.reason, "ok")
        # Capped at horizon − backoff, on the bearing (straight ahead).
        self.assertAlmostEqual(r.free_dist_m, self.cfg.horizon_m - self.cfg.backoff_m, places=6)
        self.assertAlmostEqual(r.body_xy[0], 1.7, places=6)
        self.assertAlmostEqual(r.body_xy[1], 0.0, places=6)

    def test_horizon_cap_not_exceeded(self):
        # Clear well past horizon must not push the sub-goal beyond it.
        r = furthest_free_point(_clear_grid(), META, 0.0, self.cfg)
        self.assertLessEqual(r.free_dist_m, self.cfg.horizon_m - self.cfg.backoff_m + 1e-9)

    def test_blocked_ahead_backs_off(self):
        grid = _clear_grid()
        _block(grid, 1.0, 0.0)
        r = furthest_free_point(grid, META, 0.0, self.cfg)
        self.assertTrue(r.ok)
        self.assertEqual(r.reason, "ok")
        # Sub-goal sits ~1.0 − backoff ahead of the block (within one cell).
        self.assertAlmostEqual(r.body_xy[0], 1.0 - self.cfg.backoff_m, delta=2 * RES)
        self.assertLess(r.body_xy[0], 1.0 - self.cfg.backoff_m + 1e-9)

    def test_blocked_at_origin(self):
        grid = _clear_grid()
        _block(grid, 0.0, 0.0)   # robot cell itself blocked
        r = furthest_free_point(grid, META, 0.0, self.cfg)
        self.assertFalse(r.ok)
        self.assertIsNone(r.body_xy)
        self.assertEqual(r.reason, "blocked_at_origin")

    def test_blocked_too_close_is_too_short(self):
        grid = _clear_grid()
        _block(grid, 0.35, 0.0)   # cleared a little, but < backoff+min
        r = furthest_free_point(grid, META, 0.0, self.cfg)
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "too_short")

    def test_gap_to_one_side(self):
        # Wall straight ahead at x≈0.5 covering the center/right, open up-left.
        grid = _clear_grid()
        for y in np.arange(-1.0, 0.21, RES / 2):
            _block(grid, 0.5, float(y))
        straight = furthest_free_point(grid, META, 0.0, self.cfg)
        # A bearing through the gap (toward +y/left) sees much further.
        side = furthest_free_point(grid, META, 0.7, self.cfg)
        self.assertGreater(side.free_dist_m, straight.free_dist_m)
        self.assertTrue(side.ok)

    def test_caps_at_waypoint_no_backoff(self):
        # Clear path; waypoint at 1.0 m → sub-goal is the waypoint itself,
        # NOT horizon-backoff and NOT past the waypoint.
        r = furthest_free_point(_clear_grid(), META, 0.0, self.cfg, max_dist_m=1.0)
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.free_dist_m, 1.0, delta=RES)
        self.assertAlmostEqual(r.body_xy[0], 1.0, delta=RES)

    def test_obstacle_before_waypoint_backs_off(self):
        # Waypoint at 1.5 m but a block at 0.8 m → stop short of the block.
        grid = _clear_grid()
        _block(grid, 0.8, 0.0)
        r = furthest_free_point(grid, META, 0.0, self.cfg, max_dist_m=1.5)
        self.assertTrue(r.ok)
        self.assertLess(r.free_dist_m, 0.8)            # backed off the obstacle
        self.assertAlmostEqual(r.body_xy[0], 0.8 - self.cfg.backoff_m, delta=2 * RES)

    def test_all_unknown_grid(self):
        r = furthest_free_point(_unknown_grid(), META, 0.0, self.cfg)
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "all_unknown")

    def test_unknown_allowed_when_not_require_clear(self):
        cfg = Tier2Config(require_clear=False)
        r = furthest_free_point(_unknown_grid(), META, 0.0, cfg)
        self.assertTrue(r.ok)   # unknown treated as traversable → reaches horizon

    def test_off_grid_bearing_treated_as_horizon(self):
        # Horizon beyond the grid extent: ray exits the window, capped, no crash.
        cfg = Tier2Config(horizon_m=3.0)   # grid only reaches 2.5
        r = furthest_free_point(_clear_grid(), META, 0.0, cfg)
        self.assertTrue(r.ok)
        self.assertLess(r.free_dist_m, HALF)   # capped near the grid edge

    def test_backoff_clears_swept_footprint(self):
        # The backed-off sub-goal leaves room for Tier-3's swept-footprint check.
        foot = FootprintConfig(footprint_radius_m=0.22)
        far = _clear_grid()
        _block(far, 1.0, 0.0)
        r = furthest_free_point(far, META, 0.0, self.cfg)
        self.assertTrue(r.ok)
        # A straight push over this grid is not vetoed (block is well beyond preview).
        self.assertFalse(swept_path_blocked(far, META, v_mps=0.18, omega_radps=0.0, config=foot))
        # Whereas a block right in front is vetoed.
        near = _clear_grid()
        for y in np.arange(-0.3, 0.31, RES / 2):
            _block(near, 0.3, float(y))
        self.assertTrue(swept_path_blocked(near, META, v_mps=0.18, omega_radps=0.0, config=foot))


class TestPlanTier2(unittest.TestCase):
    def setUp(self):
        self.cfg = Tier2Config(horizon_m=2.0, backoff_m=0.30, min_subgoal_m=0.20)

    def test_clear_to_target_capped_no_backoff(self):
        d = plan_tier2(_clear_grid(), META, 0.0, 1.0, self.cfg)
        self.assertTrue(d.ok)
        self.assertTrue(d.capped_at_target)
        self.assertFalse(d.backoff_applied)
        self.assertAlmostEqual(d.free_dist_m, 1.0, delta=RES)
        self.assertEqual(d.max_dist_m, 1.0)

    def test_obstacle_on_direct_line_routes_around(self):
        # Wall segment on the straight line, open to the sides → the fan swings
        # off the direct bearing and finds a clear lane that still progresses.
        grid = _clear_grid()
        for y in np.arange(-0.25, 0.26, RES / 2):
            _block(grid, 0.6, float(y))
        d = plan_tier2(grid, META, 0.0, 1.5, self.cfg)
        self.assertTrue(d.ok)
        self.assertGreater(abs(d.bearing_offset_rad), 0.1)        # swung off-line
        # Sub-goal gets meaningfully closer to the target than the robot is now.
        self.assertLess(math.hypot(d.body_xy[0] - 1.5, d.body_xy[1] - 0.0), 1.5)

    def test_prefers_straighter_of_two_lanes(self):
        # Narrow wall → a small swing clears it; the fan takes the straightest.
        grid = _clear_grid()
        for y in np.arange(-0.15, 0.16, RES / 2):
            _block(grid, 0.6, float(y))
        d = plan_tier2(grid, META, 0.0, 1.5, self.cfg)
        self.assertTrue(d.ok)
        self.assertLess(abs(d.bearing_offset_rad), math.radians(25))   # small swing

    def test_arc_wall_boxes_in_not_ok(self):
        # A blocked arc ~0.3 m across the whole forward fan → no clear bearing.
        grid = _clear_grid()
        for deg in range(-90, 91, 2):
            a = math.radians(deg)
            _block(grid, 0.3 * math.cos(a), 0.3 * math.sin(a))
        d = plan_tier2(grid, META, 0.0, 1.5, self.cfg)
        self.assertFalse(d.ok)
        self.assertIsNone(d.body_xy)

    def test_clear_direct_no_swing(self):
        d = plan_tier2(_clear_grid(), META, 0.0, 1.0, self.cfg)
        self.assertTrue(d.ok)
        self.assertAlmostEqual(d.bearing_offset_rad, 0.0, places=6)   # straight shot

    def test_as_dict_roundtrips_fields(self):
        d = plan_tier2(_clear_grid(), META, 0.0, 1.0, self.cfg)
        j = d.as_dict()
        self.assertEqual(j["reason"], "ok")
        self.assertEqual(j["body_xy"][1], 0.0)
        self.assertTrue(j["capped_at_target"])


class TestBearingToWaypoint(unittest.TestCase):
    def test_straight_ahead(self):
        self.assertAlmostEqual(bearing_to_waypoint(0, 0, 0.0, 1.0, 0.0), 0.0, places=6)

    def test_left_is_positive(self):
        self.assertAlmostEqual(bearing_to_waypoint(0, 0, 0.0, 0.0, 1.0), math.pi / 2, places=6)

    def test_right_is_negative(self):
        self.assertAlmostEqual(bearing_to_waypoint(0, 0, 0.0, 0.0, -1.0), -math.pi / 2, places=6)

    def test_behind(self):
        self.assertAlmostEqual(abs(bearing_to_waypoint(0, 0, 0.0, -1.0, 0.0)), math.pi, places=6)

    def test_robot_yaw_subtracted(self):
        # Waypoint dead ahead in world, but robot faces +90° → waypoint is to its right.
        self.assertAlmostEqual(
            bearing_to_waypoint(0, 0, math.pi / 2, 1.0, 0.0), -math.pi / 2, places=6)

    def test_result_wrapped(self):
        # Waypoint behind with robot yaw near π must stay in (−π, π].
        b = bearing_to_waypoint(0, 0, math.pi - 0.01, -1.0, 0.05)
        self.assertGreaterEqual(b, -math.pi)
        self.assertLessEqual(b, math.pi)


if __name__ == "__main__":
    unittest.main()

"""Tests for Pi-side Tier-3 pure logic: drive core + swept-footprint safety."""
import math
import unittest

import numpy as np

from body.lib.drive_safety import (
    FootprintConfig, driveable_from_rows, swept_path_blocked,
)
from body.lib.local_drive_core import (
    DriveParams, LocalPlanConfig, body_to_odom, odom_to_body, plan_drive,
    rotate_to_heading, steer_to_body_point, wrap_pi,
)


# ── Frame transforms ────────────────────────────────────────────────

class TestFrames(unittest.TestCase):
    def test_odom_to_body_identity_pose(self):
        bx, by = odom_to_body((1.0, 0.0), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(bx, 1.0)
        self.assertAlmostEqual(by, 0.0)

    def test_odom_to_body_rotated(self):
        # Robot at origin facing +y (90°). A point at odom +x is on the
        # robot's right → body -y.
        bx, by = odom_to_body((1.0, 0.0), (0.0, 0.0, math.pi / 2))
        self.assertAlmostEqual(bx, 0.0, places=6)
        self.assertAlmostEqual(by, -1.0, places=6)

    def test_odom_to_body_translated(self):
        bx, by = odom_to_body((2.0, 3.0), (1.0, 3.0, 0.0))
        self.assertAlmostEqual(bx, 1.0)
        self.assertAlmostEqual(by, 0.0)

    def test_round_trip(self):
        pose = (0.5, -1.2, 0.9)
        odom_pt = (2.3, 0.7)
        b = odom_to_body(odom_pt, pose)
        back = body_to_odom(b, pose)
        self.assertAlmostEqual(back[0], odom_pt[0], places=6)
        self.assertAlmostEqual(back[1], odom_pt[1], places=6)


# ── Steering ─────────────────────────────────────────────────────────

class TestSteer(unittest.TestCase):
    def setUp(self):
        self.p = DriveParams()

    def test_straight_ahead_drives_forward(self):
        v, omega, dist, bearing = steer_to_body_point((1.0, 0.0), self.p)
        self.assertGreater(v, 0.0)
        self.assertAlmostEqual(omega, 0.0)
        self.assertAlmostEqual(dist, 1.0)

    def test_large_bearing_rotates_in_place(self):
        # Goal 90° to the left → rotate, no translation.
        v, omega, _, bearing = steer_to_body_point((0.0, 1.0), self.p)
        self.assertEqual(v, 0.0)
        self.assertGreater(omega, 0.0)
        self.assertAlmostEqual(bearing, math.pi / 2)

    def test_slowdown_near_goal(self):
        far = steer_to_body_point((1.0, 0.0), self.p)[0]
        near = steer_to_body_point((0.2, 0.0), self.p)[0]
        self.assertLessEqual(near, far)

    def test_v_min_snap(self):
        # Just outside arrival, within slowdown → should not stall below v_min.
        v, _, _, _ = steer_to_body_point((0.05, 0.0), self.p)
        self.assertGreaterEqual(v, self.p.v_min_mps - 1e-9)

    def test_omega_capped(self):
        steep = DriveParams(k_omega=100.0, omega_max=0.6)
        _, omega, _, _ = steer_to_body_point((0.0, 0.3), steep)
        self.assertLessEqual(abs(omega), 0.6 + 1e-9)

    def test_rotate_to_heading_aligned(self):
        omega, aligned = rotate_to_heading(0.0, 0.01, self.p)
        self.assertTrue(aligned)
        self.assertEqual(omega, 0.0)

    def test_rotate_to_heading_turns_shortest(self):
        # Target slightly negative → turn negative (CW).
        omega, aligned = rotate_to_heading(0.0, -0.5, self.p)
        self.assertFalse(aligned)
        self.assertLess(omega, 0.0)

    def test_wrap_pi(self):
        self.assertAlmostEqual(wrap_pi(2 * math.pi + 0.5), 0.5)
        self.assertAlmostEqual(wrap_pi(-2 * math.pi - 0.5), -0.5)
        self.assertAlmostEqual(abs(wrap_pi(3 * math.pi)), math.pi)


# ── Swept-footprint safety (Pi port) ─────────────────────────────────

def _meta(res=0.08, n=80):
    return {"resolution_m": res, "origin_x_m": -(n * res) / 2.0,
            "origin_y_m": -(n * res) / 2.0}


def _grid(n=80, fill=1):
    return np.full((n, n), fill, dtype=np.int8)


def _set(grid, meta, x, y, val, res=0.08):
    i = int(math.floor((x - meta["origin_x_m"]) / res))
    j = int(math.floor((y - meta["origin_y_m"]) / res))
    grid[i, j] = val


class TestSweptSafety(unittest.TestCase):
    def setUp(self):
        self.cfg = FootprintConfig(footprint_radius_m=0.22)

    def test_clear_not_blocked(self):
        self.assertFalse(swept_path_blocked(
            _grid(), _meta(), v_mps=0.18, omega_radps=0.0, config=self.cfg))

    def test_obstacle_ahead_blocks(self):
        g, m = _grid(), _meta()
        _set(g, m, 0.30, 0.0, 0)
        self.assertTrue(swept_path_blocked(
            g, m, v_mps=0.18, omega_radps=0.0, config=self.cfg))

    def test_pure_rotation_never_blocks(self):
        g, m = _grid(), _meta()
        _set(g, m, 0.20, 0.0, 0)
        self.assertFalse(swept_path_blocked(
            g, m, v_mps=0.0, omega_radps=0.5, config=self.cfg))

    def test_unknown_close_blocks(self):
        g, m = _grid(), _meta()
        _set(g, m, 0.15, 0.0, -1)
        self.assertTrue(swept_path_blocked(
            g, m, v_mps=0.18, omega_radps=0.0, config=self.cfg))

    def test_empty_map_guard(self):
        cfg = FootprintConfig(footprint_radius_m=0.22, block_on_unknown=False,
                              min_observed_cells=3)
        self.assertTrue(swept_path_blocked(
            _grid(fill=-1), _meta(), v_mps=0.18, omega_radps=0.0, config=cfg))

    def test_reverse_checks_behind(self):
        g, m = _grid(), _meta()
        _set(g, m, -0.25, 0.0, 0)
        self.assertTrue(swept_path_blocked(
            g, m, v_mps=-0.10, omega_radps=0.0, config=self.cfg))

    def test_directional_side_behind_obstacle_does_not_block_forward(self):
        # #3: an obstacle beside-and-behind the robot, within the footprint
        # radius, must NOT veto forward motion (we're driving away from it).
        g, m = _grid(), _meta()
        _set(g, m, -0.10, 0.05, 0)   # behind-left, within r_foot
        self.assertFalse(swept_path_blocked(
            g, m, v_mps=0.18, omega_radps=0.0, config=self.cfg))
        # But it DOES block reverse motion (we'd back into it).
        self.assertTrue(swept_path_blocked(
            g, m, v_mps=-0.10, omega_radps=0.0, config=self.cfg))

    def test_directional_obstacle_ahead_still_blocks(self):
        # The forward hemisphere is still checked: an obstacle dead ahead
        # within the footprint blocks (regression guard for the #3 change).
        g, m = _grid(), _meta()
        _set(g, m, 0.12, 0.0, 0)
        self.assertTrue(swept_path_blocked(
            g, m, v_mps=0.18, omega_radps=0.0, config=self.cfg))


class TestPlanDrive(unittest.TestCase):
    def setUp(self):
        self.params = DriveParams()
        self.foot = FootprintConfig(footprint_radius_m=0.14)
        self.cfg = LocalPlanConfig()

    def _plan(self, grid, meta, goal):
        return plan_drive(grid, meta, goal, self.params, self.foot, self.cfg)

    def test_open_field_goal_ahead_pursues(self):
        g, m = _grid(fill=1), _meta()
        v, omega, mode, _ = self._plan(g, m, (0.8, 0.0))
        self.assertEqual(mode, "pursue")
        self.assertGreater(v, 0.0)
        self.assertAlmostEqual(omega, 0.0, places=2)

    def test_goal_behind_rotates(self):
        g, m = _grid(fill=1), _meta()
        v, omega, mode, _ = self._plan(g, m, (-0.8, 0.1))  # ~173° bearing
        self.assertEqual(mode, "rotate")
        self.assertEqual(v, 0.0)
        self.assertNotEqual(omega, 0.0)

    def test_centers_away_from_left_wall(self):
        # Obstacle close on the left only; goal straight ahead. Should steer
        # right (negative omega) proactively, before any block.
        g, m = _grid(fill=1), _meta()
        for x in (0.25, 0.33, 0.41):
            for y in (0.20, 0.28):
                _set(g, m, x, y, 0)
        v, omega, mode, _ = self._plan(g, m, (1.0, 0.0))
        self.assertEqual(mode, "center")
        self.assertLess(omega, 0.0)            # steering right, off the wall
        self.assertGreater(v, 0.0)

    def test_dead_ahead_within_footprint_blocks(self):
        g, m = _grid(fill=1), _meta()
        _set(g, m, 0.10, 0.0, 0)               # inside the footprint, ahead
        v, omega, mode, _ = self._plan(g, m, (0.8, 0.0))
        self.assertEqual(mode, "blocked")
        self.assertEqual((v, omega), (0.0, 0.0))

    def test_offcenter_obstacle_nudges_around(self):
        # An obstacle just left of center blocks the straight path; a turn
        # right curves clear → nudge (not stop), steering away from it.
        g, m = _grid(fill=1), _meta()
        for x in (0.30, 0.36):
            _set(g, m, x, 0.06, 0)
        v, omega, mode, _ = self._plan(g, m, (0.8, 0.0))
        self.assertEqual(mode, "nudge")
        self.assertGreater(v, 0.0)
        self.assertLess(omega, 0.0)            # turning right, around the obstacle

    def test_wide_dead_ahead_blocks_for_escalation(self):
        # A wide obstacle dead ahead can't be cleared within the preview
        # horizon → blocked, so the caller escalates (Tier-2 picks a
        # subgoal to the side) rather than the bottom tier path-finding.
        g, m = _grid(fill=1), _meta()
        for x in (0.30, 0.38):
            for y in (-0.06, 0.02, 0.06):
                _set(g, m, x, y, 0)
        _, _, mode, _ = self._plan(g, m, (0.8, 0.0))
        self.assertEqual(mode, "blocked")

    def test_seeks_open_arm_beyond_fan(self):
        # A wall blocks the goal direction and the whole fan range; the only
        # opening is a side arm well beyond the fan (~60° left). plan_drive
        # should SEEK it (rotate toward the gap), not give up.
        g, m = _grid(fill=1), _meta()
        # Solid block ahead and to the right; the only opening is far left.
        x = 0.30
        while x <= 0.60 + 1e-9:
            y = -0.6
            while y <= 0.55 + 1e-9:
                _set(g, m, x, y, 0)
                y += 0.04
            x += 0.04
        v, omega, mode, seek_target = self._plan(g, m, (1.0, 0.0))
        self.assertEqual(mode, "seek")
        self.assertEqual(v, 0.0)
        self.assertGreater(seek_target, math.radians(50))   # opening is left, beyond fan
        self.assertGreater(omega, 0.0)                       # turning toward it


class TestDriveableFromRows(unittest.TestCase):
    def test_conversion(self):
        rows = [[True, False, None], [None, True, False]]
        arr = driveable_from_rows(rows, 2, 3)
        self.assertIsNotNone(arr)
        np.testing.assert_array_equal(arr, np.array([[1, 0, -1], [-1, 1, 0]], dtype=np.int8))

    def test_bad_shape_returns_none(self):
        self.assertIsNone(driveable_from_rows([[True]], 2, 3))


if __name__ == "__main__":
    unittest.main()

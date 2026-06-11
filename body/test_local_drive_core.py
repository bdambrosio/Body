"""Tests for Pi-side Tier-3 pure logic: drive core + swept-footprint safety."""
import math
import unittest

import numpy as np

from body.lib.drive_safety import (
    FootprintConfig, driveable_from_rows, swept_path_blocked,
)
from body.lib.local_drive_core import (
    DriveParams, ImuYawCorrector, body_to_odom, odom_to_body,
    quat_wxyz_to_yaw, rotate_to_heading, steer_to_body_point,
    swept_block_response, wrap_pi,
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
        v, omega, dist, bearing, rot = steer_to_body_point((1.0, 0.0), self.p)
        self.assertGreater(v, 0.0)
        self.assertAlmostEqual(omega, 0.0)
        self.assertAlmostEqual(dist, 1.0)
        self.assertFalse(rot)

    def test_large_bearing_rotates_in_place(self):
        # Goal 90° to the left → rotate, no translation.
        v, omega, _, bearing, rot = steer_to_body_point((0.0, 1.0), self.p)
        self.assertEqual(v, 0.0)
        self.assertGreater(omega, 0.0)
        self.assertAlmostEqual(bearing, math.pi / 2)
        self.assertTrue(rot)

    def test_rotate_hysteresis_band_is_sticky(self):
        # Bearing 0.45 rad sits in the band (exit 0.26 < 0.45 < enter 0.61):
        # the mode should be sticky to the incoming `rotating` state.
        band = (math.cos(0.45), math.sin(0.45))  # dist 1, bearing 0.45
        v_d, _, _, _, rot_d = steer_to_body_point(band, self.p, rotating=False)
        self.assertGreater(v_d, 0.0)      # was driving → keep driving
        self.assertFalse(rot_d)
        v_r, _, _, _, rot_r = steer_to_body_point(band, self.p, rotating=True)
        self.assertEqual(v_r, 0.0)        # was rotating → keep rotating
        self.assertTrue(rot_r)

    def test_slowdown_near_goal(self):
        far = steer_to_body_point((1.0, 0.0), self.p)[0]
        near = steer_to_body_point((0.2, 0.0), self.p)[0]
        self.assertLessEqual(near, far)

    def test_v_min_snap(self):
        # Just outside arrival, within slowdown → should not stall below v_min.
        v, _, _, _, _ = steer_to_body_point((0.05, 0.0), self.p)
        self.assertGreaterEqual(v, self.p.v_min_mps - 1e-9)

    def test_omega_capped(self):
        steep = DriveParams(k_omega=100.0, omega_max=0.6)
        _, omega, _, _, _ = steer_to_body_point((0.0, 0.3), steep)
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


class TestDriveableFromRows(unittest.TestCase):
    def test_conversion(self):
        rows = [[True, False, None], [None, True, False]]
        arr = driveable_from_rows(rows, 2, 3)
        self.assertIsNotNone(arr)
        np.testing.assert_array_equal(arr, np.array([[1, 0, -1], [-1, 1, 0]], dtype=np.int8))

    def test_bad_shape_returns_none(self):
        self.assertIsNone(driveable_from_rows([[True]], 2, 3))


class TestSweptBlockResponse(unittest.TestCase):
    KW = dict(thresh_rad=0.10, timeout_s=2.0, k_omega=1.5, omega_max=0.6)

    def test_offaxis_realigns_toward_lookahead(self):
        # Lookahead 30° to the left → rotate left (positive omega) in place.
        resp, omega = swept_block_response(math.radians(30), 0.0, **self.KW)
        self.assertEqual(resp, "realign")
        self.assertGreater(omega, 0.0)
        # …and to the right → rotate right.
        resp, omega = swept_block_response(math.radians(-30), 0.0, **self.KW)
        self.assertEqual(resp, "realign")
        self.assertLess(omega, 0.0)

    def test_omega_clamped(self):
        _resp, omega = swept_block_response(math.radians(170), 0.0, **self.KW)
        self.assertAlmostEqual(omega, 0.6, places=6)        # clamped to omega_max

    def test_aligned_is_a_genuine_block(self):
        # Within the bearing threshold and still blocked → real dead-end.
        resp, omega = swept_block_response(math.radians(3), 0.0, **self.KW)
        self.assertEqual(resp, "block")
        self.assertEqual(omega, 0.0)

    def test_realign_times_out(self):
        # Off-axis but re-aiming too long → give up (bounded episode).
        resp, _omega = swept_block_response(math.radians(30), 2.5, **self.KW)
        self.assertEqual(resp, "block")


class TestImuYawCorrector(unittest.TestCase):
    def test_no_imu_passthrough(self):
        c = ImuYawCorrector()
        self.assertAlmostEqual(c.corrected_theta(0.7, None), 0.7)

    def test_agreement_is_identity(self):
        # Wheels and IMU report the same rotation → no correction, regardless
        # of the (arbitrary) constant offset between the two yaw references.
        c = ImuYawCorrector()
        self.assertAlmostEqual(c.corrected_theta(0.0, 1.0), 0.0)   # baseline
        self.assertAlmostEqual(c.corrected_theta(0.5, 1.5), 0.5)
        self.assertAlmostEqual(c.corrected_theta(-0.3, 0.7), -0.3)

    def test_bump_wheels_missed(self):
        # Ridge kick: chassis rotates +12° but wheel odom is unchanged. The
        # corrected heading must follow the IMU.
        c = ImuYawCorrector()
        bump = math.radians(12)
        self.assertAlmostEqual(c.corrected_theta(0.2, 1.0), 0.2)   # baseline
        self.assertAlmostEqual(c.corrected_theta(0.2, 1.0 + bump), 0.2 + bump)
        # Subsequent commanded rotation stacks on top of the bump correction.
        self.assertAlmostEqual(
            c.corrected_theta(0.2 + 0.3, 1.0 + bump + 0.3), 0.2 + bump + 0.3)

    def test_reset_rebaselines(self):
        # A new goal swallows the accumulated divergence: the goal frame is
        # the odom frame as of the re-pick (the sender used the odom pose).
        c = ImuYawCorrector()
        c.corrected_theta(0.0, 0.0)
        self.assertAlmostEqual(c.corrected_theta(0.0, 0.4), 0.4)
        c.reset()
        self.assertAlmostEqual(c.corrected_theta(0.0, 0.4), 0.0)   # new baseline

    def test_imu_gap_rebaselines(self):
        # IMU goes stale mid-goal → fall back to wheel heading and never
        # difference across the gap when it returns.
        c = ImuYawCorrector()
        c.corrected_theta(0.0, 0.0)
        self.assertAlmostEqual(c.corrected_theta(0.1, None), 0.1)
        self.assertAlmostEqual(c.corrected_theta(0.1, 5.0), 0.1)   # re-baseline
        self.assertAlmostEqual(c.corrected_theta(0.1, 5.2), 0.1 + 0.2)

    def test_frozen_imu_falls_back_to_wheels(self):
        # Hung BNO085: fresh messages, constant yaw, wheels turning. Heading
        # froze and the follower spun in place forever (observed live
        # 2026-06-11). Past MAX_DIVERGENCE the guard must hand back wheel
        # heading.
        c = ImuYawCorrector()
        c.corrected_theta(0.0, 1.0)                                 # baseline
        step = math.radians(5)
        out = 0.0
        for i in range(1, 13):                                      # 60° sweep
            out = c.corrected_theta(i * step, 1.0)
        self.assertAlmostEqual(out, 12 * step)

    def test_frozen_trip_latches_across_goals(self):
        # reset() (new goal) must not hand a hung IMU 30° of fresh trust.
        c = ImuYawCorrector()
        c.corrected_theta(0.0, 1.0)
        for i in range(1, 13):
            c.corrected_theta(i * math.radians(5), 1.0)             # trip
        c.reset()
        c.corrected_theta(0.0, 1.0)
        self.assertAlmostEqual(c.corrected_theta(0.3, 1.0), 0.3)

    def test_frozen_imu_revives_on_motion(self):
        # IMU motion past REVIVE re-baselines and restores trust.
        c = ImuYawCorrector()
        c.corrected_theta(0.0, 1.0)
        for i in range(1, 13):
            c.corrected_theta(i * math.radians(5), 1.0)             # trip
        th, yaw = math.radians(60), 1.0
        for _ in range(4):                                          # alive
            th += math.radians(3)
            yaw += math.radians(3)
            c.corrected_theta(th, yaw)
        c.corrected_theta(th, yaw)                                  # re-baseline
        bump = math.radians(12)
        self.assertAlmostEqual(c.corrected_theta(th, yaw + bump), th + bump)

    def test_pickup_rotation_still_trusted(self):
        # The case the corrector exists for, writ large: the chassis is
        # physically turned 40° (IMU sees it, wheels don't). Big divergence
        # WITH IMU motion is real rotation, not a hung sensor.
        c = ImuYawCorrector()
        c.corrected_theta(0.2, 1.0)                                 # baseline
        yaw = 1.0
        for _ in range(8):
            yaw += math.radians(5)
            out = c.corrected_theta(0.2, yaw)
        self.assertAlmostEqual(out, 0.2 + math.radians(40))

    def test_wraps_across_pi(self):
        c = ImuYawCorrector()
        c.corrected_theta(3.0, 3.0)                                 # baseline 0
        got = c.corrected_theta(3.1, 3.3)                           # +0.2 bump
        self.assertAlmostEqual(got, wrap_pi(3.3), places=6)


class TestQuatToYaw(unittest.TestCase):
    def test_identity(self):
        self.assertAlmostEqual(quat_wxyz_to_yaw(1.0, 0.0, 0.0, 0.0), 0.0)

    def test_pure_z_rotation(self):
        for deg in (-170, -90, -10, 45, 90, 135):
            th = math.radians(deg)
            w, z = math.cos(th / 2), math.sin(th / 2)
            self.assertAlmostEqual(quat_wxyz_to_yaw(w, 0.0, 0.0, z), th,
                                   places=6, msg=f"deg={deg}")


if __name__ == "__main__":
    unittest.main()

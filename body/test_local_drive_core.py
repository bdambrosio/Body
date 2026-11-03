"""Tests for Pi-side Tier-3 pure logic: drive core + swept-footprint safety."""
import math
import unittest

import numpy as np

from body.lib.drive_safety import (
    FootprintConfig, swept_path_blocked,
)
from body.lib.local_drive_core import (
    STATE_CANCELED, STATE_FAULT, STATE_IDLE,
    DriveParams, EStopMonitor, GotoArbiter, ImuYawCorrector, OdomRing,
    body_to_odom, latched_rotate_omega, odom_to_body, quat_wxyz_to_yaw,
    relative_pose, rotate_to_heading, steer_to_body_point,
    swept_block_response, wrap_pi,
)


# ── Frame transforms ────────────────────────────────────────────────

class TestFrames(unittest.TestCase):
    def test_odom_to_body_rotated(self):
        # Robot at origin facing +y (90°). A point at odom +x is on the
        # robot's right → body -y.
        bx, by = odom_to_body((1.0, 0.0), (0.0, 0.0, math.pi / 2))
        self.assertAlmostEqual(bx, 0.0, places=6)
        self.assertAlmostEqual(by, -1.0, places=6)

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
    def test_pure_z_rotation(self):
        for deg in (-170, -90, -10, 45, 90, 135):
            th = math.radians(deg)
            w, z = math.cos(th / 2), math.sin(th / 2)
            self.assertAlmostEqual(quat_wxyz_to_yaw(w, 0.0, 0.0, z), th,
                                   places=6, msg=f"deg={deg}")


class TestLatchedRotate(unittest.TestCase):
    P = DriveParams()

    def test_first_call_follows_bearing(self):
        w, d = latched_rotate_omega(1.0, 0, self.P)
        self.assertGreater(w, 0)
        self.assertEqual(d, 1)

    def test_latch_holds_near_pi(self):
        # Goal behind: bearing noise flips ±π every tick; keep turning +.
        w, d = latched_rotate_omega(-3.0, 1, self.P)
        self.assertGreater(w, 0)
        self.assertEqual(d, 1)

    def test_clear_side_change_relatches(self):
        w, d = latched_rotate_omega(-1.0, 1, self.P)
        self.assertLess(w, 0)
        self.assertEqual(d, -1)

    def test_min_omega(self):
        w, _d = latched_rotate_omega(0.01, 0, self.P, min_omega=0.25)
        self.assertAlmostEqual(w, 0.25)


class TestOdomRing(unittest.TestCase):
    def _msg(self, ts, x, th=0.0, **kw):
        m = {"ts": ts, "x": x, "y": 0.0, "theta": th}
        m.update(kw)
        return m

    def test_interpolates(self):
        r = OdomRing()
        r.push(self._msg(10.0, 0.0, 0.0))
        r.push(self._msg(10.1, 0.1, 0.2))
        p = r.pose_at(10.05)
        self.assertAlmostEqual(p[0], 0.05)
        self.assertAlmostEqual(p[2], 0.1)

    def test_interpolates_heading_across_pi(self):
        r = OdomRing()
        r.push(self._msg(0.0, 0.0, 3.1))
        r.push(self._msg(1.0, 0.0, -3.1))
        self.assertAlmostEqual(abs(r.pose_at(0.5)[2]), math.pi, places=3)

    def test_backward_clock_step_restarts_ring(self):
        r = OdomRing()
        r.push(self._msg(100.0, 0.0))
        r.push(self._msg(100.1, 0.1))
        r.push(self._msg(50.0, 0.2))          # clock stepped back 50 s
        r.push(self._msg(50.1, 0.3))
        self.assertAlmostEqual(r.latest()[1][0], 0.3)
        self.assertIsNotNone(r.pose_at(50.05))

    def test_outside_buffer_is_none(self):
        r = OdomRing()
        r.push(self._msg(10.0, 0.0))
        r.push(self._msg(10.1, 0.1))
        self.assertIsNone(r.pose_at(9.0))
        self.assertIsNone(r.pose_at(11.0))
        self.assertIsNotNone(r.pose_at(10.12))      # within tolerance

    def test_reset_on_session_change_or_seq_decrease(self):
        r = OdomRing()
        self.assertFalse(r.push(self._msg(1.0, 0.0, session_id="aaaa0000", seq=5)))
        self.assertFalse(r.push(self._msg(1.1, 0.1, session_id="aaaa0000", seq=6)))
        self.assertTrue(r.push(self._msg(1.2, 0.0, session_id="bbbb1111", seq=0)))
        self.assertIsNone(r.pose_at(1.05))           # old frame dropped
        self.assertFalse(r.push(self._msg(1.3, 0.0, session_id="bbbb1111", seq=1)))
        self.assertTrue(r.push(self._msg(1.4, 0.0, session_id="bbbb1111", seq=0)))

    def test_legacy_odom_without_fields_never_resets(self):
        r = OdomRing()
        for k in range(5):
            self.assertFalse(r.push(self._msg(k * 0.1, 0.0)))

    def test_relative_pose(self):
        rel = relative_pose((1.0, 0.0, math.pi / 2), (1.0, 1.0, math.pi / 2 + 0.1))
        self.assertAlmostEqual(rel[0], 1.0)
        self.assertAlmostEqual(rel[1], 0.0, places=9)
        self.assertAlmostEqual(rel[2], 0.1)


class TestGotoArbiter(unittest.TestCase):
    def _goto(self, cid, **kw):
        m = {"kind": "goto", "cmd_id": cid, "frame": "odom", "x_m": 1.0, "y_m": 0.0}
        m.update(kw)
        return m

    def test_accept_and_supersede(self):
        a = GotoArbiter()
        self.assertEqual(a.on_command(self._goto(0)), "accepted")
        self.assertEqual(a.on_command(self._goto(3)), "accepted")
        self.assertEqual(a.goal["cmd_id"], 3)
        self.assertEqual(a.watermark, 3)

    def test_stale_goto_rejected_with_watermark(self):
        a = GotoArbiter()
        a.on_command(self._goto(7))
        self.assertEqual(a.on_command(self._goto(5)), "rejected")
        self.assertEqual(a.take_rejects(), [5])
        self.assertEqual(a.take_rejects(), [])
        self.assertEqual(a.goal["cmd_id"], 7)        # active goal untouched

    def test_resend_is_idempotent(self):
        # Desktop resends until it sees a status with its id: same id again
        # must neither restart the goal nor be REJECTED.
        a = GotoArbiter()
        a.on_command(self._goto(4, x_m=1.0))
        goal = a.goal
        self.assertEqual(a.on_command(self._goto(4, x_m=1.0)), "duplicate")
        self.assertIs(a.goal, goal)
        self.assertEqual(a.take_rejects(), [])

    def test_body_frame_accepted_bad_frame_rejected(self):
        a = GotoArbiter()
        self.assertEqual(a.on_command(self._goto(1, frame="body", anchor_ts=12.5)), "accepted")
        self.assertEqual(a.on_command(self._goto(2, frame="map")), "malformed")
        self.assertEqual(a.on_command(self._goto(3, x_m="nan?")), "malformed")
        self.assertEqual(a.take_rejects(), [2, 3])

    def test_cancel_announces_once_then_idle(self):
        a = GotoArbiter()
        a.on_command(self._goto(2))
        a.on_command({"kind": "cancel", "cmd_id": 3})
        self.assertIsNone(a.goal)
        self.assertEqual(a.idle_status(), (STATE_CANCELED, 2, None))
        self.assertEqual(a.idle_status(), (STATE_IDLE, 3, None))
        self.assertEqual(a.on_command(self._goto(3)), "duplicate")

    def test_drop_latches_fault_until_next_command(self):
        a = GotoArbiter()
        a.on_command(self._goto(2))
        self.assertEqual(a.drop("odom_reset"), 2)
        self.assertEqual(a.idle_status(), (STATE_FAULT, 2, "odom_reset"))
        self.assertEqual(a.idle_status(), (STATE_FAULT, 2, "odom_reset"))
        a.on_command(self._goto(5))
        self.assertIsNone(a.fault)

    def test_refused_while_estop(self):
        a = GotoArbiter()
        self.assertEqual(a.on_command(self._goto(9), estop_active=True), "refused_estop")
        self.assertIsNone(a.goal)
        self.assertEqual(a.idle_status(), (STATE_FAULT, 9, "e_stop"))

    def test_finish_only_matching(self):
        a = GotoArbiter()
        a.on_command(self._goto(2))
        a.finish(1)
        self.assertIsNotNone(a.goal)
        a.finish(2)
        self.assertIsNone(a.goal)


class TestEStopMonitor(unittest.TestCase):
    def test_boot_latched_then_clear(self):
        m = EStopMonitor()
        m.on_status({"e_stop_active": True})
        self.assertTrue(m.active())
        m.on_status({"e_stop_active": False})
        m.on_motor_state({"e_stop_active": False, "stall_detected": False})
        self.assertFalse(m.active())

    def test_event_active_until_a_later_clear_report(self):
        m = EStopMonitor()
        m.on_event({"reason": "heartbeat_timeout"})
        self.assertTrue(m.active())
        m.on_motor_state({"e_stop_active": True})
        self.assertTrue(m.active())
        m.on_motor_state({"e_stop_active": False})
        self.assertFalse(m.active())

    def test_stall_counts(self):
        m = EStopMonitor()
        m.on_motor_state({"e_stop_active": False, "stall_detected": True})
        self.assertTrue(m.active())
        self.assertEqual(m.describe(), "stall")


class TestImuWindowGuard(unittest.TestCase):
    def test_wheels_turn_imu_still_falls_back_then_rebinds(self):
        c = ImuYawCorrector()
        t, th, yaw = 0.0, 0.0, 1.0
        c.corrected_theta(th, yaw, t=t)                     # baseline
        out = None
        for _ in range(10):                                 # 0.6 rad/s for 1 s, IMU frozen
            t += 0.1
            th += 0.06
            out = c.corrected_theta(th, yaw, t=t)
        self.assertTrue(c.distrusted)
        self.assertAlmostEqual(out, th)                     # wheel heading
        for _ in range(15):                                 # IMU tracks again
            t += 0.1
            th += 0.03
            yaw += 0.03
            out = c.corrected_theta(th, yaw, t=t)
        self.assertFalse(c.distrusted)
        bump = 0.1                                          # and is trusted again
        self.assertAlmostEqual(c.corrected_theta(th, yaw + bump, t=t + 0.1), th + bump, places=6)

    def test_wheel_glitch_step_ignored(self):
        c = ImuYawCorrector()
        c.corrected_theta(0.0, 1.0, t=0.0)
        c.corrected_theta(1.0, 1.0, t=0.1)                  # 10 rad/s "step" = glitch
        self.assertFalse(c.distrusted)

    def test_rebaseline_no_step(self):
        c = ImuYawCorrector()
        c.corrected_theta(0.0, 0.0)
        self.assertAlmostEqual(c.corrected_theta(0.0, 0.2), 0.2)
        c.rebaseline()                                      # IMU driver restarted
        self.assertAlmostEqual(c.corrected_theta(0.0, 2.5), 0.0)

    def test_baseline_deferred_while_turning_fast(self):
        c = ImuYawCorrector()
        self.assertAlmostEqual(c.corrected_theta(0.0, 1.0, yaw_rate=0.5), 0.0)
        self.assertAlmostEqual(c.corrected_theta(0.1, 1.3, yaw_rate=0.5), 0.1)  # no ref yet
        c.corrected_theta(0.2, 1.4, yaw_rate=0.0)                              # baseline now
        self.assertAlmostEqual(c.corrected_theta(0.2, 1.5, yaw_rate=0.0), 0.3)


class TestGotoArbiterPath(unittest.TestCase):
    def _p(self, cid, seq, n=3, **kw):
        m = {"kind": "path", "cmd_id": cid, "frame": "body", "path_seq": seq, "route_id": 1,
             "path_xy": [[0.1 * k, 0.0] for k in range(n)]}
        m.update(kw)
        return m

    def test_refresh_updates_in_place_and_old_seq_is_absorbed(self):
        a = GotoArbiter()
        self.assertEqual(a.on_command(self._p(7, 1)), "accepted")
        self.assertEqual(a.on_command(self._p(7, 2, n=5)), "path_update")
        self.assertEqual(len(a.goal["path_xy"]), 5)
        self.assertEqual(a.on_command(self._p(7, 2)), "duplicate")
        self.assertEqual(a.on_command(self._p(7, 1)), "duplicate")
        self.assertEqual(a.goal["path_seq"], 2)

    def test_malformed_paths_rejected(self):
        a = GotoArbiter()
        self.assertEqual(a.on_command(self._p(1, 1, n=1)), "malformed")
        self.assertEqual(a.on_command(self._p(2, 1, path_xy=[[0, 0], [1, "x"]])), "malformed")
        self.assertEqual(a.on_command(self._p(3, 1, ttl_s=0)), "malformed")
        self.assertEqual(a.on_command(self._p(4, 1, path_xy=[[0, 0], [float("nan"), 1]])), "malformed")
        self.assertIsNone(a.goal)

    def test_refresh_cannot_revive_a_dropped_goal(self):
        a = GotoArbiter()
        a.on_command(self._p(7, 1))
        a.drop("e_stop")
        self.assertEqual(a.on_command(self._p(7, 2)), "duplicate")
        self.assertIsNone(a.goal)

if __name__ == "__main__":
    unittest.main()

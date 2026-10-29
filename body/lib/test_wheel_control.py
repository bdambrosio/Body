"""Tests for motor-control math: WheelPI, command parsing, arbitration, e-stop latch."""
import math
import unittest

from body.lib.wheel_control import (
    CMD_TIMEOUT_MIN_MS,
    CommandArbiter,
    CommandError,
    EStopLatch,
    WheelPI,
    clamp_timeout_ms,
    parse_cmd_direct,
    parse_cmd_vel,
)

DT = 0.02
MAX_V = 0.9
TICK_M = 2 * math.pi * 0.0325 / 1920


def _signed(out):
    duty, d = out
    return duty if d == "fwd" else -duty


def _pi(**kw):
    base = dict(kp=0.8, ki=2.0, integ_limit=0.5, min_drive_pwm=0.16)
    base.update(kw)
    return WheelPI(**base)


def simulate(pi, cmds, d_static=0.20, d_kin=0.12, tau=0.05):
    """Coulomb-friction wheel, encoder-quantised speed measurement. Returns [(cmd, v_true)]."""
    v = pos = 0.0
    ticks_prev = 0
    out = []
    for c in cmds:
        ticks = int(math.floor(pos / TICK_M))
        vm = (ticks - ticks_prev) * TICK_M / DT
        ticks_prev = ticks
        u = _signed(pi.step(c, vm, DT, MAX_V))
        slope = MAX_V / (1 - d_kin)
        if abs(v) < 1e-4:
            tgt = math.copysign((abs(u) - d_kin) * slope, u) if abs(u) > d_static else 0.0
        else:
            tgt = math.copysign((abs(u) - d_kin) * slope, u) if abs(u) > d_kin else 0.0
            if tgt * v < 0:
                tgt = 0.0
        v += (tgt - v) * DT / tau
        if abs(v) < 1e-3 and tgt == 0.0:
            v = 0.0
        pos += v * DT
        out.append((c, v))
    return out


class TestWheelPI(unittest.TestCase):
    def test_zero_command_outputs_zero_and_resets(self):
        pi = _pi()
        for _ in range(10):
            pi.step(0.1, 0.0, DT, MAX_V)
        self.assertNotEqual(pi.integ, 0.0)
        self.assertEqual(pi.step(0.0, 0.05, DT, MAX_V), (0.0, "fwd"))
        self.assertEqual(pi.integ, 0.0)
        self.assertEqual(pi.kick_tick_count, 0)

    def test_legacy_feedforward_when_kinetic_zero(self):
        pi = _pi(min_drive_pwm=0.0, kp=0.0, ki=0.0)
        duty, d = pi.step(0.45, 0.45, DT, MAX_V)
        self.assertAlmostEqual(duty, 0.5)
        self.assertEqual(d, "fwd")

    def test_coulomb_feedforward(self):
        pi = _pi(min_drive_pwm=0.0, kp=0.0, ki=0.0, kinetic_pwm=0.1)
        pi.moving = True
        duty, _ = pi.step(0.09, 0.09, DT, MAX_V)
        self.assertAlmostEqual(duty, 0.1 + 0.09 * 0.9 / MAX_V)
        duty, d = pi.step(-0.09, -0.09, DT, MAX_V)
        self.assertAlmostEqual(duty, 0.1 + 0.09 * 0.9 / MAX_V)
        self.assertEqual(d, "rev")
        # Full speed still maps to full duty.
        pi2 = _pi(min_drive_pwm=0.0, kp=0.0, ki=0.0, kinetic_pwm=0.1)
        self.assertAlmostEqual(pi2.step(MAX_V, MAX_V, DT, MAX_V)[0], 1.0)

    def test_kick_ramps_then_holds(self):
        pi = _pi(kp=0.0, ki=0.0, kick_ramp_ticks=4)
        duties = [pi.step(0.01, 0.0, DT, MAX_V)[0] for _ in range(6)]
        self.assertAlmostEqual(duties[0], 0.04)
        self.assertAlmostEqual(duties[3], 0.16)
        self.assertAlmostEqual(duties[5], 0.16)

    def test_single_fast_tick_does_not_restart_kick(self):
        pi = _pi(kp=0.0, ki=0.0, kick_ramp_ticks=5, regime_confirm_ticks=2)
        for _ in range(5):
            pi.step(0.03, 0.0, DT, MAX_V)
        self.assertFalse(pi.moving)
        pi.step(0.03, 0.05, DT, MAX_V)  # one noisy fast tick
        self.assertFalse(pi.moving)
        duty, _ = pi.step(0.03, 0.0, DT, MAX_V)
        self.assertAlmostEqual(duty, 0.16)  # floor still fully up, not re-ramping from 0.032

    def test_regime_hysteresis(self):
        pi = _pi(kinetic_threshold_ms=0.02, static_threshold_ms=0.01, regime_confirm_ticks=2)
        for _ in range(2):
            pi.step(0.05, 0.03, DT, MAX_V)
        self.assertTrue(pi.moving)
        for _ in range(5):  # between thresholds: stays moving
            pi.step(0.05, 0.015, DT, MAX_V)
        self.assertTrue(pi.moving)
        for _ in range(2):
            pi.step(0.05, 0.0, DT, MAX_V)
        self.assertFalse(pi.moving)

    def test_moving_backwards_counts_as_static(self):
        pi = _pi(regime_confirm_ticks=1)
        pi.step(0.05, -0.1, DT, MAX_V)
        self.assertFalse(pi.moving)

    def test_integrator_reset_on_sign_change(self):
        pi = _pi(min_drive_pwm=0.0)
        for _ in range(50):
            pi.step(0.05, 0.0, DT, MAX_V)
        self.assertGreater(pi.integ, 0.0)
        duty, d = pi.step(-0.05, 0.0, DT, MAX_V)
        self.assertLessEqual(pi.integ, 0.0)
        self.assertEqual(d, "rev")

    def test_integral_authority_capped(self):
        pi = _pi(min_drive_pwm=0.0, integ_max_duty=0.2)
        for _ in range(1000):
            pi.step(0.05, 0.0, DT, MAX_V)
        self.assertLessEqual(pi.ki * abs(pi.integ), 0.2 + 1e-9)

    def test_no_windup_while_saturated(self):
        pi = _pi(min_drive_pwm=0.0, kp=5.0, integ_max_duty=1.0)
        for _ in range(100):
            duty, _ = pi.step(MAX_V, 0.0, DT, MAX_V)
            self.assertEqual(duty, 1.0)
        self.assertEqual(pi.integ, 0.0)

    def test_no_integration_while_kick_ramping(self):
        pi = _pi(kp=0.0, kick_ramp_ticks=5)
        for _ in range(4):
            pi.step(0.02, 0.0, DT, MAX_V)
        self.assertEqual(pi.integ, 0.0)

    def test_kick_escalation_bounded(self):
        pi = _pi(kp=0.0, ki=0.0, kick_ramp_ticks=1, kick_escalation_per_s=0.5, integ_max_duty=0.2)
        duty = 0.0
        for _ in range(200):
            duty, _ = pi.step(0.02, 0.0, DT, MAX_V)
        self.assertAlmostEqual(duty, 0.16 + 0.2)

    def test_nonfinite_command_is_zero(self):
        pi = _pi()
        self.assertEqual(pi.step(float("nan"), 0.0, DT, MAX_V), (0.0, "fwd"))

    def test_low_speed_tracking_without_stick_slip(self):
        """Measured-friction config tracks 0.035 m/s quickly and never stalls once moving."""
        pi = _pi(min_drive_pwm=0.20, kinetic_pwm=0.12, kick_escalation_per_s=0.25)
        out = simulate(pi, [0.035] * 150)
        first = next(i for i, (_, v) in enumerate(out) if abs(v) > 0.005)
        self.assertLess(first, 20)  # < 0.4 s to break away
        tail = out[75:]
        self.assertEqual(sum(1 for _, v in tail if abs(v) < 0.002), 0)
        mean = sum(v for _, v in tail) / len(tail)
        self.assertAlmostEqual(mean, 0.035, delta=0.01)

    def test_underestimated_breakaway_still_starts(self):
        """Breakaway (0.20) above min_drive_pwm (0.16): escalation still gets it moving."""
        pi = _pi(min_drive_pwm=0.16, kinetic_pwm=0.08, kick_escalation_per_s=0.25)
        out = simulate(pi, [0.035] * 100)
        first = next((i for i, (_, v) in enumerate(out) if abs(v) > 0.005), None)
        self.assertIsNotNone(first)
        self.assertLess(first, 25)

    def test_reversal_does_not_drive_wrong_way(self):
        pi = _pi(min_drive_pwm=0.20, kinetic_pwm=0.12, kick_escalation_per_s=0.25)
        out = simulate(pi, [0.05] * 75 + [-0.05] * 75)
        tail = out[75 + 40:]
        self.assertTrue(all(v <= 0.0 for _, v in tail))
        self.assertLess(sum(v for _, v in tail) / len(tail), -0.03)


class TestParse(unittest.TestCase):
    def test_cmd_vel_ok_and_default_timeout(self):
        self.assertEqual(parse_cmd_vel({"linear": 0.1, "angular": -1}, 500), (0.1, -1.0, 500.0))
        self.assertEqual(parse_cmd_vel({"linear": 0, "angular": 0, "timeout_ms": 200}), (0.0, 0.0, 200.0))

    def test_rejects_missing_nonfinite_bool_string(self):
        bad = [
            {"angular": 0.0},
            {"linear": float("nan"), "angular": 0.0},
            {"linear": float("inf"), "angular": 0.0},
            {"linear": True, "angular": 0.0},
            {"linear": "0.1", "angular": 0.0},
            {"linear": 0.1, "angular": 0.0, "timeout_ms": "x"},
            {"linear": 0.1, "angular": 0.0, "timeout_ms": float("nan")},
        ]
        for msg in bad:
            with self.assertRaises(CommandError, msg=msg):
                parse_cmd_vel(msg)

    def test_cmd_direct(self):
        self.assertEqual(parse_cmd_direct({"left": 0.1, "right": 0.2}), (0.1, 0.2, 500.0))
        with self.assertRaises(CommandError):
            parse_cmd_direct({"right": 0.2, "ts": 1.0})

    def test_clamp_timeout(self):
        self.assertEqual(clamp_timeout_ms(10000, 1000), 1000)
        self.assertEqual(clamp_timeout_ms(0, 1000), CMD_TIMEOUT_MIN_MS)
        self.assertEqual(clamp_timeout_ms(-5, 1000), CMD_TIMEOUT_MIN_MS)
        self.assertEqual(clamp_timeout_ms(300, 1000), 300)


class TestArbiter(unittest.TestCase):
    def test_latest_receipt_wins_regardless_of_sender_ts(self):
        arb = CommandArbiter(max_timeout_ms=1000)
        # A desktop-stamped zero cmd_direct (sender clock far ahead is irrelevant now).
        arb.store("direct", 0.0, 0.0, 500, now_mono=10.0)
        arb.store("twist", 0.2, 0.0, 500, now_mono=10.1)
        self.assertEqual(arb.select(10.15).kind, "twist")

    def test_each_topic_expires_on_its_own_timer(self):
        arb = CommandArbiter(max_timeout_ms=1000)
        arb.store("direct", 0.0, 0.0, 500, now_mono=10.0)
        for i in range(10):  # Tier-3 cmd_vel stream must not keep the direct alive
            arb.store("twist", 0.2, 0.0, 500, now_mono=10.1 + 0.1 * i)
        arb.select(10.6)
        self.assertNotIn("direct", arb._cmds)
        self.assertEqual(arb.select(11.0).kind, "twist")
        self.assertIsNone(arb.select(11.6))

    def test_older_unexpired_command_resumes(self):
        arb = CommandArbiter(max_timeout_ms=1000)
        arb.store("direct", 0.1, 0.1, 1000, now_mono=0.0)
        arb.store("twist", 0.2, 0.0, 50, now_mono=0.1)
        self.assertEqual(arb.select(0.12).kind, "twist")
        self.assertEqual(arb.select(0.2).kind, "direct")

    def test_timeout_clamped(self):
        arb = CommandArbiter(max_timeout_ms=1000)
        arb.store("twist", 0.2, 0.0, 10000, now_mono=0.0)
        self.assertIsNotNone(arb.select(0.99))
        self.assertIsNone(arb.select(1.01))


class TestEStopLatch(unittest.TestCase):
    def test_motion_cannot_clear_and_rearm_needs_heartbeat(self):
        e = EStopLatch()
        e.on_emergency_stop()
        self.assertTrue(e.active)
        self.assertFalse(e.on_rearm(heartbeat_ok=False))
        self.assertTrue(e.active)
        self.assertTrue(e.on_rearm(heartbeat_ok=True))
        self.assertFalse(e.active)

    def test_status_level_and_watchdog_clear_transition(self):
        e = EStopLatch()
        e.on_emergency_stop()
        e.on_status(True)
        self.assertTrue(e.active)
        e.on_status(True)
        self.assertTrue(e.active)
        e.on_status(False)  # watchdog accepted a re-arm
        self.assertFalse(e.active)

    def test_status_false_steady_does_not_clear_local_latch(self):
        e = EStopLatch()
        e.on_status(False)
        e.on_emergency_stop()
        e.on_status(False)
        self.assertTrue(e.active)


if __name__ == "__main__":
    unittest.main()

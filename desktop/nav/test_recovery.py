"""Tests for the recovery supervisor (pure state machine)."""

from __future__ import annotations

import unittest

from desktop.localization.pose_verify import BAD, OK, PENDING, UNKNOWN, Verdict
from desktop.nav.recovery import (
    RESUME, WAIT, RecoveryConfig, RecoveryHooks, RecoverySupervisor,
)


class FakeVerifier:
    def __init__(self, seq=None):
        self.seq = list(seq or [Verdict(OK, "fine")])
        self.marks = 0
        self.restarts = 0

    def mark_fault(self):
        self.marks += 1

    def restart_window(self):
        self.restarts += 1

    def poll(self):
        pass

    def verdict(self, *, final=False):
        v = self.seq[0]
        if len(self.seq) > 1:
            self.seq.pop(0)
        if v.kind == PENDING and final:
            return Verdict(UNKNOWN, "timeout")
        return v


def _sup(verifier=None, **hooks):
    return RecoverySupervisor(RecoveryHooks(verifier=verifier or FakeVerifier(), **hooks))


def _run(sup, t0, t1, *, link_ok=True, dt=0.1):
    t, out = t0, WAIT
    while t <= t1:
        out = sup.step(t, link_ok=link_ok)
        if out == RESUME:
            return t, out
        t += dt
    return t, out


class TestRecoverySupervisor(unittest.TestCase):
    def test_idle_never_resumes(self):
        self.assertEqual(_sup().step(0.0, link_ok=True), WAIT)

    def test_resumes_after_stable_link_and_ok_pose(self):
        s = _sup()
        s.begin(0.0, "pose_lost")
        _t, out = _run(s, 0.0, 1.9)
        self.assertEqual(out, WAIT)               # link not stable 2 s yet
        t, out = _run(s, 2.0, 3.0)
        self.assertEqual(out, RESUME)
        self.assertFalse(s.active())
        self.assertIn("ok", s.resume_detail)

    def test_link_flicker_restarts_stability_clock(self):
        s = _sup()
        s.begin(0.0, "pose_lost")
        _run(s, 0.0, 1.5)
        s.step(1.6, link_ok=False)
        _t, out = _run(s, 1.7, 3.5)
        self.assertEqual(out, WAIT)
        t, out = _run(s, 3.6, 4.0)
        self.assertEqual(out, RESUME)

    def test_flapping_link_raises_stability_requirement(self):
        s = _sup()
        for i in range(3):
            s.begin(i * 10.0, "pose_lost")
        self.assertEqual(s.status(30.0)["stable_req_s"], RecoveryConfig().stable_link_s[2])
        # Faults age out of the flap window.
        s.begin(1000.0, "pose_lost")
        self.assertEqual(s.status(1000.0)["stable_req_s"], RecoveryConfig().stable_link_s[0])

    def test_pending_then_unknown_after_timeout_resumes(self):
        s = _sup(FakeVerifier([Verdict(PENDING, "waiting")]))
        s.begin(0.0, "pose_lost")
        t, out = _run(s, 0.0, 6.0)
        self.assertEqual(out, WAIT)
        t, out = _run(s, 6.0, 8.0)
        self.assertEqual(out, RESUME)
        self.assertIn("unknown", s.resume_detail)

    def test_bad_pose_relocalizes_then_verifies_again(self):
        calls = []
        ver = FakeVerifier([Verdict(BAD, "off"), Verdict(OK, "fine")])
        s = _sup(ver, relocalize=lambda: calls.append(1) or (True, "cp_003"))
        s.begin(0.0, "pose_lost")
        t, out = _run(s, 0.0, 4.0)
        self.assertEqual(out, RESUME)
        self.assertEqual(calls, [1])
        self.assertGreaterEqual(ver.restarts, 2)

    def test_relocalize_failure_backs_off_and_keeps_trying(self):
        calls = []
        ver = FakeVerifier([Verdict(BAD, "off")])
        s = _sup(ver, relocalize=lambda: calls.append(1) or (False, "no checkpoint"))
        s.begin(0.0, "pose_lost")
        _run(s, 0.0, 2.5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(s.status(2.5)["phase"], "relocalize_wait")
        _run(s, 2.6, 12.0)                         # first backoff 10 s
        self.assertEqual(len(calls), 1)
        _run(s, 12.1, 13.0)
        self.assertEqual(len(calls), 2)
        _run(s, 13.1, 200.0)
        self.assertGreaterEqual(len(calls), 4)      # never gives up
        self.assertTrue(s.active())

    def test_no_resume_while_estop_latched_and_rearm_rate_limited(self):
        latched = [True]
        rearms = []
        s = _sup(estop_latched=lambda: latched[0],
                 rearm=lambda: rearms.append(1) or True)
        s.begin(0.0, "pose_lost")
        t, out = _run(s, 0.0, 9.0)
        self.assertEqual(out, WAIT)
        self.assertEqual(len(rearms), 1)            # first at 2 s, next not before 12 s
        _run(s, 9.1, 12.5)
        self.assertEqual(len(rearms), 2)
        latched[0] = False
        t, out = _run(s, 12.6, 14.0)
        self.assertEqual(out, RESUME)

    def test_no_rearm_while_link_down_or_goal_live(self):
        rearms = []
        s = _sup(estop_latched=lambda: True, rearm=lambda: rearms.append(1) or True)
        s.begin(0.0, "pose_lost")
        _run(s, 0.0, 5.0, link_ok=False)
        t = 5.0
        while t < 10.0:
            s.step(t, link_ok=True, goal_live=True)
            t += 0.1
        self.assertEqual(rearms, [])

    def test_cancel_stops_everything(self):
        s = _sup()
        s.begin(0.0, "pose_lost")
        s.cancel()
        _t, out = _run(s, 0.0, 5.0)
        self.assertEqual(out, WAIT)


if __name__ == "__main__":
    unittest.main()


class TestLinkEStopGuard(unittest.TestCase):
    def _g(self, active_at_start=False):
        from desktop.nav.recovery import LinkEStopGuard
        g = LinkEStopGuard()
        g.on_fault_start(100.0, estop_active=active_at_start)
        return g

    def test_heartbeat_timeout_during_fault_may_rearm(self):
        g = self._g()
        self.assertTrue(g.may_rearm(
            estop_active=True,
            emergency={"reason": "heartbeat_timeout", "source": "watchdog"},
            emergency_ts=101.0))

    def test_lost_emergency_message_still_link_caused(self):
        self.assertTrue(self._g().may_rearm(
            estop_active=True, emergency=None, emergency_ts=0.0))
        # An emergency from long before the fault doesn't count against it.
        self.assertTrue(self._g().may_rearm(
            estop_active=True, emergency={"reason": "operator"}, emergency_ts=10.0))

    def test_operator_or_other_estop_never(self):
        g = self._g()
        self.assertFalse(g.may_rearm(
            estop_active=True, emergency={"reason": "operator", "source": "desktop"},
            emergency_ts=101.0))
        self.assertFalse(g.may_rearm(
            estop_active=True,
            emergency={"reason": "heartbeat_timeout", "source": "desktop"},
            emergency_ts=101.0))

    def test_already_latched_at_fault_start_never(self):
        self.assertFalse(self._g(active_at_start=True).may_rearm(
            estop_active=True, emergency=None, emergency_ts=0.0))

    def test_not_latched_nothing_to_do(self):
        self.assertFalse(self._g().may_rearm(
            estop_active=False, emergency=None, emergency_ts=0.0))

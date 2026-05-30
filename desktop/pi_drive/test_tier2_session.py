"""Tests for the pure Tier-2 debug session (no Qt, no zenoh)."""
import math
import unittest

import numpy as np

from desktop.pi_drive.tier2_session import Tier2Session, Tier2SessionConfig

RES, HALF = 0.08, 2.5
N = 2 * int(math.ceil(HALF / RES))
META = {"resolution_m": RES, "origin_x_m": -HALF, "origin_y_m": -HALF,
        "nx": N, "ny": N, "frame": "body"}


def _clear():
    return np.ones((N, N), dtype=np.int8)


def _codes(tick):
    return [e.code for e in tick.events]


class FakeIO:
    def __init__(self, start=1000):
        self.sent = []
        self.cancels = 0
        self._cid = start

    def send_goto_from_body(self, bx, by, *, arrival_tol_m=None, v_max=None):
        self._cid += 1
        self.sent.append((bx, by, self._cid))
        return self._cid

    def cancel(self):
        self.cancels += 1


def _kw(**over):
    kw = dict(odom_pose=(0.0, 0.0, 0.0), grid=_clear(), meta=META, scan_age_s=0.05,
              tier3_status=None, e_stop_active=False, heartbeat_ok=True)
    kw.update(over)
    return kw


class TestTier2Session(unittest.TestCase):
    def _sess(self):
        return Tier2Session(FakeIO(), Tier2SessionConfig())

    def test_no_target_no_decision(self):
        s = self._sess()
        t = s.tick(0.0, **_kw())
        self.assertFalse(t.has_target)
        self.assertIsNone(t.decision)

    def test_dry_computes_decision_no_send(self):
        s = self._sess()
        s.set_target_from_body(1.0, 0.0, (0.0, 0.0, 0.0))
        t = s.tick(0.0, **_kw())
        self.assertIsNotNone(t.decision)
        self.assertTrue(t.decision.ok)
        self.assertTrue(t.decision.capped_at_target)
        self.assertIsNone(t.sent_cmd_id)          # drive off → dry
        self.assertEqual(s._io.cancels, 0)

    def test_drive_sends_goto(self):
        s = self._sess()
        s.set_drive(True)
        s.set_target_from_body(1.0, 0.0, (0.0, 0.0, 0.0))
        t = s.tick(0.0, **_kw())
        self.assertIsNotNone(t.sent_cmd_id)
        self.assertEqual(len(s._io.sent), 1)

    def test_repicks_after_arrived(self):
        s = self._sess()
        s.set_drive(True)
        s.set_target_from_body(1.0, 0.0, (0.0, 0.0, 0.0))
        c1 = s.tick(0.0, **_kw()).sent_cmd_id
        # Tier-3 reports DRIVING for our cmd → no re-send.
        t2 = s.tick(0.1, **_kw(tier3_status={"cmd_id": c1, "state": "DRIVING"}))
        self.assertIsNone(t2.sent_cmd_id)
        # Tier-3 ARRIVED → re-pick toward the same target.
        t3 = s.tick(0.2, **_kw(tier3_status={"cmd_id": c1, "state": "ARRIVED"}))
        self.assertIsNotNone(t3.sent_cmd_id)
        self.assertEqual(len(s._io.sent), 2)

    def test_cmd_id_mismatch_event(self):
        s = self._sess()
        s.set_drive(True)
        s.set_target_from_body(1.0, 0.0, (0.0, 0.0, 0.0))
        c1 = s.tick(0.0, **_kw()).sent_cmd_id
        # Tier-3 stuck servicing a much older id (the 129-collision pattern).
        t = s.tick(0.1, **_kw(tier3_status={"cmd_id": c1 - 500, "state": "IDLE"}))
        self.assertIn("cmd_id_mismatch", _codes(t))

    def test_estop_event_edge_triggered(self):
        s = self._sess()
        s.set_target_from_body(1.0, 0.0, (0.0, 0.0, 0.0))
        t1 = s.tick(0.0, **_kw(e_stop_active=True))
        self.assertIn("e_stop_active", _codes(t1))
        # Same state next tick → no duplicate event.
        t2 = s.tick(0.1, **_kw(e_stop_active=True))
        self.assertNotIn("e_stop_active", _codes(t2))

    def test_swept_block_surfaces_from_tier3(self):
        s = self._sess()
        s.set_target_from_body(1.0, 0.0, (0.0, 0.0, 0.0))
        t = s.tick(0.0, **_kw(tier3_status={
            "cmd_id": 1, "state": "BLOCKED", "blocked_reason": "swept_block", "mode": "blocked"}))
        self.assertIn("tier3_swept_block", _codes(t))

    def test_no_scan_event(self):
        s = self._sess()
        s.set_target_from_body(1.0, 0.0, (0.0, 0.0, 0.0))
        t = s.tick(0.0, **_kw(grid=None, meta=None))
        self.assertIsNone(t.decision)
        self.assertIn("no_scan", _codes(t))

    def test_target_reached_cancels(self):
        s = self._sess()
        s.set_drive(True)
        s.set_target_from_body(0.10, 0.0, (0.0, 0.0, 0.0))   # within arrival tol
        t = s.tick(0.0, **_kw())
        self.assertIn("target_reached", _codes(t))
        self.assertIsNone(t.decision)
        self.assertGreaterEqual(s._io.cancels, 1)

    def test_clear_target_cancels(self):
        s = self._sess()
        s.set_target_from_body(1.0, 0.0, (0.0, 0.0, 0.0))
        s.clear_target()
        self.assertFalse(s.has_target)
        self.assertGreaterEqual(s._io.cancels, 1)


if __name__ == "__main__":
    unittest.main()

"""DriveClient goto delivery: ack/resend, REJECTED re-seed, body-frame
anchoring (X5, X6). No zenoh: a fake session records puts."""
from __future__ import annotations

import json
import unittest

from desktop.pi_drive.drive_client import DriveClient


class _Payload:
    def __init__(self, obj):
        self._s = json.dumps(obj)

    def to_string(self):
        return self._s


class _Sample:
    def __init__(self, obj):
        self.payload = _Payload(obj)


class _FakeSession:
    def __init__(self):
        self.puts = []

    def put(self, key, payload, **kw):
        self.puts.append((key, json.loads(payload)))


def _client():
    c = DriveClient("tcp/127.0.0.1:1")
    c._session = _FakeSession()
    c._on_odom(_Sample({"x": 1.0, "y": 2.0, "theta": 0.0}))
    return c


class TestDriveClient(unittest.TestCase):
    def test_body_frame_goal_with_anchor_ts(self):
        c = _client()
        cid = c.send_goto_from_body(0.5, 0.1, anchor_ts=123.5)
        key, msg = c._session.puts[-1]
        self.assertEqual(key, "body/drive/goto")
        self.assertEqual(msg["frame"], "body")
        self.assertEqual(msg["anchor_ts"], 123.5)
        self.assertEqual((msg["x_m"], msg["y_m"]), (0.5, 0.1))
        self.assertEqual(msg["cmd_id"], cid)

    def test_odom_frame_without_anchor(self):
        c = _client()
        c.send_goto_from_body(0.5, 0.0)
        _key, msg = c._session.puts[-1]
        self.assertEqual(msg["frame"], "odom")
        self.assertAlmostEqual(msg["x_m"], 1.5)
        self.assertNotIn("anchor_ts", msg)

    def test_resend_until_acked(self):
        c = _client()
        cid = c.send_goto_from_body(0.5, 0.0, anchor_ts=1.0)
        t0 = c._pending[2]
        c.pump_resends(t0 + 0.5)
        self.assertEqual(len(c._session.puts), 1)      # not yet
        c.pump_resends(t0 + 1.1)
        self.assertEqual(len(c._session.puts), 2)      # resent, same cmd_id
        self.assertEqual(c._session.puts[-1][1]["cmd_id"], cid)
        c._on_status(_Sample({"cmd_id": cid, "state": "DRIVING"}))
        self.assertTrue(c.is_acked(cid))
        c.pump_resends(t0 + 5.0)
        self.assertEqual(len(c._session.puts), 2)
        self.assertFalse(c.ack_failed(cid))

    def test_ack_failed_after_max_resends(self):
        c = _client()
        cid = c.send_goto_from_body(0.5, 0.0)
        t = c._pending[2]
        for _ in range(DriveClient.MAX_RESENDS):
            t += 1.05
            c.pump_resends(t)
        self.assertFalse(c.ack_failed(cid))
        t += 1.05
        c.pump_resends(t)
        self.assertTrue(c.ack_failed(cid))
        self.assertEqual(len(c._session.puts), 1 + DriveClient.MAX_RESENDS)

    def test_rejected_reseeds_past_watermark(self):
        c = _client()
        cid = c.send_goto_from_body(0.5, 0.0)
        wm = cid + 1000
        c._on_status(_Sample({"cmd_id": cid, "state": "REJECTED", "watermark": wm}))
        cid2 = c.send_goto_from_body(0.5, 0.0)
        self.assertEqual(cid2, wm + 1)

    def test_seeds_past_last_seen_status(self):
        c = _client()
        c._on_status(_Sample({"cmd_id": c._cmd_id + 500, "state": "IDLE"}))
        before = c._cmd_id + 500
        c.cancel()
        self.assertEqual(c._session.puts[-1][1]["cmd_id"], before + 1)
        self.assertEqual(c._session.puts[-1][1]["kind"], "cancel")


class TestDepthEventLatch(unittest.TestCase):
    def test_short_stop_between_redraws_is_latched_once(self):
        c = _client()
        drive = {"state": "DRIVING", "cmd_id": 1}
        stop = {"state": "BLOCKED", "blocked_reason": "depth_block", "cmd_id": 1, "depth_hits": 30}
        self.assertIsNone(c.pop_depth_event())
        c._on_status(_Sample(drive))
        c._on_status(_Sample(stop))       # stop starts and ends before any redraw
        c._on_status(_Sample(stop))
        c._on_status(_Sample(drive))
        ev = c.pop_depth_event()
        self.assertEqual(ev["blocked_reason"], "depth_block")
        self.assertIsNone(c.pop_depth_event())
        self.assertEqual(c.latest_status()["state"], "DRIVING")
        c._on_status(_Sample({"state": "DRIVING", "depth_would_block": True, "cmd_id": 1}))
        self.assertTrue(c.pop_depth_event()["depth_would_block"])


if __name__ == "__main__":
    unittest.main()

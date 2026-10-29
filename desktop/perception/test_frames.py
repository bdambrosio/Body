"""Tagged RGB capture: routing past the shared slot, pose stamped at arrival."""
from __future__ import annotations

import base64
import json
import types
import unittest

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.perception.frames import FrameGrabber


class _Session:
    def __init__(self):
        self.sent = []

    def put(self, topic, payload):
        self.sent.append((topic, json.loads(payload)))


def _rgb_sample(rid, ok=True, jpeg=b"\xff\xd8fake", ts=123.0):
    msg = {"ts": ts, "request_id": rid, "ok": ok, "format": "jpeg",
           "encoding": "base64", "width": 640, "height": 400}
    if ok:
        msg["data"] = base64.b64encode(jpeg).decode("ascii")
    else:
        msg["error"] = "no frame"
    return types.SimpleNamespace(payload=json.dumps(msg).encode("utf-8"))


class TestTaggedCapture(unittest.TestCase):
    def setUp(self):
        self.ctl = StubController(StubConfig())
        self.ctl._session = _Session()
        self.pose = (1.0, 2.0, 0.5)
        self.grabber = FrameGrabber(self.ctl, lambda: self.pose)

    def test_request_payload(self):
        rid = self.grabber.grab(lambda f: None, max_width=320)
        topic, body = self.ctl._session.sent[-1]
        self.assertEqual(topic, self.ctl.config.topics.oakd_config)
        self.assertEqual(body["action"], "capture_rgb")
        self.assertEqual(body["request_id"], rid)
        self.assertEqual(body["max_width"], 320)
        self.assertNotIn("jpeg_quality", body)

    def test_reply_goes_to_callback_not_shared_slot(self):
        got = []
        rid = self.grabber.grab(got.append)
        self.ctl._on_oakd_rgb(_rgb_sample(rid))
        self.assertEqual(len(got), 1)
        f = got[0]
        self.assertEqual(f.request_id, rid)
        self.assertEqual(f.jpeg, b"\xff\xd8fake")
        self.assertEqual((f.width, f.height), (640, 400))
        self.assertEqual(f.pi_ts, 123.0)
        self.assertGreater(f.recv_ts, 0.0)
        self.assertEqual(f.pose, self.pose)
        self.assertIsNone(self.ctl.state.rgb_jpeg)

    def test_pose_read_at_arrival_not_at_request(self):
        got = []
        rid = self.grabber.grab(got.append)
        self.pose = (5.0, 6.0, -1.0)
        self.ctl._on_oakd_rgb(_rgb_sample(rid))
        self.assertEqual(got[0].pose, (5.0, 6.0, -1.0))

    def test_tagged_reply_survives_pending_on_demand_request(self):
        # An on-demand request in flight would reject other ids in the
        # shared path; tagged replies must still be delivered.
        got = []
        rid = self.grabber.grab(got.append)
        self.ctl.request_rgb()
        self.ctl._on_oakd_rgb(_rgb_sample(rid))
        self.assertEqual(len(got), 1)

    def test_failed_capture_calls_on_fail(self):
        got, fails = [], []
        rid = self.grabber.grab(got.append,
                                on_fail=lambda r, e: fails.append((r, e)))
        self.ctl._on_oakd_rgb(_rgb_sample(rid, ok=False))
        self.assertEqual(got, [])
        self.assertEqual(fails, [(rid, "no frame")])

    def test_untagged_reply_still_fills_shared_slot(self):
        self.ctl._on_oakd_rgb(_rgb_sample("streaming-1"))
        self.assertEqual(self.ctl.state.rgb_jpeg, b"\xff\xd8fake")

    def test_reply_delivered_once(self):
        got = []
        rid = self.grabber.grab(got.append)
        self.ctl._on_oakd_rgb(_rgb_sample(rid))
        self.ctl._on_oakd_rgb(_rgb_sample(rid))
        self.assertEqual(len(got), 1)
        self.assertEqual(self.ctl.state.rgb_jpeg, b"\xff\xd8fake")

    def test_expired_requests_are_pruned(self):
        self.grabber.grab(lambda f: None)
        self.assertEqual(len(self.ctl._tagged_rgb), 1)
        rid0 = next(iter(self.ctl._tagged_rgb))
        cb, _ = self.ctl._tagged_rgb[rid0]
        self.ctl._tagged_rgb[rid0] = (cb, 0.0)
        self.grabber.grab(lambda f: None)
        self.assertNotIn(rid0, self.ctl._tagged_rgb)
        self.assertEqual(len(self.ctl._tagged_rgb), 1)

    def test_no_session(self):
        ctl = StubController(StubConfig())
        self.assertIsNone(FrameGrabber(ctl, lambda: None).grab(lambda f: None))

    def test_pose_unavailable(self):
        got = []
        g = FrameGrabber(self.ctl, lambda: None)
        rid = g.grab(got.append)
        self.ctl._on_oakd_rgb(_rgb_sample(rid))
        self.assertIsNone(got[0].pose)


if __name__ == "__main__":
    unittest.main()

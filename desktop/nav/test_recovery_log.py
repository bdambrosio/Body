"""Tests for recovery diagnostic dumps."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from desktop.nav.recovery_log import RecoveryLogger
from desktop.perception.frames import PosedFrame


class TestRecoveryLogger(unittest.TestCase):
    def test_writes_event_frames_and_scan_rate_limited(self):
        with tempfile.TemporaryDirectory() as d:
            t = [100.0]
            lg = RecoveryLogger(d, clock=lambda: t[0])
            fr = PosedFrame("r1", b"\xff\xd8jpeg", 320, 200, 1.0, 99.0, (1.0, 2.0, 0.5))
            p = lg.event("wait_link", "pose_lost", frames=[fr],
                         scan={"ranges": [1.0, None, 2.0], "angle_min": 0.0,
                               "angle_increment": 0.1},
                         pose=(1.0, 2.0, 0.5), status={"phase": "wait_link"})
            self.assertIsNotNone(p)
            meta = json.load(open(os.path.join(p, "event.json")))
            self.assertEqual(meta["tag"], "wait_link")
            self.assertEqual(meta["frames"][0]["pose"], [1.0, 2.0, 0.5])
            self.assertTrue(os.path.isfile(os.path.join(p, "frame_0.jpg")))
            self.assertTrue(os.path.isfile(os.path.join(p, "scan.npz")))
            self.assertIsNone(lg.event("wait_link", "again"))     # rate limit
            self.assertIsNone(lg.event("verify", "skipped tag"))
            t[0] = 103.0
            self.assertIsNotNone(lg.event("wait_link", "later"))

    def test_disabled_without_dir(self):
        self.assertIsNone(RecoveryLogger(None).event("idle", "x"))


if __name__ == "__main__":
    unittest.main()


class TestSnapshotRing(unittest.TestCase):
    def test_rate_size_and_after_filter(self):
        from desktop.perception.frames import SnapshotRing

        class Grabber:
            def __init__(self):
                self.calls = 0

            def grab(self, on_frame, **kw):
                self.calls += 1
                on_frame(PosedFrame(str(self.calls), b"j", 320, 200, 0.0,
                                    float(self.calls), (0.0, 0.0, 0.0)))
                return str(self.calls)

        g = Grabber()
        ring = SnapshotRing(g, size=3, period_s=1.0)
        for i in range(50):
            ring.tick(i * 0.1, active=True)          # 5 s → 5 grabs
        self.assertEqual(g.calls, 5)
        self.assertEqual([f.recv_ts for f in ring.frames()], [3.0, 4.0, 5.0])
        self.assertEqual(ring.latest(after=3.5).recv_ts, 5.0)
        self.assertIsNone(ring.latest(after=9.0))
        ring.tick(100.0, active=False)
        self.assertEqual(g.calls, 5)

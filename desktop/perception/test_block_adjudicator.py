"""BlockAdjudicator: one query per block run, verdict → caution-only advisory."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest

from desktop.nav.hierarchical_drive import HierarchicalDrive, HierConfig, HierState
from desktop.nav.test_hierarchical_drive import (
    FakeDriveIO, FakePose, _clear_scan, _runner,
)
from desktop.perception.block_adjudicator import (
    AdjudicatorConfig, BlockAdjudicator, advisory_from_verdict,
)
from desktop.perception.vlm_worker import VlmWorker

CFG = AdjudicatorConfig()


class TestAdvisoryMapping(unittest.TestCase):
    def test_moving_thing_holds(self):
        a = advisory_from_verdict(
            {"blocked": True, "what": "person", "confidence": 0.9}, CFG)
        self.assertEqual((a.action, a.label, a.hold_s), ("hold", "person", CFG.hold_s))

    def test_static_obstacle_pauses(self):
        a = advisory_from_verdict(
            {"blocked": True, "what": "box", "confidence": 0.8}, CFG)
        self.assertEqual((a.action, a.label), ("pause", "box"))

    def test_low_confidence_is_note_only(self):
        a = advisory_from_verdict(
            {"blocked": True, "what": "box", "confidence": 0.5}, CFG)
        self.assertEqual(a.action, "note")
        a = advisory_from_verdict(
            {"blocked": True, "what": "person", "confidence": 0.5}, CFG)
        self.assertEqual(a.action, "note")

    def test_clear_path_is_note_only(self):
        a = advisory_from_verdict(
            {"blocked": False, "what": "rug_edge", "confidence": 0.95}, CFG)
        self.assertEqual(a.action, "note")
        self.assertIn("rug_edge", a.label)

    def test_artifact_never_acts_even_if_blocked(self):
        a = advisory_from_verdict(
            {"blocked": True, "what": "shadow_or_reflection", "confidence": 1.0}, CFG)
        self.assertEqual(a.action, "note")

    def test_garbage_is_none(self):
        for v in (None, "x", {}, {"blocked": "yes", "what": "box", "confidence": 1},
                  {"blocked": True, "what": "dragon", "confidence": 1},
                  {"blocked": True, "what": "box", "confidence": "high"}):
            self.assertIsNone(advisory_from_verdict(v, CFG), v)


class _FakeController:
    """request_rgb_tagged that the test answers by hand."""
    def __init__(self):
        self.pending = []      # list of (rid, callback)

    def request_rgb_tagged(self, on_reply, max_width=None, jpeg_quality=None):
        rid = f"rid{len(self.pending)}"
        self.pending.append((rid, on_reply))
        return rid

    def answer(self, i=-1, ok=True):
        rid, cb = self.pending[i]
        cb({"request_id": rid, "ok": ok, "jpeg": b"JPEG" if ok else None,
            "width": 640, "height": 400, "ts": time.time(),
            "recv_ts": time.time(), "error": None if ok else "no frame"})


class TestBlockAdjudicator(unittest.TestCase):
    def setUp(self):
        self.ctl = _FakeController()
        self.worker = VlmWorker()
        self.worker.start()
        self.addCleanup(self.worker.stop)
        self.verdict = {"blocked": True, "what": "box", "confidence": 0.9}
        self.answered = threading.Event()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.emitted = []

    def _adj(self, **cfg):
        def query(jpeg):
            self.assertEqual(jpeg, b"JPEG")
            return self.verdict
        adj = BlockAdjudicator(
            self.ctl, self.worker, cfg=AdjudicatorConfig(**cfg), query_fn=query,
            emit=lambda c, e, d: self.emitted.append((c, e, d)),
            log_dir=self.tmp.name)
        return adj

    def _blocked_drive(self, reason="depth_block"):
        io = FakeDriveIO(scan=_clear_scan())
        hd = HierarchicalDrive(_runner(((5.0, 0.0),)), FakePose((0.0, 0.0, 0.0)), io,
                               HierConfig(blocked_retry_window_s=30.0))
        hd.start()
        hd.tick(0.0)
        hd.tick(0.0)
        io.set_status(cmd_id=1, state="BLOCKED", blocked_reason=reason,
                      depth_hits=42, depth_seq=7)
        hd.tick(1.0)
        self.assertEqual(hd.state(), HierState.BLOCKED)
        return hd, io

    def _settle(self, adj, hd, io, timeout=2.0):
        """Poll until the worker's answer has been applied."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            adj.poll(hd, io.latest_status(), (1.0, 2.0, 0.5))
            if adj.last_verdict() is not None:
                return adj.last_verdict()
            time.sleep(0.01)
        raise AssertionError("no verdict")

    def test_queries_once_per_run_and_pauses(self):
        adj = self._adj()
        hd, io = self._blocked_drive()
        adj.poll(hd, io.latest_status(), (1.0, 2.0, 0.5))
        adj.poll(hd, io.latest_status(), (1.0, 2.0, 0.5))
        self.assertEqual(len(self.ctl.pending), 1)          # one capture per run
        self.ctl.answer()
        rec = self._settle(adj, hd, io)
        self.assertTrue(rec["applied"])
        self.assertEqual(rec["advisory"]["action"], "pause")
        self.assertEqual(rec["depth_hits"], 42)
        self.assertEqual(rec["depth_seq"], 7)
        self.assertEqual(rec["pose"], [1.0, 2.0, 0.5])
        self.assertEqual(hd.block_advisory().label, "box")
        sent = len(io.sent)
        hd.tick(5.0)
        self.assertEqual(len(io.sent), sent)                # paused: no retries
        self.assertEqual(self.emitted[0][:2], ("safety", "depth_block_verdict"))

    def test_writes_jpeg_and_jsonl(self):
        adj = self._adj()
        hd, io = self._blocked_drive()
        adj.poll(hd, io.latest_status(), None)
        self.ctl.answer()
        rec = self._settle(adj, hd, io)
        with open(rec["jpeg"], "rb") as fh:
            self.assertEqual(fh.read(), b"JPEG")
        with open(os.path.join(self.tmp.name, "verdicts.jsonl")) as fh:
            lines = [json.loads(x) for x in fh]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["verdict"], self.verdict)

    def test_log_only_mode_never_advises(self):
        adj = self._adj(act=False)
        hd, io = self._blocked_drive()
        adj.poll(hd, io.latest_status(), None)
        self.ctl.answer()
        rec = self._settle(adj, hd, io)
        self.assertFalse(rec["applied"])
        self.assertEqual(rec["advisory"]["action"], "pause")   # computed, not applied
        self.assertIsNone(hd.block_advisory())

    def test_other_block_reasons_not_queried(self):
        adj = self._adj()
        hd, io = self._blocked_drive(reason="swept_block")
        adj.poll(hd, io.latest_status(), None)
        self.assertEqual(self.ctl.pending, [])

    def test_late_answer_for_passed_block_is_not_applied(self):
        adj = self._adj()
        hd, io = self._blocked_drive()
        adj.poll(hd, io.latest_status(), None)
        hd.tick(1.6)                                        # retry → goto 2
        io.set_status(cmd_id=2, state="ARRIVED")            # got past it
        hd.tick(1.7)
        self.ctl.answer()
        rec = self._settle(adj, hd, io)
        self.assertFalse(rec["applied"])
        self.assertNotEqual(hd.state(), HierState.BLOCKED)

    def test_answer_for_replaced_drive_is_not_applied(self):
        adj = self._adj()
        hd, io = self._blocked_drive()
        adj.poll(hd, io.latest_status(), None)
        hd2, io2 = self._blocked_drive()                    # new mission
        self.ctl.answer()
        rec = self._settle(adj, hd2, io2)
        self.assertFalse(rec["applied"])
        self.assertIsNone(hd2.block_advisory())

    def test_capture_failure_recorded(self):
        adj = self._adj()
        hd, io = self._blocked_drive()
        adj.poll(hd, io.latest_status(), None)
        self.ctl.answer(ok=False)
        rec = self._settle(adj, hd, io)
        self.assertIn("capture", rec["error"])
        self.assertFalse(rec["applied"])

    def test_query_failure_recorded(self):
        adj = self._adj()
        hd, io = self._blocked_drive()

        def boom(_):
            raise ConnectionError("vlm down")
        adj._query_fn = boom
        adj.poll(hd, io.latest_status(), None)
        self.ctl.answer()
        rec = self._settle(adj, hd, io)
        self.assertIn("vlm down", rec["error"])
        self.assertIsNone(rec["advisory"])


if __name__ == "__main__":
    unittest.main()

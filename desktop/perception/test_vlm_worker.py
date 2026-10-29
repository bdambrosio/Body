"""VlmWorker: one result per job, staleness, bounded queue, failures, stop."""
from __future__ import annotations

import threading
import time
import unittest

from desktop.perception.frames import PosedFrame
from desktop.perception.vlm_worker import VlmJob, VlmWorker


def _frame(rid="r1", recv_ts=None):
    return PosedFrame(request_id=rid, jpeg=b"jpg-" + rid.encode(), width=640,
                      height=400, pi_ts=0.0,
                      recv_ts=time.time() if recv_ts is None else recv_ts,
                      pose=(1.0, 2.0, 0.3))


class _Collector:
    def __init__(self):
        self.results = []
        self.cv = threading.Condition()

    def __call__(self, res):
        with self.cv:
            self.results.append(res)
            self.cv.notify_all()

    def wait_for(self, n, timeout=2.0):
        with self.cv:
            ok = self.cv.wait_for(lambda: len(self.results) >= n, timeout)
        if not ok:
            raise AssertionError(f"got {len(self.results)} results, want {n}")
        return self.results


class TestVlmWorker(unittest.TestCase):
    def setUp(self):
        self.w = VlmWorker(max_queue=2)
        self.w.start()
        self.addCleanup(self.w.stop)

    def test_answer_carries_frame_pose_and_verdict(self):
        out = _Collector()
        f = _frame("a")
        self.w.submit(VlmJob(f, lambda jpeg: {"seen": jpeg.decode()}, out,
                             kind="clear_path"))
        res = out.wait_for(1)[0]
        self.assertIsNone(res.error)
        self.assertEqual(res.kind, "clear_path")
        self.assertEqual(res.verdict, {"seen": "jpg-a"})
        self.assertEqual(res.request_id, "a")
        self.assertEqual(res.pose, (1.0, 2.0, 0.3))
        self.assertEqual(res.frame_ts, f.recv_ts)
        self.assertGreaterEqual(res.latency_s, 0.0)
        self.assertEqual(self.w.stats()["answered"], 1)

    def test_stale_frame_skipped_without_query(self):
        out = _Collector()
        called = []
        self.w.submit(VlmJob(_frame(recv_ts=time.time() - 10.0),
                             lambda j: called.append(j), out, max_age_s=3.0))
        res = out.wait_for(1)[0]
        self.assertEqual(res.error, "stale")
        self.assertEqual(called, [])
        self.assertEqual(self.w.stats()["stale"], 1)

    def test_query_exception_reported(self):
        out = _Collector()

        def boom(_):
            raise RuntimeError("server down")
        self.w.submit(VlmJob(_frame(), boom, out))
        res = out.wait_for(1)[0]
        self.assertIsNone(res.verdict)
        self.assertIn("server down", res.error)
        self.assertEqual(self.w.stats()["failed"], 1)

    def test_full_queue_drops_oldest(self):
        out = _Collector()
        gate = threading.Event()
        started = threading.Event()

        def blocking(_):
            started.set()
            gate.wait(2.0)
            return "done"
        self.w.submit(VlmJob(_frame("busy"), blocking, out))
        self.assertTrue(started.wait(2.0))
        for rid in ("q1", "q2", "q3"):
            self.w.submit(VlmJob(_frame(rid), lambda j: "ok", out))
        gate.set()
        res = out.wait_for(4)
        by_id = {r.request_id: r for r in res}
        self.assertEqual(by_id["q1"].error, "dropped")
        self.assertEqual(by_id["busy"].verdict, "done")
        self.assertEqual(by_id["q2"].verdict, "ok")
        self.assertEqual(by_id["q3"].verdict, "ok")
        self.assertEqual(self.w.stats()["dropped"], 1)

    def test_stop_reports_queued_and_rejects_new(self):
        out = _Collector()
        gate = threading.Event()
        started = threading.Event()

        def blocking(_):
            started.set()
            gate.wait(2.0)
        self.w.submit(VlmJob(_frame("busy"), blocking, out))
        self.assertTrue(started.wait(2.0))
        self.w.submit(VlmJob(_frame("queued"), lambda j: "ok", out))
        stopper = threading.Thread(target=self.w.stop)
        stopper.start()
        gate.set()
        stopper.join(3.0)
        self.assertFalse(self.w.submit(VlmJob(_frame("late"), lambda j: 1, out)))
        res = out.wait_for(3)
        by_id = {r.request_id: r for r in res}
        self.assertEqual(by_id["queued"].error, "stopped")
        self.assertEqual(by_id["late"].error, "stopped")

    def test_on_result_exception_does_not_kill_worker(self):
        out = _Collector()

        def bad(_res):
            raise ValueError("ui gone")
        self.w.submit(VlmJob(_frame("x"), lambda j: 1, bad))
        self.w.submit(VlmJob(_frame("y"), lambda j: 2, out))
        self.assertEqual(out.wait_for(1)[0].verdict, 2)


if __name__ == "__main__":
    unittest.main()

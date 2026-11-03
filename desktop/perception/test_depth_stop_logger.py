import os
import tempfile
import unittest

import numpy as np

from desktop.perception.depth_stop_logger import DepthStopLogger


class TestDepthStopLogger(unittest.TestCase):
    def test_rising_edge_rate_limited(self):
        t = [100.0]
        with tempfile.TemporaryDirectory() as d:
            lg = DepthStopLogger(d, min_interval_s=2.0, clock=lambda: t[0])
            depth = {"depth": np.full((90, 120), 500, np.uint16), "ts": 99.9}
            st = {"depth_would_block": True, "depth_hits": 42, "ts": 100.0,
                  "depth_bbox_px": [1, 2, 3, 4, 120, 90], "depth_hit_body": [[0.5, 0.0]]}
            p = lg.observe(st, depth=depth, rgb={"jpeg": b"JPEG", "ts": 99.8}, pose=(1, 2, 0.3))
            self.assertIsNotNone(p)
            self.assertTrue(os.path.exists(p + ".npz") and os.path.exists(p + ".jpg"))
            z = np.load(p + ".npz")
            self.assertEqual(int(z["depth_hits"]), 42)
            self.assertEqual(z["depth"].shape, (90, 120))
            self.assertIsNone(lg.observe(st, depth=depth, rgb=None, pose=None))   # steady: no
            t[0] += 0.5
            self.assertIsNone(lg.observe({"depth_would_block": False}, depth=depth, rgb=None, pose=None))
            self.assertIsNone(lg.observe(st, depth=depth, rgb=None, pose=None))   # too soon
            t[0] += 3.0
            lg.observe({"depth_would_block": False}, depth=depth, rgb=None, pose=None)
            self.assertIsNotNone(lg.observe(st, depth=depth, rgb=None, pose=None))
            self.assertEqual(lg.count, 2)
        self.assertIsNone(DepthStopLogger(None).observe(st, depth=depth, rgb=None, pose=None))

    def test_burst_pre_and_post_frames(self):
        t = [100.0]
        with tempfile.TemporaryDirectory() as d:
            lg = DepthStopLogger(d, pre_s=2.0, post_s=1.0, clock=lambda: t[0])
            idle = {"state": "DRIVING", "v_mps": 0.18, "omega_radps": 0.0}
            blk = {"state": "BLOCKED", "blocked_reason": "depth_block", "depth_hits": 30}

            def frame(k):
                return {"depth": np.full((90, 120), 500 + k, np.uint16), "ts": 50.0 + k * 0.1}
            k = 0
            for _ in range(40):                     # 4 s of driving at 10 Hz
                lg.observe(idle, depth=frame(k), rgb=None, pose=(k * 0.01, 0, 0))
                lg.observe(idle, depth=frame(k), rgb=None, pose=None)   # re-render: deduped
                t[0] += 0.1
                k += 1
            base = lg.observe(blk, depth=frame(k), rgb=None, pose=(k * 0.01, 0, 0))
            self.assertTrue(os.path.basename(base).startswith("depth_block_"))
            self.assertTrue(bool(np.load(base + ".npz")["acted"]))
            for _ in range(12):                     # 1.2 s stopped
                t[0] += 0.1
                k += 1
                lg.observe(blk, depth=frame(k), rgb=None, pose=(0.4, 0, 0))
            z = np.load(base + "_burst.npz")
            n = z["depth"].shape[0]
            ti = int(z["trigger_index"])
            self.assertEqual(z["depth"].shape[1:], (90, 120))
            self.assertGreaterEqual(ti, 19)          # ~2 s of pre frames
            self.assertGreaterEqual(n - 1 - ti, 9)   # ~1 s of post frames
            self.assertEqual(len(set(z["depth_ts"].tolist())), n)   # no duplicates
            self.assertEqual(z["pose"].shape, (n, 3))

    def test_no_burst_without_dir(self):
        lg = DepthStopLogger(None)
        st = {"depth_would_block": True}
        lg.observe(st, depth={"depth": np.zeros((2, 2), np.uint16), "ts": 1.0}, rgb=None, pose=None)
        self.assertIsNone(lg.flush())


if __name__ == "__main__":
    unittest.main()

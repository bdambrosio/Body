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


if __name__ == "__main__":
    unittest.main()

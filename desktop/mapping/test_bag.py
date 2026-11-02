"""BagRecorder / iter_bag round trip (mapping plan Stage 0)."""
import json
import os
import tempfile
import unittest

import numpy as np

from desktop.mapping.bag import BagRecorder, bag_frames, iter_bag


class TestBag(unittest.TestCase):
    def test_round_trip_in_ts_order(self):
        with tempfile.TemporaryDirectory() as d:
            bag = BagRecorder(d, chunk=2, info={"session_id": "abc"})
            self.assertTrue(bag.enabled)
            bag.add_odom({"ts": 10.0, "x": 0.0, "y": 0.0, "theta": 0.0})
            bag.add_imu({"ts": 10.05, "yaw": 0.1})
            for k in range(3):
                bag.add_scan({"ts": 10.1 + k, "angle_min": -3.14, "angle_increment": 0.0174,
                              "ranges": [1.0 + k, None, 2.5], "intensities": [10, 0, 200],
                              "unknown_bins": [1]})
            bag.add_odom({"ts": 10.5, "x": 0.1, "y": 0.0, "theta": 0.0})
            name = bag.add_frame(11.0, (0.1, 0.0, 0.0), b"jpeg", pi_ts=10.9)
            bag.close()
            self.assertEqual(sorted(f for f in os.listdir(d) if f.startswith("scans_")),
                             ["scans_0000.npz", "scans_0001.npz"])
            with open(os.path.join(d, "bag.json")) as f:
                self.assertEqual(json.load(f)["session_id"], "abc")
            rows = list(iter_bag(d))
            frames = bag_frames(d)
        self.assertEqual([k for k, _ in rows], ["odom", "imu", "scan", "odom", "scan", "scan"])
        scan = rows[2][1]
        self.assertEqual(scan["ranges"], [1.0, None, 2.5])
        self.assertEqual(scan["intensities"], [10, 0, 200])
        self.assertEqual(scan["unknown_bins"], [1])
        self.assertAlmostEqual(scan["angle_increment"], 0.0174)
        self.assertEqual(frames[0]["path"], name)
        self.assertEqual(frames[0]["pose"], [0.1, 0.0, 0.0])

    def test_disabled(self):
        bag = BagRecorder(None)
        self.assertFalse(bag.enabled)
        bag.add_scan({"ts": 1.0, "ranges": [1.0]})
        bag.add_odom({"ts": 1.0})
        self.assertIsNone(bag.add_frame(1.0, None, b"x"))
        self.assertIsNone(bag.flush_scans())
        bag.close()


if __name__ == "__main__":
    unittest.main()

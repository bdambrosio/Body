"""MappingController ingestion: bag recording, keyframe request, prior
layers carried into the export (mapping plan Stage 0)."""
import json
import math
import os
import tempfile
import unittest

import numpy as np

from desktop.mapping.controller import MappingConfig, MappingController
from desktop.reference_map.reference_map import (
    build_reference_map_from_log_odds, save_reference_map,
)


def _scan_msg(ts, dist=2.0):
    n = 90
    return {"ts": ts, "angle_min": -math.pi, "angle_increment": 2 * math.pi / n,
            "ranges": [dist] * n, "intensities": [50] * n, "unknown_bins": []}


class TestController(unittest.TestCase):
    def _settled(self, ctl, t0=100.0):
        for i in range(250):
            ctl.ingest_imu(_imu_payload(t0 + i * 0.01))
        ctl.ingest_odom({"ts": t0 + 2.5, "x": 0.0, "y": 0.0, "theta": 0.0})

    def test_bag_records_and_nodes_follow_motion(self):
        with tempfile.TemporaryDirectory() as d:
            ctl = MappingController(MappingConfig(bag_dir=d))
            ctl.mapper._clock = lambda: 0.0
            self._settled(ctl)
            added = []
            for k in range(6):
                ts = 102.6 + 0.5 * k
                ctl.ingest_odom({"ts": ts, "x": 0.3 * k, "y": 0.0, "theta": 0.0})
                ctl.mapper._last_match_mono = -10.0
                added.append(ctl.ingest_scan(_scan_msg(ts)))
            self.assertTrue(any(added))
            st = ctl.status_summary()
            self.assertEqual(st["bag_scans"], 6)
            self.assertTrue(ctl.wants_keyframe())
            ctl.record_frame(b"jpeg", pi_ts=103.0)
            self.assertFalse(ctl.wants_keyframe())
            ctl.shutdown()
            files = os.listdir(d)
            self.assertIn("scans_0000.npz", files)
            self.assertIn("odom.jsonl", files)
            self.assertIn("imu.jsonl", files)
            self.assertIn("frame_00000.jpg", files)

    def test_prior_layers_carried(self):
        with tempfile.TemporaryDirectory() as d:
            slam = MappingController(MappingConfig(record=False))._slam_cfg.slam
            n = 2 * int(math.ceil(slam.extent_m / slam.resolution_m / 2.0))
            nogo = np.zeros((n, n), dtype=bool)
            nogo[10:20, 10:20] = True
            prior = build_reference_map_from_log_odds(
                np.zeros((n, n), dtype=np.float32), resolution_m=slam.resolution_m,
                origin_x_m=-slam.extent_m / 2, origin_y_m=-slam.extent_m / 2,
                session_id="old", metadata={"checkpoints": [{"id": "cp1"}]}, nogo_mask=nogo)
            path = os.path.join(d, "reference_map.npz")
            save_reference_map(path, prior)
            ctl = MappingController(MappingConfig(record=False, prior_map=path))
            ctl.mapper._clock = lambda: 0.0
            self._settled(ctl)
            ctl.ingest_scan(_scan_msg(102.6))
            ref = ctl.finalize_map()
            self.assertEqual(ref.metadata["checkpoints"], [{"id": "cp1"}])
            self.assertTrue(ref.nogo_mask[15, 15])


def _imu_payload(ts):
    return {"ts": ts, "gyro": {"z": 0.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "fusion": {"mode": "game_rotation_vector", "accuracy_rad": 0.175}}


if __name__ == "__main__":
    unittest.main()

"""Unit tests for ReferenceMap load/save and field builders."""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np

from desktop.reference_map.reference_map import (
    ReferenceMap,
    build_distance_field,
    build_likelihood_field,
    build_reference_map_from_log_odds,
    driveable_from_occupancy,
    load_reference_map,
    save_reference_map,
)
from desktop.reference_map.legacy_convert import convert_layers_npz


class TestLikelihoodField(unittest.TestCase):
    def test_peak_at_obstacle(self):
        occ = np.zeros((20, 20), dtype=bool)
        occ[10, 10] = True
        field = build_likelihood_field(occ, resolution_m=0.05)
        self.assertAlmostEqual(float(field[10, 10]), 1.0, places=3)
        self.assertGreater(float(field[10, 11]), 0.1)
        self.assertLess(float(field[0, 0]), 0.01)

    def test_empty_map_zero_field(self):
        occ = np.zeros((10, 10), dtype=bool)
        field = build_likelihood_field(occ, resolution_m=0.05)
        self.assertEqual(float(field.max()), 0.0)


class TestReferenceMapRoundTrip(unittest.TestCase):
    def test_save_load(self):
        log_odds = np.zeros((50, 50), dtype=np.float32)
        log_odds[24:27, 24:27] = 3.0
        log_odds[10:15, 10:15] = -2.0
        ref = build_reference_map_from_log_odds(
            log_odds,
            resolution_m=0.08,
            origin_x_m=-2.0,
            origin_y_m=-2.0,
            session_id="test01",
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "reference_map.npz")
            save_reference_map(path, ref)
            loaded = load_reference_map(path)
        self.assertEqual(loaded.session_id, "test01")
        self.assertAlmostEqual(loaded.resolution_m, 0.08)
        np.testing.assert_allclose(
            loaded.occupancy_log_odds, ref.occupancy_log_odds,
        )
        self.assertGreater(float(loaded.likelihood_field[25, 25]), 0.9)

    def test_driveable_layers(self):
        log_odds = np.zeros((10, 10), dtype=np.float32)
        log_odds[2, 2] = 3.0
        log_odds[5, 5] = -3.0
        ref = build_reference_map_from_log_odds(
            log_odds, resolution_m=0.05, origin_x_m=0, origin_y_m=0,
        )
        drive = ref.driveable_int8()
        self.assertEqual(int(drive[2, 2]), 0)
        self.assertEqual(int(drive[5, 5]), 1)
        self.assertEqual(int(drive[0, 0]), -1)

    def test_legacy_layers_convert(self):
        meta = {
            "resolution_m": 0.08,
            "origin_x_m": -1.6,
            "origin_y_m": -1.6,
            "nx": 40,
            "ny": 40,
            "frame": "world",
        }
        drive = np.full((40, 40), -1, dtype=np.int8)
        drive[20, 20:30] = 0  # wall
        drive[10:15, 10:15] = 1  # clear
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "layers.npz")
            np.savez_compressed(
                path,
                max_height_m=np.full((40, 40), np.nan, dtype=np.float32),
                clear_votes=np.zeros((40, 40), dtype=np.float32),
                block_votes=np.zeros((40, 40), dtype=np.float32),
                traversed_ts=np.full((40, 40), np.nan, dtype=np.float32),
                last_observed_ts=np.full((40, 40), np.nan, dtype=np.float32),
                observation_count=np.zeros((40, 40), dtype=np.int32),
                driveable=drive,
                meta_json=np.array(__import__("json").dumps(meta)),
                session_id=np.array("legacy01"),
                bounds_ij=np.array([10, 30, 10, 30], dtype=np.int32),
            )
            ref = convert_layers_npz(path)
        self.assertEqual(ref.metadata.get("source_session_id"), "legacy01")
        occ = ref.occupied_mask()
        self.assertTrue(bool(occ[20, 25]))


if __name__ == "__main__":
    unittest.main()

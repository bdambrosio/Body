"""Tests for LPR checkpoint persistence (metadata round-trip + upsert)."""
from __future__ import annotations

import math
import unittest

from desktop.localization.checkpoints import (
    Checkpoint,
    checkpoints_from_metadata,
    upsert_checkpoint,
    write_checkpoints_to_metadata,
)


class TestCheckpoints(unittest.TestCase):
    def test_metadata_round_trip(self):
        meta: dict = {"map_version": 1}
        cps = [
            Checkpoint("cp_000", 1.0, 2.0, 0.5, 2.0, 100.0),
            Checkpoint("cp_001", -3.0, 0.0, -1.2, 2.0, 200.0),
        ]
        write_checkpoints_to_metadata(meta, cps)
        # metadata stays JSON-friendly (list of plain dicts).
        self.assertIsInstance(meta["checkpoints"], list)
        self.assertIsInstance(meta["checkpoints"][0], dict)
        back = checkpoints_from_metadata(meta)
        self.assertEqual(back, cps)

    def test_from_metadata_empty_and_missing(self):
        self.assertEqual(checkpoints_from_metadata(None), [])
        self.assertEqual(checkpoints_from_metadata({}), [])

    def test_from_metadata_skips_malformed(self):
        meta = {"checkpoints": [{"id": "bad"}, Checkpoint(
            "cp_000", 0, 0, 0, 2.0).to_dict()]}
        self.assertEqual(len(checkpoints_from_metadata(meta)), 1)

    def test_upsert_adds_with_incrementing_id(self):
        cps, c0 = upsert_checkpoint([], (0.0, 0.0, 0.0), 2.0, created_ts=1.0)
        self.assertEqual(c0.id, "cp_000")
        cps, c1 = upsert_checkpoint(cps, (5.0, 0.0, 0.0), 2.0, created_ts=2.0)
        self.assertEqual(c1.id, "cp_001")
        self.assertEqual(len(cps), 2)

    def test_upsert_updates_nearby_same_id(self):
        cps, c0 = upsert_checkpoint([], (0.0, 0.0, 0.0), 2.0, created_ts=1.0)
        # Re-Recognize 0.2 m away (< merge_dist) corrects the same checkpoint.
        cps, c0b = upsert_checkpoint(
            cps, (0.2, 0.0, math.radians(10)), 2.0, created_ts=3.0)
        self.assertEqual(len(cps), 1)
        self.assertEqual(c0b.id, "cp_000")
        self.assertAlmostEqual(c0b.x_m, 0.2)
        self.assertAlmostEqual(c0b.theta_rad, math.radians(10))

    def test_upsert_far_adds_new(self):
        cps, _ = upsert_checkpoint([], (0.0, 0.0, 0.0), 2.0)
        cps, c1 = upsert_checkpoint(cps, (1.0, 0.0, 0.0), 2.0, merge_dist_m=0.5)
        self.assertEqual(len(cps), 2)
        self.assertEqual(c1.id, "cp_001")


if __name__ == "__main__":
    unittest.main()

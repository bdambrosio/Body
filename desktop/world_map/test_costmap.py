"""Costmap keep-out (no-go) behaviour.

Run: PYTHONPATH=. python3 -m unittest desktop.world_map.test_costmap -v
"""
from __future__ import annotations

import unittest

import numpy as np

from desktop.world_map.costmap import CostmapConfig, build_costmap


def _cfg() -> CostmapConfig:
    # No inflation / no denoise so lethal == exactly the blocked+nogo cells.
    return CostmapConfig(
        footprint_radius_m=0.0, safety_margin_m=0.0, inflation_decay_m=0.01,
        denoise=False, unknown_is_lethal=False,
    )


def _snap(drive, nogo=None, res=0.05):
    snap = {"driveable": drive, "meta": {"resolution_m": res}}
    if nogo is not None:
        snap["nogo"] = nogo
    return snap


class TestNoGoCostmap(unittest.TestCase):
    def test_nogo_cell_becomes_lethal(self):
        drive = np.ones((10, 10), np.int8)        # everything clear
        nogo = np.zeros((10, 10), bool)
        nogo[5, 5] = True
        cm = build_costmap(_snap(drive, nogo), _cfg())
        self.assertTrue(cm.lethal[5, 5])
        self.assertFalse(cm.lethal[0, 0])
        self.assertTrue(np.isinf(cm.cost[5, 5]))

    def test_nogo_not_inflated(self):
        # Decision (1): no footprint dilation — only the painted cell.
        drive = np.ones((10, 10), np.int8)
        nogo = np.zeros((10, 10), bool)
        nogo[5, 5] = True
        cm = build_costmap(_snap(drive, nogo), _cfg())
        self.assertTrue(cm.lethal[5, 5])
        self.assertFalse(cm.lethal[5, 6])
        self.assertFalse(cm.lethal[4, 5])

    def test_nogo_survives_denoise(self):
        # Decision (2): a lone no-go cell is NOT speckle-dropped, unlike a
        # lone perception-blocked cell would be.
        drive = np.ones((10, 10), np.int8)
        nogo = np.zeros((10, 10), bool)
        nogo[5, 5] = True
        cfg = CostmapConfig(
            footprint_radius_m=0.0, safety_margin_m=0.0,
            inflation_decay_m=0.01, denoise=True, denoise_min_neighbors=2,
            unknown_is_lethal=False,
        )
        cm = build_costmap(_snap(drive, nogo), cfg)
        self.assertTrue(cm.lethal[5, 5])

    def test_missing_nogo_key_back_compat(self):
        drive = np.ones((10, 10), np.int8)
        cm = build_costmap(_snap(drive), _cfg())   # no "nogo" key
        self.assertFalse(cm.lethal.any())


if __name__ == "__main__":
    unittest.main()

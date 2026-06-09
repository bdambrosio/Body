"""Tests for the shared Tier-3 config builders (body/lib/drive_config.py).

These pin the desktop↔Pi config seam: both halves must build their planner/
raster configs from config.json through the same functions, and every key in
the config.json sections must actually be consumed (a key nobody reads is a
knob an operator will tune to no effect — goal_clearance_cells sat dead for
weeks exactly that way).
"""
from __future__ import annotations

import dataclasses
import unittest

from body.lib.drive_config import local_plan_config, scan_raster_config
from body.lib.local_costmap import LocalCostmapConfig
from body.lib.local_planner import LocalPlanConfig
from body.lib.scan_raster import ScanRasterConfig
from body.lib.zenoh_helpers import load_body_config


class TestBuildersReadRepoConfig(unittest.TestCase):
    def test_local_plan_config_reflects_config_json(self):
        cfg = load_body_config()
        lp = cfg["local_drive"]["local_planner"]
        pc = local_plan_config(cfg)
        self.assertAlmostEqual(pc.costmap.footprint_radius_m,
                               float(lp["footprint_radius_m"]))
        self.assertAlmostEqual(pc.costmap.safety_margin_m,
                               float(lp["safety_margin_m"]))
        self.assertEqual(pc.min_clearance_cells, int(lp["min_clearance_cells"]))
        # The key that used to be dead: config.json must now drive it.
        self.assertEqual(pc.goal_clearance_cells, int(lp["goal_clearance_cells"]))

    def test_scan_raster_config_reflects_config_json(self):
        cfg = load_body_config()
        scan = cfg["local_drive"]["scan"]
        rc = scan_raster_config(cfg)
        self.assertAlmostEqual(rc.resolution_m, float(scan["resolution_m"]))
        self.assertAlmostEqual(rc.half_extent_m, float(scan["half_extent_m"]))

    def test_empty_config_yields_defaults(self):
        # Builders must be total: an empty dict gives pure dataclass defaults
        # (so unit tests / fakes don't need a config file).
        self.assertEqual(local_plan_config({}),
                         LocalPlanConfig(costmap=LocalCostmapConfig()))
        self.assertEqual(scan_raster_config({}), ScanRasterConfig())


class TestNoDeadConfigKeys(unittest.TestCase):
    """Every key in config.json's planner/scan sections must map to a real
    consumer — a dataclass field of the built configs or an explicitly listed
    extra read by body.local_drive."""

    def test_local_planner_keys_all_consumed(self):
        cfg = load_body_config()
        lp = cfg["local_drive"]["local_planner"]
        costmap_fields = {f.name for f in dataclasses.fields(LocalCostmapConfig)}
        plan_fields = {f.name for f in dataclasses.fields(LocalPlanConfig)}
        extras = {"lookahead_m"}        # read directly by body.local_drive
        for key in lp:
            self.assertIn(
                key, costmap_fields | plan_fields | extras,
                f"config.json local_drive.local_planner.{key} is read by nothing",
            )

    def test_scan_keys_all_consumed(self):
        cfg = load_body_config()
        scan = cfg["local_drive"]["scan"]
        raster_fields = {f.name for f in dataclasses.fields(ScanRasterConfig)}
        extras = {"scan_stale_s"}       # read directly by body.local_drive
        for key in scan:
            self.assertIn(
                key, raster_fields | extras,
                f"config.json local_drive.scan.{key} is read by nothing",
            )


if __name__ == "__main__":
    unittest.main()

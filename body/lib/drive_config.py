"""Tier-3 planner/raster configs built from config.json — one source of truth.

Both halves must construct these through the SAME functions: the Pi's
``body.local_drive`` service, and every desktop consumer that re-models
Tier-3 (the hierarchical drive's Tier-2 clear-run, the pi_drive consoles).
Constructing from parallel dataclass defaults is how the two sides drift
apart silently: the Pi builds from config.json while the desktop builds
from defaults, and they agree only while config.json happens to match the
defaults. A sub-goal chosen on a different lethal mask than the one Tier-3
plans on recreates the corner-stall class of bug the I3 redesign removed
(docs/tier_contract.md).

Pure: takes the already-loaded config dict (``zenoh_helpers.load_body_config``
resolves config.json from the repo root on either host).
"""
from __future__ import annotations

from typing import Any, Dict

from body.lib.local_costmap import LocalCostmapConfig
from body.lib.local_planner import LocalPlanConfig
from body.lib.scan_raster import ScanRasterConfig


def scan_raster_config(body_cfg: Dict[str, Any]) -> ScanRasterConfig:
    """Tier-3's obstacle-field raster. Lidar mount/range come from the
    existing lidar / local_map sections so there's one source of truth."""
    cfg = body_cfg.get("local_drive", {})
    lidar_cfg = body_cfg.get("lidar", {})
    lm_cfg = body_cfg.get("local_map", {})
    scan_cfg = cfg.get("scan", {})
    return ScanRasterConfig(
        resolution_m=float(scan_cfg.get("resolution_m", 0.08)),
        half_extent_m=float(scan_cfg.get("half_extent_m", 2.5)),
        lidar_x_m=float(lm_cfg.get("lidar_x_body_m", 0.0)),
        lidar_y_m=float(lm_cfg.get("lidar_y_body_m", 0.0)),
        lidar_yaw_rad=float(lm_cfg.get("lidar_yaw_rad", 0.0)),
        range_min_m=float(lidar_cfg.get("range_min_m", 0.05)),
        range_max_m=float(scan_cfg.get("range_max_m", 8.0)),
        max_clear_range_m=float(
            scan_cfg.get("max_clear_range_m", lm_cfg.get("lidar_max_clear_range_m", 6.0))
        ),
        clear_buffer_cells=float(scan_cfg.get("clear_buffer_cells", 2.0)),
    )


def local_plan_config(body_cfg: Dict[str, Any]) -> LocalPlanConfig:
    """Tier-3's local A* planner config (the single local-routing authority).
    The costmap footprint here is THE footprint model; the swept-veto
    FootprintConfig is derived ≤ it in ``body.local_drive`` so they agree."""
    lp = body_cfg.get("local_drive", {}).get("local_planner", {})
    return LocalPlanConfig(
        costmap=LocalCostmapConfig(
            footprint_radius_m=float(lp.get("footprint_radius_m", 0.11)),
            safety_margin_m=float(lp.get("safety_margin_m", 0.08)),
            inflation_decay_m=float(lp.get("inflation_decay_m", 0.20)),
            unknown_cost=float(lp.get("unknown_cost", 25.0)),
            unknown_is_lethal=bool(lp.get("unknown_is_lethal", False)),
        ),
        min_clearance_cells=int(lp.get("min_clearance_cells", 0)),
        goal_clearance_cells=int(lp.get("goal_clearance_cells", 0)),
        cost_per_unit=float(lp.get("cost_per_unit", 0.10)),
        max_expansions=int(lp.get("max_expansions", 50000)),
    )

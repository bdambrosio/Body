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

from body.lib.depth_veto import DepthVetoConfig
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


def depth_veto_config(body_cfg: Dict[str, Any]) -> DepthVetoConfig:
    """Tier-3 near-field depth veto. Camera mount / slab heights default from
    ``local_map`` / ``oakd`` so operator calibration stays in one place; envelope
    and gating live under ``local_drive.depth_veto``."""
    ld = body_cfg.get("local_drive", {})
    dv = ld.get("depth_veto", {})
    lm = body_cfg.get("local_map", {})
    oakd = body_cfg.get("oakd", {})
    depth_z = float(
        dv.get(
            "depth_z_body_m",
            lm.get(
                "depth_z_body_m",
                oakd.get("depth_camera_height_above_ground_m", 0.09),
            ),
        )
    )
    return DepthVetoConfig(
        enabled=bool(dv.get("enabled", True)),
        stale_s=float(dv.get("stale_s", 0.5)),
        min_range_m=float(dv.get("min_range_m", 0.08)),
        max_range_m=float(dv.get("max_range_m", 0.80)),
        lateral_half_width_m=float(dv.get("lateral_half_width_m", 0.12)),
        floor_band_m=float(
            dv.get("floor_band_m", lm.get("driveable_floor_band_m", 0.04))
        ),
        clearance_height_m=float(
            dv.get(
                "clearance_height_m",
                lm.get("driveable_clearance_height_m", 0.35),
            )
        ),
        ground_z_body_m=float(
            dv.get("ground_z_body_m", lm.get("ground_z_body_m", 0.0))
        ),
        min_hits=int(dv.get("min_hits", 8)),
        hit_streak=int(dv.get("hit_streak", 2)),
        max_abs_omega_radps=float(dv.get("max_abs_omega_radps", 0.40)),
        roi_u0=float(dv.get("roi_u0", 0.20)),
        roi_u1=float(dv.get("roi_u1", 0.80)),
        roi_v0=float(dv.get("roi_v0", 0.25)),
        roi_v1=float(dv.get("roi_v1", 0.85)),
        depth_median_kernel=int(
            dv.get("depth_median_kernel", lm.get("depth_median_kernel", 3))
        ),
        depth_hfov_deg=float(dv.get("depth_hfov_deg", lm.get("depth_hfov_deg", 70.0))),
        depth_vfov_deg=float(dv.get("depth_vfov_deg", lm.get("depth_vfov_deg", 55.0))),
        depth_x_body_m=float(dv.get("depth_x_body_m", lm.get("depth_x_body_m", 0.0))),
        depth_y_body_m=float(dv.get("depth_y_body_m", lm.get("depth_y_body_m", 0.0))),
        depth_z_body_m=depth_z,
        depth_yaw_rad=float(dv.get("depth_yaw_rad", lm.get("depth_yaw_rad", 0.0))),
        depth_pitch_rad=float(
            dv.get("depth_pitch_rad", lm.get("depth_pitch_rad", 0.14))
        ),
        depth_roll_rad=float(dv.get("depth_roll_rad", lm.get("depth_roll_rad", 0.0))),
    )

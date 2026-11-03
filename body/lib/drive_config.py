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

import math
from typing import Any, Dict

from body.lib.depth_veto import DepthVetoConfig
from body.lib.drive_safety import FootprintConfig
from body.lib.local_costmap import LocalCostmapConfig
from body.lib.local_drive_core import DriveParams
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
        robot_clear_radius_m=float(scan_cfg.get("robot_clear_radius_m", 0.0)),
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
            quantization_pad=bool(lp.get("quantization_pad", True)),
            denoise=bool(lp.get("denoise", False)),
            denoise_min_range_m=float(lp.get("denoise_min_range_m", 1.0)),
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
        floor_band_per_m=float(dv.get("floor_band_per_m", 0.0)),
        min_vertical_extent_m=float(dv.get("min_vertical_extent_m", 0.03)),
        structure_free_height_m=float(dv.get("structure_free_height_m", 0.10)),
        structure_cell_m=float(dv.get("structure_cell_m", 0.04)),
    )


def drive_params(body_cfg: Dict[str, Any]) -> DriveParams:
    """Tier-3 follower speeds / steering gains (``local_drive`` section)."""
    cfg = body_cfg.get("local_drive", {})
    d = DriveParams()
    return DriveParams(
        v_max=float(cfg.get("v_max", d.v_max)),
        omega_max=float(cfg.get("omega_max", d.omega_max)),
        v_min_mps=float(cfg.get("v_min_mps", d.v_min_mps)),
        arrival_tol_m=float(cfg.get("arrival_tol_m", d.arrival_tol_m)),
        rotate_in_place_thresh_rad=float(
            cfg.get("rotate_in_place_thresh_rad", d.rotate_in_place_thresh_rad)),
        rotate_exit_thresh_rad=float(
            cfg.get("rotate_exit_thresh_rad", d.rotate_exit_thresh_rad)),
        k_omega=float(cfg.get("k_omega", d.k_omega)),
        slowdown_distance_m=float(cfg.get("slowdown_distance_m", d.slowdown_distance_m)),
    )


def footprint_config(body_cfg: Dict[str, Any]) -> FootprintConfig:
    """Last-resort swept veto, derived from the A* footprint so the two can't
    drift: the swept check adds ½ cell internally, so subtract it to keep the
    veto's effective radius == the A* footprint radius; the planner's lethal
    radius adds the res/√2 quantization pad on top (``quantization_pad``), so
    every A* path clears the veto with margin."""
    cfg = body_cfg.get("local_drive", {})
    raster = scan_raster_config(body_cfg)
    plan = local_plan_config(body_cfg)
    half_cell = 0.5 * raster.resolution_m
    veto_foot = max(0.02, plan.costmap.footprint_radius_m - half_cell)
    explained_default = int(math.ceil(raster.clear_buffer_cells)) + 1
    return FootprintConfig(
        footprint_radius_m=veto_foot,
        preview_distance_m=float(cfg.get("preview_distance_m", 0.35)),
        preview_min_distance_m=float(cfg.get("preview_min_distance_m", 0.15)),
        preview_time_s=float(cfg.get("preview_time_s", 0.5)),
        forward_cone_rad=math.radians(float(cfg.get("forward_cone_deg", 60.0))),
        hard_radius_m=min(float(cfg.get("hard_radius_m", 0.05)), veto_foot),
        block_on_unknown=bool(cfg.get("block_on_unknown", True)),
        unknown_block_range_m=float(cfg.get("unknown_block_range_m", 0.25)),
        min_observed_cells=int(cfg.get("min_observed_cells", 3)),
        unknown_explained_cells=int(cfg.get("unknown_explained_cells", explained_default)),
    )


def denoise_config(body_cfg: Dict[str, Any]):
    """``local_drive.depth_veto.denoise`` → DenoiseConfig (defaults on)."""
    from body.lib.depth_denoise import DenoiseConfig
    d = body_cfg.get("local_drive", {}).get("depth_veto", {}).get("denoise", {}) or {}
    base = DenoiseConfig()
    kw = {f: type(getattr(base, f))(d[f]) for f in base.__dataclass_fields__ if f in d}
    return DenoiseConfig(**kw)


def tier3_config(body_cfg: Dict[str, Any]):
    """Everything ``Tier3Controller`` needs, from the ``local_drive`` section
    (imported lazily: tier3_controller depends on this module's siblings)."""
    from body.lib.depth_memory import memory_config_from_dict
    from body.lib.tier3_controller import Tier3Config
    cfg = body_cfg.get("local_drive", {})
    d = Tier3Config()
    plan = local_plan_config(body_cfg)
    return Tier3Config(
        params=drive_params(body_cfg),
        plan=plan,
        foot=footprint_config(body_cfg),
        depth=depth_veto_config(body_cfg),
        memory=memory_config_from_dict(cfg.get("depth_memory")),
        depth_act=bool(cfg.get("depth_veto", {}).get("act", True)),
        denoise=denoise_config(body_cfg),
        lookahead_m=float(cfg.get("local_planner", {}).get("lookahead_m", d.lookahead_m)),
        no_progress_timeout_s=float(cfg.get("no_progress_timeout_s", d.no_progress_timeout_s)),
        no_progress_eps_m=float(cfg.get("no_progress_eps_m", d.no_progress_eps_m)),
        goal_deadline_s=float(cfg.get("goal_deadline_s", d.goal_deadline_s)),
        swept_realign_thresh_rad=float(
            cfg.get("swept_realign_thresh_rad", d.swept_realign_thresh_rad)),
        swept_realign_timeout_s=float(
            cfg.get("swept_realign_timeout_s", d.swept_realign_timeout_s)),
        rotate_timeout_s=float(cfg.get("rotate_timeout_s", d.rotate_timeout_s)),
        translate_eps_m=float(cfg.get("translate_eps_m", d.translate_eps_m)),
        rotate_flip_band_rad=float(cfg.get("rotate_flip_band_rad", d.rotate_flip_band_rad)),
        min_rotate_omega_radps=float(
            cfg.get("min_rotate_omega_radps", d.min_rotate_omega_radps)),
        path_hysteresis=float(cfg.get("path_hysteresis", d.path_hysteresis)),
        path_max_deviation_m=float(cfg.get("path_max_deviation_m", d.path_max_deviation_m)),
        stop_margin_m=float(cfg.get("stop_margin_m", d.stop_margin_m)),
        seed_clear_radius_m=float(
            cfg.get("seed_clear_radius_m", plan.costmap.footprint_radius_m)),
        slow_retry=bool(cfg.get("slow_retry", d.slow_retry)),
        path=path_follow_config(body_cfg),
    )


def path_follow_config(body_cfg: Dict[str, Any]):
    """``local_drive.path`` — Tier-3 path mode (nav redesign step 3)."""
    from body.lib.path_follow import PathFollowConfig
    c = body_cfg.get("local_drive", {}).get("path", {}) or {}
    d = PathFollowConfig()
    return PathFollowConfig(**{k: type(getattr(d, k))(c[k]) for k in c if hasattr(d, k)})

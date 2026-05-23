"""Load fusion and SLAM parameters from config.json."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from desktop.world_map.particle_filter_pose import (
    ALPHA_ROT_PER_M,
    ALPHA_ROT_PER_RAD,
    ALPHA_TRANS_PER_M,
    IMU_SIGMA_PER_SAMPLE_RAD,
)

try:
    from body.lib.zenoh_helpers import load_body_config, repo_root
except ImportError:
    def repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def load_body_config(path: Path | None = None) -> dict[str, Any]:
        import json

        cfg_path = path or (repo_root() / "config.json")
        with open(cfg_path, encoding="utf-8") as fh:
            return json.load(fh)


@dataclass(frozen=True)
class FusionNoiseConfig:
    alpha_trans_per_m: float = ALPHA_TRANS_PER_M
    alpha_rot_per_m: float = ALPHA_ROT_PER_M
    alpha_rot_per_rad: float = ALPHA_ROT_PER_RAD
    imu_sigma_rad: float = IMU_SIGMA_PER_SAMPLE_RAD
    imu_obs_hz: float = 50.0
    odom_obs_hz: float = 50.0
    wheel_base_m: float = 0.181
    imu_settle_time_s: float = 2.0
    mag_idle_fusion_enabled: bool = True
    mag_sigma_rad: float = 0.035
    mag_update_min_interval_s: float = 0.2


@dataclass(frozen=True)
class SlamConfig:
    match_hz: float = 2.0
    scan_match_xy_half_m: float = 0.30
    scan_match_theta_half_deg: float = 8.0
    min_improvement: float = 10.0
    loop_closure_search_radius_m: float = 2.0
    loop_closure_min_response: float = 0.35
    graph_optimize_every_n_nodes: int = 5
    submap_node_count: int = 20
    min_submap_evidence_cells: int = 200
    resolution_m: float = 0.05
    extent_m: float = 40.0
    relocate_xy_half_m: float = 3.0
    relocate_theta_half_deg: float = 180.0


@dataclass(frozen=True)
class SlamFusionConfig:
    fusion: FusionNoiseConfig
    slam: SlamConfig


def _float(section: dict[str, Any], key: str, default: float) -> float:
    val = section.get(key, default)
    return float(val)


def _int(section: dict[str, Any], key: str, default: int) -> int:
    val = section.get(key, default)
    return int(val)


def load_slam_config(path: Optional[Path] = None) -> SlamFusionConfig:
    """Parse motor/imu/lidar plus fusion/slam sections from config.json."""
    body = load_body_config(path)
    motor = body.get("motor", {})
    imu = body.get("imu", {})
    fusion_sec = body.get("fusion", {})
    slam_sec = body.get("slam", {})

    fusion = FusionNoiseConfig(
        alpha_trans_per_m=_float(fusion_sec, "alpha_trans_per_m", ALPHA_TRANS_PER_M),
        alpha_rot_per_m=_float(fusion_sec, "alpha_rot_per_m", ALPHA_ROT_PER_M),
        alpha_rot_per_rad=_float(fusion_sec, "alpha_rot_per_rad", ALPHA_ROT_PER_RAD),
        imu_sigma_rad=_float(fusion_sec, "imu_sigma_rad", IMU_SIGMA_PER_SAMPLE_RAD),
        imu_obs_hz=_float(fusion_sec, "imu_obs_hz", 50.0),
        odom_obs_hz=_float(fusion_sec, "odom_obs_hz", 50.0),
        wheel_base_m=_float(motor, "wheel_base_m", 0.181),
        imu_settle_time_s=_float(imu, "settle_time_s", 2.0),
        mag_idle_fusion_enabled=bool(fusion_sec.get("mag_idle_fusion_enabled", True)),
        mag_sigma_rad=_float(fusion_sec, "mag_sigma_rad", 0.035),
        mag_update_min_interval_s=_float(
            fusion_sec, "mag_update_min_interval_s", 0.2,
        ),
    )
    slam = SlamConfig(
        match_hz=_float(slam_sec, "match_hz", 2.0),
        scan_match_xy_half_m=_float(slam_sec, "scan_match_xy_half_m", 0.30),
        scan_match_theta_half_deg=_float(
            slam_sec, "scan_match_theta_half_deg", 8.0,
        ),
        min_improvement=_float(slam_sec, "min_improvement", 10.0),
        loop_closure_search_radius_m=_float(
            slam_sec, "loop_closure_search_radius_m", 2.0,
        ),
        loop_closure_min_response=_float(
            slam_sec, "loop_closure_min_response", 0.35,
        ),
        graph_optimize_every_n_nodes=_int(
            slam_sec, "graph_optimize_every_n_nodes", 5,
        ),
        submap_node_count=_int(slam_sec, "submap_node_count", 20),
        min_submap_evidence_cells=_int(
            slam_sec, "min_submap_evidence_cells", 200,
        ),
        resolution_m=_float(slam_sec, "resolution_m", 0.05),
        extent_m=_float(slam_sec, "extent_m", 40.0),
        relocate_xy_half_m=_float(slam_sec, "relocate_xy_half_m", 3.0),
        relocate_theta_half_deg=_float(
            slam_sec, "relocate_theta_half_deg", 180.0,
        ),
    )
    return SlamFusionConfig(fusion=fusion, slam=slam)


def scan_matcher_config_from_slam(slam: SlamConfig):
    """Build ScanMatcherConfig from SlamConfig."""
    from desktop.nav.slam.scan_matcher import ScanMatcherConfig

    return ScanMatcherConfig(
        xy_half_m=slam.scan_match_xy_half_m,
        theta_half_rad=math.radians(slam.scan_match_theta_half_deg),
        min_improvement=slam.min_improvement,
    )


def loop_closure_matcher_config_from_slam(slam: SlamConfig):
    """Wide search window for loop-closure scan match."""
    from desktop.nav.slam.scan_matcher import ScanMatcherConfig

    return ScanMatcherConfig(
        xy_half_m=slam.relocate_xy_half_m,
        xy_step_m=0.10,
        theta_half_rad=math.radians(slam.relocate_theta_half_deg),
        theta_step_rad=math.radians(5.0),
        min_improvement=slam.min_improvement * 2.0,
    )

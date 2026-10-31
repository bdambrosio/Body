"""Localization controller configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

# Router resolution (CLI > ZENOH_CONNECT env > default) is shared with the
# chassis side — one implementation, re-exported here for existing callers.
from desktop.chassis.config import DEFAULT_ROUTER, ENV_VAR, resolve_router

__all__ = [
    "DEFAULT_ROUTER", "ENV_VAR", "resolve_router",
    "Topics", "LocalizationConfig",
]


@dataclass
class Topics:
    odom: str = "body/odom"
    lidar_scan: str = "body/lidar/scan"
    world_cmd: str = "body/world_map/cmd"
    world_status: str = "body/world_map/status"
    world_driveable: str = "body/map/world_driveable"


@dataclass
class LocalizationConfig:
    router: str = DEFAULT_ROUTER
    map_path: str = ""

    status_hz: float = 1.0
    publish_hz: float = 2.0
    stale_odom_s: float = 0.25
    input_timeout_s: float = 2.0

    pf_device: str = "cpu"
    pf_n_particles: int = 5000
    pf_imu_obs_hz: float = 5.0
    # Lidar scan-match rate cap. Pi publishes ~10 Hz; match every scan.
    scan_hz: float = 10.0
    # Scan reweight strength: lower → scan match moves posterior faster.
    pf_scan_temperature_log_ratio: float = 1.5
    # Per-odom-tick process blur (correlated odom uncertainty).
    pf_odom_blur_xy_m: float = 0.004
    pf_odom_blur_theta_rad: float = 0.001

    teleport_distance_m: float = 0.5
    teleport_rotation_rad: float = 3.14159 / 4.0

    pose_trail_seconds: float = 60.0
    pose_trail_min_dxy_m: float = 0.05
    pose_trail_min_dtheta_rad: float = 0.0873
    pose_trail_min_period_s: float = 1.0
    pose_trail_max_points: int = 1024

    map_stale_s: float = 2.0
    ui_redraw_hz: float = 5.0
    # Global-planner lethal radius = the robot's true half-width (220 mm
    # wide; config.json local_planner.footprint_radius_m = 0.11), the same
    # as Tier-3's. Safety lives in the halo COST, not in lethal: with 0.15
    # + the planner's 1-cell clearance every gap under 0.40 m was a wall
    # and each speckle pixel a 0.4 m disc (10-31).
    footprint_radius_m: float = 0.11

    topics: Topics = field(default_factory=Topics)

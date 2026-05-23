"""Localization controller configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_ROUTER = "tcp/127.0.0.1:7447"
ENV_VAR = "ZENOH_CONNECT"


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
    scan_hz: float = 5.0

    teleport_distance_m: float = 0.5
    teleport_rotation_rad: float = 3.14159 / 4.0

    pose_trail_seconds: float = 60.0
    pose_trail_min_dxy_m: float = 0.05
    pose_trail_min_dtheta_rad: float = 0.0873
    pose_trail_min_period_s: float = 1.0
    pose_trail_max_points: int = 1024

    map_stale_s: float = 2.0
    ui_redraw_hz: float = 5.0
    footprint_radius_m: float = 0.15

    topics: Topics = field(default_factory=Topics)


def resolve_router(cli_value: Optional[str]) -> str:
    if cli_value:
        return cli_value
    env = os.environ.get(ENV_VAR)
    if env:
        return env
    return DEFAULT_ROUTER

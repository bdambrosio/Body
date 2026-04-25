"""World-map fuser configuration.

Precedence: CLI --router > ZENOH_CONNECT env > default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_ROUTER = "tcp/127.0.0.1:7447"
ENV_VAR = "ZENOH_CONNECT"


@dataclass
class Topics:
    local_map: str = "body/map/local_2p5d"
    odom: str = "body/odom"
    lidar_scan: str = "body/lidar/scan"
    status_in: str = "body/status"
    world_cmd: str = "body/world_map/cmd"
    world_status: str = "body/world_map/status"
    world_driveable: str = "body/map/world_driveable"


@dataclass
class FuserConfig:
    router: str = DEFAULT_ROUTER

    world_extent_m: float = 40.0
    world_resolution_m: float = 0.08
    publish_hz: float = 2.0
    publish_margin_cells: int = 4
    status_hz: float = 1.0

    stale_odom_s: float = 0.25
    input_timeout_s: float = 2.0

    pose_source: str = "odom"  # v1.1: "odom+scanmatch"
    # When True, FuserController constructs ImuPlusScanMatchPose
    # instead of OdomPose and calls .connect(session, grid) after
    # the zenoh session is open. See docs/slam_pi_contract.md.
    slam_enabled: bool = False

    vote_margin: int = 2
    # Sum-bounded vote model ("FIFO of length vote_capacity"):
    # per cell, clear_votes + block_votes ≤ vote_capacity. New
    # observations on one side displace existing evidence on the
    # other side, scaling both proportionally so the total returns
    # to the cap. There is no time-based decay — values persist
    # forever until contradicted by fresh observations. This is what
    # makes a saved snapshot a strong prior on reload: confidently-
    # observed cells stay at their saved values until the next
    # session's observations actively contradict them.
    #
    # Override cost: ~10 contradicting observations to flip a fully
    # saturated cell, ~2 s of fresh observation at 5 Hz fusion.
    vote_capacity: float = 10.0
    traversal_stamp_hz: float = 10.0
    traversal_vote_weight: int = 3
    footprint_radius_m: float = 0.15

    ui_redraw_hz: float = 5.0
    map_stale_s: float = 2.0

    # Pose-trail buffer (rendered as a polyline overlay). Trimmed by
    # both age and a min-displacement gate so a stationary robot
    # doesn't fill the buffer with duplicate poses.
    pose_trail_seconds: float = 60.0
    pose_trail_min_dxy_m: float = 0.05
    pose_trail_min_dtheta_rad: float = 0.0873  # ~5°
    pose_trail_min_period_s: float = 1.0
    pose_trail_max_points: int = 1024

    topics: Topics = field(default_factory=Topics)


def resolve_router(cli_value: Optional[str]) -> str:
    if cli_value:
        return cli_value
    env = os.environ.get(ENV_VAR)
    if env:
        return env
    return DEFAULT_ROUTER

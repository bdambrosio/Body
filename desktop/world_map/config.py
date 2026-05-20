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

    # PoseSource selection:
    #   "odom"     — OdomPose (encoder integration only, no SLAM)
    #   "slam"     — ImuPlusScanMatchPose (encoder + BNO085 yaw + scan-match
    #                snap corrections; docs/slam_pi_contract.md)
    #   "particle" — ParticleFilterPoseSource (Bayesian filter:
    #                encoder predict + IMU obs + scan-likelihood obs;
    #                docs/bayesian_localization_redesign.md Phase 8)
    pose_source_type: str = "odom"
    # Back-compat: True forces pose_source_type="slam". Pre-existing
    # callers that set slam_enabled=True still work; new callers should
    # set pose_source_type directly. Removed in a future cleanup pass.
    slam_enabled: bool = False

    # Particle filter knobs (only consulted when pose_source_type ==
    # "particle"). pf_n_particles default bumped 1000 → 20000 on
    # 2026-05-17 after the Phase 6.3 shadow trace showed N_eff
    # thrashing below 5 between 2 Hz scan-tick resamples. More
    # particles don't fix the IMU-over-counting root cause (also
    # addressed via ParticleFilterConfig.imu_sigma_rad 5→15 mrad)
    # but they give the cloud more headroom to absorb sharp
    # observations and survive between resamples. On Bruce's RTX 6000
    # the cycle cost is flat from 10k to 100k particles, so 20k is
    # essentially free.
    pf_device: str = "cpu"
    pf_n_particles: int = 20000

    # Phase 6.4.1 — defensive resample fraction. Diverts a fraction
    # of particles at every resample to fresh draws from a wide
    # Gaussian around the posterior mean, preserving tail support
    # so observations like VPR (σ ~ 0.5 m) can actually discriminate
    # among particles. Default 0.05 (5%) — first-cut experiment;
    # the 6.4 shadow trace showed 100% of σ-gated rejections at
    # 6 mm cloud spread, and defensive injection is the next lever.
    # Set to 0.0 to disable.
    pf_defensive_fraction: float = 0.05

    # Phase 6.4.1.5 — IMU observation rate gate. 5 Hz default puts
    # IMU observations 200 ms apart so each one is approximately
    # independent (past the BNO085's white-noise correlation time).
    # 0 = disable IMU observations entirely; 50 = old per-odom-tick
    # behavior. See ParticleFilterPoseSourceConfig.imu_obs_hz.
    pf_imu_obs_hz: float = 5.0

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
    # Override cost: ~vote_capacity contradicting observations to
    # flip a fully saturated cell.
    vote_capacity: float = 6.0
    # Per-fusion vote weights. Historically asymmetric (block=0.5)
    # to compensate for the depth-driven clear regime where clear
    # required 4 consecutive frames vs. single-frame block. Since
    # 842b7c3 lidar is the clear source (single-ray = clear evidence)
    # and lidar terminal hits dominate blocks, both sides are now
    # single-frame, so symmetric weights are right. Asymmetry was
    # also letting phantom clears (from PF jitter past walls) beat
    # the rare block votes those cells see — symmetrizing reduces
    # smear-through-walls.
    clear_vote_weight: float = 1.0
    block_vote_weight: float = 1.0
    # Block-lock threshold. Once a cell's block_votes reach this
    # value, further clear votes from fuse_local_map are refused
    # (the cell is treated as a confirmed wall). Set above
    # vote_margin+1 so a cell only locks after it has confidently
    # displayed as blocked. stamp_traversal still adds clears
    # directly — driving over a cell breaks the lock, so a chair
    # pulled away from a wall is still recoverable.
    block_lock_threshold: float = 4.0
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

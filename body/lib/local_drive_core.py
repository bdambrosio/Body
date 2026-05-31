"""Pure Tier-3 drive logic: odom-frame goal → body-frame steering.

No I/O, no zenoh. Given the current odom pose and a goal point in the odom
frame, decide the body-frame steering command. The process wrapper
(``body/local_drive.py``) adds transport, the local A* planner + path follow,
the swept-footprint safety veto, no-progress / odom-stale timers, and status
publishing. Everything here is deterministic and unit-tested.

Frames: odom is a drifting world frame the Pi integrates from wheels/IMU;
body is +x forward, +y left, robot at origin. A goal is held in odom so it
stays fixed as the robot moves; each tick we re-express it in the live body
frame and steer toward it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

# Drive states (mirrored in body/drive/status).
STATE_IDLE = "IDLE"
STATE_DRIVING = "DRIVING"
STATE_ARRIVED = "ARRIVED"
STATE_BLOCKED = "BLOCKED"
STATE_CANCELED = "CANCELED"
STATE_FAULT = "FAULT"

Pose = Tuple[float, float, float]   # (x, y, theta) in odom
Point2 = Tuple[float, float]


@dataclass(frozen=True)
class DriveParams:
    v_max: float = 0.18
    omega_max: float = 0.6
    v_min_mps: float = 0.08
    arrival_tol_m: float = 0.12
    # Bearing beyond which we rotate in place rather than arc toward the goal.
    rotate_in_place_thresh_rad: float = 0.61   # ~35° — ENTER rotate-in-place
    # Hysteresis: once rotating, keep rotating until the bearing falls below
    # this (lower) threshold before resuming drive. Prevents rotate↔drive
    # chatter when the bearing hovers at the enter threshold.
    rotate_exit_thresh_rad: float = 0.26        # ~15° — EXIT rotate-in-place
    k_omega: float = 1.5                        # proportional heading gain
    slowdown_distance_m: float = 0.4
    # Heading tolerance for the optional final-heading rotate on arrival.
    heading_tol_rad: float = 0.087              # ~5°


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def odom_to_body(goal_odom: Point2, pose: Pose) -> Point2:
    """Express an odom-frame point in the body frame of `pose`."""
    dx = goal_odom[0] - pose[0]
    dy = goal_odom[1] - pose[1]
    c = math.cos(-pose[2])
    s = math.sin(-pose[2])
    return (dx * c - dy * s, dx * s + dy * c)


def body_to_odom(body_pt: Point2, pose: Pose) -> Point2:
    """Inverse of `odom_to_body`: a body-frame point → odom frame."""
    c = math.cos(pose[2])
    s = math.sin(pose[2])
    return (
        pose[0] + body_pt[0] * c - body_pt[1] * s,
        pose[1] + body_pt[0] * s + body_pt[1] * c,
    )


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def steer_to_body_point(
    goal_body: Point2, params: DriveParams, rotating: bool = False,
) -> Tuple[float, float, float, float, bool]:
    """Pure-pursuit-style command toward a body-frame point.

    Returns (v_mps, omega_radps, dist_m, bearing_rad, rotating_next).
    Rotates in place when the bearing exceeds the threshold (so it faces the
    goal before driving), otherwise arcs toward it with a goal-distance
    slowdown. ``rotating`` is the caller-held rotate-in-place state from the
    previous tick; the returned ``rotating_next`` must be fed back next tick.
    Hysteresis (enter at ``rotate_in_place_thresh_rad``, exit only below
    ``rotate_exit_thresh_rad``) prevents rotate↔drive chatter at the boundary.
    Caller decides arrival/safety; this only shapes the velocity.
    """
    bx, by = goal_body
    dist = math.hypot(bx, by)
    bearing = math.atan2(by, bx)
    omega = _clip(params.k_omega * bearing, -params.omega_max, params.omega_max)
    thresh = (params.rotate_exit_thresh_rad if rotating
              else params.rotate_in_place_thresh_rad)
    if abs(bearing) > thresh:
        return 0.0, omega, dist, bearing, True
    v_factor = _clip(dist / max(params.slowdown_distance_m, 1e-6), 0.0, 1.0)
    v = params.v_max * v_factor
    if 0.0 < v < params.v_min_mps:
        v = params.v_min_mps
    return v, omega, dist, bearing, False


def rotate_to_heading(
    current_theta: float, target_theta: float, params: DriveParams,
) -> Tuple[float, bool]:
    """For the optional final-heading turn on arrival. Returns
    (omega_radps, aligned)."""
    err = wrap_pi(target_theta - current_theta)
    if abs(err) <= params.heading_tol_rad:
        return 0.0, True
    sign = 1.0 if err >= 0.0 else -1.0
    return sign * min(params.omega_max, params.k_omega * abs(err)), False

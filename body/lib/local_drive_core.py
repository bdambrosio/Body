"""Pure Tier-3 drive logic: odom-frame goal → body-frame steering.

No I/O, no zenoh. Given the current odom pose and a goal point in the odom
frame, decide the body-frame steering command. The process wrapper
(``body/local_drive.py``) adds transport, the swept-footprint safety veto,
no-progress / odom-stale timers, and status publishing. Everything here is
deterministic and unit-tested.

Frames: odom is a drifting world frame the Pi integrates from wheels/IMU;
body is +x forward, +y left, robot at origin. A goal is held in odom so it
stays fixed as the robot moves; each tick we re-express it in the live body
frame and steer toward it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from body.lib.drive_safety import FootprintConfig, swept_path_blocked

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
    rotate_in_place_thresh_rad: float = 0.61   # ~35°
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
    goal_body: Point2, params: DriveParams,
) -> Tuple[float, float, float, float]:
    """Pure-pursuit-style command toward a body-frame point.

    Returns (v_mps, omega_radps, dist_m, bearing_rad). Rotates in place when
    the bearing exceeds the threshold (so it faces the goal before driving),
    otherwise arcs toward it with a goal-distance slowdown. Caller decides
    arrival/safety; this only shapes the velocity toward the point.
    """
    bx, by = goal_body
    dist = math.hypot(bx, by)
    bearing = math.atan2(by, bx)
    omega = _clip(params.k_omega * bearing, -params.omega_max, params.omega_max)
    if abs(bearing) > params.rotate_in_place_thresh_rad:
        return 0.0, omega, dist, bearing
    v_factor = _clip(dist / max(params.slowdown_distance_m, 1e-6), 0.0, 1.0)
    v = params.v_max * v_factor
    if 0.0 < v < params.v_min_mps:
        v = params.v_min_mps
    return v, omega, dist, bearing


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


# ── Proactive local steering (Tier-3 local planner) ──────────────────


@dataclass(frozen=True)
class LocalPlanConfig:
    # Proactive corridor centering: probe forward-left / forward-right for
    # the nearest obstacle; when one is within `center_trigger_m`, bias the
    # heading away from the closer side (so the body stays off the walls
    # *before* it would clip — not after).
    center_probe_m: float = 0.35
    center_probe_rad: float = 0.52      # ~30° off the body axis
    center_trigger_m: float = 0.45
    k_center: float = 1.0               # rad/s per metre of L/R imbalance
    # Reactive fan: when pursuit (even with centering) is blocked, search
    # arcs at growing turn offsets for a feasible one to steer around.
    fan_max_rad: float = 0.87           # ~50°
    fan_step_rad: float = 0.21          # ~12°
    nudge_v_floor: float = 0.4          # min speed scale on a sharp nudge


def _fan_offsets(max_rad: float, step_rad: float):
    """[+s, -s, +2s, -2s, …] up to max_rad (0 handled separately)."""
    offs = []
    k = 1
    while k * step_rad <= max_rad + 1e-9:
        offs.append(k * step_rad)
        offs.append(-k * step_rad)
        k += 1
    return offs


def _nearest_blocked(bxs, bys, px: float, py: float) -> float:
    return float(np.min(np.hypot(bxs - px, bys - py)))


def plan_drive(
    grid: np.ndarray,
    meta: dict,
    goal_body: Point2,
    params: DriveParams,
    foot: FootprintConfig,
    cfg: LocalPlanConfig,
) -> Tuple[float, float, str]:
    """Proactive clearance-aware local steering toward `goal_body`.

    Returns (v_mps, omega_radps, mode) where mode is one of
    'rotate' | 'pursue' | 'center' | 'nudge' | 'blocked':
      * rotate  — bearing too large; turn in place to face the goal.
      * pursue  — straight pursuit, path clear.
      * center  — pursuit + a clearance bias steering off a nearby wall.
      * nudge   — pursuit blocked; a turning arc around the obstacle.
      * blocked — no feasible forward arc (dead-end); caller stops/escalates.

    The swept-footprint check (#3, directional) is the hard feasibility gate;
    centering and the fan only ever choose among collision-free motions.
    """
    v_des, omega_des, dist, bearing = steer_to_body_point(goal_body, params)
    if v_des < 1e-3:
        return v_des, omega_des, "rotate"

    res = float(meta["resolution_m"])
    ox = float(meta["origin_x_m"])
    oy = float(meta["origin_y_m"])
    bi, bj = np.where(grid == 0)
    has_blocked = bi.size > 0
    bxs = ox + (bi + 0.5) * res if has_blocked else None
    bys = oy + (bj + 0.5) * res if has_blocked else None

    # Proactive centering bias from forward L/R clearance imbalance.
    omega_center = 0.0
    if has_blocked:
        a = cfg.center_probe_rad
        lc = cfg.center_probe_m
        dl = _nearest_blocked(bxs, bys, lc * math.cos(a), lc * math.sin(a))
        dr = _nearest_blocked(bxs, bys, lc * math.cos(-a), lc * math.sin(-a))
        if min(dl, dr) < cfg.center_trigger_m:
            # dl < dr → wall on the left → steer right (negative ω).
            omega_center = _clip(
                cfg.k_center * (dl - dr), -params.omega_max, params.omega_max
            )

    # Try pursuit (with centering) first.
    omega = _clip(omega_des + omega_center, -params.omega_max, params.omega_max)
    if not swept_path_blocked(grid, meta, v_mps=v_des, omega_radps=omega, config=foot):
        return v_des, omega, ("center" if abs(omega_center) > 1e-3 else "pursue")

    # Blocked → reactive fan: pure pursuit (0) first, then growing offsets.
    for d in [0.0] + _fan_offsets(cfg.fan_max_rad, cfg.fan_step_rad):
        th = bearing + d
        om = _clip(params.k_omega * th, -params.omega_max, params.omega_max)
        v = max(params.v_min_mps, v_des * max(cfg.nudge_v_floor, math.cos(d)))
        if not swept_path_blocked(grid, meta, v_mps=v, omega_radps=om, config=foot):
            return v, om, ("pursue" if abs(d) < 1e-9 else "nudge")

    return 0.0, 0.0, "blocked"

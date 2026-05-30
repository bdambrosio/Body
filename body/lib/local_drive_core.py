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
    # Proactive corridor centering: each frame, measure the nearest obstacle
    # on the LEFT vs RIGHT of the heading within a forward+abeam window, and
    # steer away from whichever side is closer than `center_target_clear_m`.
    # Unlike a forward-only probe, this sees walls directly abeam (a doorjamb
    # you're driving alongside) — which the hard veto deliberately ignores so
    # you can pass them — so centering is what actually keeps side clearance.
    center_range_m: float = 0.8         # consider obstacles within this radius
    center_back_margin_m: float = 0.2   # include cells slightly behind abeam
    center_target_clear_m: float = 0.3  # desired clearance from each side
    k_center: float = 1.5               # rad/s per metre of clearance deficit
    # Forward-clearance speed governor: ease v down as the nearest obstacle in
    # the forward cone closes in, so the robot doesn't barrel into tight spots
    # (and centering gets more ticks to act). Full speed ≥ full_clear, crawl
    # at v_min by min_clear.
    gov_full_clear_m: float = 1.1
    gov_min_clear_m: float = 0.5        # ~ swept reach, so it's crawling by then
    gov_cone_rad: float = 0.35          # ±20° forward cone
    # Reactive fan: when pursuit (even with centering) is blocked, search
    # arcs at growing turn offsets for a feasible one to steer around.
    fan_max_rad: float = 0.87           # ~50°
    fan_step_rad: float = 0.21          # ~12°
    nudge_v_floor: float = 0.4          # min speed scale on a sharp nudge
    # Gap seeking: when the fan is fully blocked, look for an open corridor
    # *beyond* the fan's reach (a side arm) and rotate to face it. A
    # direction is "open" if its ray-clearance ≥ gap_min_m. If the only
    # opening is within the fan (already tried, swept-infeasible), it's a
    # narrow/wedged spot → blocked instead (caller escalates).
    gap_scan_range_m: float = 1.2
    gap_min_m: float = 0.6
    gap_step_rad: float = 0.21          # ~12° scan resolution
    gap_max_rad: float = 1.92           # ~110° each side of the goal bearing


def _fan_offsets(max_rad: float, step_rad: float):
    """[+s, -s, +2s, -2s, …] up to max_rad (0 handled separately)."""
    offs = []
    k = 1
    while k * step_rad <= max_rad + 1e-9:
        offs.append(k * step_rad)
        offs.append(-k * step_rad)
        k += 1
    return offs


def _side_clearances(bxs, bys, cfg):
    """Nearest obstacle distance on the left (by>0) vs right (by<0) of the
    heading, within a forward+abeam window. Returns (left_clear, right_clear);
    math.inf when a side is clear. Sees abeam walls, not just forward ones."""
    d = np.hypot(bxs, bys)
    region = (bxs > -cfg.center_back_margin_m) & (d < cfg.center_range_m)
    left = region & (bys > 0.0)
    right = region & (bys < 0.0)
    lc = float(d[left].min()) if np.any(left) else math.inf
    rc = float(d[right].min()) if np.any(right) else math.inf
    return lc, rc


def _forward_clearance(bxs, bys, cfg):
    """Nearest obstacle distance within a ±gov_cone_rad cone ahead of the body
    (math.inf if none) — drives the speed governor."""
    ang = np.arctan2(bys, bxs)
    d = np.hypot(bxs, bys)
    mask = (np.abs(ang) < cfg.gov_cone_rad) & (d < cfg.gov_full_clear_m + 0.3)
    return float(d[mask].min()) if np.any(mask) else math.inf


def _ray_clearance(grid, res, ox, oy, nx, ny, th, max_range):
    """Distance from the body origin to the first blocked cell along bearing
    `th` (body frame), capped at `max_range`. Unknown cells pass through."""
    steps = max(1, int(max_range / res))
    cth, sth = math.cos(th), math.sin(th)
    for s in range(1, steps + 1):
        d = s * res
        i = int((d * cth - ox) / res)
        j = int((d * sth - oy) / res)
        if not (0 <= i < nx and 0 <= j < ny):
            return d
        if grid[i, j] == 0:
            return d
    return max_range


def _find_open_heading(grid, res, ox, oy, nx, ny, bearing, cfg):
    """Body-frame bearing of the open corridor nearest the goal bearing
    (ray-clearance ≥ gap_min_m), and its offset from the goal bearing.
    Scans outward from the goal bearing so the first hit is the nearest.
    Returns (None, None) if nothing is open."""
    offs = [0.0]
    k = 1
    while k * cfg.gap_step_rad <= cfg.gap_max_rad + 1e-9:
        offs.append(k * cfg.gap_step_rad)
        offs.append(-k * cfg.gap_step_rad)
        k += 1
    for off in offs:
        th = bearing + off
        clr = _ray_clearance(grid, res, ox, oy, nx, ny, th, cfg.gap_scan_range_m)
        if clr >= cfg.gap_min_m:
            return th, off
    return None, None


def plan_drive(
    grid: np.ndarray,
    meta: dict,
    goal_body: Point2,
    params: DriveParams,
    foot: FootprintConfig,
    cfg: LocalPlanConfig,
) -> Tuple[float, float, str, float]:
    """Proactive clearance-aware local steering toward `goal_body`.

    Returns (v_mps, omega_radps, mode, seek_target) where mode is one of
    'rotate' | 'pursue' | 'center' | 'nudge' | 'seek' | 'blocked':
      * rotate  — bearing too large; turn in place to face the goal.
      * pursue  — straight pursuit, path clear.
      * center  — pursuit + a clearance bias steering off a nearby wall.
      * nudge   — pursuit blocked; a turning arc around the obstacle.
      * seek    — fan blocked, but an open corridor exists *beyond* the fan;
                  `seek_target` is its body-frame bearing — the caller should
                  rotate to face it (with commitment, to avoid flip-flopping
                  back toward the goal) then resume driving.
      * blocked — no feasible forward arc and no out-of-fan opening (dead-end
                  or too narrow); caller stops/escalates.

    The swept-footprint check (#3, directional) is the hard feasibility gate;
    centering, the fan, and gap-seeking only choose collision-free motions.
    `seek_target` is meaningful only when mode == 'seek'.
    """
    v_des, omega_des, dist, bearing = steer_to_body_point(goal_body, params)
    if v_des < 1e-3:
        return v_des, omega_des, "rotate", 0.0

    res = float(meta["resolution_m"])
    ox = float(meta["origin_x_m"])
    oy = float(meta["origin_y_m"])
    nx, ny = grid.shape
    bi, bj = np.where(grid == 0)
    has_blocked = bi.size > 0
    bxs = ox + (bi + 0.5) * res if has_blocked else None
    bys = oy + (bj + 0.5) * res if has_blocked else None

    # Proactive centering: steer away from whichever side is closer than the
    # target clearance, by how far below target it is (deficit). Sees abeam
    # walls, and only acts when a side is actually tight (no wander in the open).
    omega_center = 0.0
    if has_blocked:
        lc, rc = _side_clearances(bxs, bys, cfg)
        left_def = max(0.0, cfg.center_target_clear_m - lc)
        right_def = max(0.0, cfg.center_target_clear_m - rc)
        # right closer (right_def larger) → steer left (positive ω).
        omega_center = _clip(
            cfg.k_center * (right_def - left_def), -params.omega_max, params.omega_max
        )

    # Forward-clearance speed governor: scale a forward speed by how much room
    # is ahead (full ≥ full_clear, crawl at v_min by min_clear).
    def _gov(v: float) -> float:
        if v <= 1e-6 or not has_blocked:
            return v
        fc = _forward_clearance(bxs, bys, cfg)
        if fc >= cfg.gov_full_clear_m:
            return v
        span = max(1e-6, cfg.gov_full_clear_m - cfg.gov_min_clear_m)
        scale = min(1.0, max(0.0, (fc - cfg.gov_min_clear_m) / span))
        return max(params.v_min_mps, v * scale)

    # Try pursuit (with centering) first.
    omega = _clip(omega_des + omega_center, -params.omega_max, params.omega_max)
    if not swept_path_blocked(grid, meta, v_mps=v_des, omega_radps=omega, config=foot):
        return _gov(v_des), omega, ("center" if abs(omega_center) > 1e-3 else "pursue"), 0.0

    # Blocked → reactive fan: pure pursuit (0) first, then growing offsets.
    for d in [0.0] + _fan_offsets(cfg.fan_max_rad, cfg.fan_step_rad):
        th = bearing + d
        om = _clip(params.k_omega * th, -params.omega_max, params.omega_max)
        v = max(params.v_min_mps, v_des * max(cfg.nudge_v_floor, math.cos(d)))
        if not swept_path_blocked(grid, meta, v_mps=v, omega_radps=om, config=foot):
            return _gov(v), om, ("pursue" if abs(d) < 1e-9 else "nudge"), 0.0

    # Fan exhausted → gap-seek: is there an open corridor beyond the fan?
    th, off = _find_open_heading(grid, res, ox, oy, nx, ny, bearing, cfg)
    if th is None or abs(off) <= cfg.fan_max_rad:
        # Nothing open, or the only opening is within the fan we just tried
        # (so it's narrow/wedged, not a missed side arm) → escalate.
        return 0.0, 0.0, "blocked", 0.0
    omega = _clip(params.k_omega * th, -params.omega_max, params.omega_max)
    return 0.0, omega, "seek", th

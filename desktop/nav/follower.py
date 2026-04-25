"""Pure-pursuit path follower.

Given a planned path (list of world-frame waypoints) and the current
robot pose, produce a `cmd_vel` (linear, angular) that should steer
the robot along the path. Diff-drive friendly: when the lookahead
target is too far off the current heading, the follower commands an
in-place rotation rather than a curved arc.

Stage 4 of the nav stack runs this in **dry-run mode**: the result
is rendered on the map as the would-be drive direction, but is NOT
published to the chassis. Stage 5 wires the same output to a real
heartbeat-and-cmd_vel publisher.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

Pose = Tuple[float, float, float]            # (x_m, y_m, theta_rad)
Point2 = Tuple[float, float]                 # (x_m, y_m)


# ── Config + state types ────────────────────────────────────────────


@dataclass
class FollowerConfig:
    # Pure-pursuit lookahead distance — the point on the path the
    # follower aims for. Adaptive: scales with current speed (longer
    # lookahead at higher speed for smoother arcs), clamped to a
    # sane range.
    lookahead_min_m: float = 0.30
    lookahead_max_m: float = 0.80
    lookahead_t_s: float = 1.5     # lookahead = clamp(v * lookahead_t_s)

    # Velocity caps. Stage 4 dry-run uses these to size the rendered
    # arc; Stage 5 will actually publish these to chassis.
    v_max: float = 0.20            # m/s — conservative for first nav
    omega_max: float = 0.60        # rad/s

    # Heading-error gate for in-place rotation. If the lookahead point
    # is more than this many radians off the robot's current heading,
    # don't try to drive a curve toward it — rotate in place first.
    rotate_in_place_threshold_rad: float = math.radians(35.0)

    # Arrival: distance to last waypoint within which we declare done.
    arrival_tolerance_m: float = 0.20

    # Slow-down ramp as we approach the goal: linear scaling on v from
    # v_max at >= ramp_distance_m down to ~0 at arrival_tolerance_m.
    slowdown_distance_m: float = 0.50


# Status enum kept stringly-typed so the UI can render it directly
# without a separate enum import.
STATUS_FOLLOWING = "FOLLOWING"
STATUS_ROTATING = "ROTATING"
STATUS_ARRIVED = "ARRIVED"
STATUS_NO_PATH = "NO_PATH"


@dataclass
class FollowerOutput:
    status: str                     # one of the STATUS_* constants
    v_mps: float                    # commanded linear velocity
    omega_radps: float              # commanded angular velocity
    lookahead_world: Optional[Point2] = None
    distance_to_goal_m: float = 0.0
    heading_error_rad: float = 0.0
    note: str = ""

    @classmethod
    def stop(cls, status: str, *, note: str = "") -> "FollowerOutput":
        return cls(status=status, v_mps=0.0, omega_radps=0.0, note=note)


# ── Follower ────────────────────────────────────────────────────────


class Follower:
    """Stateless pure-pursuit follower. The whole tick state is in
    the inputs (path + pose) and the config; nothing is carried
    across calls. That makes it trivial to dry-run: same inputs →
    same outputs, no warmup.
    """

    def __init__(self, config: Optional[FollowerConfig] = None):
        self.config = config or FollowerConfig()
        self._last_v_mps: float = 0.0  # for adaptive-lookahead

    def update(
        self,
        path_world: List[Point2],
        pose: Optional[Pose],
    ) -> FollowerOutput:
        cfg = self.config
        if pose is None:
            return FollowerOutput.stop(STATUS_NO_PATH, note="no pose")
        if not path_world or len(path_world) < 2:
            return FollowerOutput.stop(STATUS_NO_PATH, note="no path")

        x_w, y_w, theta_w = pose
        goal = path_world[-1]
        dist_to_goal = math.hypot(goal[0] - x_w, goal[1] - y_w)

        # Arrival check.
        if dist_to_goal <= cfg.arrival_tolerance_m:
            return FollowerOutput(
                status=STATUS_ARRIVED,
                v_mps=0.0, omega_radps=0.0,
                lookahead_world=None,
                distance_to_goal_m=dist_to_goal,
                heading_error_rad=0.0,
                note="at goal",
            )

        # Adaptive lookahead from last commanded speed.
        L = max(
            cfg.lookahead_min_m,
            min(cfg.lookahead_max_m,
                self._last_v_mps * cfg.lookahead_t_s),
        )

        lookahead = _find_lookahead_point(path_world, (x_w, y_w), L)
        if lookahead is None:
            # The whole remaining path is shorter than L — aim for
            # the last waypoint (it's not the same as "arrived", it's
            # just nearer than L).
            lookahead = goal

        # Heading error to the lookahead point.
        dx_w = lookahead[0] - x_w
        dy_w = lookahead[1] - y_w
        bearing_w = math.atan2(dy_w, dx_w)
        alpha = _wrap_pi(bearing_w - theta_w)
        L_actual = math.hypot(dx_w, dy_w)

        # In-place rotation when the lookahead is too far off-axis.
        if abs(alpha) > cfg.rotate_in_place_threshold_rad:
            omega = _clip(alpha * 2.0, -cfg.omega_max, cfg.omega_max)
            self._last_v_mps = 0.0
            return FollowerOutput(
                status=STATUS_ROTATING,
                v_mps=0.0,
                omega_radps=omega,
                lookahead_world=lookahead,
                distance_to_goal_m=dist_to_goal,
                heading_error_rad=alpha,
                note=f"|α|={math.degrees(alpha):+.1f}° > "
                     f"{math.degrees(cfg.rotate_in_place_threshold_rad):.0f}°",
            )

        # Pure-pursuit curvature: κ = 2·sin(α) / L_actual.
        if L_actual < 1e-6:
            curvature = 0.0
        else:
            curvature = 2.0 * math.sin(alpha) / L_actual

        # Slow down as we approach the goal.
        v_factor = _clip(
            (dist_to_goal - cfg.arrival_tolerance_m)
            / max(cfg.slowdown_distance_m, 1e-6),
            0.0, 1.0,
        )
        v = cfg.v_max * v_factor
        omega = _clip(v * curvature, -cfg.omega_max, cfg.omega_max)
        # If the |omega| cap saturated and curvature is very high,
        # cap v so the arc is still well-defined (don't drive faster
        # than the angular cap can sustain on a tight turn).
        if abs(curvature) > 1e-3:
            v_for_omega_cap = cfg.omega_max / abs(curvature)
            v = min(v, v_for_omega_cap)

        self._last_v_mps = v
        return FollowerOutput(
            status=STATUS_FOLLOWING,
            v_mps=v,
            omega_radps=omega,
            lookahead_world=lookahead,
            distance_to_goal_m=dist_to_goal,
            heading_error_rad=alpha,
            note="",
        )


# ── Helpers ────────────────────────────────────────────────────────


def _find_lookahead_point(
    path: List[Point2],
    robot_xy: Point2,
    lookahead_m: float,
) -> Optional[Point2]:
    """Return the point on `path` that is approximately `lookahead_m`
    ahead of the robot along the path.

    Strategy: find the nearest path index to the robot, then walk
    forward along the path accumulating segment lengths until we
    reach `lookahead_m`. Interpolate along the final segment so the
    returned point sits exactly at distance `lookahead_m` along the
    path from the projection point.
    """
    if len(path) < 2:
        return None
    nearest_idx = _nearest_index(path, robot_xy)
    # If the nearest index is the last point, there's no path ahead.
    if nearest_idx >= len(path) - 1:
        return path[-1]

    # Project robot onto the segment starting at nearest_idx — that's
    # the "anchor" along the path from which we measure forward.
    a = path[nearest_idx]
    b = path[nearest_idx + 1]
    seg_proj, seg_t = _project_onto_segment(robot_xy, a, b)
    remaining = max(
        0.0,
        lookahead_m - math.hypot(
            seg_proj[0] - robot_xy[0], seg_proj[1] - robot_xy[1],
        ),
    )

    cur = seg_proj
    i = nearest_idx
    while i < len(path) - 1:
        nxt = path[i + 1]
        seg_len = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
        if seg_len >= remaining:
            if seg_len < 1e-9:
                return nxt
            t = remaining / seg_len
            return (
                cur[0] + t * (nxt[0] - cur[0]),
                cur[1] + t * (nxt[1] - cur[1]),
            )
        remaining -= seg_len
        cur = nxt
        i += 1
    return path[-1]


def _nearest_index(path: List[Point2], xy: Point2) -> int:
    best_i = 0
    best_d2 = float("inf")
    for i, (px, py) in enumerate(path):
        d2 = (px - xy[0]) ** 2 + (py - xy[1]) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def _project_onto_segment(
    xy: Point2, a: Point2, b: Point2,
) -> Tuple[Point2, float]:
    """Closest point on segment a–b to xy, and the parameter t∈[0,1]."""
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return a, 0.0
    t = ((xy[0] - a[0]) * abx + (xy[1] - a[1]) * aby) / denom
    t = max(0.0, min(1.0, t))
    return (a[0] + t * abx, a[1] + t * aby), t


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

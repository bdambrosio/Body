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
import time
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
    # arc; Stage 5 actually publishes these to chassis.
    v_max: float = 0.20            # m/s — conservative for first nav
    omega_max: float = 0.60        # rad/s

    # Acceleration caps — applied symmetrically to both directions of
    # change so the robot doesn't pop a wheelie on start and doesn't
    # snap to zero on a rotate-in-place transition. Tuned for an
    # indoor diff-drive at v_max=0.20 m/s: full speed is reached over
    # 2 redraw ticks (~0.4 s), ω_max over ~1.6 ticks.
    linear_accel_max_mps2: float = 0.50
    angular_accel_max_radps2: float = 1.50

    # Heading-error gate for in-place rotation. If the lookahead point
    # is more than this many radians off the robot's current heading,
    # don't try to drive a curve toward it — rotate in place first.
    rotate_in_place_threshold_rad: float = math.radians(35.0)

    # Arrival: distance to last waypoint within which we declare done.
    # 0.20 m is tight for a small house. The acceleration rate limit
    # already smooths the approach so the robot decelerates instead
    # of overshoot-then-circle; if pose-noise jitter near goal turns
    # out to be a problem at this tolerance, the next lever is to
    # cap pure-pursuit curvature when dist_to_goal is small (a "final
    # approach: aim straight" mode), not to grow this tolerance.
    arrival_tolerance_m: float = 0.20

    # Slow-down ramp as we approach the goal: linear scaling on v from
    # v_max at >= (arrival + slowdown) down to ~0 at arrival.
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
        # Last *commanded* values — fed forward into the next tick
        # for adaptive lookahead and for acceleration rate-limiting.
        self._last_v_mps: float = 0.0
        self._last_omega_radps: float = 0.0
        self._last_call_t: Optional[float] = None

    def update(
        self,
        path_world: List[Point2],
        pose: Optional[Pose],
    ) -> FollowerOutput:
        cfg = self.config
        # Measured tick interval, used for the acceleration rate
        # limiter. First call defaults to 0.20 s (the nominal 5 Hz
        # redraw tick); subsequent calls measure for real.
        now = time.monotonic()
        if self._last_call_t is None:
            dt = 0.20
        else:
            dt = max(0.05, min(1.0, now - self._last_call_t))
        self._last_call_t = now

        if pose is None:
            return self._stop_output(STATUS_NO_PATH, note="no pose")
        if not path_world or len(path_world) < 2:
            return self._stop_output(STATUS_NO_PATH, note="no path")

        x_w, y_w, theta_w = pose
        goal = path_world[-1]
        dist_to_goal = math.hypot(goal[0] - x_w, goal[1] - y_w)

        # Arrival check.
        if dist_to_goal <= cfg.arrival_tolerance_m:
            return self._stop_output(
                STATUS_ARRIVED,
                distance_to_goal_m=dist_to_goal,
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
            omega_target = _clip(alpha * 2.0, -cfg.omega_max, cfg.omega_max)
            v, omega = self._rate_limit(0.0, omega_target, dt)
            return FollowerOutput(
                status=STATUS_ROTATING,
                v_mps=v,
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
        v_target = cfg.v_max * v_factor
        omega_target = _clip(
            v_target * curvature, -cfg.omega_max, cfg.omega_max,
        )
        # If the |omega| cap saturated and curvature is very high,
        # cap v so the arc is still well-defined (don't drive faster
        # than the angular cap can sustain on a tight turn).
        if abs(curvature) > 1e-3:
            v_for_omega_cap = cfg.omega_max / abs(curvature)
            v_target = min(v_target, v_for_omega_cap)

        v, omega = self._rate_limit(v_target, omega_target, dt)
        return FollowerOutput(
            status=STATUS_FOLLOWING,
            v_mps=v,
            omega_radps=omega,
            lookahead_world=lookahead,
            distance_to_goal_m=dist_to_goal,
            heading_error_rad=alpha,
            note="",
        )

    # ── Rate-limit helpers ──────────────────────────────────────────

    def _rate_limit(
        self, v_target: float, omega_target: float, dt: float,
    ) -> Tuple[float, float]:
        """Clip the change since last command to the configured
        accel caps. Updates the cached "last commanded" values for
        the next tick. Returns (v, omega) actually-commanded.
        """
        cfg = self.config
        v_step = cfg.linear_accel_max_mps2 * dt
        omega_step = cfg.angular_accel_max_radps2 * dt
        v = _clip(v_target,
                  self._last_v_mps - v_step,
                  self._last_v_mps + v_step)
        omega = _clip(omega_target,
                      self._last_omega_radps - omega_step,
                      self._last_omega_radps + omega_step)
        # Also enforce absolute caps (in case _last_* drifted weird).
        v = _clip(v, 0.0, cfg.v_max)
        omega = _clip(omega, -cfg.omega_max, cfg.omega_max)
        self._last_v_mps = v
        self._last_omega_radps = omega
        return v, omega

    def _stop_output(
        self,
        status: str,
        *,
        distance_to_goal_m: float = 0.0,
        note: str = "",
    ) -> FollowerOutput:
        """Hard-stop output. Updates the cached last-commanded values
        to zero so a subsequent FOLLOWING tick ramps up from rest
        rather than from the previous cruise speed."""
        self._last_v_mps = 0.0
        self._last_omega_radps = 0.0
        return FollowerOutput(
            status=status,
            v_mps=0.0,
            omega_radps=0.0,
            lookahead_world=None,
            distance_to_goal_m=distance_to_goal_m,
            heading_error_rad=0.0,
            note=note,
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

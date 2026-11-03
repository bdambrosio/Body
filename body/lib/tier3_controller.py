"""Tier-3 per-goal control step (pure: no zenoh, no clocks, no threads).

``body/local_drive.py`` owns transport (subscriptions, odom ring buffer, IMU
yaw correction, e-stop / odom-reset / cancel handling, status publishing);
everything that decides what the wheels do for the active goal lives here so
it can be driven closed-loop in tests (``body/test_tier3_controller.py``).

One obstacle model (fix 3): the planner and the last-resort vetoes see the
same grid — the live scan raster in the scan's own body frame, with the
robot's footprint disk seeded clear, unexplained near-range unknown made
impassable for the planner (the veto refuses it), and a planner lethal radius
padded by the cell quantization (res/√2) so an A* path always clears the
swept veto's footprint.

Bounded rotation (fix 5): rotate-in-place has a direction latch; the swept
re-aim ("realign") and every other non-translating rotation share a per-goal
budget that only real translation (odom displacement) resets → BLOCKED
``rotate_timeout``; the previous A* path is kept while still valid and not
clearly worse (path hysteresis); progress is remaining path length; the
deadline is checked before the final-heading rotate.

Obstacle-aware speed (fix 4): on a swept block, retry the pure-pursuit arc,
then ``v_min``, then a shorter lookahead before re-aiming; the preview arc is
capped at the remaining path length; reaching a snapped / frontier endpoint
reports ARRIVED (``plan_reason`` says which) instead of a fake dead-end.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from body.lib.depth_denoise import DenoiseConfig
from body.lib.depth_memory import (
    DepthMemoryConfig, DepthObstacleMemory, plan_cells, stamp_blocked,
)
from body.lib.depth_veto import DepthVetoConfig, DepthVetoTracker
from body.lib.drive_safety import (
    FootprintConfig, explained_unknown_mask, seed_clear_disk,
    swept_path_blocked, unknown_block_mask,
)
from body.lib.local_costmap import build_local_costmap, dilate_bool, dilate_disk
from body.lib.local_drive_core import (
    STATE_ARRIVED, STATE_BLOCKED, STATE_DRIVING, DriveParams,
    body_to_odom, latched_rotate_omega, odom_to_body, relative_pose, rotate_to_heading,
    steer_to_body_point, swept_block_response,
)
from body.lib.local_planner import (
    LocalPlanConfig, lookahead_on_path, path_cost, path_remaining_m,
    plan_local, transform_path, trim_path,
)
from body.lib.path_follow import (
    PathFollowConfig, cumulative, first_blocked, max_deviation, point_at, project,
    raster_lookup, rejoin_index, resample,
)

Point2 = Tuple[float, float]
Pose = Tuple[float, float, float]

# Endpoint kinds that are a best-effort stand-in for the requested goal.
PARTIAL_ENDPOINTS = ("snapped", "frontier")
# Consecutive goal-reaching plans before a snapped/frontier latch is cleared.
LATCH_OK_TICKS = 3
# A latched endpoint whose path exceeds this × the straight goal distance is
# a detour: the fresh (shorter) plan wins. Same rule as the desktop's
# detour re-pick (HierConfig.detour_ratio).
LATCH_DETOUR_RATIO = 3.0


@dataclass(frozen=True)
class Tier3Config:
    params: DriveParams = field(default_factory=DriveParams)
    plan: LocalPlanConfig = field(default_factory=LocalPlanConfig)
    foot: FootprintConfig = field(default_factory=FootprintConfig)
    depth: DepthVetoConfig = field(default_factory=lambda: DepthVetoConfig(enabled=False))
    # Odom-frame memory of depth obstacles, stamped into the planning grid
    # (body/lib/depth_memory.py). Uses ``depth``'s camera model / ROI.
    memory: DepthMemoryConfig = field(default_factory=lambda: DepthMemoryConfig(enabled=False))
    # Observe-only veto: score every frame and report depth_would_block /
    # hits / bbox in the status, but never BLOCK on it. Off since 10-31 (~92
    # % of depth stops were floor glare / transitions); the reports feed an
    # offline denoiser until the false-positive rate is measured low.
    depth_act: bool = True
    # Blob gates on the veto's evidence (body/lib/depth_denoise.py).
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    lookahead_m: float = 0.4
    no_progress_timeout_s: float = 4.0
    no_progress_eps_m: float = 0.03
    goal_deadline_s: float = 30.0          # 0 disables
    swept_realign_thresh_rad: float = 0.10
    swept_realign_timeout_s: float = 2.0   # cumulative re-aim time between translations
    # Per-goal budget for rotating without translating (rotate-in-place,
    # swept re-aim, final-heading turn). Only odom displacement ≥
    # translate_eps_m resets it. 0 disables.
    rotate_timeout_s: float = 10.0
    translate_eps_m: float = 0.05
    rotate_flip_band_rad: float = 0.5      # direction-latch band around ±π
    min_rotate_omega_radps: float = 0.0    # floor for in-place turns (motor dead zone)
    # Keep the previous A* path while it stays collision-free and costs at
    # most (1 + path_hysteresis) × the fresh plan. 0 = always replan.
    path_hysteresis: float = 0.25
    path_max_deviation_m: float = 0.15     # robot farther than this from it → replan
    stop_margin_m: float = 0.05            # swept preview beyond the path end
    seed_clear_radius_m: float = 0.11      # robot footprint disk seeded clear
    terminal_zero_ticks: int = 3           # zero cmds after a terminal block, then silent
    slow_retry: bool = True                # fix 4 retries before re-aiming
    seal_unknown_next_to_hits: bool = True # planner: unknown touching a hit is wall
    path: PathFollowConfig = PathFollowConfig()   # path mode (goto kind "path", step 3)


@dataclass(frozen=True)
class Tier3Goal:
    cmd_id: int
    x: float                               # odom frame
    y: float
    tol: float
    final_heading: Optional[float] = None
    v_max: Optional[float] = None
    omega_max: Optional[float] = None
    # Path mode (nav redesign step 3): ``path`` is a robot-first polyline in
    # the odom frame and (x, y) is its end. ``route_id`` changes only when the
    # sender re-planned (a refresh with the same route_id is just a newer
    # copy); ``end_is_goal`` False = the end is the sender's horizon, not a
    # place to stop.
    path: Optional[Tuple[Point2, ...]] = None
    end_is_goal: bool = True
    ttl_s: Optional[float] = None
    route_id: int = 0
    path_seq: int = 0


def goal_params(base: DriveParams, g: "Tier3Goal") -> DriveParams:
    """Per-goal speed caps can only slow the robot: v_max is clamped to
    [v_min_mps, configured v_max], omega_max to (0, configured omega_max]."""
    kw = {}
    if g.v_max is not None:
        kw["v_max"] = max(base.v_min_mps, min(base.v_max, float(g.v_max)))
    if g.omega_max is not None and float(g.omega_max) > 0.0:
        kw["omega_max"] = min(base.omega_max, float(g.omega_max))
    return replace(base, **kw) if kw else base


@dataclass
class TickResult:
    state: str
    cmd: Optional[Tuple[float, float]]     # None → publish no cmd_vel this tick
    reason: Optional[str] = None
    mode: Optional[str] = None
    goal_body: Optional[Point2] = None
    dist: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    path_body: Optional[List[Point2]] = None
    plan_reason: Optional[str] = None
    done: bool = False                     # goal finished — caller clears it
    terminal: bool = False
    depth_hits: Optional[int] = None       # depth_block evidence (slab hit count)
    depth_seq: Optional[int] = None        # … and the depth frame's seq
    depth_bbox_px: Optional[Tuple[int, int, int, int, int, int]] = None  # u0,v0,u1,v1,w,h
    depth_hit_body: Optional[Any] = None   # (N, 2) body x, y of the veto hits
    depth_would_block: bool = False        # veto fired but depth_act is off (observe-only)
    # Path mode
    path_seq: Optional[int] = None         # the refresh being followed
    path_progress_m: Optional[float] = None  # arc length done along it
    path_age_s: Optional[float] = None     # time since the last refresh
    block_body_xy: Optional[Point2] = None  # path_blocked: where (live body frame)


class Tier3Controller:
    """Per-goal state machine for one Tier-3 goal at a time.

    ``gate`` (optional) is a ``HandoffGate``-like object (record / is_armed /
    should_hold / consume_continue) for the HO-3 inspector breakpoint."""

    def __init__(self, cfg: Tier3Config, gate: Any = None) -> None:
        self.cfg = cfg
        self.gate = gate
        self.goal: Optional[Tier3Goal] = None
        self._depth = DepthVetoTracker(cfg.depth, denoise=cfg.denoise)
        # Survives goals on purpose: a retry is a new goal, and forgetting
        # what stopped the last one is what made the robot creep (10-31).
        self.memory = DepthObstacleMemory(cfg.memory, cfg.depth)
        self._reset_state(0.0, (0.0, 0.0, 0.0))

    # ── lifecycle ────────────────────────────────────────────────────
    def _reset_state(self, now: float, pose: Pose) -> None:
        self.rotating = False
        self.rot_dir = 0
        self.realign_time = 0.0
        self.final_aligned = False
        self.goal_started_at = now
        self.last_translate_at = now
        self.anchor_xy = (pose[0], pose[1])
        self.prog_best: Optional[float] = None
        self.prog_at = now
        self.prev_path_odom: Optional[List[Point2]] = None
        self.prev_reason: Optional[str] = None
        self.endpoint_odom: Optional[Point2] = None
        self.goal_ok_ticks = 0                 # consecutive plans that reached the goal
        self.terminal: Optional[str] = None
        self.terminal_ticks = 0
        self.last_tick: Optional[float] = None
        self._depth.reset()
        self.path_rx_at = now
        self._path_cum = None
        self.path_s: Optional[float] = None
        self.path_stats = {"refreshes": 0, "max_gap_s": 0.0, "starved_episodes": 0,
                           "stale_blocks": 0}
        self._starved = False
        self._stale = False
        self.rejoin_side: Optional[int] = None
        self._path_refreshed = False

    def start_goal(self, goal: Tier3Goal, now_mono: float, pose_now: Pose) -> None:
        self.goal = goal
        self._reset_state(now_mono, pose_now)
        if goal.path is not None:
            self._path_cum = cumulative(goal.path)

    def update_path(self, goal: Tier3Goal, now_mono: float) -> None:
        """A newer copy of the active path goal (same cmd_id, higher
        path_seq): swap the polyline, keep progress / rotation state. A new
        ``route_id`` means the sender re-planned (e.g. after a block): that
        resets the progress clock and clears a terminal block — it is the
        resume."""
        old = self.goal
        self.goal = goal
        gap = now_mono - self.path_rx_at
        st = self.path_stats
        st["refreshes"] += 1
        st["max_gap_s"] = max(st["max_gap_s"], gap)
        self.path_rx_at = now_mono
        self._path_cum = cumulative(goal.path) if goal.path is not None else None
        self.path_s = None
        self.prev_path_odom = None
        self._path_refreshed = True          # re-reference progress, keep its timer
        if old is None or goal.route_id != old.route_id:
            self.prog_best = None
            self.prog_at = now_mono
            self.terminal = None
            self.terminal_ticks = 0
            self.realign_time = 0.0
            self.last_translate_at = now_mono

    def clear(self) -> None:
        self.goal = None

    def observe_depth(self, depth: Optional[Dict[str, Any]], *, now_wall: float,
                      pose_now: Pose, pose_depth: Optional[Pose],
                      omega_radps: float) -> bool:
        """Feed the depth memory. Call every control tick, with or without a
        goal, with raw wheel-odom poses (the memory's frame)."""
        return self.memory.update(depth, now_wall=now_wall, pose_now=pose_now,
                                  pose_depth=pose_depth, omega_radps=omega_radps)

    # ── one control tick ─────────────────────────────────────────────
    def step(
        self, *, now_mono: float, now_wall: float, pose_now: Pose,
        pose_scan: Optional[Pose], grid: Optional[np.ndarray],
        meta: Optional[Dict[str, Any]], depth: Optional[Dict[str, Any]] = None,
        pose_scan_raw: Optional[Pose] = None,
    ) -> TickResult:
        """``pose_scan_raw``: raw wheel-odom pose at the scan ts (the depth
        memory's frame); defaults to ``pose_scan``."""
        g = self.goal
        if g is None:
            raise RuntimeError("Tier3Controller.step without a goal")
        cfg = self.cfg
        gp = goal_params(cfg.params, g)
        dt = 0.0 if self.last_tick is None else min(0.5, max(0.0, now_mono - self.last_tick))
        self.last_tick = now_mono

        # Real translation (odom displacement) is the only thing that resets
        # the rotation budget and the cumulative re-aim time.
        if math.hypot(pose_now[0] - self.anchor_xy[0],
                      pose_now[1] - self.anchor_xy[1]) >= cfg.translate_eps_m:
            self.anchor_xy = (pose_now[0], pose_now[1])
            self.last_translate_at = now_mono
            self.realign_time = 0.0

        bx, by = odom_to_body((g.x, g.y), pose_now)
        dist = math.hypot(bx, by)
        gb = (bx, by)

        path_mode = g.path is not None and self._path_cum is not None
        path_len = path_rem = s_rob = 0.0
        age = ttl = 0.0
        if path_mode:
            s_rob, _lat = project(g.path, self._path_cum, (pose_now[0], pose_now[1]),
                                  self.path_s, cfg.path.back_window_m)
            self.path_s = s_rob
            path_len = float(self._path_cum[-1])
            path_rem = max(0.0, path_len - s_rob)
            age = now_mono - self.path_rx_at
            ttl = float(g.ttl_s or cfg.path.default_ttl_s)

        def res(state: str, cmd, **kw) -> TickResult:
            if path_mode:
                kw.setdefault("path_seq", g.path_seq)
                kw.setdefault("path_progress_m", s_rob)
                kw.setdefault("path_age_s", age)
            return TickResult(state=state, cmd=cmd, goal_body=gb, dist=dist, **kw)

        if self.terminal is not None:
            return self._terminal(gb, dist)

        # Path mode: a route leg can take minutes; staleness, no_progress and
        # the sender's route watchdog bound it instead of the deadline.
        deadline_hit = (not path_mode and cfg.goal_deadline_s > 0
                        and now_mono - self.goal_started_at > cfg.goal_deadline_s)
        rot_budget_hit = (cfg.rotate_timeout_s > 0
                          and now_mono - self.last_translate_at > cfg.rotate_timeout_s)

        # Arrival (+ optional final-heading rotate). The deadline and the
        # rotation budget bound the final-heading turn too (T13).
        at_goal = (dist <= g.tol and (not path_mode
                                       or (g.end_is_goal and path_rem <= g.tol + 0.3)))
        if at_goal:
            if g.final_heading is None or self.final_aligned:
                return res(STATE_ARRIVED, (0.0, 0.0), done=True)
            if deadline_hit:
                return self._enter_terminal("deadline", gb, dist)
            if rot_budget_hit:
                return self._enter_terminal("rotate_timeout", gb, dist)
            omega, aligned = rotate_to_heading(pose_now[2], float(g.final_heading), gp)
            if aligned:
                self.final_aligned = True
                return res(STATE_ARRIVED, (0.0, 0.0), done=True)
            if cfg.min_rotate_omega_radps > 0 and 0 < abs(omega) < cfg.min_rotate_omega_radps:
                omega = math.copysign(cfg.min_rotate_omega_radps, omega)
            return res(STATE_DRIVING, (0.0, omega), omega=omega, mode="rotate")

        if deadline_hit:
            return self._enter_terminal("deadline", gb, dist)

        starved = False
        if path_mode:
            # Staleness: the sender keeps the path fresh; without refreshes,
            # crawl at v_min, then stop (the desktop link may be up while its
            # planner is wedged — 11-03).
            if age > cfg.path.stale_factor * ttl:
                if not self._stale:
                    self.path_stats["stale_blocks"] += 1
                self._stale = True
                self.prog_at = now_mono
                return res(STATE_BLOCKED, (0.0, 0.0), reason="path_stale")
            self._stale = False
            at_horizon = not g.end_is_goal and path_rem <= cfg.path.end_hold_m
            starved = age > ttl or at_horizon
            if starved and not self._starved:
                self.path_stats["starved_episodes"] += 1
            self._starved = starved
            if at_horizon:
                # At the sender's horizon: wait for the path to be extended.
                self.prog_at = now_mono
                self.last_translate_at = now_mono
                return res(STATE_DRIVING, (0.0, 0.0), mode="starved")

        if grid is None or meta is None or pose_scan is None:
            self.prog_at = now_mono
            return res(STATE_BLOCKED, (0.0, 0.0), reason="no_scan")

        # ── one obstacle model, in the scan's own body frame ──
        rel = relative_pose(pose_scan, pose_now)          # robot now, in scan frame
        goal_s = odom_to_body((g.x, g.y), pose_scan)
        foot = cfg.foot
        work = np.array(grid, dtype=np.int8, copy=True)
        seed_clear_disk(work, meta, (0.0, 0.0), cfg.seed_clear_radius_m)
        seed_clear_disk(work, meta, (rel[0], rel[1]), cfg.seed_clear_radius_m)
        if len(self.memory):
            # Remembered depth obstacles are hits for the planner AND the
            # swept veto: an arc into one is refused before the robot moves.
            cells = plan_cells(self.memory.cells_body(pose_scan_raw or pose_scan),
                               rel, cfg.memory.plan_exclude_radius_m,
                               cfg.memory.plan_exclude_half_angle_deg)
            stamp_blocked(work, meta, cells, keep_clear_xy=(rel[0], rel[1]),
                          keep_clear_radius_m=cfg.seed_clear_radius_m)
        explained = (explained_unknown_mask(work, foot.unknown_explained_cells)
                     if foot.block_on_unknown else None)
        cm = build_local_costmap(work, meta, cfg.plan.costmap)
        ub = unknown_block_mask(work, meta, (rel[0], rel[1]), foot, explained)
        if cfg.seal_unknown_next_to_hits:
            # T6: an unknown cell touching a hit is a hole in a dotted wall
            # (grazing-angle hits are > 1 cell apart), not a doorway.
            ub = ub | ((work == -1) & dilate_bool(work == 0, iters=1))
        if np.any(ub):
            # The veto refuses these; the planner must not route through them.
            cm.lethal = cm.lethal | ub
            cm.cost[ub] = np.inf
        beyond = 0.0
        if path_mode:
            got = self._plan_on_path(g, work, meta, cm, rel, pose_scan, s_rob, path_len)
            if got[0] is None:
                self.prog_at = now_mono
                return res(STATE_BLOCKED, (0.0, 0.0), reason="path_blocked", mode="plan",
                           block_body_xy=_to_frame(got[1], rel))
            chosen, plan_reason, beyond = got
            adopted_new = plan_reason != self.prev_reason
            self.prev_reason = plan_reason
        else:
            plan = plan_local(work, meta, goal_s, cfg.plan, costmap=cm,
                              start_body=(rel[0], rel[1]))
            plan = self._latch_endpoint(plan, work, meta, cm, goal_s, rel, pose_scan)

            if not plan.ok:
                self.prev_path_odom = None
                self.prog_at = now_mono
                if plan.reason == "at_frontier":
                    # Already at the closest reachable cell to the goal.
                    return res(STATE_ARRIVED, (0.0, 0.0), done=True, plan_reason="frontier")
                return res(STATE_BLOCKED, (0.0, 0.0), reason=plan.reason, mode="plan")

            # Path hysteresis: keep following the previous path while it is
            # collision-free, we are still on it, its endpoint is as good as the
            # fresh plan's (no more than ~a cell farther from the goal — frontier
            # endpoints along a wall are near-ties that flip with quantization),
            # and it is not clearly more expensive (T4 homotopy flip-flop).
            chosen = plan.path_body
            plan_reason = plan.reason
            adopted_new = True
            resm = float(meta["resolution_m"])
            if (self.prev_path_odom and cfg.path_hysteresis > 0
                    and _endpoint_class(self.prev_reason) == _endpoint_class(plan.reason)):
                old_s = [odom_to_body(p, pose_scan) for p in self.prev_path_odom]
                old_trim, dev = trim_path(old_s, (rel[0], rel[1]))
                end_new = plan.path_body[-1]
                core = dilate_disk(work == 0, cfg.plan.costmap.footprint_radius_m / resm) | (work == 0)
                if ub is not None:
                    core = core | ub
                if dev <= cfg.path_max_deviation_m and old_trim:
                    c_old = path_cost(old_trim, cm, cfg.plan, lethal=core)
                    if self.rotating and math.isfinite(c_old):
                        # Turning in place to line up with a committed path: the
                        # body-aligned grid re-quantizes under the rotation and
                        # every fresh 8-connected path rotates with the robot, so
                        # re-planning here chases its own tail. Keep the path
                        # while it is still collision-free.
                        keep = True
                    else:
                        good_end = (math.hypot(old_trim[-1][0] - goal_s[0], old_trim[-1][1] - goal_s[1])
                                    <= math.hypot(end_new[0] - goal_s[0], end_new[1] - goal_s[1])
                                    + 1.5 * resm)
                        new_trim, _ = trim_path(plan.path_body, (rel[0], rel[1]))
                        c_new = path_cost(new_trim, cm, cfg.plan)
                        keep = (good_end and math.isfinite(c_old)
                                and c_old <= c_new * (1.0 + cfg.path_hysteresis))
                    if keep:
                        chosen = old_trim
                        plan_reason = self.prev_reason or plan.reason
                        adopted_new = False
            self.prev_path_odom = transform_path(chosen, pose_scan)
            self.prev_reason = plan_reason

            # A best-effort endpoint (snapped onto free space / closest reachable
            # cell) that the robot has reached is an arrival, not a dead-end (T8).
            if plan_reason in PARTIAL_ENDPOINTS:
                ex, ey = chosen[-1]
                if math.hypot(ex - rel[0], ey - rel[1]) <= max(0.5 * g.tol, 0.6 * resm):
                    return res(STATE_ARRIVED, (0.0, 0.0), done=True, plan_reason=plan_reason)

        # Everything below is in the CURRENT body frame.
        path_now = [_to_frame(p, rel) for p in chosen]
        remaining = path_remaining_m(path_now) + beyond
        look = lookahead_on_path(path_now, cfg.lookahead_m) or (bx, by)
        v, omega, _ld, bearing, rot_next = steer_to_body_point(look, gp, self.rotating)
        if starved and v > gp.v_min_mps:
            omega *= gp.v_min_mps / v           # same arc, slower
            v = gp.v_min_mps
        if rot_next:
            omega, self.rot_dir = latched_rotate_omega(
                bearing, self.rot_dir if self.rotating else 0, gp,
                flip_band_rad=cfg.rotate_flip_band_rad,
                min_omega=cfg.min_rotate_omega_radps)
        else:
            self.rot_dir = 0
        self.rotating = rot_next

        # Last-resort vetoes (never steer). Fix 4: before giving up on the
        # arc, try gentler commands that still follow the path.
        blocked = False
        if v > 1e-3:
            cand = self._candidates(v, omega, look, path_now, gp) if cfg.slow_retry else [(v, omega)]
            reach_cap = remaining + cfg.stop_margin_m
            pick = None
            for cv, cw in cand:
                if not swept_path_blocked(
                        work, meta, v_mps=cv, omega_radps=cw, config=foot,
                        start_pose=rel, max_reach_m=reach_cap,
                        explained_unknown=explained):
                    pick = (cv, cw)
                    break
            if pick is None:
                blocked = True
            else:
                v, omega = pick
        depth_blocked = (not blocked) and self._depth.update(
            depth, now_wall=now_wall, v_mps=v, omega_radps=omega,
            path_body=path_now)

        if self.gate is not None:
            self._record(g, gb, plan_reason, path_now, look, v, omega,
                         blocked or depth_blocked, work, meta)

        depth_report: Dict[str, Any] = {}
        if depth_blocked:
            if cfg.depth_act:
                self.prog_at = now_mono
                return res(STATE_BLOCKED, (0.0, 0.0), reason="depth_block", mode="follow",
                           depth_hits=self._depth.last_hits,
                           depth_seq=self._depth.last_seq,
                           depth_bbox_px=self._depth.last_bbox_px,
                           depth_hit_body=self._depth.last_hit_body)
            depth_report = dict(depth_would_block=True, depth_hits=self._depth.last_hits,
                                depth_seq=self._depth.last_seq,
                                depth_bbox_px=self._depth.last_bbox_px,
                                depth_hit_body=self._depth.last_hit_body)

        if blocked:
            look_bearing = math.atan2(look[1], look[0])
            resp, omega_r = swept_block_response(
                look_bearing, self.realign_time,
                thresh_rad=cfg.swept_realign_thresh_rad,
                timeout_s=cfg.swept_realign_timeout_s,
                k_omega=gp.k_omega, omega_max=gp.omega_max)
            self.prog_at = now_mono
            if resp == "realign":
                if rot_budget_hit:
                    return self._enter_terminal("rotate_timeout", gb, dist)
                self.realign_time += dt
                # Realign is its own mode: it does NOT arm the rotate-in-place
                # hysteresis (T3), so the next tick re-tests the forward arc.
                self.rotating = False
                self.rot_dir = 0
                return res(STATE_DRIVING, (0.0, omega_r), omega=omega_r, mode="realign")
            return res(STATE_BLOCKED, (0.0, 0.0), reason="swept_block", mode="follow")

        # HO-3 breakpoint: hold with motors at 0 until the inspector steps.
        if self.gate is not None:
            if self.gate.should_hold(3):
                self.prog_at = now_mono
                self.last_translate_at = now_mono
                return res(STATE_DRIVING, (0.0, 0.0), mode="held")
            self.gate.consume_continue(3)

        if v <= 1e-3:
            # Rotate in place toward the path — bounded by the rotation budget.
            if rot_budget_hit:
                return self._enter_terminal("rotate_timeout", gb, dist)
            self.prog_best = remaining
            self.prog_at = now_mono
            return res(STATE_DRIVING, (0.0, omega), omega=omega, mode="rotate",
                       path_body=path_now, plan_reason=plan_reason)

        # No-progress watchdog on the remaining PATH length (T9): a detour
        # that walks away from the goal but along the path is progress.
        if self.prog_best is None or remaining < self.prog_best - cfg.no_progress_eps_m:
            self.prog_best = remaining
            self.prog_at = now_mono
        elif adopted_new or self._path_refreshed:
            self.prog_best = remaining          # new route / refresh: new reference, same timer
        elif now_mono - self.prog_at > cfg.no_progress_timeout_s:
            return res(STATE_BLOCKED, (0.0, 0.0), reason="no_progress")

        self._path_refreshed = False
        mode = "follow"
        if path_mode:
            mode = "starved" if starved else ("rejoin" if plan_reason == "rejoin" else "follow")
        return res(STATE_DRIVING, (v, omega), v=v, omega=omega, mode=mode,
                   **depth_report,
                   path_body=path_now, plan_reason=plan_reason)

    # ── helpers ──────────────────────────────────────────────────────
    def _plan_on_path(self, g: Tier3Goal, work, meta, cm, rel: Pose, pose_scan: Pose,
                      s_rob: float, path_len: float):
        """Path mode planning, in the scan frame. Returns (chosen path,
        plan_reason "ok" | "rejoin", path length beyond the checked horizon)
        or (None, block point) when the path is obstructed with no rejoin
        point within the raster and ``max_rejoin_deviation_m``."""
        pc = self.cfg.path
        path_scan = [odom_to_body(p, pose_scan) for p in g.path]
        cum = self._path_cum
        lethal = raster_lookup(cm.lethal, meta)
        samples = resample(path_scan, cum, s_rob, min(path_len, s_rob + pc.horizon_m), pc.step_m)
        i0, n_on = first_blocked(samples, lethal)
        n_on = max(1, min(n_on, len(samples)))
        beyond = max(0.0, path_len - samples[n_on - 1][0])
        if i0 is None:
            self.rejoin_side = None
            return [(rel[0], rel[1])] + [p for _s, p in samples[:n_on]], "ok", beyond
        block_pt = samples[i0][1]
        j = rejoin_index(samples, i0, lethal, pc.rejoin_clear_m)
        if j is None:
            return None, block_pt
        # Aim past the obstruction, not at its edge: a target right where the
        # path clears forces a sharp turn back that pure pursuit cuts into the
        # obstacle's footprint (swept_block beside a post).
        k = j
        while (k + 1 < n_on and samples[k + 1][0] - samples[j][0] <= pc.rejoin_lead_m
               and not lethal(samples[k + 1][1])):
            k += 1
        j = k
        rest = [p for _s, p in samples[j + 1:n_on]]
        span = samples[max(0, i0 - 3):j + 1]
        # Side latch: an obstacle centred on the path has two near-equal
        # sides and a fresh A* each tick flips between them (turning in place
        # until the rotation budget runs out). Once a detour has picked a
        # side, later plans stay on it: the other side of the obstructed
        # stretch is closed off for the planner. Fresh every tick otherwise,
        # so planner and swept veto see the same footprint.
        detour = None
        if self.rejoin_side is not None:
            detour = self._rejoin_plan(work, meta, cm, rel, samples[j][1], path_scan, cum,
                                       close_side=-self.rejoin_side, span=span)
        if detour is None:
            detour = self._rejoin_plan(work, meta, cm, rel, samples[j][1], path_scan, cum)
        if detour is None:
            self.rejoin_side = None
            return None, block_pt
        side = _detour_side(detour, path_scan, cum)
        if side != 0:
            self.rejoin_side = side
        return detour + rest, "rejoin", beyond

    def _rejoin_plan(self, work, meta, cm, rel: Pose, target: Point2, path_scan, cum,
                     close_side: Optional[int] = None, span=None) -> Optional[List[Point2]]:
        """Local A* from the robot to ``target`` within the stray limit, or
        None. ``close_side`` (+1 left / −1 right of the path) blocks that side
        of ``span`` for this plan."""
        if close_side is not None and span:
            res_m = float(meta["resolution_m"])
            ox, oy = float(meta["origin_x_m"]), float(meta["origin_y_m"])
            nx, ny = cm.lethal.shape
            extra = np.zeros_like(cm.lethal)
            pts = [p for _s, p in span]
            for a, b in zip(pts, pts[1:]):
                tx, ty = b[0] - a[0], b[1] - a[1]
                n = math.hypot(tx, ty)
                if n < 1e-9:
                    continue
                lx, ly = -ty / n * close_side, tx / n * close_side
                for d in np.arange(0.0, self.cfg.path.max_rejoin_deviation_m + res_m, 0.5 * res_m):
                    for t in (0.0, 0.5):
                        px = a[0] + t * tx + lx * d
                        py = a[1] + t * ty + ly * d
                        ci, cj = int((px - ox) // res_m), int((py - oy) // res_m)
                        if 0 <= ci < nx and 0 <= cj < ny:
                            extra[ci, cj] = True
            near = (np.hypot(ox + (np.arange(nx)[:, None] + 0.5) * res_m - rel[0],
                             oy + (np.arange(ny)[None, :] + 0.5) * res_m - rel[1])
                    <= self.cfg.seed_clear_radius_m)
            extra &= ~near                       # never wall the robot in
            cm = replace(cm, lethal=cm.lethal | extra,
                         cost=np.where(extra, np.inf, cm.cost).astype(cm.cost.dtype))
        plan = plan_local(work, meta, target, self.cfg.plan, costmap=cm,
                          start_body=(rel[0], rel[1]))
        if not plan.ok or plan.reason != "ok":
            return None
        if max_deviation(plan.path_body, path_scan, cum) > self.cfg.path.max_rejoin_deviation_m:
            return None
        return list(plan.path_body)

    def _latch_endpoint(self, plan, work, meta, cm, goal_s, rel, pose_scan):
        """Hold a best-effort endpoint fixed in odom (fix 3 "snap once").

        When the goal is lethal/unreachable, the best reachable cells form a
        ring or a line of near-ties that re-quantizes differently every tick
        as the robot turns, flipping the endpoint (and the path's side). The
        first snapped/frontier endpoint is latched in odom and re-used while
        it stays reachable; it is replaced only when a fresh endpoint is
        meaningfully closer to the goal (> 2 cells) or the latch becomes
        unreachable. A plan that reaches the goal itself clears the latch."""
        if not plan.ok:
            if plan.reason != "at_frontier":
                self.endpoint_odom = None
            self.goal_ok_ticks = 0
            return plan
        self.goal_ok_ticks = self.goal_ok_ticks + 1 if plan.reason == "ok" else 0
        res_m = float(meta["resolution_m"])
        new_end = plan.goal_body_snapped or plan.path_body[-1]
        new_d = math.hypot(new_end[0] - goal_s[0], new_end[1] - goal_s[1])
        if plan.reason == "ok" and self.endpoint_odom is None:
            return plan
        if self.endpoint_odom is not None:
            lx, ly = odom_to_body(self.endpoint_odom, pose_scan)
            old_d = math.hypot(lx - goal_s[0], ly - goal_s[1])
            p2 = plan_local(work, meta, (lx, ly), self.cfg.plan, costmap=cm,
                            start_body=(rel[0], rel[1]))
            kept = None
            if p2.ok:
                e2 = p2.goal_body_snapped or p2.path_body[-1]
                if math.hypot(e2[0] - lx, e2[1] - ly) <= 1.5 * res_m:
                    kept = replace(p2, reason=("snapped" if plan.reason == "ok"
                                               else plan.reason))
            elif p2.reason == "at_frontier":
                kept = replace(p2, reason="at_frontier")
            if kept is not None:
                # The goal cell flickers lethal/non-lethal as the body-aligned
                # grid re-quantizes (inflation edge): an "ok" plan clears the
                # latch only after LATCH_OK_TICKS consecutive ok plans.
                if plan.reason == "ok" and self.goal_ok_ticks < LATCH_OK_TICKS:
                    return kept
                # Switch only to an endpoint that is clearly closer to the
                # goal AND not bought with a long detour (a far gap in a
                # dotted wall is not worth 20 cm of goal distance locally) —
                # or when the LATCHED path has become a real detour: longer
                # than LATCH_DETOUR_RATIO × the straight distance to the goal
                # and longer than the fresh path (one-sided, the old code kept
                # a 6.6 m loop over a 0.6 m path, 10-31). Ratio to the goal
                # distance, not to the fresh path: around a small post the
                # two sides' paths differ by up to ~1 m and alternate as the
                # grid re-quantizes — switching on that flips the rotation.
                gain = old_d - new_d
                len_new = path_remaining_m(plan.path_body, (rel[0], rel[1]))
                len_old = (path_remaining_m(p2.path_body, (rel[0], rel[1]))
                           if p2.ok else 0.0)
                d_goal = math.hypot(goal_s[0] - rel[0], goal_s[1] - rel[1])
                old_is_detour = (len_old > LATCH_DETOUR_RATIO * max(d_goal, 0.2) + 0.3
                                 and len_old > len_new + 0.3)
                if not old_is_detour and (
                        gain <= 1.0 * res_m or len_new > len_old + 2.0 * gain + 0.3):
                    return kept
        # The goal itself (reason ok) or a clearly better endpoint wins.
        self.endpoint_odom = None if plan.reason == "ok" else body_to_odom(new_end, pose_scan)
        return plan

    def _candidates(self, v: float, omega: float, look: Point2,
                    path_now: List[Point2], gp: DriveParams) -> List[Tuple[float, float]]:
        """(v, ω) to try in order: the nominal command, the pure-pursuit arc
        through the lookahead (stays on the path), the same at v_min, and a
        half-length lookahead at v_min (hugs a bending path)."""
        out = [(v, omega)]

        def pp(pt: Optional[Point2]) -> Optional[float]:
            if pt is None:
                return None
            l2 = pt[0] * pt[0] + pt[1] * pt[1]
            if l2 < 1e-6 or pt[0] <= 0.0:
                return None
            return 2.0 * pt[1] / l2

        def clip(w: float) -> float:
            return max(-gp.omega_max, min(gp.omega_max, w))

        k1 = pp(look)
        vmin = min(v, gp.v_min_mps)
        if k1 is not None:
            out.append((v, clip(v * k1)))
            if vmin < v - 1e-6:
                out.append((vmin, clip(vmin * k1)))
        elif vmin < v - 1e-6:
            out.append((vmin, clip(omega * vmin / v)))
        k2 = pp(lookahead_on_path(path_now, 0.5 * self.cfg.lookahead_m))
        if k2 is not None:
            out.append((vmin, clip(vmin * k2)))
        return out

    def _enter_terminal(self, reason: str, gb: Point2, dist: float) -> TickResult:
        self.terminal = reason
        self.terminal_ticks = 0
        return self._terminal(gb, dist)

    def _terminal(self, gb: Point2, dist: float) -> TickResult:
        # Stop, then go quiet on cmd_vel so a terminal block never fights
        # operator teleop with a 10 Hz stream of zeros (X3). The goal stays
        # held (status keeps reporting BLOCKED) until superseded / canceled.
        cmd = (0.0, 0.0) if self.terminal_ticks < self.cfg.terminal_zero_ticks else None
        self.terminal_ticks += 1
        return TickResult(state=STATE_BLOCKED, cmd=cmd, reason=self.terminal,
                          goal_body=gb, dist=dist, terminal=True)

    def _record(self, g: Tier3Goal, gb, plan_reason, path_now, look, v, omega,
                swept_blocked, work, meta) -> None:
        # HO-3 Tier-3 → motors. The costmap rides along only while BP3 is
        # armed (64×64 cells of JSON at 10 Hz is real jitter).
        from body.lib import schemas   # local: keeps this module zenoh-free
        try:
            armed = bool(self.gate.is_armed(3))
            self.gate.record(3, schemas.handoff_t3(
                cmd_id=g.cmd_id, goal_body=gb, plan_reason=plan_reason,
                path_body=path_now, lookahead=look, v_mps=v, omega_radps=omega,
                swept_blocked=swept_blocked,
                grid_rows=(work.tolist() if armed else None), meta=meta))
        except Exception as e:  # noqa: BLE001 — telemetry must not stop the drive
            print(f"local_drive: HO-3 record failed: {e!r}", flush=True)


def _endpoint_class(reason: Optional[str]) -> Optional[str]:
    # ok / snapped / frontier flip with cell quantization as the grid rotates
    # under a goal near the lethal edge; the endpoint-distance test in the
    # hysteresis decides whether an old path is still as good.
    return "endpoint" if reason in ("ok", "snapped", "frontier") else reason


def _detour_side(detour: List[Point2], path: List[Point2], cum) -> int:
    """+1 if the detour passes left of the path (its direction of travel),
    −1 right, 0 if it never leaves the path by more than 5 cm."""
    acc = 0.0
    for p in detour:
        sp, lat = project(path, cum, p)
        if lat < 0.05:
            continue
        a = point_at(path, cum, max(0.0, sp - 0.02))
        b = point_at(path, cum, sp + 0.02)
        q = point_at(path, cum, sp)
        acc += (b[0] - a[0]) * (p[1] - q[1]) - (b[1] - a[1]) * (p[0] - q[0])
    return 0 if acc == 0.0 else (1 if acc > 0 else -1)


def _to_frame(p: Point2, rel: Pose) -> Point2:
    """Point in the scan frame → body frame of the robot at ``rel``."""
    return odom_to_body(p, rel)

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


@dataclass(frozen=True)
class Tier3Goal:
    cmd_id: int
    x: float                               # odom frame
    y: float
    tol: float
    final_heading: Optional[float] = None
    v_max: Optional[float] = None
    omega_max: Optional[float] = None


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

    def start_goal(self, goal: Tier3Goal, now_mono: float, pose_now: Pose) -> None:
        self.goal = goal
        self._reset_state(now_mono, pose_now)

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

        def res(state: str, cmd, **kw) -> TickResult:
            return TickResult(state=state, cmd=cmd, goal_body=gb, dist=dist, **kw)

        if self.terminal is not None:
            return self._terminal(gb, dist)

        deadline_hit = cfg.goal_deadline_s > 0 and now_mono - self.goal_started_at > cfg.goal_deadline_s
        rot_budget_hit = (cfg.rotate_timeout_s > 0
                          and now_mono - self.last_translate_at > cfg.rotate_timeout_s)

        # Arrival (+ optional final-heading rotate). The deadline and the
        # rotation budget bound the final-heading turn too (T13).
        if dist <= g.tol:
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
        remaining = path_remaining_m(path_now)
        look = lookahead_on_path(path_now, cfg.lookahead_m) or (bx, by)
        v, omega, _ld, bearing, rot_next = steer_to_body_point(look, gp, self.rotating)
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
        elif adopted_new:
            self.prog_best = remaining          # new route: new reference, same timer
        elif now_mono - self.prog_at > cfg.no_progress_timeout_s:
            return res(STATE_BLOCKED, (0.0, 0.0), reason="no_progress")

        return res(STATE_DRIVING, (v, omega), v=v, omega=omega, mode="follow",
                   **depth_report,
                   path_body=path_now, plan_reason=plan_reason)

    # ── helpers ──────────────────────────────────────────────────────
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


def _to_frame(p: Point2, rel: Pose) -> Point2:
    """Point in the scan frame → body frame of the robot at ``rel``."""
    return odom_to_body(p, rel)

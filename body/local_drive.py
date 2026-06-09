"""Tier-3 reactive drive (Pi-side).

Drives the robot to one observable subgoal using the live body-frame
``local_map``, with no dependence on the global map or PF pose. Subscribes
``body/drive/goto`` + ``body/odom`` + ``body/map/local_2p5d``; publishes
``body/cmd_vel`` (only while a goal is active — see cmd_vel arbitration in
docs/drive_tier3_spec.md) and ``body/drive/status`` every tick.

v1 (Stage A): straight-line pure-pursuit toward the goal, gated by the
swept-footprint check. No dynamic avoidance yet — a swept block stops and
reports BLOCKED for the sender to re-pick. The watchdog/e-stop/motor
timeout remain supreme; this process is just another cmd_vel producer.
"""
from __future__ import annotations

import math
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

from body.lib import schemas, zenoh_helpers
from body.lib.buildinfo import git_sha
from body.lib.drive_safety import FootprintConfig, swept_path_blocked
from body.lib.handoff_gate import HandoffGate
from body.lib.scan_raster import ScanRasterConfig, rasterize_scan
from body.lib.local_costmap import LocalCostmapConfig
from body.lib.local_planner import LocalPlanConfig, lookahead_on_path, plan_local
from body.lib.local_drive_core import (
    STATE_ARRIVED, STATE_BLOCKED, STATE_DRIVING, STATE_FAULT, STATE_IDLE,
    DriveParams, odom_to_body, rotate_to_heading, steer_to_body_point,
    swept_block_response, wrap_pi,
)


def _idle_until_signal(reason: str) -> None:
    print(f"local_drive: {reason}; idling so launcher does not respawn.", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda _s, _f: stop.set())
    signal.signal(signal.SIGINT, lambda _s, _f: stop.set())
    while not stop.is_set():
        time.sleep(1.0)
    sys.exit(0)


def _with_vmax(params: DriveParams, v_max: float) -> DriveParams:
    return DriveParams(
        v_max=v_max, omega_max=params.omega_max, v_min_mps=params.v_min_mps,
        arrival_tol_m=params.arrival_tol_m,
        rotate_in_place_thresh_rad=params.rotate_in_place_thresh_rad,
        rotate_exit_thresh_rad=params.rotate_exit_thresh_rad,
        k_omega=params.k_omega, slowdown_distance_m=params.slowdown_distance_m,
        heading_tol_rad=params.heading_tol_rad,
    )


def _sleep_to(next_tick: float, period: float) -> float:
    next_tick += period
    sleep_for = next_tick - time.monotonic()
    if sleep_for > 0:
        time.sleep(sleep_for)
        return next_tick
    return time.monotonic()


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    cfg = body_cfg.get("local_drive", {})
    if not bool(cfg.get("enabled", False)):
        _idle_until_signal("disabled (local_drive.enabled=false)")
        return

    params = DriveParams(
        v_max=float(cfg.get("v_max", 0.18)),
        omega_max=float(cfg.get("omega_max", 0.6)),
        v_min_mps=float(cfg.get("v_min_mps", 0.08)),
        arrival_tol_m=float(cfg.get("arrival_tol_m", 0.12)),
        rotate_in_place_thresh_rad=float(cfg.get("rotate_in_place_thresh_rad", 0.61)),
        rotate_exit_thresh_rad=float(cfg.get("rotate_exit_thresh_rad", 0.26)),
        k_omega=float(cfg.get("k_omega", 1.5)),
        slowdown_distance_m=float(cfg.get("slowdown_distance_m", 0.4)),
    )
    # Tier-3 obstacle field = the live lidar scan rasterized each tick (see
    # body/lib/scan_raster.py). Lidar mount/range come from the existing
    # lidar / local_map config so there's one source of truth.
    lidar_cfg = body_cfg.get("lidar", {})
    lm_cfg = body_cfg.get("local_map", {})
    scan_cfg = cfg.get("scan", {})
    raster = ScanRasterConfig(
        resolution_m=float(scan_cfg.get("resolution_m", 0.08)),
        half_extent_m=float(scan_cfg.get("half_extent_m", 2.5)),
        lidar_x_m=float(lm_cfg.get("lidar_x_body_m", 0.0)),
        lidar_y_m=float(lm_cfg.get("lidar_y_body_m", 0.0)),
        lidar_yaw_rad=float(lm_cfg.get("lidar_yaw_rad", 0.0)),
        range_min_m=float(lidar_cfg.get("range_min_m", 0.05)),
        range_max_m=float(scan_cfg.get("range_max_m", 8.0)),
        max_clear_range_m=float(
            scan_cfg.get("max_clear_range_m", lm_cfg.get("lidar_max_clear_range_m", 6.0))
        ),
        clear_buffer_cells=float(scan_cfg.get("clear_buffer_cells", 2.0)),
    )
    # Local A* planner config (the single local-routing authority). The
    # costmap footprint is THE footprint model; keep the swept-veto FootprintConfig
    # ≤ this so they agree (see drive_safety).
    lp = cfg.get("local_planner", {})
    local_plan_cfg = LocalPlanConfig(
        costmap=LocalCostmapConfig(
            footprint_radius_m=float(lp.get("footprint_radius_m", 0.11)),
            safety_margin_m=float(lp.get("safety_margin_m", 0.08)),
            inflation_decay_m=float(lp.get("inflation_decay_m", 0.20)),
            unknown_cost=float(lp.get("unknown_cost", 25.0)),
            unknown_is_lethal=bool(lp.get("unknown_is_lethal", False)),
        ),
        min_clearance_cells=int(lp.get("min_clearance_cells", 0)),
        cost_per_unit=float(lp.get("cost_per_unit", 0.10)),
        max_expansions=int(lp.get("max_expansions", 50000)),
    )
    lookahead_m = float(lp.get("lookahead_m", 0.4))

    # Last-resort swept veto (drive_safety). It must NOT be stricter than the
    # A* costmap or it rejects A*'s own feasible paths (the doorway/near-wall
    # false swept_block). Derive its footprint from the A* footprint so they
    # can't drift: the swept check adds ½ cell internally, so subtract that to
    # keep its effective radius ≤ the A* lethal radius. The veto then only fires
    # on a genuine corner-cut off the path or a new obstacle, never on an
    # A*-blessed path.
    half_cell = 0.5 * float(scan_cfg.get("resolution_m", 0.08))
    veto_foot = max(0.02, local_plan_cfg.costmap.footprint_radius_m - half_cell)
    foot = FootprintConfig(
        footprint_radius_m=veto_foot,
        preview_distance_m=float(cfg.get("preview_distance_m", 0.35)),
        preview_min_distance_m=float(cfg.get("preview_min_distance_m", 0.15)),
        preview_time_s=float(cfg.get("preview_time_s", 1.5)),
        forward_cone_rad=math.radians(float(cfg.get("forward_cone_deg", 60.0))),
        hard_radius_m=min(float(cfg.get("hard_radius_m", 0.05)), veto_foot),
        block_on_unknown=bool(cfg.get("block_on_unknown", True)),
        unknown_block_range_m=float(cfg.get("unknown_block_range_m", 0.25)),
        min_observed_cells=int(cfg.get("min_observed_cells", 3)),
    )
    control_hz = float(cfg.get("control_hz", 10.0))
    period = 1.0 / max(1.0, control_hz)
    cmd_timeout_ms = max(500, int(3.0 * period * 1000.0))
    odom_stale_s = float(cfg.get("odom_stale_s", 0.5))
    scan_stale_s = float(scan_cfg.get("scan_stale_s", 0.5))
    no_progress_timeout_s = float(cfg.get("no_progress_timeout_s", 4.0))
    no_progress_eps_m = float(cfg.get("no_progress_eps_m", 0.03))
    # Hard per-goal deadline. The no-progress watchdog only runs while
    # translating, so a rotate/drive dither (or A* homotopy flip-flop) could
    # otherwise keep a goal alive forever. Tier-2 caps goals at ~1.5 m, so a
    # healthy leg finishes in well under this. 0 disables.
    goal_deadline_s = float(cfg.get("goal_deadline_s", 30.0))
    # On a swept-block, re-aim in place (swept-free) toward the path's lookahead
    # rather than stopping, until the lookahead is within this bearing or we've
    # been re-aiming longer than the timeout (then it's a genuine dead-end).
    swept_realign_thresh_rad = float(cfg.get("swept_realign_thresh_rad", 0.10))
    swept_realign_timeout_s = float(cfg.get("swept_realign_timeout_s", 2.0))

    # Shared with zenoh callback threads. Callbacks only ever assign these;
    # all per-goal bookkeeping lives in the main loop, keyed by cmd_id.
    lock = threading.Lock()
    last_odom: Optional[Dict[str, Any]] = None
    last_scan: Optional[Dict[str, Any]] = None
    goal: Optional[Dict[str, Any]] = None
    cur_cmd_id = 0

    session = zenoh_helpers.open_session(body_cfg)
    stop = threading.Event()

    def on_odom(_k: str, msg: Dict[str, Any]) -> None:
        nonlocal last_odom
        with lock:
            last_odom = msg

    def on_scan(_k: str, msg: Dict[str, Any]) -> None:
        nonlocal last_scan
        with lock:
            last_scan = msg

    def on_goto(_k: str, msg: Dict[str, Any]) -> None:
        nonlocal goal, cur_cmd_id
        kind = str(msg.get("kind", "goto"))
        with lock:
            if kind in ("cancel", "stop"):
                # Advance the supersede watermark to the cancel's own id so a
                # delayed/duplicate goto with an older id can't re-arm the
                # goal the operator just revoked.
                cur_cmd_id = max(cur_cmd_id, int(msg.get("cmd_id", 0)))
                goal = None
                return
            cmd_id = int(msg.get("cmd_id", 0))
            if cmd_id < cur_cmd_id:
                return  # superseded by a newer command
            if str(msg.get("frame", "odom")) != "odom":
                print(f"local_drive: rejecting goto frame={msg.get('frame')!r}", flush=True)
                return
            goal = dict(msg)
            cur_cmd_id = cmd_id

    zenoh_helpers.declare_subscriber_json(session, "body/odom", on_odom)
    zenoh_helpers.declare_subscriber_json(session, "body/lidar/scan", on_scan)
    zenoh_helpers.declare_subscriber_json(session, "body/drive/goto", on_goto)

    # HO-3 handoff gate (Tier-3 → motors): records each command + obeys the
    # standalone Handoff Inspector's arm/continue. tier 3 only on this side.
    gate = HandoffGate(session, tiers=(3,))

    signal.signal(signal.SIGTERM, lambda _s, _f: stop.set())
    signal.signal(signal.SIGINT, lambda _s, _f: stop.set())

    def publish_cmd(v: float, omega: float) -> None:
        zenoh_helpers.publish_json(
            session, "body/cmd_vel",
            schemas.cmd_vel(linear=v, angular=omega, timeout_ms=cmd_timeout_ms),
        )

    build = git_sha()

    def publish_status(state: str, *, cmd_id: int, goal_body=None,
                       dist=0.0, v=0.0, omega=0.0, reason=None, mode=None,
                       path_body=None, plan_reason=None) -> None:
        zenoh_helpers.publish_json(
            session, "body/drive/status",
            schemas.drive_status(
                cmd_id=cmd_id, state=state, goal_body_xy=goal_body,
                dist_remaining_m=dist, v_mps=v, omega_radps=omega,
                blocked_reason=reason, mode=mode, path_body_xy=path_body,
                plan_reason=plan_reason, build=build,
            ),
        )

    # Per-goal bookkeeping (main loop only).
    tracked_cmd_id = -1
    rotating = False  # rotate-in-place hysteresis state (per goal)
    best_dist: Optional[float] = None
    best_dist_at = 0.0
    final_aligned = False
    realign_since: Optional[float] = None   # start of the current swept-block re-aim
    goal_started_at = 0.0                   # for the per-goal deadline
    was_active = False                      # had a goal last tick (stop-on-cancel)

    print(f"local_drive: up; control_hz={control_hz} v_max={params.v_max}", flush=True)
    next_tick = time.monotonic()
    while not stop.is_set():
        now_wall = time.time()
        now_mono = time.monotonic()
        with lock:
            g = dict(goal) if goal is not None else None
            odom = dict(last_odom) if last_odom is not None else None
            scan = last_scan
            cmd_id = cur_cmd_id

        if g is None:
            if was_active:
                # Cancel/stop while driving: command zero NOW instead of
                # letting the motors coast on the last cmd_vel until its
                # 500 ms timeout (~9 cm at v_max).
                publish_cmd(0.0, 0.0)
                was_active = False
            publish_status(STATE_IDLE, cmd_id=cmd_id)
            next_tick = _sleep_to(next_tick, period)
            continue
        was_active = True

        # New goal → reset bookkeeping.
        if cmd_id != tracked_cmd_id:
            tracked_cmd_id = cmd_id
            best_dist = None
            best_dist_at = now_mono
            final_aligned = False
            rotating = False
            realign_since = None
            goal_started_at = now_mono

        gx, gy = float(g["x_m"]), float(g["y_m"])
        tol = float(g.get("arrival_tol_m", params.arrival_tol_m))
        gp = params if "v_max" not in g else _with_vmax(params, float(g["v_max"]))

        # Odom freshness — can't transform the goal without a live pose.
        if odom is None or (now_wall - float(odom.get("ts", 0.0))) > odom_stale_s:
            publish_cmd(0.0, 0.0)
            publish_status(STATE_FAULT, cmd_id=cmd_id, reason="odom_stale")
            next_tick = _sleep_to(next_tick, period)
            continue

        pose = (float(odom["x"]), float(odom["y"]), float(odom["theta"]))
        bx, by = odom_to_body((gx, gy), pose)
        dist = (bx * bx + by * by) ** 0.5

        # Arrival (+ optional final-heading rotate).
        if dist <= tol:
            seek_odom = None
            fh = g.get("final_heading_rad")
            if fh is not None and not final_aligned:
                omega, aligned = rotate_to_heading(pose[2], float(fh), gp)
                if aligned:
                    final_aligned = True
                    publish_cmd(0.0, 0.0)
                    publish_status(STATE_ARRIVED, cmd_id=cmd_id, goal_body=(bx, by), dist=dist)
                    with lock:
                        if cur_cmd_id == cmd_id:
                            goal = None
                else:
                    publish_cmd(0.0, omega)
                    publish_status(STATE_DRIVING, cmd_id=cmd_id, goal_body=(bx, by),
                                   dist=dist, omega=omega)
                next_tick = _sleep_to(next_tick, period)
                continue
            publish_cmd(0.0, 0.0)
            publish_status(STATE_ARRIVED, cmd_id=cmd_id, goal_body=(bx, by), dist=dist)
            with lock:
                if cur_cmd_id == cmd_id:
                    goal = None
            next_tick = _sleep_to(next_tick, period)
            continue

        # Hard deadline: arrival/no-progress didn't end this goal in time
        # (e.g. a rotate/drive dither that never translates). Stop and report;
        # the goal stays active so the status doesn't decay to IDLE (which the
        # desktop reads as success) — a superseding goto or cancel clears it.
        if goal_deadline_s > 0 and now_mono - goal_started_at > goal_deadline_s:
            publish_cmd(0.0, 0.0)
            publish_status(STATE_BLOCKED, cmd_id=cmd_id, goal_body=(bx, by),
                           dist=dist, reason="deadline")
            next_tick = _sleep_to(next_tick, period)
            continue

        # Live obstacle field (rasterized scan) — the substrate for steering.
        if scan is None or (now_wall - float(scan.get("ts", 0.0))) > scan_stale_s:
            publish_cmd(0.0, 0.0)
            publish_status(STATE_BLOCKED, cmd_id=cmd_id, goal_body=(bx, by),
                           dist=dist, reason="no_scan")
            next_tick = _sleep_to(next_tick, period)
            continue
        grid, meta = rasterize_scan(
            scan.get("ranges"), float(scan.get("angle_min", 0.0)),
            float(scan.get("angle_increment", 0.0)), raster,
        )

        # Local A* over the footprint-inflated scan grid: the single authority
        # for local feasibility/routing. Returns a body-frame path the body can
        # follow the whole way, or an honest "no path".
        plan = plan_local(grid, meta, (bx, by), local_plan_cfg)
        if not plan.ok:
            publish_cmd(0.0, 0.0)
            publish_status(STATE_BLOCKED, cmd_id=cmd_id, goal_body=(bx, by),
                           dist=dist, reason=plan.reason, mode="plan")
            best_dist = dist            # stopped → reset the no-progress window
            best_dist_at = now_mono
            next_tick = _sleep_to(next_tick, period)
            continue

        # Pure-pursuit follow: steer toward a lookahead point on the path.
        look = lookahead_on_path(plan.path_body, lookahead_m) or (bx, by)
        v, omega, _ld, _lb, rotating = steer_to_body_point(look, gp, rotating)

        # Last-resort safety veto — A* won't route through lethal, but a
        # dynamic/new obstacle can appear in the followed arc between replans.
        # Strictly subordinate: it only stops, never steers.
        blocked = swept_path_blocked(grid, meta, v_mps=v, omega_radps=omega, config=foot)

        # HO-3 Tier-3 → motors: record what we're about to command. Attach the
        # costmap only when BP3 is armed — serializing 64×64 cells to JSON at
        # the control rate is real jitter; the inspector synthesizes an
        # all-unknown grid for lean records.
        gate.record(3, schemas.handoff_t3(
            cmd_id=cmd_id, goal_body=(bx, by), plan_reason=plan.reason,
            path_body=plan.path_body, lookahead=look, v_mps=v, omega_radps=omega,
            swept_blocked=blocked,
            grid_rows=(grid.tolist() if gate.is_armed(3) else None),
            meta=meta))

        if blocked:
            # The forward arc clips a wall. A pure rotation is swept-free (the
            # footprint is a circle), so re-aim in place toward the lookahead to
            # straighten the approach instead of giving up. Only when already
            # aligned (or re-aiming too long) is it a genuine dead-end.
            if realign_since is None:
                realign_since = now_mono
            look_bearing = math.atan2(look[1], look[0])
            resp, omega_r = swept_block_response(
                look_bearing, now_mono - realign_since,
                thresh_rad=swept_realign_thresh_rad,
                timeout_s=swept_realign_timeout_s,
                k_omega=gp.k_omega, omega_max=gp.omega_max)
            if resp == "realign":
                publish_cmd(0.0, omega_r)
                publish_status(STATE_DRIVING, cmd_id=cmd_id, goal_body=(bx, by),
                               dist=dist, v=0.0, omega=omega_r, mode="realign")
                rotating = True
                next_tick = _sleep_to(next_tick, period)
                continue
            publish_cmd(0.0, 0.0)
            publish_status(STATE_BLOCKED, cmd_id=cmd_id, goal_body=(bx, by),
                           dist=dist, reason="swept_block", mode="follow")
            best_dist = dist
            best_dist_at = now_mono
            next_tick = _sleep_to(next_tick, period)
            continue

        # HO-3 breakpoint: hold here with motors at 0 (heartbeat + cmd_vel keep
        # flowing so the watchdog stays happy and the link alive) until the
        # inspector single-steps. Reset the no-progress window so holding here
        # never trips it.
        if gate.should_hold(3):
            publish_cmd(0.0, 0.0)
            publish_status(STATE_DRIVING, cmd_id=cmd_id, goal_body=(bx, by),
                           dist=dist, v=0.0, omega=0.0, mode="held")
            best_dist = dist
            best_dist_at = now_mono
            next_tick = _sleep_to(next_tick, period)
            continue
        gate.consume_continue(3)

        # No-progress watchdog — only while actually translating.
        if v < 1e-3:
            best_dist = dist
            best_dist_at = now_mono
        elif best_dist is None or dist < best_dist - no_progress_eps_m:
            best_dist = dist
            best_dist_at = now_mono
        elif now_mono - best_dist_at > no_progress_timeout_s:
            publish_cmd(0.0, 0.0)
            publish_status(STATE_BLOCKED, cmd_id=cmd_id, goal_body=(bx, by),
                           dist=dist, reason="no_progress")
            next_tick = _sleep_to(next_tick, period)
            continue

        realign_since = None        # real forward motion → reset the re-aim window
        publish_cmd(v, omega)
        publish_status(STATE_DRIVING, cmd_id=cmd_id, goal_body=(bx, by),
                       dist=dist, v=v, omega=omega, mode="follow",
                       path_body=plan.path_body, plan_reason=plan.reason)
        next_tick = _sleep_to(next_tick, period)

    publish_cmd(0.0, 0.0)
    session.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

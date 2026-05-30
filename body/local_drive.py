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
from body.lib.drive_safety import FootprintConfig
from body.lib.scan_raster import ScanRasterConfig, rasterize_scan
from body.lib.local_drive_core import (
    STATE_ARRIVED, STATE_BLOCKED, STATE_DRIVING, STATE_FAULT, STATE_IDLE,
    DriveParams, LocalPlanConfig, odom_to_body, plan_drive, rotate_to_heading,
    wrap_pi,
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
        k_omega=float(cfg.get("k_omega", 1.5)),
        slowdown_distance_m=float(cfg.get("slowdown_distance_m", 0.4)),
    )
    foot = FootprintConfig(
        footprint_radius_m=float(cfg.get("footprint_radius_m", 0.22)),
        preview_distance_m=float(cfg.get("preview_distance_m", 0.35)),
        preview_min_distance_m=float(cfg.get("preview_min_distance_m", 0.15)),
        preview_time_s=float(cfg.get("preview_time_s", 1.5)),
        forward_cone_rad=math.radians(float(cfg.get("forward_cone_deg", 60.0))),
        hard_radius_m=float(cfg.get("hard_radius_m", 0.07)),
        block_on_unknown=bool(cfg.get("block_on_unknown", True)),
        unknown_block_range_m=float(cfg.get("unknown_block_range_m", 0.25)),
        min_observed_cells=int(cfg.get("min_observed_cells", 3)),
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
    lp = cfg.get("local_plan", {})
    lpcfg = LocalPlanConfig(
        center_range_m=float(lp.get("center_range_m", 0.8)),
        center_back_margin_m=float(lp.get("center_back_margin_m", 0.2)),
        center_target_clear_m=float(lp.get("center_target_clear_m", 0.3)),
        k_center=float(lp.get("k_center", 1.5)),
        gov_full_clear_m=float(lp.get("gov_full_clear_m", 0.7)),
        gov_min_clear_m=float(lp.get("gov_min_clear_m", 0.3)),
        gov_cone_rad=math.radians(float(lp.get("gov_cone_deg", 20.0))),
        fan_max_rad=math.radians(float(lp.get("fan_max_deg", 50.0))),
        fan_step_rad=math.radians(float(lp.get("fan_step_deg", 12.0))),
        nudge_v_floor=float(lp.get("nudge_v_floor", 0.4)),
        gap_scan_range_m=float(lp.get("gap_scan_range_m", 1.2)),
        gap_min_m=float(lp.get("gap_min_m", 0.6)),
        gap_step_rad=math.radians(float(lp.get("gap_step_deg", 12.0))),
        gap_max_rad=math.radians(float(lp.get("gap_max_deg", 110.0))),
    )
    seek_facing_tol = math.radians(float(lp.get("seek_facing_tol_deg", 10.0)))
    seek_timeout_s = float(lp.get("seek_timeout_s", 6.0))
    control_hz = float(cfg.get("control_hz", 10.0))
    period = 1.0 / max(1.0, control_hz)
    cmd_timeout_ms = max(500, int(3.0 * period * 1000.0))
    odom_stale_s = float(cfg.get("odom_stale_s", 0.5))
    scan_stale_s = float(scan_cfg.get("scan_stale_s", 0.5))
    no_progress_timeout_s = float(cfg.get("no_progress_timeout_s", 4.0))
    no_progress_eps_m = float(cfg.get("no_progress_eps_m", 0.03))

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

    signal.signal(signal.SIGTERM, lambda _s, _f: stop.set())
    signal.signal(signal.SIGINT, lambda _s, _f: stop.set())

    def publish_cmd(v: float, omega: float) -> None:
        zenoh_helpers.publish_json(
            session, "body/cmd_vel",
            schemas.cmd_vel(linear=v, angular=omega, timeout_ms=cmd_timeout_ms),
        )

    def publish_status(state: str, *, cmd_id: int, goal_body=None,
                       dist=0.0, v=0.0, omega=0.0, reason=None, mode=None) -> None:
        zenoh_helpers.publish_json(
            session, "body/drive/status",
            schemas.drive_status(
                cmd_id=cmd_id, state=state, goal_body_xy=goal_body,
                dist_remaining_m=dist, v_mps=v, omega_radps=omega,
                blocked_reason=reason, mode=mode,
            ),
        )

    # Per-goal bookkeeping (main loop only).
    tracked_cmd_id = -1
    best_dist: Optional[float] = None
    best_dist_at = 0.0
    final_aligned = False
    # Gap-seek commitment: when latched onto an out-of-fan corridor, the
    # robot rotates to face this odom heading (ignoring goal pull) until
    # aligned, so it doesn't flip-flop back toward the goal mid-turn.
    seek_odom: Optional[float] = None
    seek_t0 = 0.0

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
            publish_status(STATE_IDLE, cmd_id=cmd_id)
            next_tick = _sleep_to(next_tick, period)
            continue

        # New goal → reset bookkeeping.
        if cmd_id != tracked_cmd_id:
            tracked_cmd_id = cmd_id
            best_dist = None
            best_dist_at = now_mono
            final_aligned = False
            seek_odom = None

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

        # Seek commitment: while latched onto an out-of-fan corridor, rotate
        # to face it (ignoring goal pull) until aligned or timed out, so the
        # robot doesn't flip-flop back toward the goal mid-turn.
        if seek_odom is not None:
            if now_mono - seek_t0 > seek_timeout_s:
                seek_odom = None
                publish_cmd(0.0, 0.0)
                publish_status(STATE_BLOCKED, cmd_id=cmd_id, goal_body=(bx, by),
                               dist=dist, reason="swept_block", mode="seek")
                next_tick = _sleep_to(next_tick, period)
                continue
            err = wrap_pi(seek_odom - pose[2])
            if abs(err) > seek_facing_tol:
                om = max(-gp.omega_max, min(gp.omega_max, gp.k_omega * err))
                publish_cmd(0.0, om)
                publish_status(STATE_DRIVING, cmd_id=cmd_id, goal_body=(bx, by),
                               dist=dist, omega=om, mode="seek")
                best_dist = dist            # rotating to face the gap, not stuck
                best_dist_at = now_mono
                next_tick = _sleep_to(next_tick, period)
                continue
            seek_odom = None                # aligned → drive up the corridor

        # Proactive local steering: directional swept gate (#3), corridor
        # centering, reactive nudge, and gap-seeking.
        v, omega, mode, seek_target = plan_drive(grid, meta, (bx, by), gp, foot, lpcfg)

        # No-progress watchdog — only while actually translating. Rotating
        # in place (large bearing) or a blocked stop legitimately holds
        # distance, so reset the timer whenever we're not translating; a
        # fresh window starts once the robot begins to drive.
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

        if mode == "seek":
            # Latch the out-of-fan corridor as an odom heading and commit.
            seek_odom = wrap_pi(pose[2] + seek_target)
            seek_t0 = now_mono
            om = max(-gp.omega_max, min(gp.omega_max, gp.k_omega * seek_target))
            publish_cmd(0.0, om)
            publish_status(STATE_DRIVING, cmd_id=cmd_id, goal_body=(bx, by),
                           dist=dist, omega=om, mode="seek")
            next_tick = _sleep_to(next_tick, period)
            continue

        if mode == "blocked":
            publish_cmd(0.0, 0.0)
            publish_status(STATE_BLOCKED, cmd_id=cmd_id, goal_body=(bx, by),
                           dist=dist, reason="swept_block", mode="blocked")
            next_tick = _sleep_to(next_tick, period)
            continue

        publish_cmd(v, omega)
        publish_status(STATE_DRIVING, cmd_id=cmd_id, goal_body=(bx, by),
                       dist=dist, v=v, omega=omega, mode=mode)
        next_tick = _sleep_to(next_tick, period)

    publish_cmd(0.0, 0.0)
    session.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

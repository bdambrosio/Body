"""Tier-3 reactive drive (Pi-side).

Drives the robot to one observable subgoal on the live rasterized lidar
scan (``body/lidar/scan``), with no dependence on the global map or PF pose.
Subscribes ``body/drive/goto`` + ``body/odom`` + ``body/lidar/scan`` +
``body/imu`` + ``body/oakd/depth`` + the safety layer (``body/emergency_stop``,
``body/status``, ``body/motor_state``); publishes ``body/cmd_vel`` (only while
a goal is active — see cmd_vel arbitration in docs/drive_tier3_spec.md) and
``body/drive/status`` every tick.

The per-goal control logic (local A* + pure-pursuit follow, swept/depth
vetoes, rotation budget, path hysteresis) lives in
``body/lib/tier3_controller.py``; this process is transport + lifecycle:

* goto acceptance / supersede watermark / one-shot ``REJECTED`` / CANCELED
  (``GotoArbiter``); every accepted command is acknowledged by the next
  tick's status carrying its cmd_id;
* ``frame: "body"`` goals are anchored with the odom pose interpolated at
  their ``anchor_ts`` (odom ring buffer), and every scan is planned with the
  pose the robot had at the scan's (mid-sweep) ``ts``;
* odom reset (``session_id`` change / ``seq`` decrease) → drop the goal,
  FAULT ``odom_reset``; e-stop / stall → drop the goal, FAULT ``e_stop``,
  refuse gotos until clear;
* every tick runs under try/except: an exception commands zero, drops the
  goal with FAULT ``internal_error`` and the loop keeps running.

The watchdog/e-stop/motor timeout remain supreme; this process is just another
cmd_vel producer.
"""
from __future__ import annotations

import math

import numpy as np
import signal
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional

from body.lib import drive_config, schemas, zenoh_helpers
from body.lib.buildinfo import git_sha
from body.lib.depth_memory import rounded_xy
from body.lib.handoff_gate import HandoffGate
from body.lib.local_drive_core import (
    STATE_IDLE,
    STATE_ARRIVED, STATE_FAULT, STATE_REJECTED, EStopMonitor, GotoArbiter,
    ImuYawCorrector, OdomRing, body_to_odom, quat_wxyz_to_yaw, wrap_pi,
)
from body.lib.scan_raster import rasterize_scan
from body.lib.tier3_controller import Tier3Controller, Tier3Goal

try:  # scan_raster grew a message-level entry point (honours unknown_bins)
    from body.lib.scan_raster import rasterize_scan_msg
except ImportError:  # pragma: no cover — older tree
    rasterize_scan_msg = None


def _idle_until_signal(reason: str) -> None:
    print(f"local_drive: {reason}; idling so launcher does not respawn.", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda _s, _f: stop.set())
    signal.signal(signal.SIGINT, lambda _s, _f: stop.set())
    while not stop.is_set():
        time.sleep(1.0)
    sys.exit(0)


def _sleep_to(next_tick: float, period: float) -> float:
    next_tick += period
    sleep_for = next_tick - time.monotonic()
    if sleep_for > 0:
        time.sleep(sleep_for)
        return next_tick
    return time.monotonic()


def _rasterize(scan: Dict[str, Any], raster):
    if rasterize_scan_msg is not None:
        return rasterize_scan_msg(scan, raster)
    return rasterize_scan(scan.get("ranges"), float(scan.get("angle_min", 0.0)),
                          float(scan.get("angle_increment", 0.0)), raster)


def _odom_usable(odom: Optional[Dict[str, Any]], now_wall: float, stale_s: float) -> bool:
    if odom is None:
        return False
    if str(odom.get("source", "")) == "none":
        return False   # encoders not live: the pose does not integrate motion
    try:
        return (now_wall - float(odom.get("ts", 0.0))) <= stale_s
    except (TypeError, ValueError):
        return False


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    cfg = body_cfg.get("local_drive", {})
    if not bool(cfg.get("enabled", False)):
        _idle_until_signal("disabled (local_drive.enabled=false)")
        return

    # Everything Tier-3 models comes from the SHARED builders in
    # body/lib/drive_config.py — the desktop's Tier-2 clear-run and the debug
    # consoles construct theirs from the same functions and config.json.
    t3_cfg = drive_config.tier3_config(body_cfg)
    raster = drive_config.scan_raster_config(body_cfg)
    scan_cfg = cfg.get("scan", {})
    control_hz = float(cfg.get("control_hz", 10.0))
    period = 1.0 / max(1.0, control_hz)
    cmd_timeout_ms = max(500, int(3.0 * period * 1000.0))
    odom_stale_s = float(cfg.get("odom_stale_s", 0.5))
    imu_stale_s = float(cfg.get("imu_stale_s", 0.3))
    scan_stale_s = float(scan_cfg.get("scan_stale_s", 0.5))
    arm_lease_s = float(cfg.get("handoff_arm_lease_s", 5.0))

    # Shared with zenoh callback threads (all under `lock`).
    lock = threading.Lock()
    ring = OdomRing()
    arb = GotoArbiter()
    estop = EStopMonitor()
    last_odom: Optional[Dict[str, Any]] = None
    last_scan: Optional[Dict[str, Any]] = None
    last_imu: Optional[Dict[str, Any]] = None
    last_depth: Optional[Dict[str, Any]] = None
    mem_reset_reason: Optional[str] = None  # set by body/drive/depth_memory
    last_done: Optional[tuple] = None       # (cmd_id, plan_reason) of the last ARRIVED
    odom_reset_pending = False
    imu_session: Optional[str] = None
    imu_seq: Optional[int] = None
    imu_seq_at = 0.0                        # wall time the IMU seq last advanced
    imu_rebaseline = False

    session = zenoh_helpers.open_session(body_cfg)
    stop = threading.Event()

    def on_odom(_k: str, msg: Dict[str, Any]) -> None:
        nonlocal last_odom, odom_reset_pending
        with lock:
            if ring.push(msg):
                odom_reset_pending = True
            last_odom = msg

    def on_scan(_k: str, msg: Dict[str, Any]) -> None:
        nonlocal last_scan
        with lock:
            last_scan = msg

    def on_imu(_k: str, msg: Dict[str, Any]) -> None:
        nonlocal last_imu, imu_session, imu_seq, imu_seq_at, imu_rebaseline
        with lock:
            sid = msg.get("session_id")
            if isinstance(sid, str) and sid:
                if imu_session is not None and sid != imu_session:
                    imu_rebaseline = True   # fusion yaw origin may have jumped
                imu_session = sid
            seq = msg.get("seq")
            if isinstance(seq, int) and not isinstance(seq, bool):
                if imu_seq is not None and seq < imu_seq:
                    imu_rebaseline = True   # driver restarted / sensor re-init
                if imu_seq is None or seq != imu_seq:
                    imu_seq_at = time.time()
                imu_seq = seq
            else:
                imu_seq = None              # older driver: freshness by ts only
            last_imu = msg

    def on_depth(_k: str, msg: Dict[str, Any]) -> None:
        nonlocal last_depth
        with lock:
            last_depth = msg

    def on_goto(_k: str, msg: Dict[str, Any]) -> None:
        with lock:
            outcome = arb.on_command(msg, estop_active=estop.active())
        if outcome in ("rejected", "malformed", "refused_estop"):
            print(f"local_drive: goto cmd_id={msg.get('cmd_id')!r} {outcome} "
                  f"(frame={msg.get('frame')!r})", flush=True)

    def on_depth_memory(_k: str, msg: Dict[str, Any]) -> None:
        # Desktop: forget everything (Go / relocate / robot carried / button).
        # Wheel odom can't see the robot being picked up, so remembered cells
        # would land around it wherever it is put down.
        nonlocal mem_reset_reason
        if isinstance(msg, dict) and msg.get("action") == "reset":
            with lock:
                mem_reset_reason = str(msg.get("reason") or "desktop")

    def on_estop(_k: str, msg: Dict[str, Any]) -> None:
        with lock:
            estop.on_event(msg)

    def on_status(_k: str, msg: Dict[str, Any]) -> None:
        with lock:
            estop.on_status(msg)

    def on_motor_state(_k: str, msg: Dict[str, Any]) -> None:
        with lock:
            estop.on_motor_state(msg)

    zenoh_helpers.declare_subscriber_json(session, "body/odom", on_odom)
    zenoh_helpers.declare_subscriber_json(session, "body/lidar/scan", on_scan)
    zenoh_helpers.declare_subscriber_json(session, "body/imu", on_imu)
    zenoh_helpers.declare_subscriber_json(session, "body/oakd/depth", on_depth)
    zenoh_helpers.declare_subscriber_json(session, "body/drive/goto", on_goto)
    zenoh_helpers.declare_subscriber_json(session, "body/drive/depth_memory", on_depth_memory)
    zenoh_helpers.declare_subscriber_json(session, "body/emergency_stop", on_estop)
    zenoh_helpers.declare_subscriber_json(session, "body/status", on_status)
    zenoh_helpers.declare_subscriber_json(session, "body/motor_state", on_motor_state)

    # HO-3 handoff gate (Tier-3 → motors): records each command + obeys the
    # standalone Handoff Inspector's arm/continue. The arm lease means a
    # closed inspector cannot leave Tier-3 holding forever.
    gate = HandoffGate(session, tiers=(3,), arm_lease_s=arm_lease_s)
    ctrl = Tier3Controller(t3_cfg, gate=gate)

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
                       path_body=None, plan_reason=None, watermark=None,
                       depth_hits=None, depth_seq=None, depth_bbox_px=None,
                       depth_hit_body=None, depth_would_block=False) -> None:
        zenoh_helpers.publish_json(
            session, "body/drive/status",
            schemas.drive_status(
                cmd_id=cmd_id, state=state, goal_body_xy=goal_body,
                dist_remaining_m=dist, v_mps=v, omega_radps=omega,
                blocked_reason=reason, mode=mode, path_body_xy=path_body,
                plan_reason=plan_reason, build=build, watermark=watermark,
                depth_hits=depth_hits, depth_seq=depth_seq,
                depth_bbox_px=depth_bbox_px,
                depth_hit_body=rounded_xy(depth_hit_body, 32),
                depth_mem_body=mem_body,
                depth_would_block=depth_would_block,
            ),
        )

    # Main-loop-only state.
    mem_body = None                         # depth memory, live body frame, per tick
    tracked_cmd_id: Optional[int] = None
    was_active = False                      # commanded motion last tick
    imu_guard = cfg.get("imu_guard", {})
    yaw_corr = ImuYawCorrector(             # IMU-vs-wheel yaw divergence per goal
        window_s=float(imu_guard.get("window_s", 1.0)),
        window_wheel_min_rad=float(imu_guard.get("wheel_min_rad", 0.35)),
        window_imu_max_rad=float(imu_guard.get("imu_max_rad", 0.02)),
        wheel_glitch_rate_radps=float(imu_guard.get("wheel_glitch_rate_radps", 4.0)),
    )
    last_exc_print = 0.0

    def drop_goal(reason: str) -> None:
        nonlocal tracked_cmd_id
        with lock:
            cid = arb.drop(reason)
        ctrl.clear()
        tracked_cmd_id = None
        if cid is not None:
            print(f"local_drive: goal {cid} dropped ({reason})", flush=True)

    def tick(now_wall: float, now_mono: float) -> None:
        nonlocal tracked_cmd_id, was_active, odom_reset_pending, imu_rebaseline, mem_body
        nonlocal mem_reset_reason, last_done
        with lock:
            rejects = arb.take_rejects()
            watermark = arb.watermark
            g = dict(arb.goal) if arb.goal is not None else None
            odom = dict(last_odom) if last_odom is not None else None
            scan = last_scan
            imu = last_imu
            depth = last_depth
            reset = odom_reset_pending
            odom_reset_pending = False
            rebase = imu_rebaseline
            imu_rebaseline = False
            mem_reset = mem_reset_reason
            mem_reset_reason = None
            imu_seq_fresh = imu_seq is None or (now_wall - imu_seq_at) <= imu_stale_s
            estop_on = estop.active()
            estop_desc = estop.describe()

        for rid in rejects:
            publish_status(STATE_REJECTED, cmd_id=rid, watermark=watermark)
        if rebase:
            yaw_corr.rebaseline()

        if reset:
            print("local_drive: odom frame reset (motor_controller restart)", flush=True)
            yaw_corr.reset()
            ctrl.memory.reset()
            if g is not None:
                drop_goal("odom_reset")
                g = None

        if mem_reset is not None:
            print(f"local_drive: depth memory reset ({mem_reset}, "
                  f"{len(ctrl.memory)} cells dropped)", flush=True)
            ctrl.memory.reset()

        # Depth memory learns (and forgets) every tick, goal or not, in the raw
        # wheel-odom frame — a retry is a new goal and must still see it.
        mem_body = None
        if _odom_usable(odom, now_wall, odom_stale_s):
            raw_now = (float(odom["x"]), float(odom["y"]), float(odom["theta"]))
            pose_depth = None
            if depth is not None:
                with lock:
                    pose_depth = ring.pose_at(float(depth.get("ts", 0.0) or 0.0))
            ctrl.observe_depth(depth, now_wall=now_wall, pose_now=raw_now,
                               pose_depth=pose_depth,
                               omega_radps=float(odom.get("vtheta", 0.0) or 0.0))
            # Everything Tier-2 could march into (ahead/abeam, within its
            # horizon + fan), not the 64 nearest: a truncated copy let it pick
            # goals inside cells Tier-3 plans around (snapped from tick 1).
            cells = ctrl.memory.cells_body(raw_now)
            if len(cells):
                cells = cells[(cells[:, 0] > -0.3) & (np.hypot(cells[:, 0], cells[:, 1]) < 2.0)]
            mem_body = rounded_xy(cells, ctrl.memory.cfg.max_cells)

        if estop_on:
            if g is not None:
                drop_goal("e_stop")
                print(f"local_drive: e-stop active ({estop_desc})", flush=True)
                g = None
            if was_active:
                publish_cmd(0.0, 0.0)
                was_active = False
            with lock:
                fault = arb.fault
                wm = arb.watermark
            publish_status(STATE_FAULT, cmd_id=(fault[0] if fault else max(0, wm)),
                           reason="e_stop")
            return

        if g is None:
            if was_active:
                # Cancel/stop/drop while driving: command zero NOW instead of
                # letting the motors coast on the last cmd_vel until timeout.
                publish_cmd(0.0, 0.0)
                was_active = False
            ctrl.clear()
            tracked_cmd_id = None
            with lock:
                state, cid, reason = arb.idle_status()
            # ARRIVED is one tick; the desktop usually sees the IDLE. Carry the
            # arrival's plan_reason on it so a snapped/frontier (best-effort)
            # endpoint is distinguishable from reaching the goal.
            pr = (last_done[1] if state == STATE_IDLE and last_done is not None
                  and last_done[0] == cid else None)
            publish_status(state, cmd_id=cid, reason=reason, plan_reason=pr)
            return

        cmd_id = int(g.get("cmd_id", 0))

        # Odom freshness — can't hold or transform the goal without a pose.
        if not _odom_usable(odom, now_wall, odom_stale_s):
            if was_active:
                publish_cmd(0.0, 0.0)
                was_active = False
            publish_status(STATE_FAULT, cmd_id=cmd_id, reason="odom_stale")
            return

        odom_theta = float(odom["theta"])
        imu_yaw = None
        if (imu is not None and imu_seq_fresh
                and (now_wall - float(imu.get("ts", 0.0))) <= imu_stale_s):
            q = imu.get("orientation")
            if isinstance(q, dict) and all(k in q for k in ("w", "x", "y", "z")):
                imu_yaw = quat_wxyz_to_yaw(
                    float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"]))

        # New goal → anchor it in odom and reset per-goal state.
        if cmd_id != tracked_cmd_id:
            tracked_cmd_id = cmd_id
            yaw_corr.reset()    # goal frame = odom frame as of now
            gx, gy = float(g["x_m"]), float(g["y_m"])
            if str(g.get("frame", "odom")) == "body":
                anchor = None
                with lock:
                    if g.get("anchor_ts") is not None:
                        anchor = ring.pose_at(float(g["anchor_ts"]))
                    latest = ring.latest()
                if anchor is None:
                    anchor = latest[1] if latest else (
                        float(odom["x"]), float(odom["y"]), odom_theta)
                    print(f"local_drive: body goal {cmd_id} anchor_ts "
                          f"{g.get('anchor_ts')!r} outside odom buffer — "
                          "anchored at latest odom", flush=True)
                gx, gy = body_to_odom((gx, gy), anchor)
            ctrl.start_goal(Tier3Goal(
                cmd_id=cmd_id, x=gx, y=gy,
                tol=float(g.get("arrival_tol_m") or t3_cfg.params.arrival_tol_m),
                final_heading=(float(g["final_heading_rad"])
                               if g.get("final_heading_rad") is not None else None),
                v_max=float(g["v_max"]) if g.get("v_max") is not None else None,
                omega_max=(float(g["omega_max"])
                           if g.get("omega_max") is not None else None),
            ), now_mono, (float(odom["x"]), float(odom["y"]), odom_theta))

        # Heading = wheel odom + the IMU-vs-wheel divergence since the goal
        # started (wheel odom misses externally forced rotation).
        theta = yaw_corr.corrected_theta(odom_theta, imu_yaw,
                                         yaw_rate=float(odom.get("vtheta", 0.0) or 0.0),
                                         t=now_mono)
        div = wrap_pi(theta - odom_theta)
        pose_now = (float(odom["x"]), float(odom["y"]), theta)

        grid = meta = pose_scan = pose_scan_raw = None
        if scan is not None and (now_wall - float(scan.get("ts", 0.0))) <= scan_stale_s:
            grid, meta = _rasterize(scan, raster)
            # Plan in the scan's own body frame: the pose at its (mid-sweep)
            # timestamp, not the newest one (T10).
            with lock:
                p_scan = ring.pose_at(float(scan.get("ts", 0.0)))
            pose_scan = (pose_now if p_scan is None
                         else (p_scan[0], p_scan[1], wrap_pi(p_scan[2] + div)))
            pose_scan_raw = ((pose_now[0], pose_now[1], odom_theta) if p_scan is None
                             else tuple(p_scan[:3]))

        r = ctrl.step(now_mono=now_mono, now_wall=now_wall, pose_now=pose_now,
                      pose_scan=pose_scan, grid=grid, meta=meta, depth=depth,
                      pose_scan_raw=pose_scan_raw)
        if r.cmd is not None:
            publish_cmd(r.cmd[0], r.cmd[1])
            was_active = abs(r.cmd[0]) > 0 or abs(r.cmd[1]) > 0
        publish_status(r.state, cmd_id=cmd_id, goal_body=r.goal_body, dist=r.dist,
                       v=r.v, omega=r.omega, reason=r.reason, mode=r.mode,
                       path_body=r.path_body, plan_reason=r.plan_reason,
                       depth_hits=r.depth_hits, depth_seq=r.depth_seq,
                       depth_bbox_px=r.depth_bbox_px, depth_hit_body=r.depth_hit_body,
                       depth_would_block=r.depth_would_block)
        if r.done:
            if r.state == STATE_ARRIVED and r.plan_reason:
                print(f"local_drive: goal {cmd_id} ARRIVED at {r.plan_reason} "
                      f"endpoint (residual {r.dist:.2f} m)", flush=True)
            with lock:
                arb.finish(cmd_id)
            last_done = (cmd_id, r.plan_reason) if r.state == STATE_ARRIVED else None
            ctrl.clear()
            tracked_cmd_id = None

    print(
        f"local_drive: up; control_hz={control_hz} v_max={t3_cfg.params.v_max} "
        f"depth_veto={('on' if t3_cfg.depth_act else 'observe') if t3_cfg.depth.enabled else 'off'} "
        f"depth_memory={'on' if t3_cfg.memory.enabled else 'off'}",
        flush=True,
    )
    next_tick = time.monotonic()
    while not stop.is_set():
        now_wall = time.time()
        now_mono = time.monotonic()
        try:
            tick(now_wall, now_mono)
        except Exception as e:  # noqa: BLE001 — never let one bad tick kill Tier-3
            if now_mono - last_exc_print > 5.0:
                last_exc_print = now_mono
                print(f"local_drive: tick exception {e!r}\n{traceback.format_exc()}",
                      flush=True)
            try:
                publish_cmd(0.0, 0.0)
                was_active = False
                with lock:
                    cid = arb.drop("internal_error")
                    wm = arb.watermark
                ctrl.clear()
                tracked_cmd_id = None
                publish_status(STATE_FAULT, cmd_id=(cid if cid is not None else max(0, wm)),
                               reason="internal_error")
            except Exception:  # noqa: BLE001
                pass
        next_tick = _sleep_to(next_tick, period)

    publish_cmd(0.0, 0.0)
    session.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

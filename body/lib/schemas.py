"""JSON message helpers for Body Zenoh topics (see docs/body_project_spec.md)."""

from __future__ import annotations

import math
import time
from typing import Any


def now_ts() -> float:
    return time.time()


def cmd_vel(ts: float | None = None, linear: float = 0.0, angular: float = 0.0, timeout_ms: int = 500) -> dict[str, Any]:
    return {"ts": now_ts() if ts is None else ts, "linear": linear, "angular": angular, "timeout_ms": timeout_ms}


def cmd_direct(ts: float | None = None, left: float = 0.0, right: float = 0.0, timeout_ms: int = 500) -> dict[str, Any]:
    return {"ts": now_ts() if ts is None else ts, "left": left, "right": right, "timeout_ms": timeout_ms}


def drive_goto(
    *,
    ts: float | None = None,
    cmd_id: int,
    frame: str = "odom",
    x_m: float = 0.0,
    y_m: float = 0.0,
    final_heading_rad: float | None = None,
    arrival_tol_m: float | None = None,
    v_max: float | None = None,
    kind: str = "goto",
) -> dict[str, Any]:
    """body/drive/goto — Tier-3 drive command (see docs/drive_tier3_spec.md).

    A single subgoal for the Pi-side reactive driver. ``frame`` is "odom":
    the goal is a point in the Pi's odom frame, so it stays fixed in the
    world as the robot moves (the sender converts a body-frame click via
    the local_map ``anchor_pose``). ``kind`` is "goto" | "cancel" | "stop".
    ``cmd_id`` correlates status; a higher cmd_id supersedes a lower one.
    """
    msg: dict[str, Any] = {
        "ts": now_ts() if ts is None else ts,
        "cmd_id": int(cmd_id),
        "frame": frame,
        "x_m": float(x_m),
        "y_m": float(y_m),
        "kind": kind,
    }
    if final_heading_rad is not None:
        msg["final_heading_rad"] = float(final_heading_rad)
    if arrival_tol_m is not None:
        msg["arrival_tol_m"] = float(arrival_tol_m)
    if v_max is not None:
        msg["v_max"] = float(v_max)
    return msg


def drive_status(
    *,
    ts: float | None = None,
    cmd_id: int,
    state: str,
    goal_body_xy: tuple[float, float] | list[float] | None = None,
    dist_remaining_m: float = 0.0,
    v_mps: float = 0.0,
    omega_radps: float = 0.0,
    blocked_reason: str | None = None,
    mode: str | None = None,
    path_body_xy: list | None = None,
    plan_reason: str | None = None,
    build: str | None = None,
) -> dict[str, Any]:
    """body/drive/status — Tier-3 drive status (see docs/drive_tier3_spec.md).

    ``state`` is one of IDLE | DRIVING | ARRIVED | BLOCKED | CANCELED | FAULT.
    ``goal_body_xy`` is the active goal transformed into the *live* body
    frame, for the operator UI to draw. ``blocked_reason`` is set only when
    state==BLOCKED/FAULT: no_path | swept_block | no_progress | odom_stale |
    goal_out_of_map | start_blocked | goal_unreachable. ``path_body_xy`` is the
    local A* path (body frame, downsampled) when driving, for the operator UI.
    """
    msg: dict[str, Any] = {
        "ts": now_ts() if ts is None else ts,
        "cmd_id": int(cmd_id),
        "state": state,
        "dist_remaining_m": float(dist_remaining_m),
        "v_mps": float(v_mps),
        "omega_radps": float(omega_radps),
    }
    if goal_body_xy is not None:
        msg["goal_body_xy"] = [float(goal_body_xy[0]), float(goal_body_xy[1])]
    if blocked_reason is not None:
        msg["blocked_reason"] = blocked_reason
    if mode is not None:
        msg["mode"] = mode
    if plan_reason is not None:
        msg["plan_reason"] = plan_reason          # ok|frontier (A* reach vs best-effort)
    if path_body_xy:
        msg["path_body_xy"] = [[float(p[0]), float(p[1])] for p in path_body_xy]
    if build is not None:
        msg["build"] = build
    return msg


def odom(
    ts: float | None = None,
    x: float = 0.0,
    y: float = 0.0,
    theta: float = 0.0,
    vx: float = 0.0,
    vtheta: float = 0.0,
    left_ticks: int = 0,
    right_ticks: int = 0,
    dt_ms: int = 20,
    source: str = "commanded_vel_playback",
) -> dict[str, Any]:
    """body/odom — dead-reckoned pose + raw encoder ticks.

    ``source`` identifies the origin of the pose integration so consumers can decide how much to
    trust it as a prior. Defined values:

    - ``"wheel_encoders"`` — integrated from real GPIO encoder ticks (best prior when available).
    - ``"commanded_vel_playback"`` — integrated from the last commanded velocity (no encoders
      configured or encoder read failed); usable as a coarse sanity check only.
    - ``"stub"`` — synthetic zero-motion publisher (stub mode, no motion being commanded).
    """
    ts_val = now_ts() if ts is None else ts
    th = math.atan2(math.sin(theta), math.cos(theta))
    return {
        "ts": ts_val,
        "x": x,
        "y": y,
        "theta": th,
        "vx": vx,
        "vtheta": vtheta,
        "left_ticks": left_ticks,
        "right_ticks": right_ticks,
        "dt_ms": dt_ms,
        "source": source,
    }


def motor_state(
    ts: float | None = None,
    left_pwm: float = 0.0,
    right_pwm: float = 0.0,
    left_dir: str = "fwd",
    right_dir: str = "fwd",
    e_stop_active: bool = False,
    cmd_timeout_active: bool = False,
    stall_detected: bool = False,
) -> dict[str, Any]:
    return {
        "ts": now_ts() if ts is None else ts,
        "left_pwm": left_pwm,
        "right_pwm": right_pwm,
        "left_dir": left_dir,
        "right_dir": right_dir,
        "e_stop_active": e_stop_active,
        "cmd_timeout_active": cmd_timeout_active,
        "stall_detected": stall_detected,
    }


def lidar_scan(
    ts: float | None = None,
    num_points: int = 360,
    range_const: float = 2.0,
    scan_time_ms: int = 100,
) -> dict[str, Any]:
    angle_increment = (2.0 * math.pi) / num_points
    ranges: list[float | None] = [range_const for _ in range(num_points)]
    return {
        "ts": now_ts() if ts is None else ts,
        "angle_min": 0.0,
        "angle_max": 2.0 * math.pi,
        "angle_increment": angle_increment,
        "range_min": 0.05,
        "range_max": 12.0,
        "ranges": ranges,
        "scan_time_ms": scan_time_ms,
    }


def lidar_scan_from_bins(
    ranges_m: list[float | None],
    *,
    intensities: list[int] | None = None,
    range_min_m: float = 0.05,
    range_max_m: float = 12.0,
    scan_time_ms: int = 100,
    ts: float | None = None,
) -> dict[str, Any]:
    """Build ``body/lidar/scan`` from fixed angular bins (see docs/body_project_spec.md §5.5)."""
    n = len(ranges_m)
    angle_increment = (2.0 * math.pi) / max(1, n)
    msg: dict[str, Any] = {
        "ts": now_ts() if ts is None else ts,
        "angle_min": 0.0,
        "angle_max": 2.0 * math.pi,
        "angle_increment": angle_increment,
        "range_min": range_min_m,
        "range_max": range_max_m,
        "ranges": ranges_m,
        "scan_time_ms": scan_time_ms,
    }
    if intensities is not None:
        msg["intensities"] = intensities
    return msg


def oakd_imu(ts: float | None = None) -> dict[str, Any]:
    t = now_ts() if ts is None else ts
    return {
        "ts": t,
        "accel": {"x": 0.0, "y": 0.0, "z": 9.81},
        "gyro": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def oakd_imu_report(
    ts: float,
    accel_xyz: tuple[float, float, float],
    gyro_xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Build body/oakd/imu JSON per docs/body_project_spec.md §5.6 (sensor frame)."""
    msg: dict[str, Any] = {
        "ts": ts,
        "accel": {"x": accel_xyz[0], "y": accel_xyz[1], "z": accel_xyz[2]},
        "gyro": {"x": gyro_xyz[0], "y": gyro_xyz[1], "z": gyro_xyz[2]},
    }
    if quat_wxyz is not None:
        msg["orientation"] = {
            "w": quat_wxyz[0],
            "x": quat_wxyz[1],
            "y": quat_wxyz[2],
            "z": quat_wxyz[3],
        }
    return msg


def imu_report(
    ts: float,
    accel_xyz: tuple[float, float, float],
    gyro_xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    fusion_mode: str,
    fusion_accuracy_rad: float,
    linear_accel_xyz: tuple[float, float, float] | None = None,
    calibration_status: int | None = None,
    mag_quat_wxyz: tuple[float, float, float, float] | None = None,
    mag_accuracy_rad: float | None = None,
    mag_valid: bool | None = None,
) -> dict[str, Any]:
    """Build body/imu JSON per docs/imu_integration_spec.md §2 (body frame).

    ``fusion_mode`` is one of ``"rotation_vector"``, ``"game_rotation_vector"``, or ``"raw"``.
    ``fusion_accuracy_rad`` is the BNO085 per-report accuracy estimate (consumer σ).
    ``linear_accel_xyz`` is the gravity-removed accel when the driver enables that report.
    ``calibration_status`` (0–3) is the SH-2 system calibration level when available.
    """
    msg: dict[str, Any] = {
        "ts": ts,
        "accel": {"x": accel_xyz[0], "y": accel_xyz[1], "z": accel_xyz[2]},
        "gyro": {"x": gyro_xyz[0], "y": gyro_xyz[1], "z": gyro_xyz[2]},
        "orientation": {
            "w": quat_wxyz[0],
            "x": quat_wxyz[1],
            "y": quat_wxyz[2],
            "z": quat_wxyz[3],
        },
        "fusion": {
            "mode": fusion_mode,
            "accuracy_rad": float(fusion_accuracy_rad),
        },
    }
    if linear_accel_xyz is not None:
        msg["linear_accel"] = {
            "x": linear_accel_xyz[0],
            "y": linear_accel_xyz[1],
            "z": linear_accel_xyz[2],
        }
    if calibration_status is not None:
        msg["fusion"]["calibration_status"] = int(calibration_status)
    if mag_valid is not None:
        mag_block: dict[str, Any] = {"valid": bool(mag_valid)}
        if mag_quat_wxyz is not None:
            mag_block["orientation"] = {
                "w": mag_quat_wxyz[0],
                "x": mag_quat_wxyz[1],
                "y": mag_quat_wxyz[2],
                "z": mag_quat_wxyz[3],
            }
        if mag_accuracy_rad is not None:
            mag_block["accuracy_rad"] = float(mag_accuracy_rad)
        msg["mag"] = mag_block
    return msg


def oakd_depth_placeholder(ts: float | None = None) -> dict[str, Any]:
    return {"ts": now_ts() if ts is None else ts, "format": "placeholder", "note": "TBD per docs/body_project_spec.md §5.7"}


def oakd_depth_stream_frame(
    width: int,
    height: int,
    data_base64: str,
    *,
    ts: float | None = None,
    dtype: str = "uint16",
    units: str = "mm",
    layout: str = "row_major",
    intrinsics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """body/oakd/depth — streamed depth from StereoDepth (host-resized), raw uint16 row-major.

    If ``intrinsics`` is provided (``fx``, ``fy``, ``cx``, ``cy`` for the published ``width``×``height``
    depth image, post-rotation), include it so consumers can unproject without re-deriving from an
    assumed HFOV.
    """
    msg: dict[str, Any] = {
        "ts": now_ts() if ts is None else ts,
        "format": "depth_uint16_mm",
        "width": width,
        "height": height,
        "dtype": dtype,
        "units": units,
        "layout": layout,
        "encoding": "base64",
        "data": data_base64,
    }
    if intrinsics is not None:
        msg["intrinsics"] = intrinsics
    return msg


def oakd_config_capture_rgb(request_id: str) -> dict[str, Any]:
    """body/oakd/config — request a single RGB frame (handled by oakd_driver)."""
    return {"action": "capture_rgb", "request_id": request_id}


def oakd_rgb_capture_ok(
    request_id: str,
    jpeg_base64: str,
    width: int,
    height: int,
    ts: float | None = None,
) -> dict[str, Any]:
    """body/oakd/rgb — successful on-request JPEG (base64)."""
    return {
        "ts": now_ts() if ts is None else ts,
        "request_id": request_id,
        "ok": True,
        "format": "jpeg",
        "encoding": "base64",
        "data": jpeg_base64,
        "width": width,
        "height": height,
    }


def oakd_rgb_capture_error(request_id: str, error: str, ts: float | None = None) -> dict[str, Any]:
    """body/oakd/rgb — capture failed (e.g. rgb disabled or no frame)."""
    return {
        "ts": now_ts() if ts is None else ts,
        "request_id": request_id,
        "ok": False,
        "error": error,
    }


def heartbeat(ts: float | None = None, seq: int = 0) -> dict[str, Any]:
    return {"ts": now_ts() if ts is None else ts, "seq": seq}


def status(
    processes: dict[str, str],
    heartbeat_ok: bool,
    e_stop_active: bool,
    uptime_s: float,
    ts: float | None = None,
    host: dict[str, Any] | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "ts": now_ts() if ts is None else ts,
        "processes": processes,
        "heartbeat_ok": heartbeat_ok,
        "e_stop_active": e_stop_active,
        "uptime_s": uptime_s,
    }
    if host is not None:
        msg["host"] = host
    return msg


def emergency_stop(reason: str, source: str = "watchdog", ts: float | None = None) -> dict[str, Any]:
    return {"ts": now_ts() if ts is None else ts, "reason": reason, "source": source}


def local_map_2p5d(
    *,
    ts: float,
    resolution_m: float,
    origin_x_m: float,
    origin_y_m: float,
    nx: int,
    ny: int,
    max_height_m: list[list[float | None]],
    frame: str = "body",
    sources: dict[str, Any] | None = None,
    driveable: list[list[bool | None]] | None = None,
    driveable_clearance_height_m: float | None = None,
    anchor_pose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """body/map/local_2p5d — egocentric max-height-above-ground grid (see docs/local_map_spec.md).

    ``anchor_pose`` (optional) carries the latest ``body/odom`` pose cached at publish time so
    consumers can fuse into a world frame without interpolating odom to ``ts``. Shape:
    ``{odom_ts, x, y, theta, source}`` where ``source`` mirrors ``odom.source``. Omitted if no
    odom has been received yet.
    """
    msg: dict[str, Any] = {
        "ts": ts,
        "frame": frame,
        "kind": "max_height_grid",
        "resolution_m": resolution_m,
        "origin_x_m": origin_x_m,
        "origin_y_m": origin_y_m,
        "nx": nx,
        "ny": ny,
        "max_height_m": max_height_m,
    }
    if driveable is not None:
        msg["driveable"] = driveable
    if driveable_clearance_height_m is not None:
        msg["driveable_clearance_height_m"] = driveable_clearance_height_m
    if sources is not None:
        msg["sources"] = sources
    if anchor_pose is not None:
        msg["anchor_pose"] = anchor_pose
    return msg


# ── Tier handoff records (debug: the Handoff Inspector) ──────────────
# One record per tier boundary, published to drive/handoff/t{1,2,3}. Pure
# JSON (grids passed as already-listified int8 rows so this module stays
# numpy-free). The HandoffGate stamps `seq`; producers stamp the rest.

def _grid_meta(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if meta is None:
        return None
    return {k: meta[k] for k in
            ("resolution_m", "origin_x_m", "origin_y_m", "nx", "ny")}


def handoff_t1(
    *, ts: float | None = None, pose, wp, wp_index: int, wp_total: int,
    lap_index: int = 0, terminal: bool, arrival_tol_m: float,
    bearing_rad: float, wp_dist_m: float, route=None,
) -> dict[str, Any]:
    """HO-1 Tier-1 → Tier-2: the chosen world-frame waypoint + bearing, plus the
    full dense Tier-1 route (all expanded sub-waypoints) for the global display
    — Tier-1 builds the whole route but only hands ``wp`` (index ``wp_index``)
    down each leg."""
    msg: dict[str, Any] = {
        "ts": now_ts() if ts is None else ts,
        "tier": 1,
        "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
        "wp": [float(wp[0]), float(wp[1])],
        "wp_index": int(wp_index), "wp_total": int(wp_total),
        "lap_index": int(lap_index), "terminal": bool(terminal),
        "arrival_tol_m": float(arrival_tol_m),
        "bearing_rad": float(bearing_rad), "wp_dist_m": float(wp_dist_m),
    }
    if route is not None:
        msg["route"] = [[float(x), float(y)] for x, y in route]
    return msg


def handoff_t2(
    *, ts: float | None = None, pose=None, bearing_rad: float, src: str,
    free_dist_m: float, subgoal_body, target_body=None, arrival_tol_m: float,
    v_max: float | None = None, cmd_id: int | None = None,
    grid_rows: list | None = None, meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """HO-2 Tier-2 → Tier-3: the body-frame sub-goal + the live scan it sits on."""
    msg: dict[str, Any] = {
        "ts": now_ts() if ts is None else ts,
        "tier": 2,
        "bearing_rad": float(bearing_rad), "src": src,
        "free_dist_m": float(free_dist_m),
        "subgoal_body": [float(subgoal_body[0]), float(subgoal_body[1])],
        "arrival_tol_m": float(arrival_tol_m),
    }
    if pose is not None:
        msg["pose"] = [float(pose[0]), float(pose[1]), float(pose[2])]
    if target_body is not None:
        msg["target_body"] = [float(target_body[0]), float(target_body[1])]
    if v_max is not None:
        msg["v_max"] = float(v_max)
    if cmd_id is not None:
        msg["cmd_id"] = int(cmd_id)
    if grid_rows is not None:
        msg["grid"] = grid_rows
    gm = _grid_meta(meta)
    if gm is not None:
        msg["meta"] = gm
    return msg


def handoff_t3(
    *, ts: float | None = None, cmd_id: int, goal_body, plan_reason,
    path_body=None, lookahead=None, v_mps: float, omega_radps: float,
    swept_blocked: bool, grid_rows: list | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """HO-3 Tier-3 → motors: the A* path + the (v, ω) it's about to command."""
    msg: dict[str, Any] = {
        "ts": now_ts() if ts is None else ts,
        "tier": 3,
        "cmd_id": int(cmd_id),
        "goal_body": [float(goal_body[0]), float(goal_body[1])],
        "plan_reason": plan_reason,
        "v_mps": float(v_mps), "omega_radps": float(omega_radps),
        "swept_blocked": bool(swept_blocked),
    }
    if path_body is not None:
        msg["path_body"] = [[float(p[0]), float(p[1])] for p in path_body]
    if lookahead is not None:
        msg["lookahead"] = [float(lookahead[0]), float(lookahead[1])]
    if grid_rows is not None:
        msg["grid"] = grid_rows
    gm = _grid_meta(meta)
    if gm is not None:
        msg["meta"] = gm
    return msg

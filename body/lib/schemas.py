"""JSON message helpers for Body Zenoh topics (see body_project_spec.md)."""

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
) -> dict[str, Any]:
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
    }


def motor_state(
    ts: float | None = None,
    left_pwm: float = 0.0,
    right_pwm: float = 0.0,
    left_dir: str = "fwd",
    right_dir: str = "fwd",
    e_stop_active: bool = False,
    cmd_timeout_active: bool = False,
) -> dict[str, Any]:
    return {
        "ts": now_ts() if ts is None else ts,
        "left_pwm": left_pwm,
        "right_pwm": right_pwm,
        "left_dir": left_dir,
        "right_dir": right_dir,
        "e_stop_active": e_stop_active,
        "cmd_timeout_active": cmd_timeout_active,
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
    """Build body/oakd/imu JSON per body_project_spec.md §5.6 (sensor frame)."""
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


def oakd_depth_placeholder(ts: float | None = None) -> dict[str, Any]:
    return {"ts": now_ts() if ts is None else ts, "format": "placeholder", "note": "TBD per body_project_spec.md §5.7"}


def heartbeat(ts: float | None = None, seq: int = 0) -> dict[str, Any]:
    return {"ts": now_ts() if ts is None else ts, "seq": seq}


def status(
    processes: dict[str, str],
    heartbeat_ok: bool,
    e_stop_active: bool,
    uptime_s: float,
    ts: float | None = None,
) -> dict[str, Any]:
    return {
        "ts": now_ts() if ts is None else ts,
        "processes": processes,
        "heartbeat_ok": heartbeat_ok,
        "e_stop_active": e_stop_active,
        "uptime_s": uptime_s,
    }


def emergency_stop(reason: str, source: str = "watchdog", ts: float | None = None) -> dict[str, Any]:
    return {"ts": now_ts() if ts is None else ts, "reason": reason, "source": source}

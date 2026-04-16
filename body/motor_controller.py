"""Motor controller: stub publishes synthetic odom/motor_state; subscribes to motion and safety topics."""

from __future__ import annotations

import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from body.lib import diff_drive, schemas, zenoh_helpers


@dataclass
class MotionState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_twist: dict[str, Any] | None = None
    last_direct: dict[str, Any] | None = None
    last_cmd_wall_s: float = 0.0
    status_e_stop: bool = False
    e_stop_latched: bool = False
    awaiting_cmd_vel_after_clear: bool = False


def _print_cmd(kind: str, msg: dict[str, Any]) -> None:
    print(f"motor_controller: {kind} {msg}", flush=True)


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    motor_cfg = body_cfg.get("motor", {})
    loop_hz = int(motor_cfg.get("loop_hz", 50))
    period = 1.0 / max(1, loop_hz)
    wheel_base_m = float(motor_cfg.get("wheel_base_m", 0.0))
    wheel_radius_m = float(motor_cfg.get("wheel_radius_m", 0.0))
    ticks_per_rev = int(motor_cfg.get("ticks_per_rev", 1920))
    max_wheel_vel_ms = float(motor_cfg.get("max_wheel_vel_ms", 0.0))

    state = MotionState()
    session = zenoh_helpers.open_session(body_cfg)
    stop = threading.Event()

    def on_cmd_vel(_key: str, msg: dict[str, Any]) -> None:
        _print_cmd("cmd_vel", msg)
        with state.lock:
            state.last_twist = msg
            state.last_cmd_wall_s = time.time()
            if state.awaiting_cmd_vel_after_clear:
                state.e_stop_latched = False
                state.awaiting_cmd_vel_after_clear = False

    def on_cmd_direct(_key: str, msg: dict[str, Any]) -> None:
        _print_cmd("cmd_direct", msg)
        with state.lock:
            state.last_direct = msg
            state.last_cmd_wall_s = time.time()

    def on_emergency_stop(_key: str, msg: dict[str, Any]) -> None:
        print(f"motor_controller: emergency_stop {msg}", flush=True)
        with state.lock:
            state.e_stop_latched = True
            state.awaiting_cmd_vel_after_clear = False

    def on_status(_key: str, msg: dict[str, Any]) -> None:
        e_active = bool(msg.get("e_stop_active", False))
        with state.lock:
            state.status_e_stop = e_active
            if e_active:
                state.e_stop_latched = True
                state.awaiting_cmd_vel_after_clear = False
            else:
                if state.e_stop_latched:
                    state.awaiting_cmd_vel_after_clear = True

    zenoh_helpers.declare_subscriber_json(session, "body/cmd_vel", on_cmd_vel)
    zenoh_helpers.declare_subscriber_json(session, "body/cmd_direct", on_cmd_direct)
    zenoh_helpers.declare_subscriber_json(session, "body/emergency_stop", on_emergency_stop)
    zenoh_helpers.declare_subscriber_json(session, "body/status", on_status)

    pose = diff_drive.Pose(0.0, 0.0, 0.0)

    def handle_sigterm(_sig: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    next_tick = time.monotonic()
    while not stop.is_set():
        now_wall = time.time()
        with state.lock:
            twist = state.last_twist
            direct = state.last_direct
            cmd_wall = state.last_cmd_wall_s
            e_stop = state.e_stop_latched or state.status_e_stop
            direct_ts = float(direct["ts"]) if direct else 0.0
            use_direct = direct is not None and (twist is None or (twist is not None and direct_ts >= float(twist["ts"])))
            active = direct if use_direct else twist
            timeout_ms = int(active.get("timeout_ms", 500)) if active else 500
            cmd_stale = active is None or (now_wall - cmd_wall) * 1000.0 > timeout_ms
            cmd_timeout_active = not e_stop and cmd_stale

            left_pwm = right_pwm = 0.0
            left_dir = right_dir = "fwd"
            vx = vtheta = 0.0

            if not e_stop and not cmd_stale and active is not None:
                if use_direct and direct is not None:
                    lv = float(direct["left"])
                    rv = float(direct["right"])
                    left_pwm, left_dir = diff_drive.pwm_from_velocity(lv, max_wheel_vel_ms)
                    right_pwm, right_dir = diff_drive.pwm_from_velocity(rv, max_wheel_vel_ms)
                    vx = (lv + rv) / 2.0
                    vtheta = (rv - lv) / wheel_base_m if wheel_base_m > 0 else 0.0
                elif twist is not None:
                    lin = float(twist["linear"])
                    ang = float(twist["angular"])
                    vl, vr = diff_drive.twist_to_wheel_velocities(lin, ang, wheel_base_m)
                    left_pwm, left_dir = diff_drive.pwm_from_velocity(vl, max_wheel_vel_ms)
                    right_pwm, right_dir = diff_drive.pwm_from_velocity(vr, max_wheel_vel_ms)
                    vx = lin
                    vtheta = ang

        dt_ms = int(round(period * 1000))
        pose = diff_drive.integrate_odometry(
            pose,
            diff_drive.ticks_to_delta_m(0, wheel_radius_m, ticks_per_rev),
            diff_drive.ticks_to_delta_m(0, wheel_radius_m, ticks_per_rev),
            wheel_base_m,
        )
        ts = schemas.now_ts()
        zenoh_helpers.publish_json(
            session,
            "body/odom",
            schemas.odom(ts=ts, x=pose.x, y=pose.y, theta=pose.theta, vx=vx, vtheta=vtheta, dt_ms=dt_ms),
        )
        zenoh_helpers.publish_json(
            session,
            "body/motor_state",
            schemas.motor_state(
                ts=ts,
                left_pwm=left_pwm,
                right_pwm=right_pwm,
                left_dir=left_dir,
                right_dir=right_dir,
                e_stop_active=e_stop,
                cmd_timeout_active=cmd_timeout_active,
            ),
        )

        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()

    session.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

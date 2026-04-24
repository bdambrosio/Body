"""Motor controller: Zenoh cmd_vel/cmd_direct → PWM+DIR on MDD10A (optional GPIO) + odom/motor_state.

When ``motor.gpio_enabled`` is false, duty values are published only (no Pi GPIO). When true, the Pi
drives BCM pins per docs/motor_controller_spec.md (lgpio). Set ``max_wheel_vel_ms`` > 0 for nonzero duty.
"""

from __future__ import annotations

import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from body.lib import diff_drive, motor_gpio, schemas, zenoh_helpers

STALL_PWM_THRESHOLD = 0.1
STALL_VELOCITY_EPS_MS = 0.01


@dataclass
class WheelPI:
    """Per-wheel velocity PI with feed-forward and dead-zone snap.

    ``ff = v_cmd / max_v`` (the prior open-loop mapping) is the starting duty; the
    integrator absorbs the residual so commanded wheel velocity matches measured.
    ``min_drive_pwm`` snaps any nonzero output past the static-friction threshold
    so the loop doesn't spend its first ticks waiting for the integrator to wind up.
    """

    kp: float
    ki: float
    integ_limit: float
    min_drive_pwm: float
    integ: float = 0.0

    def reset(self) -> None:
        self.integ = 0.0

    def step(
        self, v_cmd: float, v_meas: float, dt: float, max_v: float
    ) -> tuple[float, str]:
        if max_v <= 0.0 or abs(v_cmd) < 1e-6:
            self.integ = 0.0
            return 0.0, "fwd"
        ff = v_cmd / max_v
        err = v_cmd - v_meas
        self.integ += err * dt
        if self.integ > self.integ_limit:
            self.integ = self.integ_limit
        elif self.integ < -self.integ_limit:
            self.integ = -self.integ_limit
        pwm = ff + self.kp * err + self.ki * self.integ
        if self.min_drive_pwm > 0.0 and abs(pwm) < self.min_drive_pwm:
            pwm = math.copysign(self.min_drive_pwm, v_cmd)
        if pwm > 1.0:
            pwm = 1.0
        elif pwm < -1.0:
            pwm = -1.0
        return abs(pwm), "fwd" if pwm >= 0 else "rev"


@dataclass
class MotionState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_twist: dict[str, Any] | None = None
    last_direct: dict[str, Any] | None = None
    last_cmd_wall_s: float = 0.0
    status_e_stop: bool = False
    e_stop_latched: bool = False
    awaiting_cmd_vel_after_clear: bool = False
    stall_latched: bool = False


def _print_cmd(kind: str, msg: dict[str, Any]) -> None:
    print(f"motor_controller: {kind} {msg}", flush=True)


def _command_is_all_stop(msg: dict[str, Any], *, direct: bool) -> bool:
    """True if the message commands no motion (clears software stall latch)."""
    if direct:
        return abs(float(msg.get("left", 0.0))) < 1e-6 and abs(float(msg.get("right", 0.0))) < 1e-6
    return abs(float(msg.get("linear", 0.0))) < 1e-6 and abs(float(msg.get("angular", 0.0))) < 1e-6


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    motor_cfg = body_cfg.get("motor", {})
    loop_hz = int(motor_cfg.get("loop_hz", 50))
    period = 1.0 / max(1, loop_hz)
    wheel_base_m = float(motor_cfg.get("wheel_base_m", 0.0))
    wheel_radius_m = float(motor_cfg.get("wheel_radius_m", 0.0))
    ticks_per_rev = int(motor_cfg.get("ticks_per_rev", 1920))
    max_wheel_vel_ms = float(motor_cfg.get("max_wheel_vel_ms", 0.0))
    stall_detect_enabled = bool(motor_cfg.get("stall_detect_enabled", False))
    stall_detect_ms = int(motor_cfg.get("stall_detect_ms", 1000))
    gpio_enabled = bool(motor_cfg.get("gpio_enabled", False))
    encoders_wanted = bool(motor_cfg.get("encoders_enabled", False))
    encoder_log_interval_s = float(motor_cfg.get("encoder_log_interval_s", 0.0))
    velocity_loop_enabled = bool(motor_cfg.get("velocity_loop_enabled", False))
    velocity_kp = float(motor_cfg.get("velocity_kp", 0.5))
    velocity_ki = float(motor_cfg.get("velocity_ki", 2.0))
    velocity_integ_limit = float(motor_cfg.get("velocity_integ_limit", 0.5))
    min_drive_pwm = float(motor_cfg.get("min_drive_pwm", 0.0))
    pi_left = WheelPI(velocity_kp, velocity_ki, velocity_integ_limit, min_drive_pwm)
    pi_right = WheelPI(velocity_kp, velocity_ki, velocity_integ_limit, min_drive_pwm)
    if encoders_wanted and not gpio_enabled:
        print(
            "motor_controller: encoders_enabled requires gpio_enabled — encoder GPIO skipped.",
            flush=True,
        )

    gpio_h: Any = None
    gpio_pins: dict[str, Any] | None = None
    encoders: dict[str, Any] | None = None
    both_encoders_live = False
    if gpio_enabled:
        gpio_h, gpio_pins = motor_gpio.open_mdd10a(motor_cfg)
        print(
            "motor_controller: GPIO enabled — MDD10A PWM+DIR on Pi (see motor.gpio_* in config)",
            flush=True,
        )
        if encoders_wanted:
            encoders = motor_gpio.setup_quadrature_encoders(gpio_h, motor_cfg)
            if encoders is not None:
                for side in ("left", "right"):
                    port = encoders.get(side)
                    if port is not None:
                        a, b = port.pins()
                        st = port.initial_state()
                        print(
                            f"motor_controller: {side} encoder on BCM A={a} B={b}; "
                            f"initial A/B={(st >> 1) & 1}/{st & 1}",
                            flush=True,
                        )
                both_encoders_live = (
                    encoders.get("left") is not None and encoders.get("right") is not None
                )
                if not both_encoders_live:
                    print(
                        "motor_controller: only one encoder enabled — odom.source stays "
                        "'commanded_vel_playback' (see docs/encoder_integration_spec.md §6).",
                        flush=True,
                    )
            else:
                print(
                    "motor_controller: no encoder device opened — check encoders_enabled and "
                    "encoder_left_enabled / encoder_right_enabled.",
                    flush=True,
                )
    if gpio_enabled and max_wheel_vel_ms <= 0.0:
        print(
            "motor_controller: warning: max_wheel_vel_ms is 0 — commanded duty will stay 0; set a positive value to move.",
            flush=True,
        )

    state = MotionState()
    session = zenoh_helpers.open_session(body_cfg)
    stop = threading.Event()
    stall_begin_wall: float | None = None
    left_ticks_total = 0
    right_ticks_total = 0
    encoder_log_last_mono = 0.0

    def on_cmd_vel(_key: str, msg: dict[str, Any]) -> None:
        _print_cmd("cmd_vel", msg)
        with state.lock:
            if state.stall_latched and _command_is_all_stop(msg, direct=False):
                state.stall_latched = False
            state.last_twist = msg
            state.last_cmd_wall_s = time.time()
            if state.awaiting_cmd_vel_after_clear:
                state.e_stop_latched = False
                state.awaiting_cmd_vel_after_clear = False

    def on_cmd_direct(_key: str, msg: dict[str, Any]) -> None:
        _print_cmd("cmd_direct", msg)
        with state.lock:
            if state.stall_latched and _command_is_all_stop(msg, direct=True):
                state.stall_latched = False
            state.last_direct = msg
            state.last_cmd_wall_s = time.time()
            if state.awaiting_cmd_vel_after_clear:
                state.e_stop_latched = False
                state.awaiting_cmd_vel_after_clear = False

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
    try:
        while not stop.is_set():
            now_wall = time.time()
            delta_left_ticks = 0
            delta_right_ticks = 0
            if encoders is not None:
                if encoders.get("left") is not None:
                    delta_left_ticks = int(encoders["left"].drain())
                if encoders.get("right") is not None:
                    delta_right_ticks = int(encoders["right"].drain())

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
                vx_cmd = vtheta_cmd = 0.0
                vl_cmd = vr_cmd = 0.0

                dl_m = diff_drive.ticks_to_delta_m(delta_left_ticks, wheel_radius_m, ticks_per_rev)
                dr_m = diff_drive.ticks_to_delta_m(delta_right_ticks, wheel_radius_m, ticks_per_rev)
                v_left_enc = dl_m / period if period > 0 else 0.0
                v_right_enc = dr_m / period if period > 0 else 0.0

                if not e_stop and not cmd_stale and active is not None:
                    if use_direct and direct is not None:
                        vl_cmd = float(direct["left"])
                        vr_cmd = float(direct["right"])
                        vx_cmd = (vl_cmd + vr_cmd) / 2.0
                        vtheta_cmd = (vr_cmd - vl_cmd) / wheel_base_m if wheel_base_m > 0 else 0.0
                    elif twist is not None:
                        lin = float(twist["linear"])
                        ang = float(twist["angular"])
                        vl_cmd, vr_cmd = diff_drive.twist_to_wheel_velocities(lin, ang, wheel_base_m)
                        vx_cmd = lin
                        vtheta_cmd = ang
                    if velocity_loop_enabled and both_encoders_live:
                        left_pwm, left_dir = pi_left.step(vl_cmd, v_left_enc, period, max_wheel_vel_ms)
                        right_pwm, right_dir = pi_right.step(vr_cmd, v_right_enc, period, max_wheel_vel_ms)
                    else:
                        left_pwm, left_dir = diff_drive.pwm_from_velocity(vl_cmd, max_wheel_vel_ms)
                        right_pwm, right_dir = diff_drive.pwm_from_velocity(vr_cmd, max_wheel_vel_ms)
                else:
                    pi_left.reset()
                    pi_right.reset()

                max_cmd_pwm = max(left_pwm, right_pwm)

                if stall_detect_enabled and not state.stall_latched and not e_stop and not cmd_stale and active is not None:
                    if max_cmd_pwm > STALL_PWM_THRESHOLD:
                        if max(abs(v_left_enc), abs(v_right_enc)) < STALL_VELOCITY_EPS_MS:
                            if stall_begin_wall is None:
                                stall_begin_wall = now_wall
                            elif (now_wall - stall_begin_wall) * 1000.0 >= float(stall_detect_ms):
                                state.stall_latched = True
                                stall_begin_wall = None
                        else:
                            stall_begin_wall = None
                    else:
                        stall_begin_wall = None
                else:
                    stall_begin_wall = None

                stall_active = state.stall_latched
                if stall_active:
                    left_pwm = right_pwm = 0.0
                    left_dir = right_dir = "fwd"
                    pi_left.reset()
                    pi_right.reset()

                if both_encoders_live:
                    vx = (v_left_enc + v_right_enc) / 2.0
                    vtheta = (v_right_enc - v_left_enc) / wheel_base_m if wheel_base_m > 0 else 0.0
                elif dl_m != 0.0 or dr_m != 0.0:
                    vx = (v_left_enc + v_right_enc) / 2.0
                    vtheta = (v_right_enc - v_left_enc) / wheel_base_m if wheel_base_m > 0 else 0.0
                else:
                    vx = vx_cmd
                    vtheta = vtheta_cmd

                if e_stop or cmd_stale or stall_active:
                    vx = 0.0
                    vtheta = 0.0

            if gpio_h is not None and gpio_pins is not None:
                motor_gpio.apply_outputs(gpio_h, gpio_pins, left_pwm, right_pwm, left_dir, right_dir)

            left_ticks_total += delta_left_ticks
            right_ticks_total += delta_right_ticks

            dt_ms = int(round(period * 1000))
            pose = diff_drive.integrate_odometry(pose, dl_m, dr_m, wheel_base_m)
            ts = schemas.now_ts()
            odom_source = (
                "wheel_encoders" if both_encoders_live else "commanded_vel_playback"
            )
            zenoh_helpers.publish_json(
                session,
                "body/odom",
                schemas.odom(
                    ts=ts,
                    x=pose.x,
                    y=pose.y,
                    theta=pose.theta,
                    vx=vx,
                    vtheta=vtheta,
                    left_ticks=left_ticks_total,
                    right_ticks=right_ticks_total,
                    dt_ms=dt_ms,
                    source=odom_source,
                ),
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
                    stall_detected=stall_active,
                ),
            )

            if encoders is not None and encoder_log_interval_s > 0.0:
                now_mono = time.monotonic()
                if now_mono - encoder_log_last_mono >= encoder_log_interval_s:
                    lp_port = encoders.get("left")
                    rp_port = encoders.get("right")
                    l_edges = lp_port.edge_count() if lp_port is not None else 0
                    r_edges = rp_port.edge_count() if rp_port is not None else 0
                    l_state = lp_port.current_state() if lp_port is not None else -1
                    r_state = rp_port.current_state() if rp_port is not None else -1
                    print(
                        "motor_controller: encoders "
                        f"left ticks={left_ticks_total} (Δ{delta_left_ticks}) edges={l_edges} A/B={(l_state >> 1) & 1 if l_state >= 0 else '-'}/{l_state & 1 if l_state >= 0 else '-'} | "
                        f"right ticks={right_ticks_total} (Δ{delta_right_ticks}) edges={r_edges} A/B={(r_state >> 1) & 1 if r_state >= 0 else '-'}/{r_state & 1 if r_state >= 0 else '-'}",
                        flush=True,
                    )
                    encoder_log_last_mono = now_mono

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    finally:
        session.close()
        if encoders is not None:
            motor_gpio.close_quadrature_encoders(encoders)
        if gpio_h is not None and gpio_pins is not None:
            motor_gpio.shutdown(gpio_h, gpio_pins)

    sys.exit(0)


if __name__ == "__main__":
    main()

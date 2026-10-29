"""Motor controller: Zenoh cmd_vel/cmd_direct → PWM+DIR on MDD10A (optional GPIO) + odom/motor_state.

When ``motor.gpio_enabled`` is false, duty values are published only (no Pi GPIO). When true, the Pi
drives BCM pins per docs/motor_controller_spec.md (lgpio). Set ``max_wheel_vel_ms`` > 0 for nonzero duty.
"""

from __future__ import annotations

import secrets
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from body.lib import diff_drive, motor_gpio, schemas, zenoh_helpers
from body.lib.wheel_control import (
    CommandArbiter,
    CommandError,
    EStopLatch,
    WheelPI,
    parse_cmd_direct,
    parse_cmd_vel,
)

STALL_PWM_THRESHOLD = 0.1
STALL_VELOCITY_EPS_MS = 0.01
# Consecutive failing control ticks before the process exits (launcher respawns).
TICK_ERROR_EXIT_COUNT = 250

__all__ = ["WheelPI", "main"]


@dataclass
class MotionState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    arbiter: CommandArbiter = field(default_factory=CommandArbiter)
    estop: EStopLatch = field(default_factory=EStopLatch)
    last_hb_mono: float = 0.0
    stall_latched: bool = False


class _RateLimitedLog:
    def __init__(self, period_s: float = 1.0) -> None:
        self.period_s = period_s
        self._last: dict[str, float] = {}

    def __call__(self, key: str, text: str) -> None:
        now = time.monotonic()
        if now - self._last.get(key, -1e9) >= self.period_s:
            self._last[key] = now
            print(text, flush=True)


def _wheel_param(motor_cfg: dict[str, Any], key: str, side: str, default: float) -> float:
    """Per-wheel override ``<key>_<side>`` falling back to ``<key>``."""
    v = motor_cfg.get(f"{key}_{side}")
    if v is None:
        v = motor_cfg.get(key, default)
    return float(v)


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
    velocity_loop_enabled = bool(motor_cfg.get("velocity_loop_enabled", False))
    velocity_kp = float(motor_cfg.get("velocity_kp", 0.5))
    velocity_ki = float(motor_cfg.get("velocity_ki", 2.0))
    velocity_integ_limit = float(motor_cfg.get("velocity_integ_limit", 0.5))
    velocity_integ_max_duty = float(motor_cfg.get("velocity_integ_max_duty", 0.2))
    kinetic_threshold_ms = float(motor_cfg.get("kinetic_threshold_ms", 0.02))
    static_threshold_ms = float(motor_cfg.get("static_threshold_ms", 0.5 * kinetic_threshold_ms))
    regime_confirm_ticks = int(motor_cfg.get("regime_confirm_ticks", 2))
    kick_ramp_ticks = int(motor_cfg.get("kick_ramp_ticks", 5))
    kick_escalation_per_s = float(motor_cfg.get("kick_escalation_per_s", 0.0))
    cmd_timeout_default_ms = float(motor_cfg.get("cmd_timeout_ms", 500))
    cmd_timeout_max_ms = float(motor_cfg.get("cmd_timeout_max_ms", 1000))
    hb_timeout_s = float(body_cfg.get("watchdog", {}).get("heartbeat_timeout_ms", 2000)) / 1000.0

    def _make_pi(side: str) -> WheelPI:
        return WheelPI(
            velocity_kp, velocity_ki, velocity_integ_limit,
            _wheel_param(motor_cfg, "min_drive_pwm", side, 0.0),
            kinetic_threshold_ms=kinetic_threshold_ms,
            kick_ramp_ticks=kick_ramp_ticks,
            kinetic_pwm=_wheel_param(motor_cfg, "ff_kinetic_pwm", side, 0.0),
            static_threshold_ms=static_threshold_ms,
            regime_confirm_ticks=regime_confirm_ticks,
            integ_max_duty=velocity_integ_max_duty,
            kick_escalation_per_s=kick_escalation_per_s,
        )

    pi_left = _make_pi("left")
    pi_right = _make_pi("right")
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
                        "motor_controller: only one encoder enabled — pose is NOT integrated, "
                        "odom.source='none' (see docs/encoder_integration_spec.md §6).",
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

    state = MotionState(arbiter=CommandArbiter(max_timeout_ms=cmd_timeout_max_ms))
    log_rl = _RateLimitedLog(1.0)
    stop = threading.Event()

    def _zero_pwm() -> None:
        if gpio_h is not None and gpio_pins is not None:
            try:
                motor_gpio.apply_outputs(gpio_h, gpio_pins, 0.0, 0.0, "fwd", "fwd")
            except Exception:
                motor_gpio.zero_outputs(gpio_pins)

    def handle_stop_signal(_sig: int, _frame: Any) -> None:
        stop.set()

    # SIGHUP too: a launcher started from an SSH session must not leave PWM running.
    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(_sig, handle_stop_signal)

    try:
        session = zenoh_helpers.open_session(body_cfg)
    except Exception:
        if gpio_h is not None and gpio_pins is not None:
            motor_gpio.shutdown(gpio_h, gpio_pins)
        raise

    def _on_motion(kind: str, msg: dict[str, Any]) -> None:
        try:
            if kind == "twist":
                a, b, timeout_ms = parse_cmd_vel(msg, cmd_timeout_default_ms)
            else:
                a, b, timeout_ms = parse_cmd_direct(msg, cmd_timeout_default_ms)
        except CommandError as e:
            topic = "body/cmd_vel" if kind == "twist" else "body/cmd_direct"
            log_rl(f"bad_{kind}", f"motor_controller: ignoring malformed {topic}: {e}")
            return
        now_mono = time.monotonic()
        with state.lock:
            if state.stall_latched and abs(a) < 1e-6 and abs(b) < 1e-6:
                state.stall_latched = False
            state.arbiter.store(kind, a, b, timeout_ms, now_mono)

    def on_cmd_vel(_key: str, msg: dict[str, Any]) -> None:
        _on_motion("twist", msg)

    def on_cmd_direct(_key: str, msg: dict[str, Any]) -> None:
        _on_motion("direct", msg)

    def on_emergency_stop(_key: str, msg: dict[str, Any]) -> None:
        print(f"motor_controller: emergency_stop {msg}", flush=True)
        with state.lock:
            state.estop.on_emergency_stop()

    def on_status(_key: str, msg: dict[str, Any]) -> None:
        with state.lock:
            state.estop.on_status(bool(msg.get("e_stop_active", False)))

    def on_heartbeat(_key: str, _msg: dict[str, Any]) -> None:
        with state.lock:
            state.last_hb_mono = time.monotonic()

    def on_estop_rearm(_key: str, msg: dict[str, Any]) -> None:
        with state.lock:
            hb_ok = state.last_hb_mono > 0.0 and (time.monotonic() - state.last_hb_mono) < hb_timeout_s
            was = state.estop.latched
            cleared = state.estop.on_rearm(hb_ok)
        if cleared:
            print(f"motor_controller: e-stop re-armed by {msg.get('source', '?')}", flush=True)
        elif was:
            print("motor_controller: estop_rearm refused — heartbeat not ok", flush=True)

    zenoh_helpers.declare_subscriber_json(session, "body/cmd_vel", on_cmd_vel)
    zenoh_helpers.declare_subscriber_json(session, "body/cmd_direct", on_cmd_direct)
    zenoh_helpers.declare_subscriber_json(session, "body/emergency_stop", on_emergency_stop)
    zenoh_helpers.declare_subscriber_json(session, "body/status", on_status)
    zenoh_helpers.declare_subscriber_json(session, "body/heartbeat", on_heartbeat)
    zenoh_helpers.declare_subscriber_json(session, "body/estop_rearm", on_estop_rearm)

    pose = diff_drive.Pose(0.0, 0.0, 0.0)
    session_id = secrets.token_hex(4)
    odom_seq = 0
    stall_begin_mono: float | None = None
    left_ticks_total = 0
    right_ticks_total = 0
    last_drain_mono: float | None = None
    last_pub_mono: float | None = None
    consecutive_errors = 0
    print(f"motor_controller: odom session_id={session_id}", flush=True)

    def tick() -> None:
        nonlocal pose, odom_seq, stall_begin_mono, left_ticks_total, right_ticks_total
        nonlocal last_drain_mono, last_pub_mono

        delta_left_ticks = 0
        delta_right_ticks = 0
        if encoders is not None:
            if encoders.get("left") is not None:
                delta_left_ticks = int(encoders["left"].drain())
            if encoders.get("right") is not None:
                delta_right_ticks = int(encoders["right"].drain())
        now_mono = time.monotonic()
        # Measured interval between drains (velocity), not the nominal period.
        dt = period if last_drain_mono is None else max(1e-4, now_mono - last_drain_mono)
        last_drain_mono = now_mono

        dl_m = diff_drive.ticks_to_delta_m(delta_left_ticks, wheel_radius_m, ticks_per_rev)
        dr_m = diff_drive.ticks_to_delta_m(delta_right_ticks, wheel_radius_m, ticks_per_rev)
        v_left_enc = dl_m / dt
        v_right_enc = dr_m / dt

        with state.lock:
            active = state.arbiter.select(now_mono)
            e_stop = state.estop.active
            cmd_stale = active is None
            cmd_timeout_active = not e_stop and cmd_stale

            left_pwm = right_pwm = 0.0
            left_dir = right_dir = "fwd"
            vx_cmd = vtheta_cmd = 0.0
            vl_cmd = vr_cmd = 0.0

            if not e_stop and active is not None:
                if active.kind == "direct":
                    vl_cmd, vr_cmd = active.a, active.b
                    vx_cmd = (vl_cmd + vr_cmd) / 2.0
                    vtheta_cmd = (vr_cmd - vl_cmd) / wheel_base_m if wheel_base_m > 0 else 0.0
                else:
                    vl_cmd, vr_cmd = diff_drive.twist_to_wheel_velocities(active.a, active.b, wheel_base_m)
                    vx_cmd, vtheta_cmd = active.a, active.b
                if velocity_loop_enabled and both_encoders_live:
                    left_pwm, left_dir = pi_left.step(vl_cmd, v_left_enc, dt, max_wheel_vel_ms)
                    right_pwm, right_dir = pi_right.step(vr_cmd, v_right_enc, dt, max_wheel_vel_ms)
                else:
                    left_pwm, left_dir = diff_drive.pwm_from_velocity(vl_cmd, max_wheel_vel_ms)
                    right_pwm, right_dir = diff_drive.pwm_from_velocity(vr_cmd, max_wheel_vel_ms)
            else:
                pi_left.reset()
                pi_right.reset()

            max_cmd_pwm = max(left_pwm, right_pwm)

            if stall_detect_enabled and not state.stall_latched and not e_stop and active is not None:
                if max_cmd_pwm > STALL_PWM_THRESHOLD and max(abs(v_left_enc), abs(v_right_enc)) < STALL_VELOCITY_EPS_MS:
                    if stall_begin_mono is None:
                        stall_begin_mono = now_mono
                    elif (now_mono - stall_begin_mono) * 1000.0 >= float(stall_detect_ms):
                        state.stall_latched = True
                        stall_begin_mono = None
                else:
                    stall_begin_mono = None
            else:
                stall_begin_mono = None

            stall_active = state.stall_latched
            if stall_active:
                left_pwm = right_pwm = 0.0
                left_dir = right_dir = "fwd"
                pi_left.reset()
                pi_right.reset()

        if gpio_h is not None and gpio_pins is not None:
            motor_gpio.apply_outputs(gpio_h, gpio_pins, left_pwm, right_pwm, left_dir, right_dir)

        left_ticks_total += delta_left_ticks
        right_ticks_total += delta_right_ticks

        if both_encoders_live:
            # Always the measured wheel motion (coasting / pushed robot is real motion).
            vx = (v_left_enc + v_right_enc) / 2.0
            vtheta = (v_right_enc - v_left_enc) / wheel_base_m if wheel_base_m > 0 else 0.0
            pose = diff_drive.integrate_odometry(pose, dl_m, dr_m, wheel_base_m)
            odom_source = "wheel_encoders"
        else:
            # No (or one) encoder: never integrate a one-sided pose. Report the
            # commanded velocity only while a command is actually being executed.
            moving = not (e_stop or cmd_stale or stall_active)
            vx = vx_cmd if moving else 0.0
            vtheta = vtheta_cmd if moving else 0.0
            odom_source = "none"

        pub_mono = time.monotonic()
        dt_ms = period * 1000.0 if last_pub_mono is None else (pub_mono - last_pub_mono) * 1000.0
        last_pub_mono = pub_mono
        ts = schemas.now_ts()
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
                dt_ms=round(dt_ms, 2),
                source=odom_source,
                session_id=session_id,
                seq=odom_seq,
            ),
        )
        odom_seq += 1
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

    next_tick = time.monotonic()
    try:
        while not stop.is_set():
            try:
                tick()
                consecutive_errors = 0
            except Exception:
                # Never leave the motors driven on an unexpected error: zero PWM,
                # reset the loops, log, keep ticking (exit if it keeps failing).
                _zero_pwm()
                pi_left.reset()
                pi_right.reset()
                consecutive_errors += 1
                log_rl("tick_error", f"motor_controller: control tick failed (PWM zeroed):\n{traceback.format_exc()}")
                if consecutive_errors >= TICK_ERROR_EXIT_COUNT:
                    print("motor_controller: too many consecutive tick errors — exiting", flush=True)
                    raise

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    finally:
        # PWM off FIRST — session.close() can block; the motors must not wait on it.
        if gpio_h is not None and gpio_pins is not None:
            try:
                motor_gpio.shutdown(gpio_h, gpio_pins)
            except Exception:
                motor_gpio.zero_outputs(gpio_pins)
        if encoders is not None:
            try:
                motor_gpio.close_quadrature_encoders(encoders)
            except Exception:
                pass
        try:
            session.close()
        except Exception:
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()

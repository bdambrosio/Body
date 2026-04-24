#!/usr/bin/env python3
"""Motor / encoder / velocity-PID acceptance tests.

Interactive harness that drives ``body/cmd_direct`` at moderate wheel velocities
and cross-checks per-wheel measured velocity from ``body/odom`` encoder ticks
against the command. Validates §6 of ``docs/encoder_integration_spec.md``
(stationary stability, forward/reverse tick signs, left-vs-right balance) plus
the closed-loop WheelPI controller from ``motor_controller.py`` (each wheel
tracks its own commanded velocity independently, including asymmetric
left-only / right-only commands).

Assumes ``motor_controller`` (and ``watchdog``) are already running, e.g. via
``python3 -m body.launcher``. Encoders must be live — tests bail out if
``odom.source`` is not ``"wheel_encoders"``.

Run:
    sudo PYTHONPATH=. .venv/bin/python3 scripts/motor_acceptance.py

Make sure the robot has room to move or put it on blocks — forward/reverse
tests command 0.2 m/s for 2 s each by default.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from body.lib import diff_drive, schemas, zenoh_helpers

STATIONARY_TICK_BOUND = 4
LEFT_RIGHT_BALANCE_FRAC = 0.20
PID_TRACKING_FRAC = 0.30
OFF_WHEEL_ABS_VEL_MS = 0.03
SETTLE_S = 0.5


@dataclass
class OdomSample:
    ts: float
    left_ticks: int
    right_ticks: int
    source: str
    vx: float


@dataclass
class MotorStateSample:
    ts: float
    left_pwm: float
    right_pwm: float
    left_dir: str
    right_dir: str
    stall_detected: bool
    e_stop_active: bool
    cmd_timeout_active: bool


@dataclass
class Collector:
    lock: threading.Lock = field(default_factory=threading.Lock)
    odom: list[OdomSample] = field(default_factory=list)
    motor: list[MotorStateSample] = field(default_factory=list)

    def on_odom(self, _key: str, msg: dict[str, Any]) -> None:
        try:
            sample = OdomSample(
                ts=float(msg["ts"]),
                left_ticks=int(msg["left_ticks"]),
                right_ticks=int(msg["right_ticks"]),
                source=str(msg.get("source", "")),
                vx=float(msg.get("vx", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as e:
            print(f"  [warn] bad body/odom payload: {e}", file=sys.stderr, flush=True)
            return
        with self.lock:
            self.odom.append(sample)

    def on_motor_state(self, _key: str, msg: dict[str, Any]) -> None:
        try:
            sample = MotorStateSample(
                ts=float(msg["ts"]),
                left_pwm=float(msg.get("left_pwm", 0.0)),
                right_pwm=float(msg.get("right_pwm", 0.0)),
                left_dir=str(msg.get("left_dir", "fwd")),
                right_dir=str(msg.get("right_dir", "fwd")),
                stall_detected=bool(msg.get("stall_detected", False)),
                e_stop_active=bool(msg.get("e_stop_active", False)),
                cmd_timeout_active=bool(msg.get("cmd_timeout_active", False)),
            )
        except (KeyError, TypeError, ValueError):
            return
        with self.lock:
            self.motor.append(sample)

    def odom_since(self, since_ts: float) -> list[OdomSample]:
        with self.lock:
            return [s for s in self.odom if s.ts >= since_ts]

    def motor_since(self, since_ts: float) -> list[MotorStateSample]:
        with self.lock:
            return [s for s in self.motor if s.ts >= since_ts]

    def wait_for_odom(self, timeout_s: float) -> OdomSample | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            with self.lock:
                if self.odom:
                    return self.odom[-1]
            time.sleep(0.05)
        return None


def _prompt(msg: str) -> None:
    print()
    print("=" * 72)
    print(msg)
    print("=" * 72)
    input("Press Enter to continue (or Ctrl-C to abort)... ")


def _wait_estop_clear(
    session: Any, status_ref: dict[str, Any], timeout_s: float = 5.0
) -> bool:
    """Block until body/status shows heartbeat_ok and cleared e_stop."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        st = status_ref.get("latest")
        if isinstance(st, dict) and st.get("heartbeat_ok") and not st.get("e_stop_active"):
            return True
        zenoh_helpers.publish_json(
            session,
            "body/cmd_direct",
            schemas.cmd_direct(left=0.0, right=0.0, timeout_ms=500),
        )
        time.sleep(0.2)
    return False


def _drive(
    session: Any,
    left_cmd: float,
    right_cmd: float,
    duration_s: float,
    tick_hz: float = 10.0,
) -> float:
    """Publish ``cmd_direct`` at ``tick_hz`` for ``duration_s``. Return start wall-time."""
    t_start = time.time()
    end = t_start + duration_s
    period = 1.0 / tick_hz
    while time.time() < end:
        zenoh_helpers.publish_json(
            session,
            "body/cmd_direct",
            schemas.cmd_direct(left=left_cmd, right=right_cmd, timeout_ms=500),
        )
        time.sleep(period)
    zenoh_helpers.publish_json(
        session,
        "body/cmd_direct",
        schemas.cmd_direct(left=0.0, right=0.0, timeout_ms=500),
    )
    return t_start


def _per_wheel_mean_velocity(
    samples: list[OdomSample],
    wheel_radius_m: float,
    ticks_per_rev: int,
    since_ts: float | None = None,
) -> tuple[float, float, int, int]:
    """Return (mean_v_left, mean_v_right, delta_left_ticks, delta_right_ticks).

    Integrates from the first sample at/after ``since_ts`` (default: first sample)
    to the last sample. Mean velocity is (distance / elapsed) per wheel.
    """
    if len(samples) < 2:
        return 0.0, 0.0, 0, 0
    sel = samples if since_ts is None else [s for s in samples if s.ts >= since_ts]
    if len(sel) < 2:
        return 0.0, 0.0, 0, 0
    first, last = sel[0], sel[-1]
    dt = last.ts - first.ts
    if dt <= 0.0:
        return 0.0, 0.0, 0, 0
    dl_ticks = last.left_ticks - first.left_ticks
    dr_ticks = last.right_ticks - first.right_ticks
    dl_m = diff_drive.ticks_to_delta_m(dl_ticks, wheel_radius_m, ticks_per_rev)
    dr_m = diff_drive.ticks_to_delta_m(dr_ticks, wheel_radius_m, ticks_per_rev)
    return dl_m / dt, dr_m / dt, dl_ticks, dr_ticks


def _check_encoders_live(collector: Collector, timeout_s: float = 5.0) -> bool:
    sample = collector.wait_for_odom(timeout_s)
    if sample is None:
        print("  FAIL — no body/odom in 5 s. Is motor_controller running?")
        return False
    if sample.source != "wheel_encoders":
        print(
            f"  FAIL — odom.source = {sample.source!r}, expected 'wheel_encoders'. "
            "Enable motor.encoders_enabled (left+right) and rerun motor_controller.",
        )
        return False
    return True


def _test_stationary(collector: Collector, duration_s: float) -> bool:
    """§6(1) — stationary: ticks stay constant (quadrature dither only)."""
    _prompt(
        f"TEST 1 — Stationary ({duration_s:.0f} s).\n"
        "Place the robot on a flat surface, hands off, wheels free. No command\n"
        "will be sent. Ticks should stay constant (±1 from quadrature dither)."
    )
    t_start = time.time()
    print(f"  recording for {duration_s:.0f} s...", flush=True)
    time.sleep(duration_s)
    samples = collector.odom_since(t_start)
    if len(samples) < 5:
        print(f"  FAIL — only {len(samples)} odom samples captured; is motor_controller publishing?")
        return False
    first, last = samples[0], samples[-1]
    dl = last.left_ticks - first.left_ticks
    dr = last.right_ticks - first.right_ticks
    ok_left = abs(dl) <= STATIONARY_TICK_BOUND
    ok_right = abs(dr) <= STATIONARY_TICK_BOUND
    print()
    print(f"  samples:          {len(samples)}")
    print(
        f"  left Δticks:      {dl:+d}   (bound |Δ|≤{STATIONARY_TICK_BOUND}) "
        f"{'PASS' if ok_left else 'FAIL'}"
    )
    print(
        f"  right Δticks:     {dr:+d}   (bound |Δ|≤{STATIONARY_TICK_BOUND}) "
        f"{'PASS' if ok_right else 'FAIL'}"
    )
    return ok_left and ok_right


def _test_directional(
    collector: Collector,
    session: Any,
    status_ref: dict[str, Any],
    label: str,
    left_cmd: float,
    right_cmd: float,
    duration_s: float,
    wheel_radius_m: float,
    ticks_per_rev: int,
    check_sign: bool = True,
) -> bool:
    """Drive (left_cmd, right_cmd) for ``duration_s`` and validate per-wheel tracking.

    Passes when:
      - ticks moved in the commanded sign for each driven wheel (check_sign)
      - steady-state measured wheel velocity within PID_TRACKING_FRAC of cmd
      - off wheels (cmd≈0) show |mean v| < OFF_WHEEL_ABS_VEL_MS
      - when BOTH wheels are driven the same way, |v_left−v_right|/max < LEFT_RIGHT_BALANCE_FRAC
      - no stall latched during the window
    """
    _prompt(
        f"TEST {label}\n"
        f"Commanded: left={left_cmd:+.2f} m/s, right={right_cmd:+.2f} m/s for "
        f"{duration_s:.1f} s.\nEnsure the robot has clear floor space (or wheels up)."
    )
    if not _wait_estop_clear(session, status_ref, timeout_s=5.0):
        st = status_ref.get("latest", {})
        print(
            f"  FAIL — watchdog still latched: heartbeat_ok={st.get('heartbeat_ok')}, "
            f"e_stop_active={st.get('e_stop_active')}."
        )
        return False
    print("  driving...", flush=True)
    t_start = _drive(session, left_cmd, right_cmd, duration_s)
    samples = collector.odom_since(t_start)
    motor_samples = collector.motor_since(t_start)
    if len(samples) < 5:
        print(f"  FAIL — only {len(samples)} odom samples captured in {duration_s:.1f} s.")
        return False

    total_vl, total_vr, dl_ticks, dr_ticks = _per_wheel_mean_velocity(
        samples, wheel_radius_m, ticks_per_rev
    )
    settle_vl, settle_vr, _, _ = _per_wheel_mean_velocity(
        samples, wheel_radius_m, ticks_per_rev, since_ts=t_start + SETTLE_S
    )

    stalled = any(m.stall_detected for m in motor_samples)
    e_stop_hit = any(m.e_stop_active for m in motor_samples)
    timeout_hit = any(m.cmd_timeout_active for m in motor_samples)

    def _tracking_ok(cmd: float, meas: float) -> tuple[bool, str]:
        if abs(cmd) < 1e-6:
            ok = abs(meas) < OFF_WHEEL_ABS_VEL_MS
            return ok, f"|meas|<{OFF_WHEEL_ABS_VEL_MS}"
        err = abs(meas - cmd) / abs(cmd)
        return err < PID_TRACKING_FRAC, f"|err|/|cmd|={err:.2f}<{PID_TRACKING_FRAC}"

    def _sign_ok(cmd: float, delta_ticks: int) -> tuple[bool, str]:
        if abs(cmd) < 1e-6:
            return True, "cmd≈0 (n/a)"
        if cmd > 0:
            return delta_ticks > 0, "Δticks>0"
        return delta_ticks < 0, "Δticks<0"

    left_sign_ok, left_sign_desc = _sign_ok(left_cmd, dl_ticks)
    right_sign_ok, right_sign_desc = _sign_ok(right_cmd, dr_ticks)
    left_track_ok, left_track_desc = _tracking_ok(left_cmd, settle_vl)
    right_track_ok, right_track_desc = _tracking_ok(right_cmd, settle_vr)

    both_driven_same_sign = (
        abs(left_cmd) > 1e-6
        and abs(right_cmd) > 1e-6
        and (left_cmd > 0) == (right_cmd > 0)
    )
    balance_ok = True
    balance_desc = "n/a (asymmetric or single-wheel cmd)"
    if both_driven_same_sign:
        denom = max(abs(settle_vl), abs(settle_vr))
        if denom < 1e-6:
            balance_ok = False
            balance_desc = "both wheels stopped"
        else:
            frac = abs(settle_vl - settle_vr) / denom
            balance_ok = frac < LEFT_RIGHT_BALANCE_FRAC
            balance_desc = f"|L-R|/max={frac:.2f}<{LEFT_RIGHT_BALANCE_FRAC}"

    print()
    print(f"  odom samples:     {len(samples)}")
    print(f"  left ticks Δ:     {dl_ticks:+d}")
    print(f"  right ticks Δ:    {dr_ticks:+d}")
    print(
        f"  left  v (window): total={total_vl:+.3f}  settle={settle_vl:+.3f}  "
        f"cmd={left_cmd:+.3f} m/s"
    )
    print(
        f"  right v (window): total={total_vr:+.3f}  settle={settle_vr:+.3f}  "
        f"cmd={right_cmd:+.3f} m/s"
    )
    if check_sign:
        print(f"  left direction:   {left_sign_desc}  {'PASS' if left_sign_ok else 'FAIL'}")
        print(f"  right direction:  {right_sign_desc}  {'PASS' if right_sign_ok else 'FAIL'}")
    print(f"  left tracking:    {left_track_desc}  {'PASS' if left_track_ok else 'FAIL'}")
    print(f"  right tracking:   {right_track_desc}  {'PASS' if right_track_ok else 'FAIL'}")
    print(f"  L/R balance:      {balance_desc}  {'PASS' if balance_ok else 'FAIL'}")
    print(f"  stall_detected:   {stalled}  {'PASS' if not stalled else 'FAIL'}")
    if e_stop_hit:
        print("  e_stop fired during window — FAIL")
    if timeout_hit:
        print("  cmd_timeout fired during window — FAIL")
    ok = (
        (not check_sign or (left_sign_ok and right_sign_ok))
        and left_track_ok
        and right_track_ok
        and balance_ok
        and not stalled
        and not e_stop_hit
        and not timeout_hit
    )
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Motor/encoder/PID acceptance tests (see docs/encoder_integration_spec.md §6).",
    )
    ap.add_argument("--velocity", type=float, default=0.2,
                    help="Wheel velocity magnitude for motion tests, m/s (default: 0.2).")
    ap.add_argument("--duration", type=float, default=2.0,
                    help="Motion test duration in seconds (default: 2.0).")
    ap.add_argument("--stationary-s", type=float, default=5.0,
                    help="Stationary test duration in seconds (default: 5.0).")
    ap.add_argument("--skip-stationary", action="store_true")
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--skip-reverse", action="store_true")
    ap.add_argument("--skip-left-only", action="store_true")
    ap.add_argument("--skip-right-only", action="store_true")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    body_cfg = zenoh_helpers.load_body_config()
    motor_cfg = body_cfg.get("motor", {})
    wheel_radius_m = float(motor_cfg.get("wheel_radius_m", 0.0))
    ticks_per_rev = int(motor_cfg.get("ticks_per_rev", 1920))
    max_wheel_vel_ms = float(motor_cfg.get("max_wheel_vel_ms", 0.0))
    velocity_loop_enabled = bool(motor_cfg.get("velocity_loop_enabled", False))
    if wheel_radius_m <= 0.0 or ticks_per_rev <= 0:
        print(
            "FAIL — motor.wheel_radius_m / motor.ticks_per_rev must be set in config.json.",
            file=sys.stderr,
        )
        return 2
    if max_wheel_vel_ms > 0.0 and abs(args.velocity) > max_wheel_vel_ms:
        print(
            f"FAIL — --velocity {args.velocity:.3f} exceeds motor.max_wheel_vel_ms "
            f"{max_wheel_vel_ms:.3f}; pick a smaller magnitude.",
            file=sys.stderr,
        )
        return 2
    if not velocity_loop_enabled:
        print(
            "  [note] motor.velocity_loop_enabled = false — PID tracking bounds will\n"
            "         be evaluated on the open-loop v_cmd/max_wheel_vel_ms mapping.",
            flush=True,
        )

    session = zenoh_helpers.open_session(body_cfg)
    collector = Collector()
    odom_sub = zenoh_helpers.declare_subscriber_json(session, "body/odom", collector.on_odom)
    motor_sub = zenoh_helpers.declare_subscriber_json(
        session, "body/motor_state", collector.on_motor_state
    )

    status_ref: dict[str, Any] = {"latest": None}

    def on_status(_key: str, msg: dict[str, Any]) -> None:
        status_ref["latest"] = msg

    status_sub = zenoh_helpers.declare_subscriber_json(session, "body/status", on_status)

    hb_stop = threading.Event()
    hb_seq = 0

    def _hb_loop() -> None:
        nonlocal hb_seq
        while not hb_stop.is_set():
            hb_seq += 1
            try:
                zenoh_helpers.publish_json(
                    session, "body/heartbeat", schemas.heartbeat(seq=hb_seq)
                )
            except Exception:
                return
            hb_stop.wait(0.5)

    hb_thread = threading.Thread(target=_hb_loop, daemon=True)
    hb_thread.start()

    print("waiting for first body/odom (motor_controller up, encoders live)...", flush=True)
    if not _check_encoders_live(collector, timeout_s=5.0):
        hb_stop.set()
        odom_sub.undeclare()
        motor_sub.undeclare()
        status_sub.undeclare()
        session.close()
        return 2
    print(
        f"  wheel_radius_m={wheel_radius_m:.4f}  ticks_per_rev={ticks_per_rev}  "
        f"max_wheel_vel_ms={max_wheel_vel_ms:.2f}  "
        f"velocity_loop_enabled={velocity_loop_enabled}"
    )

    v = abs(args.velocity)
    results: dict[str, bool] = {}
    try:
        if not args.skip_stationary:
            results["1_stationary"] = _test_stationary(collector, args.stationary_s)
        else:
            print("(skipping test 1 per --skip-stationary)")
        if not args.skip_forward:
            results["2_forward"] = _test_directional(
                collector, session, status_ref,
                f"2 — Forward ({v:+.2f} m/s, both wheels)",
                +v, +v, args.duration, wheel_radius_m, ticks_per_rev,
            )
        else:
            print("(skipping test 2 per --skip-forward)")
        if not args.skip_reverse:
            results["3_reverse"] = _test_directional(
                collector, session, status_ref,
                f"3 — Reverse ({-v:+.2f} m/s, both wheels)",
                -v, -v, args.duration, wheel_radius_m, ticks_per_rev,
            )
        else:
            print("(skipping test 3 per --skip-reverse)")
        if not args.skip_left_only:
            results["4_left_only"] = _test_directional(
                collector, session, status_ref,
                f"4 — Left wheel only (left={+v:+.2f}, right=0.00)",
                +v, 0.0, args.duration, wheel_radius_m, ticks_per_rev,
            )
        else:
            print("(skipping test 4 per --skip-left-only)")
        if not args.skip_right_only:
            results["5_right_only"] = _test_directional(
                collector, session, status_ref,
                f"5 — Right wheel only (left=0.00, right={+v:+.2f})",
                0.0, +v, args.duration, wheel_radius_m, ticks_per_rev,
            )
        else:
            print("(skipping test 5 per --skip-right-only)")
    except KeyboardInterrupt:
        print("\naborted by operator.", file=sys.stderr)
    finally:
        try:
            zenoh_helpers.publish_json(
                session,
                "body/cmd_direct",
                schemas.cmd_direct(left=0.0, right=0.0, timeout_ms=500),
            )
        except Exception:
            pass
        hb_stop.set()
        odom_sub.undeclare()
        motor_sub.undeclare()
        status_sub.undeclare()
        session.close()

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, ok in results.items():
        print(f"  {name:<18} {'PASS' if ok else 'FAIL'}")
    all_ok = bool(results) and all(results.values())
    print(f"  overall:           {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

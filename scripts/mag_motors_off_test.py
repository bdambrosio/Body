#!/usr/bin/env python3
"""Test magnetometer snapshots while motors are idle (mag-when-idle driver mode).

Requires Pi ``imu_driver`` with ``imu.mag_when_idle_enabled: true`` while keeping
``fusion_mode: game_rotation_vector``. The driver publishes ``body/imu.orientation``
from Game RV continuously and a ``mag`` block (rotation_vector quaternion) only when
``body/motor_state`` shows PWM below threshold for ``mag_idle_settle_ms``.

Run from desktop (or any machine with Zenoh route to the Pi):

    PYTHONPATH=. python3 scripts/mag_motors_off_test.py --router tcp/192.168.8.60:7447

Phases: idle → motor spin → idle. Pass if mag is valid with good accuracy when
stopped and invalid (or absent) while spinning.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from body.lib import schemas, zenoh_helpers


def _yaw_from_wxyz(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def _wrap_pi(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


@dataclass
class Sample:
    ts: float
    game_yaw: float
    game_accuracy_rad: float
    mag_present: bool
    mag_valid: bool
    mag_yaw: float | None
    mag_accuracy_rad: float | None
    left_pwm: float
    right_pwm: float


@dataclass
class Collector:
    lock: threading.Lock = field(default_factory=threading.Lock)
    samples: list[Sample] = field(default_factory=list)
    latest: Sample | None = None
    motor_ref: dict[str, float] = field(
        default_factory=lambda: {"left_pwm": 0.0, "right_pwm": 0.0}
    )

    def on_imu(self, _key: str, msg: dict[str, Any]) -> None:
        try:
            ts = float(msg["ts"])
            q = msg["orientation"]
            fusion = msg["fusion"]
            mag = msg.get("mag")
            mag_present = isinstance(mag, dict)
            mag_block = mag if mag_present else {}
            mag_valid = bool(mag_block.get("valid", False))
            mag_yaw: float | None = None
            mag_accuracy: float | None = None
            if mag_valid and isinstance(mag_block.get("orientation"), dict):
                mq = mag_block["orientation"]
                mag_yaw = _yaw_from_wxyz(
                    float(mq["w"]), float(mq["x"]), float(mq["y"]), float(mq["z"])
                )
                if "accuracy_rad" in mag_block:
                    mag_accuracy = float(mag_block["accuracy_rad"])
            sample = Sample(
                ts=ts,
                game_yaw=_yaw_from_wxyz(
                    float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])
                ),
                game_accuracy_rad=float(fusion["accuracy_rad"]),
                mag_present=mag_present,
                mag_valid=mag_valid,
                mag_yaw=mag_yaw,
                mag_accuracy_rad=mag_accuracy,
                left_pwm=float(self.motor_ref["left_pwm"]),
                right_pwm=float(self.motor_ref["right_pwm"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            print(f"  [warn] bad body/imu payload: {e}", file=sys.stderr, flush=True)
            return
        with self.lock:
            self.samples.append(sample)
            self.latest = sample

    def on_motor(self, _key: str, msg: dict[str, Any]) -> None:
        try:
            self.motor_ref["left_pwm"] = abs(float(msg.get("left_pwm", 0.0)))
            self.motor_ref["right_pwm"] = abs(float(msg.get("right_pwm", 0.0)))
        except (TypeError, ValueError):
            return

    def snapshot_since(self, since_ts: float) -> list[Sample]:
        with self.lock:
            return [s for s in self.samples if s.ts >= since_ts]

    def wait_for_first(self, timeout_s: float) -> Sample | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            with self.lock:
                if self.latest is not None:
                    return self.latest
            time.sleep(0.05)
        return None


def _prompt(msg: str) -> None:
    print()
    print("=" * 72)
    print(msg)
    print("=" * 72)
    input("Press Enter to continue (or Ctrl-C to abort)... ")


def _phase_stats(samples: list[Sample], label: str) -> dict[str, Any]:
    if not samples:
        print(f"  {label}: no samples")
        return {"ok": False}
    mag_valid = [s for s in samples if s.mag_valid]
    mag_present = [s for s in samples if s.mag_present]
    mag_frac = len(mag_valid) / len(samples)
    acc_deg: list[float] = []
    idle_acc_deg: list[float] = []
    game_mag_delta_deg: list[float] = []
    for s in mag_present:
        if s.mag_accuracy_rad is not None:
            idle_acc_deg.append(_deg(s.mag_accuracy_rad))
    for s in mag_valid:
        if s.mag_accuracy_rad is not None:
            acc_deg.append(_deg(s.mag_accuracy_rad))
        if s.mag_yaw is not None:
            game_mag_delta_deg.append(_deg(_wrap_pi(s.game_yaw - s.mag_yaw)))
    acc_max = max(acc_deg) if acc_deg else float("nan")
    acc_med = sorted(acc_deg)[len(acc_deg) // 2] if acc_deg else float("nan")
    idle_acc_max = max(idle_acc_deg) if idle_acc_deg else float("nan")
    delta_med = sorted(game_mag_delta_deg)[len(game_mag_delta_deg) // 2] if game_mag_delta_deg else float("nan")
    delta_abs_med = abs(delta_med) if game_mag_delta_deg else float("nan")
    print(
        f"  {label}: n={len(samples)}  mag_valid={len(mag_valid)} ({100.0 * mag_frac:.0f}%)"
    )
    if mag_present and not mag_valid and idle_acc_max >= 90.0:
        print(
            f"           mag accuracy ~{idle_acc_max:.0f}° while idle → magnetometer not calibrated.\n"
            "           Run figure-8 cal (motors off): body.cli imu calibrate start/save"
        )
    elif mag_valid:
        print(
            f"           mag accuracy max={acc_max:.2f}° median={acc_med:.2f}°  "
            f"|game−mag| median={delta_abs_med:.2f}° (offset expected; game RV is relative)"
        )
    return {
        "ok": True,
        "n": len(samples),
        "mag_frac": mag_frac,
        "acc_max_deg": acc_max,
        "acc_med_deg": acc_med,
        "game_mag_delta_med_deg": delta_abs_med,
    }


def _wait_estop_clear(
    session: Any, status_ref: dict[str, Any], timeout_s: float = 5.0
) -> bool:
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


def _run_phase(
    label: str,
    duration_s: float,
    collector: Collector,
    session: Any | None = None,
    duty: float = 0.0,
    status_ref: dict[str, Any] | None = None,
    rotate_in_place: bool = True,
) -> list[Sample]:
    print(f"\n--- {label} ({duration_s:.0f} s) ---", flush=True)
    if duty > 0.0:
        if session is None or status_ref is None:
            raise ValueError("motor spin requires session and status_ref")
        if not _wait_estop_clear(session, status_ref):
            print("  FAIL — watchdog e-stop still latched; publish heartbeat first.")
            return []
    t_start = time.time()
    end = t_start + duration_s
    while time.time() < end:
        if duty > 0.0 and session is not None:
            left = duty
            right = -duty if rotate_in_place else duty
            zenoh_helpers.publish_json(
                session,
                "body/cmd_direct",
                schemas.cmd_direct(left=left, right=right, timeout_ms=500),
            )
        time.sleep(0.1)
    if duty > 0.0 and session is not None:
        zenoh_helpers.publish_json(
            session,
            "body/cmd_direct",
            schemas.cmd_direct(left=0.0, right=0.0, timeout_ms=500),
        )
    return collector.snapshot_since(t_start)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mag-when-idle BNO085 validation harness.")
    ap.add_argument("--router", default=None, help="Zenoh connect endpoint, e.g. tcp/192.168.8.60:7447")
    ap.add_argument("--idle-s", type=float, default=8.0, help="Idle phase duration (default: 8).")
    ap.add_argument("--motor-s", type=float, default=4.0, help="Motor spin duration (default: 4).")
    ap.add_argument("--motor-duty", type=float, default=0.12, help="Spin duty (default: 0.12).")
    ap.add_argument(
        "--motor-forward",
        action="store_true",
        help="Drive both wheels forward (default: rotate in place).",
    )
    ap.add_argument("--skip-motor", action="store_true", help="Skip motor spin phase.")
    ap.add_argument("--no-prompt", action="store_true", help="Skip Enter prompt before motor phase.")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    body_cfg = zenoh_helpers.load_body_config()
    if args.router:
        body_cfg = dict(body_cfg)
        body_cfg["zenoh"] = dict(body_cfg.get("zenoh", {}))
        body_cfg["zenoh"]["connect_endpoints"] = [args.router]
    session = zenoh_helpers.open_session(body_cfg)

    collector = Collector()
    imu_sub = zenoh_helpers.declare_subscriber_json(session, "body/imu", collector.on_imu)
    motor_sub = zenoh_helpers.declare_subscriber_json(
        session, "body/motor_state", collector.on_motor
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

    print("waiting for body/imu...", flush=True)
    first = collector.wait_for_first(timeout_s=15.0)
    if first is None:
        print("FAIL — no body/imu in 15 s.", file=sys.stderr)
        hb_stop.set()
        imu_sub.undeclare()
        motor_sub.undeclare()
        status_sub.undeclare()
        session.close()
        return 2

    print(
        f"  first sample: mag_present={first.mag_present} mag_valid={first.mag_valid}",
        flush=True,
    )
    if not first.mag_present:
        print(
            "\nWARN — no mag block in body/imu. On the Pi, set imu.mag_when_idle_enabled: true\n"
            "  and restart imu_driver (or launcher), then re-run this script.",
            flush=True,
        )

    results: dict[str, bool] = {}
    try:
        idle1 = _run_phase("IDLE (before spin)", args.idle_s, collector)
        s1 = _phase_stats(idle1, "idle-1")
        results["idle1_mag_valid"] = s1.get("mag_frac", 0.0) >= 0.8
        results["idle1_mag_accuracy"] = s1.get("acc_max_deg", 999.0) < 5.0

        if args.skip_motor:
            print("\n(skipping motor spin per --skip-motor)")
            spin: list[Sample] = []
            results["spin_mag_invalid"] = True
        else:
            motion = "forward drive" if args.motor_forward else "rotate in place"
            if not args.no_prompt:
                _prompt(
                    f"MOTOR PHASE — {motion} for {args.motor_s:.0f} s at duty {args.motor_duty}.\n"
                    "Clear space around the robot. Wheels-up is safest.\n"
                    "Default is rotate-in-place (not forward drive)."
                )
            spin = _run_phase(
                f"MOTOR {motion} duty={args.motor_duty}",
                args.motor_s,
                collector,
                session=session,
                duty=args.motor_duty,
                status_ref=status_ref,
                rotate_in_place=not args.motor_forward,
            )
        if not args.skip_motor:
            s_spin = _phase_stats(spin, "spin")
            results["spin_mag_invalid"] = s_spin.get("mag_frac", 1.0) <= 0.05

        idle2 = _run_phase("IDLE (after spin)", args.idle_s, collector)
        s2 = _phase_stats(idle2, "idle-2")
        results["idle2_mag_valid"] = s2.get("mag_frac", 0.0) >= 0.8
        results["idle2_mag_accuracy"] = s2.get("acc_max_deg", 999.0) < 5.0

        # Compare yaw at start of idle-1 vs start of idle-2.
        mag1 = next((s for s in idle1 if s.mag_valid and s.mag_yaw is not None), None)
        mag2 = next((s for s in idle2 if s.mag_valid and s.mag_yaw is not None), None)
        if mag1 and mag2:
            mag_delta_deg = _deg(_wrap_pi(mag2.mag_yaw - mag1.mag_yaw))
            game_delta_deg = _deg(_wrap_pi(mag2.game_yaw - mag1.game_yaw))
            agree_deg = abs(mag_delta_deg - game_delta_deg)
            if agree_deg > 180.0:
                agree_deg = 360.0 - agree_deg
            print(
                f"\n  cross-idle yaw change: mag={mag_delta_deg:+.2f}°  game={game_delta_deg:+.2f}°  "
                f"|Δmag−Δgame|={agree_deg:.2f}°"
            )
            if args.skip_motor:
                results["mag_stable_across_stop"] = abs(mag_delta_deg) < 2.0
            else:
                # After intentional rotation, mag and game should change by similar amounts.
                results["mag_stable_across_stop"] = agree_deg < 5.0
        else:
            results["mag_stable_across_stop"] = False
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
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
        imu_sub.undeclare()
        motor_sub.undeclare()
        status_sub.undeclare()
        session.close()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, ok in results.items():
        print(f"  {name:<24} {'PASS' if ok else 'FAIL'}")
    all_ok = bool(results) and all(results.values())
    print(f"  {'overall':<24} {'PASS' if all_ok else 'FAIL'}")
    if all_ok:
        print(
            "\nMag-when-idle looks usable on this mount. Next: wire EKF to apply mag\n"
            "corrections only when mag.valid (optional follow-up)."
        )
    else:
        spin_ok = results.get("spin_mag_invalid", False)
        acc_ok = results.get("idle1_mag_accuracy", False) and results.get(
            "idle2_mag_accuracy", False
        )
        stable_ok = results.get("mag_stable_across_stop", False)
        print()
        if spin_ok and not args.skip_motor:
            print("Motor gating: PASS — mag suppressed during spin.")
        elif args.skip_motor:
            print("Motor gating: skipped (--skip-motor). Re-run without it to verify.")
        if stable_ok:
            print(
                "Mag yaw stability: PASS — rotation_vector heading barely moved while idle.\n"
                "  This is the useful signal even when accuracy_rad looks bad."
            )
        if not acc_ok:
            print(
                "Mag accuracy estimate: FAIL — BNO085 reports ~90° uncertainty (want <5°).\n"
                "  Causes: mag not fully calibrated, or indoor field distortion.\n"
                "  If mag_valid stays true at ~92°, sync imu_driver.py to the Pi (accuracy gate\n"
                "  should set mag.valid=false until accuracy ≤ 5°)."
            )
            print(
                "\nRe-calibrate (motors OFF):\n"
                "  PYTHONPATH=. desktop/.venv/bin/python -m body.cli imu calibrate start \\\n"
                "    --router tcp/192.168.8.60:7447\n"
                "  (figure-8 ~15 s)\n"
                "  PYTHONPATH=. desktop/.venv/bin/python -m body.cli imu calibrate save \\\n"
                "    --router tcp/192.168.8.60:7447\n"
                "  Check Pi launcher for: mag calibration_status=3 (high)"
            )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

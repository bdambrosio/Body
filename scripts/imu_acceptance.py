#!/usr/bin/env python3
"""BNO085 acceptance tests per docs/imu_integration_spec.md §8.

Interactive, three-step harness that subscribes to ``body/imu`` and walks the
operator through the stationary-drift, 90°-hand-rotation, and motor-spin tests.
Assumes ``imu_driver`` is already running (e.g. via ``python3 -m body.launcher``).

Run:
    PYTHONPATH=. python3 scripts/imu_acceptance.py

Thresholds come from §8; Rotation-Vector vs Game-mode bounds are selected from
the first ``fusion.mode`` observed on ``body/imu``. Test 3 publishes
``body/cmd_direct`` at moderate duty — make sure the robot has room to move or
pass ``--skip-motor``.
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
    """ZYX yaw in radians from a unit quaternion (w, x, y, z), z-up body frame."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class ImuSample:
    ts: float
    gyro_z: float
    yaw: float
    accuracy_rad: float
    mode: str


@dataclass
class Collector:
    """Thread-safe accumulator for body/imu samples, feeding per-test slices."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    samples: list[ImuSample] = field(default_factory=list)
    latest: ImuSample | None = None
    count: int = 0

    def on_msg(self, _key: str, msg: dict[str, Any]) -> None:
        try:
            ts = float(msg["ts"])
            gyro = msg["gyro"]
            q = msg["orientation"]
            fusion = msg["fusion"]
            sample = ImuSample(
                ts=ts,
                gyro_z=float(gyro["z"]),
                yaw=_yaw_from_wxyz(
                    float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])
                ),
                accuracy_rad=float(fusion["accuracy_rad"]),
                mode=str(fusion.get("mode", "?")),
            )
        except (KeyError, TypeError, ValueError) as e:
            print(f"  [warn] bad body/imu payload: {e}", file=sys.stderr, flush=True)
            return
        with self.lock:
            self.samples.append(sample)
            self.latest = sample
            self.count += 1

    def snapshot_since(self, since_ts: float) -> list[ImuSample]:
        with self.lock:
            return [s for s in self.samples if s.ts >= since_ts]

    def wait_for_first(self, timeout_s: float) -> ImuSample | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            with self.lock:
                if self.latest is not None:
                    return self.latest
            time.sleep(0.05)
        return None


def _unwrap(yaws: list[float]) -> list[float]:
    """Unwrap a wrapped-angle sequence to a continuous cumulative yaw."""
    if not yaws:
        return []
    out = [yaws[0]]
    for y in yaws[1:]:
        d = y - out[-1] + math.pi
        d = d - 2.0 * math.pi * math.floor(d / (2.0 * math.pi)) - math.pi
        out.append(out[-1] + d)
    return out


def _prompt(msg: str) -> None:
    print()
    print("=" * 72)
    print(msg)
    print("=" * 72)
    input("Press Enter to continue (or Ctrl-C to abort)... ")


def _deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def _test1_stationary(collector: Collector, duration_s: float, mode: str) -> bool:
    """§8(1) — 30 s stationary: |gyro.z mean| < 0.005 rad/s, yaw drift below mode bound."""
    _prompt(
        f"TEST 1 — Stationary drift ({duration_s:.0f} s).\n"
        "Place the robot on a flat surface, hands off. Do NOT touch it during the test."
    )
    t_start = time.time()
    print(f"  recording for {duration_s:.0f} s...", flush=True)
    last_print = time.monotonic()
    while time.time() - t_start < duration_s:
        time.sleep(0.25)
        now = time.monotonic()
        if now - last_print >= 5.0:
            last_print = now
            with collector.lock:
                n = sum(1 for s in collector.samples if s.ts >= t_start)
            print(f"    ...{time.time() - t_start:5.1f}s   samples={n}", flush=True)

    samples = collector.snapshot_since(t_start)
    if len(samples) < 10:
        print(f"  FAIL — only {len(samples)} samples captured; is imu_driver publishing?")
        return False

    gyro_z_mean = sum(s.gyro_z for s in samples) / len(samples)
    yaws_unwrapped = _unwrap([s.yaw for s in samples])
    drift_rad = yaws_unwrapped[-1] - yaws_unwrapped[0]

    drift_bound_deg = 0.5 if mode == "rotation_vector" else 1.0
    gyro_ok = abs(gyro_z_mean) < 0.005
    drift_ok = abs(_deg(drift_rad)) < drift_bound_deg

    print()
    print(f"  samples:          {len(samples)}  ({len(samples) / duration_s:.1f} Hz)")
    print(
        f"  gyro.z mean:      {gyro_z_mean:+.5f} rad/s   "
        f"(bound |mean|<0.005) {'PASS' if gyro_ok else 'FAIL'}"
    )
    print(
        f"  yaw drift:        {_deg(drift_rad):+.3f}°        "
        f"(bound |drift|<{drift_bound_deg:.1f}° in {mode}) {'PASS' if drift_ok else 'FAIL'}"
    )
    return gyro_ok and drift_ok


def _test2_hand_rotation(collector: Collector) -> bool:
    """§8(2) — 90° CCW hand rotation in place: ∆yaw = +90° ± 2°."""
    _prompt(
        "TEST 2 — 90° hand rotation (CCW, right-hand rule around z-up).\n"
        "After you press Enter, you will have 10 s to slowly rotate the chassis\n"
        "by 90° counter-clockwise and set it down. CCW = positive (looking down)."
    )
    t_start = time.time()
    print("  rotate now... (10 s)", flush=True)
    time.sleep(10.0)

    samples = collector.snapshot_since(t_start)
    if len(samples) < 20:
        print(f"  FAIL — only {len(samples)} samples captured in 10 s.")
        return False

    yaws_unwrapped = _unwrap([s.yaw for s in samples])
    dyaw_rad = yaws_unwrapped[-1] - yaws_unwrapped[0]
    dyaw_deg = _deg(dyaw_rad)
    magnitude_ok = abs(abs(dyaw_deg) - 90.0) <= 2.0
    sign_ok = dyaw_deg > 0.0

    print()
    print(f"  samples:          {len(samples)}")
    print(
        f"  yaw change:       {dyaw_deg:+.2f}°        "
        f"(bound |∆yaw−90°|≤2°) {'PASS' if magnitude_ok else 'FAIL'}"
    )
    print(
        f"  sign (CCW→+):     {'+' if dyaw_deg >= 0 else '-'}             "
        f"{'PASS' if sign_ok else 'FAIL (rotated CW? or mount rotation inverted?)'}"
    )
    return magnitude_ok and sign_ok


def _wait_estop_clear(
    session: Any, status_ref: dict[str, Any], timeout_s: float = 5.0
) -> bool:
    """Block until body/status shows heartbeat_ok and cleared e_stop.

    Watchdog clears the latched e_stop only when a cmd_vel or cmd_direct arrives
    (with heartbeat fresh), so we publish zero cmd_direct at 5 Hz while waiting.
    """
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


def _test3_motor_spin(
    collector: Collector,
    session: Any,
    status_ref: dict[str, Any],
    odom_samples: list[tuple[float, float, str]],
    odom_lock: threading.Lock,
    mode: str,
    duration_s: float,
    duty: float,
) -> bool:
    """§8(3) — motor spin: observe fusion.accuracy_rad during commanded motion."""
    _prompt(
        f"TEST 3 — Motor spin ({duration_s:.0f} s at left={duty}, right={duty}).\n"
        "The harness will publish body/cmd_direct. Ensure the robot has clear\n"
        "floor space or wheels-up. motor_controller must be running."
    )
    if not _wait_estop_clear(session, status_ref, timeout_s=5.0):
        st = status_ref.get("latest", {})
        print(
            f"  FAIL — watchdog still latched: heartbeat_ok={st.get('heartbeat_ok')}, "
            f"e_stop_active={st.get('e_stop_active')}. Is the harness heartbeat publishing?",
            flush=True,
        )
        return False
    t_start = time.time()
    print("  driving motors...", flush=True)
    end = t_start + duration_s
    while time.time() < end:
        zenoh_helpers.publish_json(
            session,
            "body/cmd_direct",
            schemas.cmd_direct(left=duty, right=duty, timeout_ms=500),
        )
        time.sleep(0.1)
    zenoh_helpers.publish_json(
        session,
        "body/cmd_direct",
        schemas.cmd_direct(left=0.0, right=0.0, timeout_ms=500),
    )
    print("  motors stopped.", flush=True)

    samples = collector.snapshot_since(t_start)
    if len(samples) < 10:
        print(f"  FAIL — only {len(samples)} samples captured.")
        return False

    with odom_lock:
        odom_during = [(vx, src) for (ts, vx, src) in odom_samples if ts >= t_start]
    sources = {src for _, src in odom_during}
    max_abs_vx = max((abs(v) for v, _ in odom_during), default=0.0)
    encoder_backed = sources == {"wheel_encoders"}
    wheels_moved = encoder_backed and max_abs_vx > 0.05
    print()
    if not encoder_backed:
        print(
            f"  odom source:      {sources or '—'} (not encoder-backed; "
            "vx only echoes the command, cannot cross-check wheel motion)."
        )
    print(
        f"  odom samples:     {len(odom_during)}, max|vx|={max_abs_vx:.3f} m/s  "
        f"(bound |vx|>0.05, encoder-backed) "
        f"{'PASS' if wheels_moved else 'FAIL — wheels did not spin (or no encoders)'}"
    )

    acc = [s.accuracy_rad for s in samples]
    acc_max = max(acc)
    acc_median = sorted(acc)[len(acc) // 2]
    bound_rad = 3.0 * math.pi / 180.0

    print(f"  samples:          {len(samples)}")
    print(
        f"  accuracy_rad:     max={_deg(acc_max):.3f}°  median={_deg(acc_median):.3f}°  "
        f"(bound max<3°)"
    )
    if mode == "rotation_vector":
        ok = acc_max < bound_rad and wheels_moved
        print(f"  rotation_vector:  {'PASS' if ok else 'FAIL — mag contaminated or wheels did not spin'}")
        return ok
    print(
        "  game_rotation_vector: accuracy_rad is a driver constant in this mode; the\n"
        "    §8(3) threshold is only meaningful in rotation_vector. Informational —\n"
        "    motor-spin pass/fail here is driven entirely by the odom cross-check."
    )
    return wheels_moved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="BNO085 acceptance tests (see docs/imu_integration_spec.md §8).",
    )
    ap.add_argument("--stationary-s", type=float, default=30.0,
                    help="Test 1 duration in seconds (default: 30).")
    ap.add_argument("--motor-s", type=float, default=5.0,
                    help="Test 3 motor-spin duration in seconds (default: 5).")
    ap.add_argument("--motor-duty", type=float, default=0.3,
                    help="Test 3 left/right duty cycle (default: 0.3).")
    ap.add_argument("--skip-motor", action="store_true",
                    help="Skip test 3 (no motor commands issued).")
    ap.add_argument("--skip-stationary", action="store_true",
                    help="Skip test 1 (stationary drift).")
    ap.add_argument("--skip-rotation", action="store_true",
                    help="Skip test 2 (90° hand rotation).")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    body_cfg = zenoh_helpers.load_body_config()
    session = zenoh_helpers.open_session(body_cfg)
    collector = Collector()
    sub = zenoh_helpers.declare_subscriber_json(session, "body/imu", collector.on_msg)

    status_ref: dict[str, Any] = {"latest": None}

    def on_status(_key: str, msg: dict[str, Any]) -> None:
        status_ref["latest"] = msg

    status_sub = zenoh_helpers.declare_subscriber_json(session, "body/status", on_status)

    odom_samples: list[tuple[float, float, str]] = []
    odom_lock = threading.Lock()

    def on_odom(_key: str, msg: dict[str, Any]) -> None:
        try:
            with odom_lock:
                odom_samples.append(
                    (
                        float(msg["ts"]),
                        float(msg["vx"]),
                        str(msg.get("source", "")),
                    )
                )
        except (KeyError, TypeError, ValueError):
            return

    odom_sub = zenoh_helpers.declare_subscriber_json(session, "body/odom", on_odom)

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

    print("waiting for first body/imu message (driver settled)...", flush=True)
    first = collector.wait_for_first(timeout_s=15.0)
    if first is None:
        print(
            "FAIL — no body/imu in 15 s. Is imu_driver running and settled?\n"
            "  Check: ps -ef | grep imu_driver; it should print "
            "'settled after Ns' on stdout.",
            file=sys.stderr,
        )
        hb_stop.set()
        sub.undeclare()
        status_sub.undeclare()
        odom_sub.undeclare()
        session.close()
        return 2

    mode = first.mode
    print(f"  fusion.mode = {mode!r}, first accuracy_rad={first.accuracy_rad:.4f}")

    results: dict[str, bool] = {}
    try:
        if args.skip_stationary:
            print("(skipping test 1 per --skip-stationary)")
        else:
            results["1_stationary"] = _test1_stationary(collector, args.stationary_s, mode)
        if args.skip_rotation:
            print("(skipping test 2 per --skip-rotation)")
        else:
            results["2_hand_rotation"] = _test2_hand_rotation(collector)
        if args.skip_motor:
            print("\n(skipping test 3 per --skip-motor)")
        else:
            results["3_motor_spin"] = _test3_motor_spin(
                collector, session, status_ref, odom_samples, odom_lock,
                mode, args.motor_s, args.motor_duty,
            )
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
        sub.undeclare()
        status_sub.undeclare()
        odom_sub.undeclare()
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

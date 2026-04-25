#!/usr/bin/env python3
"""Automated planar figure-8 drive for BNO085 Rotation Vector calibration.

This script starts SH-2 calibration, drives repeated left/right constant-speed
loops via body/cmd_vel, prompts for a manual tilt phase, monitors body/imu
fusion status, then stops the robot.

Run with the robot on a clear floor. A wheeled figure-8 gives repeatable yaw
coverage, but full magnetometer calibration can still benefit from manually
tilting/lifting the chassis to cover pitch and roll axes before saving.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from typing import Any

from body.lib import schemas, zenoh_helpers

CAL_STATUS_LABELS = {
    0: "unreliable",
    1: "low",
    2: "medium",
    3: "high",
}


class ImuMonitor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: dict[str, Any] | None = None

    def on_imu(self, _key: str, msg: dict[str, Any]) -> None:
        with self.lock:
            self.latest = msg

    def snapshot(self) -> dict[str, Any] | None:
        with self.lock:
            return dict(self.latest) if self.latest is not None else None


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
            "body/cmd_vel",
            schemas.cmd_vel(linear=0.0, angular=0.0, timeout_ms=500),
        )
        time.sleep(0.2)
    return False


def _publish_stop(session: Any) -> None:
    for _ in range(3):
        zenoh_helpers.publish_json(
            session,
            "body/cmd_vel",
            schemas.cmd_vel(linear=0.0, angular=0.0, timeout_ms=500),
        )
        time.sleep(0.05)


def _format_imu_status(msg: dict[str, Any] | None) -> str:
    if msg is None:
        return "imu=no samples yet"
    fusion = msg.get("fusion", {})
    mode = str(fusion.get("mode", "?"))
    accuracy = fusion.get("accuracy_rad")
    acc_s = "?"
    if isinstance(accuracy, int | float):
        acc_s = f"{math.degrees(float(accuracy)):.1f}deg"
    cal = fusion.get("calibration_status")
    cal_s = "?"
    if isinstance(cal, int):
        cal_s = f"{cal} {CAL_STATUS_LABELS.get(cal, '?')}"
    return f"mode={mode} accuracy={acc_s} mag_status={cal_s}"


def _drive_figure8(
    session: Any,
    monitor: ImuMonitor,
    *,
    duration_s: float,
    linear_ms: float,
    radius_m: float,
    publish_hz: float,
    timeout_ms: int,
) -> None:
    angular = linear_ms / radius_m
    period = 1.0 / publish_hz
    loop_s = (2.0 * math.pi) / angular
    t0 = time.monotonic()
    next_print_s = 0.0

    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= duration_s:
            return
        direction = 1.0 if int(elapsed / loop_s) % 2 == 0 else -1.0
        zenoh_helpers.publish_json(
            session,
            "body/cmd_vel",
            schemas.cmd_vel(
                linear=linear_ms,
                angular=direction * angular,
                timeout_ms=timeout_ms,
            ),
        )
        if elapsed >= next_print_s:
            turn = "left" if direction > 0 else "right"
            print(
                f"  t={elapsed:5.1f}s/{duration_s:.1f}s turn={turn:<5} "
                f"{_format_imu_status(monitor.snapshot())}",
                flush=True,
            )
            next_print_s = elapsed + 1.0
        time.sleep(period)


def _manual_tilt_phase(
    session: Any,
    monitor: ImuMonitor,
    *,
    duration_s: float,
    publish_hz: float,
    timeout_ms: int,
    prompt: bool,
) -> None:
    if duration_s <= 0.0:
        return
    print()
    print("=" * 72)
    print("Manual tilt phase")
    print()
    print("Robot drive is stopped. Lift or hold the chassis safely and slowly")
    print("tilt through pitch and roll: nose up/down, left/right side up, then")
    print("return it level. Keep it away from large metal or magnets.")
    print("=" * 72)
    if prompt:
        input("Press Enter when you are ready to start tilting...")

    period = 1.0 / publish_hz
    t0 = time.monotonic()
    next_print_s = 0.0
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= duration_s:
            return
        zenoh_helpers.publish_json(
            session,
            "body/cmd_vel",
            schemas.cmd_vel(linear=0.0, angular=0.0, timeout_ms=timeout_ms),
        )
        if elapsed >= next_print_s:
            print(
                f"  tilt t={elapsed:5.1f}s/{duration_s:.1f}s "
                f"{_format_imu_status(monitor.snapshot())}",
                flush=True,
            )
            next_print_s = elapsed + 1.0
        time.sleep(period)


def _prompt(args: argparse.Namespace) -> None:
    angular = args.linear / args.radius
    loop_s = (2.0 * math.pi) / angular
    print()
    print("=" * 72)
    print("BNO085 Rotation Vector figure-8 calibration")
    print()
    print(f"Duration:       {args.duration:.1f} s")
    print(f"Linear speed:   {args.linear:.3f} m/s")
    print(f"Loop radius:    {args.radius:.3f} m")
    print(f"Angular speed:  {angular:.3f} rad/s")
    print(f"Loop period:    {loop_s:.1f} s per left/right circle")
    print(f"Tilt phase:     {args.tilt_duration:.1f} s after driving")
    print()
    print("Clear a floor area, keep hands clear, and be ready to Ctrl-C.")
    print("This automates yaw coverage, then prompts you to manually tilt the")
    print("stopped chassis for pitch/roll coverage before saving.")
    print("=" * 72)
    if not args.yes:
        input("Press Enter to start (or Ctrl-C to abort)... ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Drive a repeatable figure-8 while running BNO085 calibration.",
    )
    ap.add_argument("--duration", type=float, default=45.0, help="Drive duration, seconds.")
    ap.add_argument("--linear", type=float, default=0.08, help="Forward speed, m/s.")
    ap.add_argument("--radius", type=float, default=0.20, help="Loop radius, meters.")
    ap.add_argument(
        "--tilt-duration",
        type=float,
        default=20.0,
        help="Manual post-drive pitch/roll tilt duration, seconds. Use 0 to skip.",
    )
    ap.add_argument("--publish-hz", type=float, default=10.0, help="cmd_vel publish rate.")
    ap.add_argument("--timeout-ms", type=int, default=500, help="cmd_vel timeout_ms.")
    ap.add_argument(
        "--save-if-high",
        action="store_true",
        help="Save BNO085 calibration to flash if final mag status is high.",
    )
    ap.add_argument("-y", "--yes", action="store_true", help="Do not prompt before driving.")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if args.duration <= 0.0 or args.linear <= 0.0 or args.radius <= 0.0:
        print("FAIL - --duration, --linear, and --radius must be positive.", file=sys.stderr)
        return 2
    if args.tilt_duration < 0.0:
        print("FAIL - --tilt-duration must be nonnegative.", file=sys.stderr)
        return 2
    if args.publish_hz <= 0.0 or args.timeout_ms <= 0:
        print("FAIL - --publish-hz and --timeout-ms must be positive.", file=sys.stderr)
        return 2

    body_cfg = zenoh_helpers.load_body_config()
    imu_mode = str(body_cfg.get("imu", {}).get("fusion_mode", ""))
    if imu_mode != "rotation_vector":
        print(
            f"WARNING - config imu.fusion_mode is {imu_mode!r}; "
            "restart imu_driver with 'rotation_vector' for magnetometer calibration.",
            file=sys.stderr,
            flush=True,
        )

    _prompt(args)

    session = zenoh_helpers.open_session(body_cfg)
    monitor = ImuMonitor()
    status_ref: dict[str, Any] = {"latest": None}

    def on_status(_key: str, msg: dict[str, Any]) -> None:
        status_ref["latest"] = msg

    imu_sub = zenoh_helpers.declare_subscriber_json(session, "body/imu", monitor.on_imu)
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

    try:
        print("waiting for watchdog e-stop clear...", flush=True)
        if not _wait_estop_clear(session, status_ref, timeout_s=5.0):
            st = status_ref.get("latest", {})
            print(
                f"FAIL - watchdog still latched: heartbeat_ok={st.get('heartbeat_ok')}, "
                f"e_stop_active={st.get('e_stop_active')}.",
                file=sys.stderr,
            )
            return 1

        print("starting BNO085 calibration...", flush=True)
        zenoh_helpers.publish_json(session, "body/imu/calibrate", {"action": "start"})
        time.sleep(0.5)

        _drive_figure8(
            session,
            monitor,
            duration_s=float(args.duration),
            linear_ms=float(args.linear),
            radius_m=float(args.radius),
            publish_hz=float(args.publish_hz),
            timeout_ms=int(args.timeout_ms),
        )
        _publish_stop(session)

        _manual_tilt_phase(
            session,
            monitor,
            duration_s=float(args.tilt_duration),
            publish_hz=float(args.publish_hz),
            timeout_ms=int(args.timeout_ms),
            prompt=not bool(args.yes),
        )
        _publish_stop(session)

        final = monitor.snapshot()
        print()
        print(f"final {_format_imu_status(final)}", flush=True)
        zenoh_helpers.publish_json(session, "body/imu/calibrate", {"action": "status"})

        fusion = final.get("fusion", {}) if final is not None else {}
        cal = fusion.get("calibration_status")
        if args.save_if_high:
            if cal == 3:
                zenoh_helpers.publish_json(session, "body/imu/calibrate", {"action": "save"})
                print("save requested because final mag_status is high.", flush=True)
            else:
                print("not saving: final mag_status is not high.", flush=True)
        else:
            print("not saving automatically; rerun with --save-if-high to persist high calibration.")

        return 0
    except KeyboardInterrupt:
        print("\ninterrupted; stopping robot.", file=sys.stderr, flush=True)
        _publish_stop(session)
        return 130
    finally:
        hb_stop.set()
        hb_thread.join(timeout=1.0)
        imu_sub.undeclare()
        status_sub.undeclare()
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())

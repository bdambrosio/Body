"""Standalone teleop client: publishes body/heartbeat and body/cmd_vel without Jill.

Run from the repo root with PYTHONPATH set (same as launcher). Connects to the Zenoh router
from config.json or ZENOH_CONNECT.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from typing import Any

from body.lib import schemas, zenoh_helpers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish body/heartbeat and body/cmd_vel for local or remote Body stacks.",
    )
    parser.add_argument("--heartbeat-hz", type=float, default=2.0, help="Minimum spec is 2 Hz.")
    parser.add_argument("--cmd-vel-hz", type=float, default=20.0)
    parser.add_argument("--timeout-ms", type=int, default=500, help="cmd_vel timeout_ms field (robot-side).")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every body/status sample (noisy). emergency_stop is always printed.",
    )
    args = parser.parse_args()

    body_cfg = zenoh_helpers.load_body_config()
    session = zenoh_helpers.open_session(body_cfg)

    stop = threading.Event()
    lock = threading.Lock()
    linear_mps = 0.0
    angular_rps = 0.0
    hb_seq = 0
    latest_status: dict[str, Any] | None = None

    def on_status(_key: str, msg: dict[str, Any]) -> None:
        nonlocal latest_status
        with lock:
            latest_status = msg
        if args.verbose:
            print(f"[status] {msg}", flush=True)

    def on_emergency_stop(_key: str, msg: dict[str, Any]) -> None:
        print(f"[emergency_stop] {msg}", flush=True)

    zenoh_helpers.declare_subscriber_json(session, "body/status", on_status)
    zenoh_helpers.declare_subscriber_json(session, "body/emergency_stop", on_emergency_stop)

    hb_period = 1.0 / max(0.5, args.heartbeat_hz)
    cv_period = 1.0 / max(1.0, args.cmd_vel_hz)

    def heartbeat_loop() -> None:
        nonlocal hb_seq
        next_tick = time.monotonic()
        while not stop.is_set():
            with lock:
                seq = hb_seq
                hb_seq += 1
            zenoh_helpers.publish_json(session, "body/heartbeat", schemas.heartbeat(seq=seq))
            next_tick += hb_period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                stop.wait(sleep_for)
            else:
                next_tick = time.monotonic()

    def cmd_vel_loop() -> None:
        next_tick = time.monotonic()
        while not stop.is_set():
            with lock:
                lin = linear_mps
                ang = angular_rps
            zenoh_helpers.publish_json(
                session,
                "body/cmd_vel",
                schemas.cmd_vel(linear=lin, angular=ang, timeout_ms=args.timeout_ms),
            )
            next_tick += cv_period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                stop.wait(sleep_for)
            else:
                next_tick = time.monotonic()

    hb_thread = threading.Thread(target=heartbeat_loop, name="heartbeat", daemon=True)
    cv_thread = threading.Thread(target=cmd_vel_loop, name="cmd_vel", daemon=True)
    hb_thread.start()
    cv_thread.start()

    def handle_signal(_sig: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(
        "Body teleop: vel LINEAR_MPS [ANGULAR_RPS] | stop | status | help | quit",
        flush=True,
    )
    try:
        while not stop.is_set():
            try:
                line = input("teleop> ")
            except EOFError:
                break
            parts = line.strip().split()
            if not parts:
                continue
            cmd = parts[0].lower()
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "help":
                print(
                    "  vel LINEAR [ANGULAR] — linear m/s, angular rad/s (CCW +); "
                    "default angular 0\n"
                    "  stop — latch zero velocity\n"
                    "  status — last body/status JSON (if received)\n"
                    "  quit — exit",
                    flush=True,
                )
                continue
            if cmd == "stop":
                with lock:
                    linear_mps = 0.0
                    angular_rps = 0.0
                print("latched linear=0 angular=0", flush=True)
                continue
            if cmd == "status":
                with lock:
                    st = latest_status
                print(st if st is not None else "(no body/status yet)", flush=True)
                continue
            if cmd == "vel" and len(parts) >= 2:
                try:
                    lin = float(parts[1])
                    ang = float(parts[2]) if len(parts) >= 3 else 0.0
                except ValueError:
                    print("vel: expected numbers", flush=True)
                    continue
                with lock:
                    linear_mps = lin
                    angular_rps = ang
                print(f"latched linear={linear_mps} angular={angular_rps}", flush=True)
                continue
            print("unknown command; try help", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        hb_thread.join(timeout=2.0)
        cv_thread.join(timeout=2.0)
        session.close()

    sys.exit(0)


if __name__ == "__main__":
    main()

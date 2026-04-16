"""Watchdog: heartbeat timeout, process topic staleness, status and emergency_stop."""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Any

from body.lib import schemas, zenoh_helpers

TOPIC_TO_PROCESS: dict[str, str] = {
    "body/odom": "motor_controller",
    "body/lidar/scan": "lidar_driver",
    "body/oakd/imu": "oakd_driver",
}


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    wd_cfg = body_cfg.get("watchdog", {})
    hb_timeout_ms = float(wd_cfg.get("heartbeat_timeout_ms", 2000))
    proc_timeout_ms = float(wd_cfg.get("process_timeout_ms", 5000))
    status_hz = float(wd_cfg.get("status_publish_hz", 1))
    monitored: list[str] = list(wd_cfg.get("monitored_topics", []))

    last_hb = 0.0
    last_seen: dict[str, float] = {t: 0.0 for t in monitored}
    lock = threading.Lock()
    e_stop_active = False
    hb_lost_announced = False
    start = time.time()

    session = zenoh_helpers.open_session(body_cfg)

    def on_heartbeat(_key: str, _msg: dict[str, Any]) -> None:
        nonlocal last_hb
        with lock:
            last_hb = time.time()

    def on_cmd_vel(_key: str, _msg: dict[str, Any]) -> None:
        nonlocal e_stop_active
        with lock:
            hb_ok = (time.time() - last_hb) * 1000.0 < hb_timeout_ms if last_hb > 0.0 else False
            if e_stop_active and hb_ok:
                e_stop_active = False

    def make_topic_handler(topic: str):
        def _on(_key: str, _msg: dict[str, Any]) -> None:
            with lock:
                last_seen[topic] = time.time()

        return _on

    zenoh_helpers.declare_subscriber_json(session, "body/heartbeat", on_heartbeat)
    zenoh_helpers.declare_subscriber_json(session, "body/cmd_vel", on_cmd_vel)
    for topic in monitored:
        zenoh_helpers.declare_subscriber_json(session, topic, make_topic_handler(topic))

    stop = threading.Event()

    def handle_sigterm(_sig: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    status_period = 1.0 / max(1.0, status_hz)
    next_status = time.monotonic()
    while not stop.is_set():
        now = time.time()
        emergency: dict[str, Any] | None = None
        with lock:
            hb_ok = (now - last_hb) * 1000.0 < hb_timeout_ms if last_hb > 0.0 else False
            if not hb_ok:
                if not hb_lost_announced:
                    e_stop_active = True
                    hb_lost_announced = True
                    emergency = schemas.emergency_stop("heartbeat_timeout")
            else:
                hb_lost_announced = False

            processes: dict[str, str] = {}
            for topic, proc_name in TOPIC_TO_PROCESS.items():
                if topic not in monitored:
                    continue
                seen = last_seen.get(topic, 0.0)
                if seen <= 0.0:
                    processes[proc_name] = "missing"
                else:
                    age_ms = (now - seen) * 1000.0
                    processes[proc_name] = "ok" if age_ms <= proc_timeout_ms else "missing"

            uptime_s = now - start
            status_msg = schemas.status(
                processes=processes,
                heartbeat_ok=hb_ok,
                e_stop_active=e_stop_active,
                uptime_s=uptime_s,
                ts=now,
            )

        if emergency is not None:
            zenoh_helpers.publish_json(session, "body/emergency_stop", emergency)
        zenoh_helpers.publish_json(session, "body/status", status_msg)

        next_status += status_period
        sleep_for = next_status - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_status = time.monotonic()

    session.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

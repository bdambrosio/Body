"""Watchdog: heartbeat timeout, process topic staleness, status and emergency_stop."""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Any

from body.lib import host_metrics, schemas, zenoh_helpers

TOPIC_TO_PROCESS: dict[str, str] = {
    "body/odom": "motor_controller",
    "body/lidar/scan": "lidar_driver",
    "body/imu": "imu_driver",
    "body/oakd/depth": "oakd_driver",
    "body/map/local_2p5d": "local_map",
    "body/drive/status": "local_drive",
}


class EStopState:
    """Watchdog e-stop latch (pure; unit-tested).

    Latches on heartbeat loss. Clears ONLY on ``body/estop_rearm`` while the
    heartbeat is ok — motion commands never clear it (body_project_spec §5.10).
    """

    def __init__(self) -> None:
        self.e_stop_active = False
        self.hb_lost_announced = False

    def update_heartbeat(self, hb_ok: bool) -> bool:
        """Returns True when a new heartbeat_timeout e-stop should be announced."""
        if not hb_ok:
            if not self.hb_lost_announced:
                self.e_stop_active = True
                self.hb_lost_announced = True
                return True
            return False
        self.hb_lost_announced = False
        return False

    def rearm(self, hb_ok: bool) -> bool:
        """Returns True if the latch was cleared."""
        if self.e_stop_active and hb_ok:
            self.e_stop_active = False
            return True
        return False


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    wd_cfg = body_cfg.get("watchdog", {})
    hb_timeout_ms = float(wd_cfg.get("heartbeat_timeout_ms", 2000))
    proc_timeout_ms = float(wd_cfg.get("process_timeout_ms", 5000))
    status_hz = float(wd_cfg.get("status_publish_hz", 1))
    monitored: list[str] = list(wd_cfg.get("monitored_topics", []))
    host_metrics_enabled = bool(wd_cfg.get("host_metrics", True))

    # All intra-process timing is monotonic: a wall-clock step (NTP/chrony on a
    # Pi without RTC) must neither fake a heartbeat loss nor hide one.
    last_hb = 0.0
    last_seen: dict[str, float] = {t: 0.0 for t in monitored}
    lock = threading.Lock()
    estop = EStopState()
    start = time.monotonic()
    publish_now = threading.Event()

    session = zenoh_helpers.open_session(body_cfg)

    def _hb_ok(now: float) -> bool:
        return (now - last_hb) * 1000.0 < hb_timeout_ms if last_hb > 0.0 else False

    def on_heartbeat(_key: str, _msg: dict[str, Any]) -> None:
        nonlocal last_hb
        with lock:
            last_hb = time.monotonic()

    def on_estop_rearm(_key: str, msg: dict[str, Any]) -> None:
        with lock:
            hb_ok = _hb_ok(time.monotonic())
            was = estop.e_stop_active
            cleared = estop.rearm(hb_ok)
        if cleared:
            print(f"watchdog: e-stop re-armed by {msg.get('source', '?')}", flush=True)
            publish_now.set()
        elif was:
            print("watchdog: estop_rearm refused — heartbeat not ok", flush=True)

    def make_topic_handler(topic: str):
        # Raw subscriber: only the arrival time matters, so don't JSON-decode
        # (depth frames / local maps are large).
        def _on(_sample: Any) -> None:
            with lock:
                last_seen[topic] = time.monotonic()

        return _on

    zenoh_helpers.declare_subscriber_json(session, "body/heartbeat", on_heartbeat)
    zenoh_helpers.declare_subscriber_json(session, "body/estop_rearm", on_estop_rearm)
    for topic in monitored:
        session.declare_subscriber(topic, make_topic_handler(topic))

    stop = threading.Event()

    def handle_sigterm(_sig: int, _frame: object) -> None:
        stop.set()
        publish_now.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGHUP, handle_sigterm)

    status_period = 1.0 / max(1.0, status_hz)
    next_status = time.monotonic()
    while not stop.is_set():
        now = time.monotonic()
        emergency: dict[str, Any] | None = None
        try:
            with lock:
                hb_ok = _hb_ok(now)
                if estop.update_heartbeat(hb_ok):
                    emergency = schemas.emergency_stop("heartbeat_timeout")

                processes: dict[str, str] = {}
                for topic in monitored:
                    proc_name = TOPIC_TO_PROCESS.get(topic, topic)
                    seen = last_seen.get(topic, 0.0)
                    if seen <= 0.0:
                        processes[proc_name] = "missing"
                    else:
                        age_ms = (now - seen) * 1000.0
                        processes[proc_name] = "ok" if age_ms <= proc_timeout_ms else "missing"
                e_stop_active = estop.e_stop_active

            host = host_metrics.read_host_metrics_dict() if host_metrics_enabled else None
            status_msg = schemas.status(
                processes=processes,
                heartbeat_ok=hb_ok,
                e_stop_active=e_stop_active,
                uptime_s=now - start,
                host=host,
            )

            if emergency is not None:
                zenoh_helpers.publish_json(session, "body/emergency_stop", emergency)
            zenoh_helpers.publish_json(session, "body/status", status_msg)
        except Exception as e:
            print(f"watchdog: status tick failed: {e!r}", flush=True)

        next_status += status_period
        # Wake early when a re-arm changed the latch so consumers see it at once.
        # The heartbeat check runs at >= 10 Hz so a loss is latched promptly
        # even though status is only published at status_publish_hz.
        while not stop.is_set():
            remaining = next_status - time.monotonic()
            if remaining <= 0:
                break
            if publish_now.wait(min(remaining, 0.1)):
                publish_now.clear()
                next_status = time.monotonic()
                break
            with lock:
                hb_newly_lost = not _hb_ok(time.monotonic()) and not estop.hb_lost_announced
            if hb_newly_lost:
                next_status = time.monotonic()
                break
        if time.monotonic() - next_status > status_period:
            next_status = time.monotonic()

    try:
        session.close()
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()

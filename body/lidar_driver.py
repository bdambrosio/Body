"""LIDAR driver: stub publishes synthetic scans at ~10 Hz."""

from __future__ import annotations

import signal
import sys
import threading
import time

from body.lib import schemas, zenoh_helpers


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    lidar_cfg = body_cfg.get("lidar", {})
    hz = float(lidar_cfg.get("publish_hz", 10))
    period = 1.0 / max(1.0, hz)

    session = zenoh_helpers.open_session(body_cfg)
    stop = threading.Event()

    def handle_sigterm(_sig: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    next_tick = time.monotonic()
    while not stop.is_set():
        zenoh_helpers.publish_json(session, "body/lidar/scan", schemas.lidar_scan(scan_time_ms=int(round(period * 1000))))
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

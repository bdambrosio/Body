"""OAK-D driver: stub publishes synthetic IMU @ 100 Hz and placeholder depth @ configured FPS."""

from __future__ import annotations

import signal
import sys
import threading
import time

from body.lib import schemas, zenoh_helpers


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    oakd_cfg = body_cfg.get("oakd", {})
    depth_fps = float(oakd_cfg.get("depth_fps", 15))
    imu_period = 1.0 / 100.0
    depth_period = 1.0 / max(1.0, depth_fps)

    session = zenoh_helpers.open_session(body_cfg)
    stop = threading.Event()

    def handle_sigterm(_sig: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    next_imu = time.monotonic()
    next_depth = time.monotonic()
    while not stop.is_set():
        now = time.monotonic()
        if now >= next_imu:
            zenoh_helpers.publish_json(session, "body/oakd/imu", schemas.oakd_imu())
            next_imu += imu_period
            if next_imu < now:
                next_imu = now
        if now >= next_depth:
            zenoh_helpers.publish_json(session, "body/oakd/depth", schemas.oakd_depth_placeholder())
            next_depth += depth_period
            if next_depth < now:
                next_depth = now

        sleep_for = min(next_imu, next_depth) - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        elif stop.wait(0.001):
            break

    session.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

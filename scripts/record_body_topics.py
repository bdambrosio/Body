#!/usr/bin/env python3
"""Record key Body topics to a JSONL file for offline replay.

Subscribes to body/odom, body/imu (and body/oakd/imu for legacy),
body/lidar/scan, and body/map/local_2p5d. Every incoming message is
written as a line:

    {"topic": "body/odom", "recv_ts": 1713264000.123, "payload": {...}}

Run:
    PYTHONPATH=. python3 scripts/record_body_topics.py \\
        --router tcp/192.168.68.59:7447 \\
        --out ~/body-logs/session-2026-04-23-pm.jsonl

Stop with Ctrl-C. Playback is a separate module (desktop/nav/slam/
replay.py) that reads this file and republishes or feeds a pipeline
directly.

Design notes:
- Payloads are stored parsed-then-reserialized JSON, not raw bytes.
  Simpler to replay, costs a little fidelity on floats. For lidar
  scans at 10 Hz that's fine; we don't need sub-ms jitter.
- recv_ts is desktop wall-clock at zenoh callback time. The payload's
  own `ts` field (sensor time) is preserved inside payload.
- No compression. Body runs produce ~10 MB/min; disk is cheap.
- Binary fields (depth images) are NOT recorded — we'd need a
  separate sidecar file and replay logic. v1 scope is pose-relevant
  data only.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# Topics to capture. Add body/imu once live; keep body/oakd/imu for
# legacy compat until the rename lands.
DEFAULT_TOPICS = [
    "body/odom",
    "body/imu",
    "body/oakd/imu",
    "body/lidar/scan",
    "body/map/local_2p5d",
    "body/status",
    "body/motor_state",
]


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="record_body_topics",
        description="Record Body zenoh topics to JSONL for offline replay.",
    )
    p.add_argument("--router", required=True,
                   help="Zenoh router endpoint, e.g. tcp/192.168.68.59:7447")
    p.add_argument("--out", required=True,
                   help="Output JSONL file path.")
    p.add_argument("--topics", nargs="*", default=None,
                   help=f"Topics to record (default: {' '.join(DEFAULT_TOPICS)})")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    topics = args.topics if args.topics else DEFAULT_TOPICS

    out_path = Path(os.path.expanduser(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"recording {len(topics)} topics → {out_path}")
    for t in topics:
        logger.info(f"  {t}")

    # Import here so --help works without zenoh installed.
    from desktop.chassis.transport import open_session

    session = open_session(args.router)
    stop = threading.Event()
    write_lock = threading.Lock()
    counts: dict[str, int] = {t: 0 for t in topics}

    out_f = out_path.open("w", buffering=1)  # line-buffered

    def make_cb(topic: str):
        def cb(sample):
            recv_ts = time.time()
            try:
                raw = bytes(sample.payload.to_bytes())
            except AttributeError:
                raw = bytes(sample.payload)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as e:
                logger.debug(f"{topic}: non-JSON payload ({e}); skipping")
                return
            line = json.dumps({
                "topic": topic, "recv_ts": recv_ts, "payload": payload,
            })
            with write_lock:
                out_f.write(line + "\n")
                counts[topic] += 1
        return cb

    subs = []
    for t in topics:
        try:
            subs.append(session.declare_subscriber(t, make_cb(t)))
        except Exception:
            logger.exception(f"subscribe failed: {t}")

    def handle_sig(*_a):
        stop.set()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    # Periodic status so the user can see it's alive.
    t_start = time.time()
    try:
        while not stop.is_set():
            stop.wait(5.0)
            elapsed = time.time() - t_start
            parts = [f"{t}:{counts[t]}" for t in topics if counts[t] > 0]
            logger.info(f"t={elapsed:.1f}s  " + ("  ".join(parts) or "no messages yet"))
    finally:
        logger.info("stopping…")
        for s in subs:
            try:
                s.undeclare()
            except Exception:
                pass
        try:
            session.close()
        except Exception:
            pass
        out_f.close()
        total = sum(counts.values())
        logger.info(f"wrote {total} messages to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Record an RGB+pose drive for Phase 6 (VPR) bank building.

Subscribes to ``body/oakd/rgb`` (the Pi's request-gated JPEG stream)
and ``body/world_map/status`` (for ``pose_world``). For each RGB
frame, writes the JPEG to ``<out>/rgb/NNNNNN.jpg`` and appends a
line to ``<out>/frames.jsonl``::

    {"idx": 0, "rgb_ts": ..., "pose_ts": ..., "pose_age_s": ...,
     "pose_world": {"x_m": ..., "y_m": ..., "theta_rad": ...},
     "pose_source": "particle", "jpeg": "rgb/000000.jpg"}

Frames whose latest ``pose_world`` is older than ``--max-pose-age``
are dropped (counter: ``frames_dropped_stale_pose``).

Operationally
-------------
1. Start the nav stack (or world_map controller) with ``--pf`` so
   ``body/world_map/status`` is publishing pose.
2. ``rebind_world_to_current`` at your intended bank origin.
3. Run this script with ``--rate`` matching the desired bank density
   (1–2 Hz is plenty for a single-room loop; the OAK-D RGB stream is
   request-gated so the script publishes ``capture_rgb`` ticks).
4. Drive the loop. Ctrl-C to stop.

Output dir layout
-----------------
::

    <out>/
      meta.json         # session id, args, counters, pose_source name
      frames.jsonl      # one line per recorded frame (see above)
      rgb/NNNNNN.jpg    # the JPEGs themselves, zero-padded indices

This is what ``vpr_build_bank.py`` ingests next.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


RGB_TOPIC = "body/oakd/rgb"
OAKD_CONFIG_TOPIC = "body/oakd/config"
WORLD_STATUS_TOPIC = "body/world_map/status"


@dataclass
class RecorderState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_pose: Optional[Dict[str, float]] = None  # x_m, y_m, theta_rad
    # pose_ts is the publisher-side timestamp (desktop wall-clock for
    # world_status); pose_recv_ts is *our* wall-clock at callback time.
    # We compare ages against pose_recv_ts so that any Pi-vs-desktop
    # clock skew on other topics (e.g. body/oakd/rgb's Pi-stamped ts)
    # can't make every frame look stale.
    latest_pose_ts: float = 0.0
    latest_pose_recv_ts: float = 0.0
    pose_source_name: str = "unknown"
    frame_idx: int = 0
    counters: Dict[str, int] = field(default_factory=lambda: {
        "rgb_received": 0,
        "rgb_malformed": 0,
        "rgb_error_payload": 0,
        "frames_written": 0,
        "frames_dropped_stale_pose": 0,
        "frames_dropped_no_pose": 0,
        "world_status_received": 0,
        "capture_requests_sent": 0,
    })


def _payload_bytes(sample: Any) -> bytes:
    try:
        return bytes(sample.payload.to_bytes())
    except AttributeError:
        return bytes(sample.payload)


def _decode_b64_jpeg(b64: str) -> bytes:
    s = b64.strip()
    if s.startswith("data:"):
        comma = s.find(",")
        if comma >= 0:
            s = s[comma + 1:]
    return base64.standard_b64decode(s)


def _make_world_status_cb(state: RecorderState):
    def cb(sample: Any) -> None:
        try:
            msg = json.loads(_payload_bytes(sample).decode("utf-8"))
        except Exception:
            return
        pose = msg.get("pose_world")
        if not isinstance(pose, dict):
            return
        recv_ts = time.time()
        ts = float(msg.get("ts") or recv_ts)
        source = str(msg.get("pose_source") or "unknown")
        with state.lock:
            state.latest_pose = {
                "x_m": float(pose["x_m"]),
                "y_m": float(pose["y_m"]),
                "theta_rad": float(pose["theta_rad"]),
            }
            state.latest_pose_ts = ts
            state.latest_pose_recv_ts = recv_ts
            state.pose_source_name = source
            state.counters["world_status_received"] += 1
    return cb


def _make_rgb_cb(
    state: RecorderState,
    out_dir: Path,
    frames_f,
    write_lock: threading.Lock,
    max_pose_age_s: float,
):
    rgb_dir = out_dir / "rgb"

    def cb(sample: Any) -> None:
        with state.lock:
            state.counters["rgb_received"] += 1
        try:
            msg = json.loads(_payload_bytes(sample).decode("utf-8"))
        except Exception:
            with state.lock:
                state.counters["rgb_malformed"] += 1
            return
        if not msg.get("ok"):
            with state.lock:
                state.counters["rgb_error_payload"] += 1
            return
        b64 = msg.get("data")
        if not isinstance(b64, str):
            with state.lock:
                state.counters["rgb_malformed"] += 1
            return
        try:
            jpeg_bytes = _decode_b64_jpeg(b64)
        except Exception:
            with state.lock:
                state.counters["rgb_malformed"] += 1
            return

        rgb_recv_ts = time.time()
        rgb_ts = float(msg.get("ts") or rgb_recv_ts)

        # Snapshot pose under lock.
        with state.lock:
            pose = state.latest_pose
            pose_ts = state.latest_pose_ts
            pose_recv_ts = state.latest_pose_recv_ts
            pose_source = state.pose_source_name

        if pose is None:
            with state.lock:
                state.counters["frames_dropped_no_pose"] += 1
            return
        # Compare desktop-side receive times; Pi clock for rgb_ts
        # vs desktop clock for pose_ts can be skewed arbitrarily.
        age = max(0.0, rgb_recv_ts - pose_recv_ts)
        if age > max_pose_age_s:
            with state.lock:
                state.counters["frames_dropped_stale_pose"] += 1
            return

        with state.lock:
            idx = state.frame_idx
            state.frame_idx += 1
            state.counters["frames_written"] += 1

        jpeg_name = f"{idx:06d}.jpg"
        jpeg_path = rgb_dir / jpeg_name
        try:
            jpeg_path.write_bytes(jpeg_bytes)
        except Exception:
            logger.exception("vpr_record: failed to write %s", jpeg_path)
            return

        line = json.dumps({
            "idx": idx,
            "rgb_ts": rgb_ts,            # Pi sensor clock
            "rgb_recv_ts": rgb_recv_ts,  # desktop wall clock at arrival
            "pose_ts": pose_ts,          # publisher (desktop) clock
            "pose_recv_ts": pose_recv_ts,  # desktop wall clock at arrival
            "pose_age_s": age,           # desktop-side staleness
            "pose_world": pose,
            "pose_source": pose_source,
            "jpeg": f"rgb/{jpeg_name}",
        })
        with write_lock:
            frames_f.write(line + "\n")

    return cb


def _request_loop(state: RecorderState, publisher: Any, period_s: float,
                  stop: threading.Event) -> None:
    # Initial small wait to avoid colliding with stack startup.
    if stop.wait(0.5):
        return
    while not stop.is_set():
        try:
            payload = json.dumps({
                "action": "capture_rgb",
                "request_id": uuid.uuid4().hex,
            }).encode("utf-8")
            publisher.put(payload)
            with state.lock:
                state.counters["capture_requests_sent"] += 1
        except Exception:
            logger.exception("vpr_record: capture request failed")
        if stop.wait(period_s):
            return


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="vpr_record",
        description=(
            "Record RGB+pose into a session dir for Phase 6 VPR bank "
            "building. Subscribes to body/oakd/rgb and tags each frame "
            "with the latest pose_world from body/world_map/status."
        ),
    )
    p.add_argument("--router", required=True,
                   help="Zenoh router endpoint, e.g. tcp/192.168.68.59:7447")
    p.add_argument("--out", required=True,
                   help="Output session directory (created if missing).")
    p.add_argument("--rate", type=float, default=1.0,
                   help="capture_rgb request rate in Hz. 0 = passive "
                        "(only record frames driven by someone else).")
    p.add_argument("--max-pose-age", type=float, default=0.75,
                   help="Drop frames whose latest pose_world is older than "
                        "this (seconds). Default 0.75 — comfortable for the "
                        "controller's ~2 Hz world_status cadence.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir = Path(os.path.expanduser(args.out)).resolve()
    rgb_dir = out_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.jsonl"
    meta_path = out_dir / "meta.json"

    session_id = uuid.uuid4().hex[:12]
    logger.info("vpr_record: session=%s out=%s rate=%.2f Hz",
                session_id, out_dir, args.rate)

    # Import here so --help works without zenoh installed.
    from desktop.chassis.transport import open_session

    z_session = open_session(args.router)
    state = RecorderState()
    write_lock = threading.Lock()
    stop = threading.Event()

    frames_f = frames_path.open("a", buffering=1)  # line-buffered append
    subs = []
    pub_config = None
    request_thread: Optional[threading.Thread] = None
    t_start = time.time()

    try:
        subs.append(z_session.declare_subscriber(
            WORLD_STATUS_TOPIC, _make_world_status_cb(state),
        ))
        subs.append(z_session.declare_subscriber(
            RGB_TOPIC,
            _make_rgb_cb(
                state, out_dir, frames_f, write_lock, args.max_pose_age,
            ),
        ))
        if args.rate > 0.0:
            pub_config = z_session.declare_publisher(OAKD_CONFIG_TOPIC)
            request_thread = threading.Thread(
                target=_request_loop,
                args=(state, pub_config, 1.0 / args.rate, stop),
                name="vpr-rgb-requester", daemon=True,
            )
            request_thread.start()

        def handle_sig(*_a):
            stop.set()
        signal.signal(signal.SIGINT, handle_sig)
        signal.signal(signal.SIGTERM, handle_sig)

        while not stop.is_set():
            stop.wait(5.0)
            with state.lock:
                c = dict(state.counters)
                pose = state.latest_pose
                pose_src = state.pose_source_name
            elapsed = time.time() - t_start
            logger.info(
                "t=%.1fs rgb=%d written=%d dropped_stale=%d dropped_nopose=%d "
                "pose_src=%s pose=%s",
                elapsed, c["rgb_received"], c["frames_written"],
                c["frames_dropped_stale_pose"],
                c["frames_dropped_no_pose"], pose_src,
                "yes" if pose else "no",
            )
    finally:
        logger.info("vpr_record: stopping…")
        stop.set()
        for s in subs:
            try:
                s.undeclare()
            except Exception:
                pass
        if pub_config is not None:
            try:
                pub_config.undeclare()
            except Exception:
                pass
        if request_thread is not None:
            request_thread.join(timeout=1.0)
        try:
            z_session.close()
        except Exception:
            pass
        frames_f.close()

        # Write meta.json summarizing the session.
        with state.lock:
            counters = dict(state.counters)
            pose_source = state.pose_source_name
            n_frames = state.frame_idx
        meta_payload = {
            "session_id": session_id,
            "started_ts": t_start,
            "stopped_ts": time.time(),
            "args": {
                "router": args.router,
                "out": str(out_dir),
                "rate_hz": args.rate,
                "max_pose_age_s": args.max_pose_age,
            },
            "pose_source": pose_source,
            "frame_count": n_frames,
            "counters": counters,
            "schema_version": 1,
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2))
        logger.info("vpr_record: wrote %d frames → %s", n_frames, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

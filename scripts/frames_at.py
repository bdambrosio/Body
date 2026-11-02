#!/usr/bin/env python
"""Which recorded camera frames were looking at a spot (or a change)?

    PYTHONPATH=. .venv/bin/python scripts/frames_at.py RUN_DIR --change 54
    PYTHONPATH=. .venv/bin/python scripts/frames_at.py RUN_DIR --xy 3.43 -2.01 \\
        [--range 4] [--fov 30] [--sheet out.jpg]

RUN_DIR is a nav explore folder (~/body-logs/explore/<ts>/) or a mapping bag
(~/Body/maps/<sid>/bag_<ts>/): anything with frames.jsonl + frame_*.jpg.
With --change N the spot and the time window (appeared → gone, ±20 s) come
from events.jsonl. Prints each frame whose camera had the spot within
±FOV° and RANGE m, and writes a contact sheet (default RUN_DIR/at_<x>_<y>.jpg)
with time, range and bearing on each tile.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--change", type=int)
    g.add_argument("--xy", nargs=2, type=float)
    ap.add_argument("--range", type=float, default=4.0)
    ap.add_argument("--fov", type=float, default=30.0, help="half-angle, degrees")
    ap.add_argument("--margin-s", type=float, default=20.0)
    ap.add_argument("--max-tiles", type=int, default=24)
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args()

    frames = load_jsonl(os.path.join(args.run_dir, "frames.jsonl"))
    if not frames:
        print("no frames.jsonl in", args.run_dir, file=sys.stderr)
        return 1
    t_lo, t_hi = -math.inf, math.inf
    if args.change is not None:
        ev = [e for e in load_jsonl(os.path.join(args.run_dir, "events.jsonl"))
              if e.get("type") == "change" and e.get("change_id") == args.change]
        if not ev:
            print(f"change #{args.change} not in events.jsonl", file=sys.stderr)
            return 1
        x, y = ev[0]["x"], ev[0]["y"]
        t_lo = min(e["wall_ts"] for e in ev) - args.margin_s
        t_hi = max(e["wall_ts"] for e in ev) + args.margin_s
        print(f"change #{args.change} {ev[0]['kind']} at ({x:.2f}, {y:.2f}): "
              + ", ".join(f"{e['status']} {e['wall_ts'] - t_lo - args.margin_s:+.0f}s" for e in ev))
    else:
        x, y = args.xy
    hits = []
    for f in frames:
        pose = f.get("pose")
        if not pose or not (t_lo <= f["ts"] <= t_hi):
            continue
        dx, dy = x - pose[0], y - pose[1]
        rng = math.hypot(dx, dy)
        b = math.degrees(math.remainder(math.atan2(dy, dx) - pose[2], 2 * math.pi))
        if rng <= args.range and abs(b) <= args.fov:
            hits.append((f, rng, b))
    t0 = frames[0]["ts"]
    print(f"{len(hits)} of {len(frames)} frames looked at ({x:.2f}, {y:.2f}):")
    for f, rng, b in hits:
        print(f"  {f['path']}  t={f['ts'] - t0:7.1f}s  range {rng:4.2f} m  bearing {b:+4.0f}°")
    if not hits:
        return 0
    tiles = []
    step = max(1, len(hits) // args.max_tiles)
    for f, rng, b in hits[::step][: args.max_tiles]:
        img = cv2.imread(os.path.join(args.run_dir, f["path"]))
        if img is None:
            continue
        img = cv2.resize(img, (320, 200))
        # Where the spot should be in the image (pinhole, fx ~ 558 @ 640 px).
        u = int(160 - 279.0 * math.tan(math.radians(b)))
        cv2.line(img, (u, 0), (u, 199), (0, 255, 255), 1)
        cv2.putText(img, f"t={f['ts'] - t0:.0f}s {rng:.1f}m {b:+.0f}deg", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(img)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = args.sheet or os.path.join(args.run_dir, f"at_{x:.2f}_{y:.2f}.jpg")
    cv2.imwrite(out, sheet)
    print("sheet:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

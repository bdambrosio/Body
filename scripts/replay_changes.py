#!/usr/bin/env python
"""Replay a recorded nav run through the session grid + change detector.

    PYTHONPATH=. .venv/bin/python scripts/replay_changes.py \\
        ~/Body/maps/<sid>/map_*/reference_map.npz ~/body-logs/explore/<ts> \\
        [--set wall_margin_m=0.4 --set new_min_cells=6 ...] [--events out.jsonl]

Uses the scans the live run recorded (``scans_*.npz``, written by
``desktop/perception/scan_recorder.py``) and, by default, the live run's own
per-scan "write into the world" verdict, so only the detector settings
change. Prints counts per kind/status, how many reports sit near reference
walls, distinct places, and how often the same place is re-reported — the
numbers used to judge the 11-02 patrol.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desktop.perception.change_detector import ChangeDetectorConfig  # noqa: E402
from desktop.perception.change_session import ChangeSession  # noqa: E402
from desktop.perception.scan_recorder import load_scans  # noqa: E402
from desktop.reference_map.reference_map import load_reference_map  # noqa: E402


def parse_sets(items):
    cfg = ChangeDetectorConfig()
    kw = {}
    for it in items or []:
        k, v = it.split("=", 1)
        cur = getattr(cfg, k)          # AttributeError on a typo
        kw[k] = type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true", "yes")
    return dataclasses.replace(cfg, **kw)


def summarize(events, rm):
    dist = rm.distance_field_m

    def d_wall(x, y):
        i = int((x - rm.origin_x_m) / rm.resolution_m)
        j = int((y - rm.origin_y_m) / rm.resolution_m)
        return float(dist[i, j]) if 0 <= i < rm.nx and 0 <= j < rm.ny else float("nan")

    c = Counter((e.kind, e.status) for e in events)
    print("events:", dict(sorted(c.items())))
    for kind in ("new", "missing", "moving"):
        app = [e for e in events if e.kind == kind and e.status == "appeared"]
        if not app:
            continue
        places = []
        for e in app:
            for p in places:
                if math.hypot(p[0] - e.x, p[1] - e.y) < 0.5:
                    p[2] += 1
                    break
            else:
                places.append([e.x, e.y, 1])
        near = sum(d_wall(e.x, e.y) < 0.25 for e in app)
        print(f"  {kind:8s} appeared {len(app):4d}  near wall(<0.25 m) {near:4d}  "
              f"places {len(places):3d}  max repeats {max(p[2] for p in places)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map")
    ap.add_argument("run_dir")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="ChangeDetectorConfig override (repeatable)")
    ap.add_argument("--all-scans", action="store_true",
                    help="integrate every scan with a pose, not just those the live run accepted")
    ap.add_argument("--events", help="write replayed events to this JSONL")
    args = ap.parse_args()

    rm = load_reference_map(args.map)
    cfg = parse_sets(args.set)
    paths = glob.glob(os.path.join(args.run_dir, "scans_*.npz"))
    if not paths:
        print(f"no scans_*.npz in {args.run_dir}", file=sys.stderr)
        return 1
    cs = ChangeSession(rm, det_cfg=cfg)
    events, n, used = [], 0, 0
    for ts, ranges, angles, pose, ok, _q, _cov in load_scans(paths):
        n += 1
        if pose is None or not (ok or args.all_scans):
            continue
        used += 1
        events += cs.observe(ts, ranges, angles, pose)
    print(f"scans {n}, integrated {used}; detector {cfg}")
    summarize(events, rm)
    if args.events:
        with open(args.events, "w") as f:
            for e in events:
                f.write(json.dumps(e.to_dict()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

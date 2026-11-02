#!/usr/bin/env python
"""Rebuild a map from a recorded mapping bag — offline, no robot.

    PYTHONPATH=. .venv/bin/python scripts/replay_mapping.py \\
        ~/Body/maps/<sid>/bag_<ts> [--out DIR] [--prior-map MAP] \\
        [--seed-pose X Y THETA_DEG] [--set match_hz=4 ...]

Feeds the bag's odom / IMU / scans through the same ``MappingController``
the live app runs (rate-limited by the scan clock, not wall time), then
exports ``reference_map.npz`` the usual way. Every mapping backend change
is judged here first, on the same drive.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desktop.fusion import load_slam_config as lsc  # noqa: E402
from desktop.mapping.bag import iter_bag  # noqa: E402
from desktop.mapping.controller import MappingConfig, MappingController  # noqa: E402
from desktop.mapping.export import export_mapping_session  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_dir")
    ap.add_argument("--out", default=None, help="map output root (default ~/Body/maps)")
    ap.add_argument("--prior-map", default=None)
    ap.add_argument("--seed-pose", nargs=3, type=float, metavar=("X", "Y", "THETA_DEG"))
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="SlamConfig override (repeatable), e.g. match_hz=4")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.set:
        base = lsc.load_slam_config()
        kw = {}
        for it in args.set:
            k, v = it.split("=", 1)
            cur = getattr(base.slam, k)
            kw[k] = type(cur)(v)
        patched = lsc.SlamFusionConfig(fusion=base.fusion,
                                       slam=dataclasses.replace(base.slam, **kw))
        import desktop.mapping.controller as ctl_mod
        ctl_mod.load_slam_config = lambda *a, **k: patched

    seed = None
    if args.seed_pose:
        seed = (args.seed_pose[0], args.seed_pose[1], math.radians(args.seed_pose[2]))
    ctl = MappingController(MappingConfig(record=False, prior_map=args.prior_map,
                                          seed_pose=seed))
    clock = {"ts": 0.0}
    ctl.mapper._clock = lambda: clock["ts"]

    t0 = time.monotonic()
    n = {"odom": 0, "imu": 0, "scan": 0, "nodes": 0}
    for kind, msg in iter_bag(args.bag_dir):
        n[kind] += 1
        if kind == "odom":
            ctl.ingest_odom(msg)
        elif kind == "imu":
            ctl.ingest_imu(msg)
        else:
            clock["ts"] = float(msg["ts"])
            if ctl.ingest_scan(msg):
                n["nodes"] += 1
    wall = time.monotonic() - t0
    diag = ctl.status_summary()
    print(f"replayed {n} in {wall:.1f} s; keyframes skipped {diag['keyframes_skipped']}, "
          f"edge rejects {diag['edge_rejects']}, contests {diag['contests_imu_wins']}/"
          f"{diag['contests_judged']}, loop closures {diag['loop_closures']}")
    if n["nodes"] == 0:
        print("no graph nodes — was the EKF ever ready (IMU settled)?", file=sys.stderr)
        return 1
    out = export_mapping_session(ctl, base_dir=args.out)
    print("map:", os.path.join(out, "reference_map.npz"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

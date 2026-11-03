#!/usr/bin/env python
"""Score a persistence gate for the depth veto on logged bursts.

    PYTHONPATH=. .venv/bin/python scripts/depth_burst_score.py ~/body-logs/vlm/<session>[...]

Input: ``*_burst.npz`` from DepthStopLogger (distinct depth frames ~2 s
before to 1 s after a depth stop, each with the desktop world pose). For
each frame the blob gates (body.lib.depth_denoise) run on the forward
corridor; surviving blobs are placed in the world frame. The proposed
persistence gate: an obstacle is real if a blob within ``--match-m`` of the
same world spot is seen on >= ``--min-frames`` frames over which the robot
moved >= ``--min-motion-m`` (parallax: a floor reflection / glare moves with
the camera, a thing stays put). Reports, per burst, how many frames saw a
gated blob, the best persistent track, and the gate's verdict at the
trigger frame (using only frames up to it — what the Pi could have known).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

from body.lib import drive_config, zenoh_helpers
from body.lib.depth_denoise import DenoiseConfig, denoised_hits
from body.lib.depth_veto import _arc_coords, depth_points_body


def _corridor(dp, vcfg, half_width=0.20, max_range=0.6):
    s, lat = _arc_coords(dp.pts[:, 0], dp.pts[:, 1], 0.18, 0.0)
    return (s >= vcfg.min_range_m) & (s <= max_range) & (lat <= half_width)


def frame_blobs(arr, pose, vcfg, dcfg):
    """World (x, y) centroids of gated blobs in one frame."""
    msg = {"width": int(arr.shape[1]), "height": int(arr.shape[0])}
    dp = depth_points_body(arr, msg, vcfg)
    if dp is None:
        return []
    _n, _idx, blobs = denoised_hits(dp, _corridor(dp, vcfg), vcfg, dcfg)
    out = []
    c, s = math.cos(pose[2]), math.sin(pose[2])
    for b in blobs:
        bx = 0.5 * (b.x_min + b.x_max)
        by = 0.5 * (b.y_min + b.y_max)
        out.append((pose[0] + c * bx - s * by, pose[1] + s * bx + c * by))
    return out


def best_track(frames, upto, match_m):
    """(n frames, robot motion over them) for the most persistent world spot
    among frames[0..upto]."""
    best = (0, 0.0)
    for k in range(upto + 1):
        for spot in frames[k][1]:
            hits = [j for j in range(upto + 1)
                    if any(math.hypot(p[0] - spot[0], p[1] - spot[1]) <= match_m
                           for p in frames[j][1])]
            if not hits:
                continue
            poses = [frames[j][0] for j in hits]
            motion = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in poses for b in poses)
            if (len(hits), motion) > best:
                best = (len(hits), motion)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--min-motion-m", type=float, default=0.10)
    ap.add_argument("--match-m", type=float, default=0.10)
    args = ap.parse_args()
    body_cfg = zenoh_helpers.load_body_config()
    vcfg = drive_config.depth_veto_config(body_cfg)
    dcfg = DenoiseConfig()
    files = []
    for d in args.dirs:
        files += sorted(glob.glob(os.path.join(os.path.expanduser(d), "**", "*_burst.npz"),
                                  recursive=True))
    for f in files:
        z = np.load(f)
        ti = int(z["trigger_index"])
        frames = [(tuple(z["pose"][k]), frame_blobs(z["depth"][k], z["pose"][k], vcfg, dcfg))
                  for k in range(len(z["depth"]))
                  if np.isfinite(z["pose"][k]).all()]
        ti = min(ti, len(frames) - 1)
        seen = sum(1 for _p, b in frames if b)
        n_all, mot_all = best_track(frames, len(frames) - 1, args.match_m)
        n_t, mot_t = best_track(frames, ti, args.match_m)
        passes = n_t >= args.min_frames and mot_t >= args.min_motion_m
        print(f"{os.path.basename(f)}: {len(frames)} frames, gated blob on {seen}; "
              f"best track {n_all} frames / {mot_all:.2f} m motion (whole burst), "
              f"{n_t} / {mot_t:.2f} m up to the trigger -> "
              f"{'KEEP (real)' if passes else 'DROP'}")
    if not files:
        print("no *_burst.npz found")
    return 0


if __name__ == "__main__":
    sys.exit(main())

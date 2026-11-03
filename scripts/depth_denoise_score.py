#!/usr/bin/env python
"""Score the depth-veto blob gates on logged frames.

    PYTHONPATH=. .venv/bin/python scripts/depth_denoise_score.py ~/body-logs/vlm/<session>[/...]

Inputs: ``would_block_*.npz`` / ``depth_block_*.npz`` (single frames; ``*_burst.npz`` skipped) (observe-only veto: frames the raw veto WOULD
have stopped on; assumed negatives unless listed in --positives) and
``run*_depth.npz`` (real stops). For each frame: raw veto hits (the 10-31
rule: >= min_hits in the commanded-arc corridor), and hits after
``body.lib.depth_denoise`` gates. Also renders synthetic positives (bag,
shoe, chair caster) to show the gates keep real things.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from body.lib import drive_config, zenoh_helpers
from body.lib.depth_denoise import DenoiseConfig, denoised_hits, find_blobs
from body.lib.depth_veto import _arc_coords, depth_points_body
from body.test_depth_memory import render_depth


def corridor(dp, vcfg, v=0.18, w=0.0):
    s, lat = _arc_coords(dp.pts[:, 0], dp.pts[:, 1], v, w)
    return (s >= vcfg.min_range_m) & (s <= vcfg.max_range_m) & (lat <= vcfg.lateral_half_width_m)


def score(arr, vcfg, dcfg):
    msg = {"width": int(arr.shape[1]), "height": int(arr.shape[0])}
    dp = depth_points_body(arr, msg, vcfg)
    if dp is None:
        return 0, 0, []
    reg = corridor(dp, vcfg)
    raw, _i, _b = denoised_hits(dp, reg, vcfg, DenoiseConfig(enabled=False))
    den, _i, blobs = denoised_hits(dp, reg, vcfg, dcfg)
    return raw, den, blobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--min-hits", type=int, default=None)
    args = ap.parse_args()
    body_cfg = zenoh_helpers.load_body_config()
    vcfg = drive_config.depth_veto_config(body_cfg)
    dcfg = DenoiseConfig()
    min_hits = args.min_hits or vcfg.min_hits
    files = []
    for d in args.dirs:
        for pat in ("would_block_*.npz", "depth_block_*.npz"):
            files += sorted(f for f in glob.glob(os.path.join(os.path.expanduser(d), "**", pat), recursive=True)
                            if not f.endswith("_burst.npz"))
        files += sorted(glob.glob(os.path.join(os.path.expanduser(d), "**", "run*_depth.npz"), recursive=True))
    raw_fire = den_fire = 0
    for f in files:
        z = np.load(f)
        arr = z["depth"]
        if arr.size == 0:
            continue
        raw, den, blobs = score(arr, vcfg, dcfg)
        rf, df = raw >= min_hits, den >= min_hits
        raw_fire += rf
        den_fire += df
        desc = "; ".join(f"{b.n_pts}pts {b.extent_m*100:.0f}cm @{b.range_m:.2f}m h{b.h_max*100:.0f}" for b in blobs[:3])
        print(f"{os.path.basename(f):38s} raw {raw:4d} {'FIRE' if rf else '    '}  gated {den:4d} {'FIRE' if df else '    '}  {desc}")
    print(f"\nlogged frames: {len(files)}  raw would fire on {raw_fire}, gated on {den_fire}")

    print("\nsynthetic positives (gates must keep these):")
    scenes = {
        "bag 30x30 cm, 30 cm tall @0.5 m": [(0.50, -0.15, 0.80, 0.15, 0.30)],
        "shoe 10x25 cm, 8 cm tall @0.45 m": [(0.45, -0.05, 0.70, 0.05, 0.08)],
        "chair caster 6 cm, 6 cm tall @0.4 m": [(0.40, -0.03, 0.46, 0.03, 0.06)],
        "box 20x20, 15 cm tall @0.6 m": [(0.60, -0.10, 0.80, 0.10, 0.15)],
        "cable 2 cm x 40 cm, 2 cm tall @0.4 m": [(0.40, -0.20, 0.42, 0.20, 0.02)],
    }
    for name, boxes in scenes.items():
        arr = render_depth((0, 0, 0), boxes, vcfg)
        raw, den, blobs = score(arr, vcfg, dcfg)
        print(f"  {name:40s} raw {raw:4d} gated {den:4d} {'KEPT' if den >= min_hits else 'dropped'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

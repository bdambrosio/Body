#!/usr/bin/env python3
"""Phase-0 benchmark for object commands: can the VLM find <target> in a frame?

Runs ``desktop.tasks.locate`` 's single-object query for each (frame, target)
pair and reports hit rate, false positives, latency and the bbox bearing.
Ground truth is an optional JSON file {"<jpeg basename>": ["person", ...]}
listing the targets actually visible; without it every answer is reported
but nothing is scored.

    python scripts/object_detect_bench.py ~/body-logs/vlm/20261030_105551/*.jpg \
        --targets person dog "red ball" chair --truth truth.json --sheet out.jpg
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "desktop"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vision_service as vs  # noqa: E402
from desktop.tasks.locate import (SCHEMA, CameraModel, prompt_for,  # noqa: E402
                                  sighting_from_answer)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--truth", help="JSON {basename: [visible targets]}")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--out", help="JSONL of every answer")
    ap.add_argument("--sheet", help="contact sheet JPEG with boxes (hits only)")
    args = ap.parse_args()

    truth = json.load(open(args.truth)) if args.truth else None
    cam = CameraModel.from_config({})
    rows, lat = [], []
    tp = fp = fn = tn = 0
    for path in args.frames:
        jpeg = open(path, "rb").read()
        w, h = vs.jpeg_size(jpeg)
        c = cam.scaled(w, h) if (w, h) != (cam.width, cam.height) else cam
        base = os.path.basename(path)
        for target in args.targets:
            t0 = time.monotonic()
            err = None
            try:
                ans = vs.chat_json([{"role": "user", "content": prompt_for(target)}],
                                   SCHEMA, images=[jpeg], max_tokens=80,
                                   name="locate")
            except Exception as e:  # noqa: BLE001 - report and keep going
                ans, err = None, f"{type(e).__name__}: {e}"
            dt = time.monotonic() - t0
            lat.append(dt)
            s = sighting_from_answer(target, ans, c, ts=0.0, min_conf=args.min_conf)
            seen = bool(s and s.present)
            mark = ""
            if truth is not None and base in truth:
                real = target in truth[base]
                if seen and real:
                    tp += 1; mark = "TP"
                elif seen and not real:
                    fp += 1; mark = "FP"
                elif real:
                    fn += 1; mark = "FN"
                else:
                    tn += 1; mark = "TN"
            row = {"frame": path, "target": target, "answer": ans, "error": err,
                   "seen": seen, "latency_s": round(dt, 3), "score": mark,
                   "bbox_px": s.bbox_px if s else None,
                   "bearing_deg": round(math.degrees(s.bearing), 1)
                   if s and s.bearing is not None else None}
            rows.append(row)
            print(f"{base:28s} {target:10s} {mark:2s} seen={seen!s:5s} "
                  f"conf={(ans or {}).get('confidence', '-')!s:5s} "
                  f"brg={row['bearing_deg']!s:6s} {dt:.2f}s"
                  f"{'  ERR ' + err if err else ''}")

    print(f"\nlatency p50={statistics.median(lat):.2f}s max={max(lat):.2f}s "
          f"n={len(lat)}")
    if truth is not None:
        print(f"TP={tp} FP={fp} FN={fn} TN={tn}  "
              f"hit={tp / max(1, tp + fn):.0%}  FP rate={fp / max(1, fp + tn):.0%}")
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    if args.sheet:
        _sheet([r for r in rows if r["seen"]], args.sheet)
    return 0


def _sheet(rows, out):
    from PIL import Image, ImageDraw
    if not rows:
        print("no hits; no sheet")
        return
    tiles = []
    for r in rows:
        im = Image.open(r["frame"]).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle(r["bbox_px"], outline=(255, 0, 0), width=3)
        d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
        d.text((3, 2), f"{os.path.basename(r['frame'])} {r['target']} "
               f"{r['answer'].get('confidence')} {r['score']}", fill=(255, 255, 0))
        tiles.append(im.resize((im.width // 2, im.height // 2)))
    tw, th = tiles[0].size
    cols = 4
    sheet = Image.new("RGB", (tw * cols, th * math.ceil(len(tiles) / cols)))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(out, quality=85)
    print(f"sheet: {out}")


if __name__ == "__main__":
    sys.exit(main())

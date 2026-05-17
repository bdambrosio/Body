#!/usr/bin/env python3
"""Ingest a vpr_record session dir → DINOv2 feature bank file.

Reads ``<session>/frames.jsonl`` and ``<session>/rgb/*.jpg``, runs
the DINOv2 extractor on every frame, and writes a single ``.pt``
bank file::

    {
      "features":   torch.float32 (N, D), L2-normalized,
      "poses":      torch.float32 (N, 3) [x_m, y_m, theta_rad],
      "timestamps": torch.float64 (N,)   wall-clock rgb_ts,
      "frame_idx":  torch.int64   (N,)   source row index in frames.jsonl,
      "metadata":   {
          "schema_version": 1,
          "source_session": "<path>",
          "extractor": {model_name, input_size, patch_size, feature_dim,
                        device, use_half_on_cuda},
          "n_frames_total":     <int>,   # rows in frames.jsonl
          "n_frames_extracted": <int>,   # rows that made it into the bank
          "build_started_ts":   <float>,
          "build_finished_ts":  <float>,
      },
    }

Phase 6.2 consumes this bank for cosine query + observation model.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from desktop.world_map.vpr.extractor import (
    DinoV2Extractor,
    ExtractorConfig,
    load_default_extractor,
)

logger = logging.getLogger(__name__)


def _load_frames(session_dir: Path) -> List[dict]:
    frames_path = session_dir / "frames.jsonl"
    if not frames_path.exists():
        raise FileNotFoundError(f"missing {frames_path}")
    out: List[dict] = []
    with frames_path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def _load_jpeg_rgb(jpeg_path: Path) -> np.ndarray:
    """JPEG → HWC uint8 RGB. Uses PIL (already a desktop dep)."""
    from PIL import Image
    img = Image.open(jpeg_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def _extract_batches(
    extractor: DinoV2Extractor,
    jpeg_paths: List[Path],
    batch_size: int,
) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    n = len(jpeg_paths)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        imgs = [_load_jpeg_rgb(p) for p in jpeg_paths[start:end]]
        batch = extractor.extract_batch(imgs)
        feats.append(batch.cpu())
        logger.info("vpr_build_bank: extracted %d/%d", end, n)
    return torch.cat(feats, dim=0) if feats else torch.empty((0,), dtype=torch.float32)


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="vpr_build_bank",
        description=(
            "Build a DINOv2 feature bank from a vpr_record session dir."
        ),
    )
    p.add_argument("session_dir", help="vpr_record session directory.")
    p.add_argument("--out", required=True,
                   help="Output bank file path (.pt).")
    p.add_argument("--model", default="dinov2_vitb14",
                   help="DINOv2 model name (default: dinov2_vitb14).")
    p.add_argument("--input-size", type=int, default=518,
                   help="Square input side in pixels (multiple of 14; "
                        "default 518).")
    p.add_argument("--device", default="auto",
                   help="cpu, cuda, or auto (default: auto).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for extraction (default: 16).")
    p.add_argument("--no-half", action="store_true",
                   help="Disable fp16 on CUDA (default: use fp16 on CUDA).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    session_dir = Path(os.path.expanduser(args.session_dir)).resolve()
    out_path = Path(os.path.expanduser(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = _load_frames(session_dir)
    if not frames:
        logger.error("no frames in %s", session_dir / "frames.jsonl")
        return 1

    device = _resolve_device(args.device)
    cfg = ExtractorConfig(
        model_name=args.model,
        input_size=args.input_size,
        patch_size=14,
        device=device,
        use_half_on_cuda=not args.no_half,
    )
    logger.info("vpr_build_bank: %d frames, device=%s, model=%s, input=%d",
                len(frames), device, cfg.model_name, cfg.input_size)
    t_build_start = time.time()
    extractor = load_default_extractor(cfg)

    jpeg_paths: List[Path] = []
    poses: List[Tuple[float, float, float]] = []
    timestamps: List[float] = []
    frame_indices: List[int] = []
    missing = 0
    for row in frames:
        jp = session_dir / row["jpeg"]
        if not jp.exists():
            missing += 1
            continue
        jpeg_paths.append(jp)
        p = row["pose_world"]
        poses.append((float(p["x_m"]), float(p["y_m"]), float(p["theta_rad"])))
        timestamps.append(float(row["rgb_ts"]))
        frame_indices.append(int(row["idx"]))
    if missing:
        logger.warning("vpr_build_bank: %d frames referenced missing JPEGs", missing)
    if not jpeg_paths:
        logger.error("no readable jpegs; aborting")
        return 1

    features = _extract_batches(extractor, jpeg_paths, args.batch_size)
    poses_t = torch.tensor(poses, dtype=torch.float32)
    ts_t = torch.tensor(timestamps, dtype=torch.float64)
    idx_t = torch.tensor(frame_indices, dtype=torch.int64)

    metadata = {
        "schema_version": 1,
        "source_session": str(session_dir),
        "extractor": asdict(cfg) | {"feature_dim": extractor.feature_dim},
        "n_frames_total": len(frames),
        "n_frames_extracted": int(features.shape[0]),
        "build_started_ts": t_build_start,
        "build_finished_ts": time.time(),
    }

    bank = {
        "features": features,
        "poses": poses_t,
        "timestamps": ts_t,
        "frame_idx": idx_t,
        "metadata": metadata,
    }
    torch.save(bank, out_path)
    logger.info(
        "vpr_build_bank: wrote %s — %d × %d-dim features (%.1f MB)",
        out_path, features.shape[0], features.shape[1],
        out_path.stat().st_size / 1e6,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Snapshot bundle exporter — writes a self-contained directory the
operator can hand to a notebook or eyeball directly.

Bundle layout:
    <base>/<session_id>/snap_<UTC ts>/
        layers.npz            # all raw layers + meta + pose trail
        summary.json          # bounds, cell counts, session id, ...
        height_full.png       # turbo, full pre-allocated grid
        height_crop.png       # turbo, cropped to populated bounds + 4-cell margin
        driveable_full.png    # clear/blocked/unknown, full grid
        driveable_crop.png    # clear/blocked/unknown, cropped
        traversal_full.png    # binary mask, full grid
        traversal_crop.png    # binary mask, cropped

PNGs use the same display flip as the live UI (world +x = up, world +y
= left), so a screenshot of the running app matches the saved file.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
from PyQt6.QtGui import QImage

if TYPE_CHECKING:
    from .controller import FuserController

logger = logging.getLogger(__name__)


# ── Colorization (mirrors map_views.py; kept private so the renderers
# stay decoupled — UI uses Qt's RGB image; export uses the same arrays
# but writes via QImage.save). ───────────────────────────────────────

_DRIVEABLE_CLEAR = (60, 170, 90)
_DRIVEABLE_BLOCKED = (180, 60, 60)
_DRIVEABLE_UNKNOWN = (60, 60, 60)
_TRAVERSAL_ON = (255, 220, 120)
_TRAVERSAL_OFF = (16, 16, 16)
_HEIGHT_NAN = (16, 16, 16)


def _turbo_rgb(x: np.ndarray) -> np.ndarray:
    """Anton Mikhailov's polynomial Turbo approximation, Apache 2.0."""
    r = 0.1357 + x*(4.5744 + x*(-42.3335 + x*(130.8988 + x*(-152.6574 + x*59.9032))))
    g = 0.0914 + x*(2.1915 + x*(  4.9271 + x*(-14.1846 + x*(  4.2755 + x* 2.8289))))
    b = 0.1067 + x*(12.5989 + x*(-60.1846 + x*(109.2364 + x*(-88.7840 + x*27.0060))))
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _height_to_rgb(height: np.ndarray, max_height_m: float = 2.2) -> np.ndarray:
    valid = ~np.isnan(height)
    norm = np.zeros_like(height, dtype=np.float32)
    np.divide(height, max_height_m, out=norm, where=valid)
    np.clip(norm, 0.0, 1.0, out=norm)
    rgb = _turbo_rgb(norm)
    rgb[~valid] = _HEIGHT_NAN
    return rgb


def _driveable_to_rgb(drive: np.ndarray) -> np.ndarray:
    nx, ny = drive.shape
    rgb = np.empty((nx, ny, 3), dtype=np.uint8)
    rgb[...] = _DRIVEABLE_UNKNOWN
    rgb[drive == 1] = _DRIVEABLE_CLEAR
    rgb[drive == 0] = _DRIVEABLE_BLOCKED
    return rgb


def _traversal_to_rgb(traversed_ts: np.ndarray) -> np.ndarray:
    on = ~np.isnan(traversed_ts)
    nx, ny = traversed_ts.shape
    rgb = np.empty((nx, ny, 3), dtype=np.uint8)
    rgb[...] = _TRAVERSAL_OFF
    rgb[on] = _TRAVERSAL_ON
    return rgb


def _save_rgb_png(rgb: np.ndarray, path: str) -> None:
    """rgb is (nx, ny, 3) uint8 in *world* (unflipped) orientation;
    we apply the display flip on the way out so the PNG matches what
    the live UI shows.
    """
    flipped = np.ascontiguousarray(rgb[::-1, ::-1])
    h, w, _ = flipped.shape
    qimg = QImage(
        flipped.data, w, h, 3 * w, QImage.Format.Format_RGB888,
    ).copy()
    if not qimg.save(path, "PNG"):
        raise IOError(f"QImage.save failed for {path}")


def _crop_layer(arr: np.ndarray, bounds_ij: Tuple[int, int, int, int],
                margin: int) -> np.ndarray:
    nx, ny = arr.shape
    i0, i1, j0, j1 = bounds_ij
    i0 = max(0, i0 - margin)
    j0 = max(0, j0 - margin)
    i1 = min(nx - 1, i1 + margin)
    j1 = min(ny - 1, j1 + margin)
    return arr[i0:i1 + 1, j0:j1 + 1]


def _default_base_dir() -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, "Body", "sessions")


def _summary(
    snap: Dict[str, Any],
    pose_trail: list,
    pose_source: str,
    extras: Dict[str, Any],
) -> Dict[str, Any]:
    meta = snap["meta"]
    bounds_ij = snap.get("bounds_ij")
    res = float(meta["resolution_m"])
    nx = int(meta["nx"])
    ny = int(meta["ny"])

    obs = int(np.count_nonzero(snap["observation_count"]))
    trav = int(np.count_nonzero(~np.isnan(snap["traversed_ts"])))
    drive = snap["driveable"]
    n_clear = int(np.count_nonzero(drive == 1))
    n_block = int(np.count_nonzero(drive == 0))
    n_unknown = int(drive.size - n_clear - n_block)

    bounds_world = None
    if bounds_ij is not None:
        i0, i1, j0, j1 = bounds_ij
        ox = float(meta["origin_x_m"])
        oy = float(meta["origin_y_m"])
        bounds_world = {
            "min_x_m": ox + i0 * res,
            "max_x_m": ox + (i1 + 1) * res,
            "min_y_m": oy + j0 * res,
            "max_y_m": oy + (j1 + 1) * res,
            "extent_x_m": (i1 - i0 + 1) * res,
            "extent_y_m": (j1 - j0 + 1) * res,
        }

    pose_summary: Optional[Dict[str, Any]] = None
    if pose_trail:
        xs = [p[0] for p in pose_trail]
        ys = [p[1] for p in pose_trail]
        # Path length as a sanity check on "did the robot actually move?"
        path_len = 0.0
        for i in range(1, len(pose_trail)):
            dx = pose_trail[i][0] - pose_trail[i - 1][0]
            dy = pose_trail[i][1] - pose_trail[i - 1][1]
            path_len += float((dx * dx + dy * dy) ** 0.5)
        pose_summary = {
            "n_samples": len(pose_trail),
            "x_min_m": min(xs),
            "x_max_m": max(xs),
            "y_min_m": min(ys),
            "y_max_m": max(ys),
            "path_length_m": path_len,
        }

    return {
        "session_id": snap.get("session_id"),
        "pose_source": pose_source,
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": {
            "resolution_m": res,
            "nx": nx,
            "ny": ny,
            "extent_m": nx * res,
            "frame": meta.get("frame", "world"),
            "vote_margin": meta.get("vote_margin"),
            "footprint_radius_m": meta.get("footprint_radius_m"),
        },
        "bounds_ij": list(bounds_ij) if bounds_ij is not None else None,
        "bounds_world": bounds_world,
        "cells": {
            "observed": obs,
            "traversed": trav,
            "driveable_clear": n_clear,
            "driveable_blocked": n_block,
            "driveable_unknown": n_unknown,
        },
        "pose_trail": pose_summary,
        "extras": extras,
    }


def write_bundle(
    controller: "FuserController",
    *,
    base_dir: Optional[str] = None,
) -> str:
    """Write a snapshot bundle. Returns the bundle directory path.

    Safe to call from the UI thread; ~few MB of I/O. Raises on
    filesystem errors so the caller can surface them.
    """
    snap = controller.grid.snapshot_for_export()
    if snap is None or snap.get("bounds_ij") is None:
        # No data fused yet — still write the bundle so the operator
        # gets a "I tried, here's what was empty" artifact.
        snap = snap or _empty_snap_from(controller)

    pose_trail = controller.pose_trail()
    pose_source_name = controller.pose_source.source_name()

    out_root = base_dir or _default_base_dir()
    sid = snap.get("session_id") or "unknown"
    ts_dir = time.strftime("snap_%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, sid, ts_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── Layers (.npz) ────────────────────────────────────────────────
    bounds_ij = snap.get("bounds_ij")
    npz_kwargs: Dict[str, Any] = {
        "max_height_m": snap["max_height_m"],
        "clear_votes": snap["clear_votes"],
        "block_votes": snap["block_votes"],
        "traversed_ts": snap["traversed_ts"],
        "last_observed_ts": snap["last_observed_ts"],
        "observation_count": snap["observation_count"],
        "driveable": snap["driveable"],
        "meta_json": np.array(json.dumps(snap["meta"])),
        "session_id": np.array(snap.get("session_id") or ""),
        "bounds_ij": (
            np.array(bounds_ij, dtype=np.int32)
            if bounds_ij is not None else np.array([], dtype=np.int32)
        ),
        "pose_trail": (
            np.array(pose_trail, dtype=np.float64).reshape(-1, 3)
            if pose_trail else np.empty((0, 3), dtype=np.float64)
        ),
    }
    np.savez_compressed(os.path.join(out_dir, "layers.npz"), **npz_kwargs)

    # ── PNGs ─────────────────────────────────────────────────────────
    height_rgb = _height_to_rgb(snap["max_height_m"])
    drive_rgb = _driveable_to_rgb(snap["driveable"])
    trav_rgb = _traversal_to_rgb(snap["traversed_ts"])
    _save_rgb_png(height_rgb, os.path.join(out_dir, "height_full.png"))
    _save_rgb_png(drive_rgb, os.path.join(out_dir, "driveable_full.png"))
    _save_rgb_png(trav_rgb, os.path.join(out_dir, "traversal_full.png"))

    if bounds_ij is not None:
        margin = 4
        h_crop = _crop_layer(snap["max_height_m"], bounds_ij, margin)
        d_crop = _crop_layer(snap["driveable"], bounds_ij, margin)
        t_crop = _crop_layer(snap["traversed_ts"], bounds_ij, margin)
        _save_rgb_png(_height_to_rgb(h_crop),
                      os.path.join(out_dir, "height_crop.png"))
        _save_rgb_png(_driveable_to_rgb(d_crop),
                      os.path.join(out_dir, "driveable_crop.png"))
        _save_rgb_png(_traversal_to_rgb(t_crop),
                      os.path.join(out_dir, "traversal_crop.png"))

    # ── Summary ──────────────────────────────────────────────────────
    extras: Dict[str, Any] = {}
    try:
        st = controller.status_summary()
        extras["status_summary"] = st
    except Exception:
        logger.exception("status_summary failed during snapshot; continuing")
    summary = _summary(snap, pose_trail, pose_source_name, extras)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)

    logger.info(
        f"snapshot bundle: {out_dir} "
        f"({summary['cells']['observed']} observed cells, "
        f"trail={len(pose_trail)})"
    )
    return out_dir


def _empty_snap_from(controller: "FuserController") -> Dict[str, Any]:
    """Best-effort placeholder when the grid has no bounds yet — we
    still want to emit a bundle so the operator sees the click did
    something.
    """
    g = controller.grid
    n = g.n_cells
    return {
        "max_height_m": np.full((n, n), np.nan, dtype=np.float32),
        "clear_votes": np.zeros((n, n), dtype=np.int32),
        "block_votes": np.zeros((n, n), dtype=np.int32),
        "traversed_ts": np.full((n, n), np.nan, dtype=np.float32),
        "last_observed_ts": np.full((n, n), np.nan, dtype=np.float32),
        "observation_count": np.zeros((n, n), dtype=np.int32),
        "driveable": np.full((n, n), -1, dtype=np.int8),
        "meta": {
            "resolution_m": g.resolution_m,
            "origin_x_m": g.origin_x_m,
            "origin_y_m": g.origin_y_m,
            "nx": n,
            "ny": n,
            "frame": "world",
        },
        "session_id": g.session_id,
        "bounds_ij": None,
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

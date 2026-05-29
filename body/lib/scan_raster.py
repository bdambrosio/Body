"""Single-frame lidar scan → body-frame occupancy grid (Tier-3 substrate).

The live 360° lidar scan *is* the local map for the reactive driver: the
freshest, motion-correct, body-centric picture of "what's around me at
lidar height, right now." This rasterizes one scan into an int8 grid
(-1 unknown, 0 blocked, 1 clear) that the swept-footprint check consumes
— no cross-frame accumulation, no decay, no odom (cf. the fused
``local_map`` vote grid, whose temporal gating is not motion-compensated
and so lags/misses blocks while moving).

Pure NumPy, no zenoh — imported by both the Pi driver (``body.local_drive``)
and the desktop debug console so both see exactly the same grid.

Design note — *no-return beams clear to the horizon.* A beam with no echo
within range means open space, so it clears free out to ``max_clear_range_m``
(unlike the mapping ``local_map``, which leaves no-return beams unknown).
This is what lets Tier-3 drive down an open hallway/doorway without the
empty-region guard refusing. The trade-off: a near surface that absorbs the
beam (black/specular/too-close) returns nothing and is then cleared as open
— a genuine false-clear risk that the depth low-obstacle overlay (later) is
meant to cover. Tune ``max_clear_range_m`` down if absorptive false-clears
show up before the overlay lands.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ScanRasterConfig:
    resolution_m: float = 0.08
    half_extent_m: float = 2.5         # body-centered square window
    lidar_x_m: float = 0.0             # lidar mount offset in body frame
    lidar_y_m: float = 0.0
    lidar_yaw_rad: float = 0.0
    range_min_m: float = 0.05          # ignore returns nearer than this
    range_max_m: float = 8.0           # returns beyond → treated as no-return
    max_clear_range_m: float = 6.0     # cap a beam's free-space clearing
    clear_buffer_cells: float = 2.0    # stop clearing this many cells short of a hit


def _meta(cfg: ScanRasterConfig, n: int) -> Dict[str, Any]:
    return {
        "resolution_m": cfg.resolution_m,
        "origin_x_m": -cfg.half_extent_m,
        "origin_y_m": -cfg.half_extent_m,
        "nx": n,
        "ny": n,
        "frame": "body",
    }


def rasterize_scan(
    ranges: Optional[Sequence[Any]],
    angle_min: float,
    angle_increment: float,
    cfg: Optional[ScanRasterConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """One scan → (int8 grid (n, n): -1 unknown / 0 blocked / 1 clear, meta).

    Robot at body (0, 0) = grid center cell, +x forward, +y left. Each valid
    return stamps a blocked cell at its endpoint; cells the beam swept through
    (to ``clear_buffer_cells`` short of the hit, or to ``max_clear_range_m``
    for a no-return beam) are cleared; everything else is unknown. Returns an
    all-unknown grid if there are no ranges (fail-safe: the swept check's
    empty-region guard then blocks).
    """
    cfg = cfg or ScanRasterConfig()
    res = cfg.resolution_m
    n = 2 * int(math.ceil(cfg.half_extent_m / res))
    origin = -cfg.half_extent_m
    grid = np.full((n, n), -1, dtype=np.int8)

    if not ranges:
        return grid, _meta(cfg, n)

    rs = np.array(
        [
            float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else 0.0
            for v in ranges
        ],
        dtype=np.float64,
    )
    N = rs.shape[0]
    thetas = angle_min + np.arange(N, dtype=np.float64) * angle_increment + cfg.lidar_yaw_rad
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    lx, ly = cfg.lidar_x_m, cfg.lidar_y_m

    has_return = (rs > cfg.range_min_m) & (rs <= cfg.range_max_m)
    # No-return / beyond-range beams clear to the horizon (open space).
    r_clear = np.where(has_return, np.minimum(rs, cfg.max_clear_range_m),
                       cfg.max_clear_range_m)

    # Ray-trace clear: sample every cell along each beam.
    max_samples = int(math.ceil(cfg.max_clear_range_m / res))
    if max_samples > 0:
        sample_d = (np.arange(1, max_samples + 1, dtype=np.float64) * res)[None, :]
        valid = sample_d < (r_clear[:, None] - cfg.clear_buffer_cells * res)
        sx = lx + sample_d * cos_t[:, None]
        sy = ly + sample_d * sin_t[:, None]
        ix = np.floor((sx - origin) / res).astype(np.int32)
        iy = np.floor((sy - origin) / res).astype(np.int32)
        inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & valid
        if np.any(inside):
            grid[ix[inside], iy[inside]] = 1

    # Hits (blocked) — written after clears so blocked wins on conflict.
    if np.any(has_return):
        rr = rs[has_return]
        hx = lx + rr * cos_t[has_return]
        hy = ly + rr * sin_t[has_return]
        ix = np.floor((hx - origin) / res).astype(np.int32)
        iy = np.floor((hy - origin) / res).astype(np.int32)
        inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        if np.any(inside):
            grid[ix[inside], iy[inside]] = 0

    return grid, _meta(cfg, n)

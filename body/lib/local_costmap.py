"""Costmap for the body-frame local scan grid (pure Python + numpy).

Pi-side sibling of ``desktop/world_map/costmap.py``, stripped to what the local
planner needs: footprint-radius lethal dilation + a graded clearance halo over
the live ``scan_raster`` grid. No pose, no traversal protection, no forward
cone, no UI rendering — the local grid is robot-centred and rebuilt every tick.

Output ``cost``/``lethal`` feed ``body/lib/astar.py``. The halo cost (decaying
with distance from lethal) is what makes A* prefer more clearance when there's
room, without forbidding tight passages.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class LocalCostmapConfig:
    # The SINGLE footprint model: cells whose centre is within this of any
    # blocked cell are lethal (robot centre cannot go there). ≈ true half-width
    # (7.5 cm) + margin. Keep the swept-veto footprint ≤ this so they agree.
    footprint_radius_m: float = 0.11
    safety_margin_m: float = 0.08      # full-cost band just outside lethal
    inflation_decay_m: float = 0.20    # exp halo length-scale (clearance pref)
    halo_max: float = 100.0
    unknown_cost: float = 25.0         # prefer observed-clear; route unknown if needed
    unknown_is_lethal: bool = False    # scan_raster clears no-return to horizon
    denoise: bool = True
    denoise_min_neighbors: int = 2


@dataclass
class LocalCostmap:
    cost: np.ndarray            # float32; +inf on lethal
    lethal: np.ndarray          # bool
    unknown: np.ndarray         # bool
    distance_m: np.ndarray      # float32, distance to nearest blocked (capped)
    meta: Dict[str, Any]


def build_local_costmap(
    grid: np.ndarray, meta: Dict[str, Any],
    cfg: Optional[LocalCostmapConfig] = None,
) -> LocalCostmap:
    """Build a costmap from an int8 scan grid (-1 unknown / 0 blocked / 1 clear)."""
    cfg = cfg or LocalCostmapConfig()
    res = float(meta["resolution_m"])
    if res <= 0:
        raise ValueError(f"bad resolution_m={res}")

    blocked = (grid == 0)
    unknown = (grid == -1)
    if cfg.denoise:
        blocked = drop_speckle(blocked, min_neighbors=cfg.denoise_min_neighbors)

    halo_extent_m = (
        cfg.footprint_radius_m + cfg.safety_margin_m + 5.0 * cfg.inflation_decay_m
    )
    max_cells = max(2, int(math.ceil(halo_extent_m / res)))
    dist_m = (wavefront_distance(blocked, max_cells=max_cells) * res).astype(np.float32)

    lethal = blocked | (dist_m < cfg.footprint_radius_m)
    if cfg.unknown_is_lethal:
        lethal = lethal | unknown

    cost = np.zeros_like(dist_m, dtype=np.float32)
    halo = (~lethal) & (dist_m < halo_extent_m)
    if np.any(halo):
        d_excess = dist_m[halo] - cfg.footprint_radius_m
        in_safety = d_excess <= cfg.safety_margin_m
        decay_d = np.maximum(0.0, d_excess - cfg.safety_margin_m)
        cost[halo] = np.where(
            in_safety, cfg.halo_max,
            cfg.halo_max * np.exp(-decay_d / cfg.inflation_decay_m))
    cost[unknown] = cfg.unknown_cost
    cost[lethal] = np.inf

    return LocalCostmap(cost=cost, lethal=lethal, unknown=unknown,
                        distance_m=dist_m, meta=dict(meta))


# ── numpy helpers (shared with the planner) ─────────────────────────


def _shift(arr: np.ndarray, di: int, dj: int, fill) -> np.ndarray:
    out = np.full_like(arr, fill)
    h, w = arr.shape
    si0 = max(0, -di); si1 = min(h, h - di)
    sj0 = max(0, -dj); sj1 = min(w, w - dj)
    di0 = max(0, di); di1 = min(h, h + di)
    dj0 = max(0, dj); dj1 = min(w, w + dj)
    if si1 > si0 and sj1 > sj0:
        out[di0:di1, dj0:dj1] = arr[si0:si1, sj0:sj1]
    return out


def dilate_bool(mask: np.ndarray, *, iters: int) -> np.ndarray:
    """8-connected boolean dilation, `iters` times."""
    out = mask
    for _ in range(iters):
        nxt = out
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nxt = nxt | _shift(out, di, dj, fill=False)
        out = nxt
    return out


def drop_speckle(mask: np.ndarray, *, min_neighbors: int = 2) -> np.ndarray:
    """Drop True cells with fewer than `min_neighbors` True 8-neighbors."""
    count = np.zeros(mask.shape, dtype=np.int8)
    src = mask.astype(np.int8)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            count += _shift(src, di, dj, fill=0)
    return mask & (count >= int(min_neighbors))


def wavefront_distance(blocked: np.ndarray, *, max_cells: int) -> np.ndarray:
    """Iterative-dilation Euclidean distance transform (cells), capped."""
    SQRT2 = math.sqrt(2.0)
    INF = float(max_cells + 1)
    dist = np.where(blocked, 0.0, INF).astype(np.float32)
    if max_cells <= 0:
        return dist
    neighbors = [
        (-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
        (0, -1, 1.0),                  (0, 1, 1.0),
        (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2),
    ]
    for _ in range(max_cells):
        cur = dist
        for di, dj, w in neighbors:
            cur = np.minimum(cur, _shift(dist, di, dj, fill=INF) + np.float32(w))
        if np.array_equal(cur, dist):
            break
        dist = cur
    return dist

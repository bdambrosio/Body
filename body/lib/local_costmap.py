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
    # Footprint quantization pad (T12): the path runs through cell CENTRES but
    # the robot (and the swept veto) is continuous, so a cell whose centre is
    # just outside the footprint can still put the body within it of an
    # obstacle — up to half a cell diagonal (res/√2). True → lethal radius is
    # footprint_radius_m + res/√2, so every A* path keeps the veto's
    # footprint clear with that margin (the planner/veto invariant).
    quantization_pad: bool = True
    safety_margin_m: float = 0.08      # full-cost band just outside lethal
    inflation_decay_m: float = 0.20    # exp halo length-scale (clearance pref)
    halo_max: float = 100.0
    # Unknown cells cost at least this, and never LESS than the clearance halo
    # they sit in (L2): the unknown pre-hit band in front of a wall must not be
    # the cheapest way past it.
    unknown_cost: float = 25.0         # prefer observed-clear; route unknown if needed
    unknown_is_lethal: bool = False    # scan_raster clears no-return to horizon
    # Speckle removal (T1). OFF by default: the planner must never see FEWER
    # obstacles than the swept veto, and at 8 cm cells legs/posts/jamb edges
    # ARE 1-2 cells. When enabled it only drops fully isolated cells (never
    # line ends) farther than denoise_min_range_m from the robot.
    denoise: bool = False
    denoise_min_neighbors: int = 1     # clamped to ≤ 1 (line ends are kept)
    denoise_min_range_m: float = 1.0


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
        isolated = blocked & ~drop_speckle(
            blocked, min_neighbors=min(1, int(cfg.denoise_min_neighbors)))
        if np.any(isolated):
            far = _cell_range_m(meta, blocked.shape) > cfg.denoise_min_range_m
            blocked = blocked & ~(isolated & far)

    lethal_r = lethal_radius_m(cfg, res)
    halo_extent_m = lethal_r + cfg.safety_margin_m + 5.0 * cfg.inflation_decay_m
    max_cells = max(2, int(math.ceil(halo_extent_m / res)))
    dist_m = (wavefront_distance(blocked, max_cells=max_cells) * res).astype(np.float32)

    # Exact-disk dilation for the lethal set (the chamfer wavefront distance
    # over-estimates off-axis distances, which is exactly where corner clips
    # hide); the wavefront distance only grades the halo.
    lethal = blocked | dilate_disk(blocked, lethal_r / res)
    if cfg.unknown_is_lethal:
        lethal = lethal | unknown

    cost = np.zeros_like(dist_m, dtype=np.float32)
    halo = (~lethal) & (dist_m < halo_extent_m)
    if np.any(halo):
        d_excess = np.maximum(0.0, dist_m[halo] - lethal_r)
        in_safety = d_excess <= cfg.safety_margin_m
        decay_d = np.maximum(0.0, d_excess - cfg.safety_margin_m)
        cost[halo] = np.where(
            in_safety, cfg.halo_max,
            cfg.halo_max * np.exp(-decay_d / cfg.inflation_decay_m))
    cost[unknown] = np.maximum(cost[unknown], np.float32(cfg.unknown_cost))
    cost[lethal] = np.inf

    return LocalCostmap(cost=cost, lethal=lethal, unknown=unknown,
                        distance_m=dist_m, meta=dict(meta))


def lethal_radius_m(cfg: LocalCostmapConfig, res: float) -> float:
    """Robot-centre exclusion radius around a blocked cell centre (strict <)."""
    pad = res / math.sqrt(2.0) if cfg.quantization_pad else 0.0
    return float(cfg.footprint_radius_m) + pad


def _cell_range_m(meta: Dict[str, Any], shape) -> np.ndarray:
    res = float(meta["resolution_m"])
    ox = float(meta.get("origin_x_m", 0.0))
    oy = float(meta.get("origin_y_m", 0.0))
    xs = ox + (np.arange(shape[0], dtype=np.float64) + 0.5) * res
    ys = oy + (np.arange(shape[1], dtype=np.float64) + 0.5) * res
    return np.hypot(xs[:, None], ys[None, :])


# ── numpy helpers (shared with the planner) ─────────────────────────


def dilate_disk(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    """True wherever a True cell's centre is strictly within ``radius_cells``
    (Euclidean, cell units) — an exact disk dilation via shifted ORs."""
    out = mask.copy()
    r = int(math.ceil(radius_cells))
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if (di == 0 and dj == 0) or math.hypot(di, dj) >= radius_cells:
                continue
            out |= _shift(mask, di, dj, fill=False)
    return out


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

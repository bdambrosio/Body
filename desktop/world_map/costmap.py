"""Costmap construction for the planner.

Inputs: a snapshot from `WorldGrid.snapshot_for_ui()` carrying the
int8 driveable layer (1=clear, 0=blocked, -1=unknown).

Output: a `Costmap` with three layers aligned to the same world frame:
    - lethal: bool, robot must not enter (blocked or within
      footprint+safety-margin of blocked).
    - cost: float32, cost of routing through each non-lethal cell.
      Free clear is 0; halo around lethal decays exponentially toward 0;
      unknown gets a configurable cost so the planner prefers
      observed-clear when one exists, but is willing to plan into
      unknown when nothing else gets to the goal.
    - unknown: bool, cells never observed (passthrough from driveable).

Implementation notes
--------------------
We avoid pulling scipy into the desktop deps; the operations needed
(3×3 morphological open + a small-radius distance transform) are
cheap to implement in numpy:

* 3×3 binary erode/dilate: shift-and-AND / shift-and-OR over the 8
  neighbors. ~10 lines each.
* Distance transform: iterative wavefront dilation, capped at the
  worst-case useful radius (lethal radius + halo decay × ~5). For
  the parameters we use (lethal_radius ≈ 0.25 m, halo_decay ≈ 0.30 m),
  ~24 cells of propagation is plenty. That's 24 × 8 = 192 numpy ops
  on a 500×500 grid per costmap build — fast enough to rebuild every
  UI tick.

If costmap construction ever shows up in a profile, the natural next
step is to crop to the bounded region (`snapshot["bounds_ij"]` plus
a halo) before building, since cells far from any data don't need
processing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ── Config ──────────────────────────────────────────────────────────


@dataclass
class CostmapConfig:
    # Robot footprint radius. Cells whose center is within this
    # distance of any blocked cell are LETHAL — robot literally
    # cannot fit. Smaller than the previous "footprint + safety"
    # combo: safety now lives in the halo, not in lethal, so a
    # speckle of blocked cells doesn't paint a giant no-go disk.
    footprint_radius_m: float = 0.15
    # Beyond the footprint, a band of width safety_margin_m carries
    # the maximum halo cost (planner strongly prefers to avoid).
    # Past that, cost decays exponentially.
    safety_margin_m: float = 0.10

    # Exponential decay length-scale beyond the safety band.
    inflation_decay_m: float = 0.30
    halo_max: float = 100.0

    # Cost assigned to cells that have never been observed. Low enough
    # to plan through if no observed-clear path exists; high enough
    # that "go through observed clear" is preferred.
    unknown_cost: float = 25.0

    # Speckle filter on the blocked layer before inflation. Drops
    # blocked cells whose number of blocked 8-neighbors is below
    # this threshold. denoise_min_neighbors=1 drops only fully-
    # isolated cells (preserves walls); =2 also drops sparse 2-cell
    # pairs that would otherwise inflate into ~50-cell lethal disks.
    # Walls and definite blobs (any cell with 2+ co-line/co-blob
    # neighbors) survive at either threshold.
    denoise: bool = True
    denoise_min_neighbors: int = 2


# ── Output type ────────────────────────────────────────────────────


@dataclass
class Costmap:
    cost: np.ndarray            # float32, finite for non-lethal
    lethal: np.ndarray          # bool
    unknown: np.ndarray         # bool
    distance_m: np.ndarray      # float32, distance to nearest blocked (capped)
    meta: Dict[str, Any]        # resolution_m, origin_x_m, origin_y_m, nx, ny
    bounds_ij: Optional[Tuple[int, int, int, int]] = None
    config: CostmapConfig = field(default_factory=CostmapConfig)


# ── Public API ─────────────────────────────────────────────────────


def build_costmap(
    snap: Dict[str, Any],
    config: Optional[CostmapConfig] = None,
) -> Costmap:
    """Construct a Costmap from a `WorldGrid.snapshot_for_ui()` result.

    `snap` must carry a `driveable` int8 layer (1=clear, 0=blocked,
    -1=unknown) and a `meta` dict with `resolution_m`. Free of side
    effects; safe to call on the UI thread once per redraw.
    """
    cfg = config or CostmapConfig()
    drive = snap["driveable"]
    meta = snap["meta"]
    res = float(meta["resolution_m"])
    if res <= 0:
        raise ValueError(f"bad resolution_m={res} in costmap input")

    blocked = (drive == 0)
    unknown = (drive == -1)

    if cfg.denoise:
        blocked = _drop_speckle(
            blocked, min_neighbors=cfg.denoise_min_neighbors,
        )

    # Lethal radius is footprint only — the safety margin contributes
    # to halo cost, not to "robot cannot enter." This stops a single
    # speckle from painting a 50-cell lethal disk.
    lethal_radius_m = cfg.footprint_radius_m
    safety_band_m = cfg.safety_margin_m
    halo_extent_m = (
        lethal_radius_m + safety_band_m + 5.0 * cfg.inflation_decay_m
    )
    max_cells = max(2, int(math.ceil(halo_extent_m / res)))
    dist_cells = _wavefront_distance(blocked, max_cells=max_cells)
    dist_m = (dist_cells * res).astype(np.float32)

    # Lethal: blocked plus footprint-radius dilation.
    lethal = blocked | (dist_m < lethal_radius_m)

    # Cost field. Inside the safety band (lethal_radius..lethal_radius
    # + safety_margin) the halo is at full halo_max — strongly avoided
    # but not lethal. Beyond, exponential decay.
    cost = np.zeros_like(dist_m, dtype=np.float32)
    halo = (~lethal) & (dist_m < halo_extent_m)
    if np.any(halo):
        d_excess = dist_m[halo] - lethal_radius_m
        in_safety = d_excess <= safety_band_m
        decay_d = np.maximum(0.0, d_excess - safety_band_m)
        cost[halo] = np.where(
            in_safety,
            cfg.halo_max,
            cfg.halo_max * np.exp(-decay_d / cfg.inflation_decay_m),
        )
    cost[unknown] = cfg.unknown_cost
    # Lethal cells: leave as +inf so any consumer that ignores `lethal`
    # but uses `cost` gets sane behavior.
    cost[lethal] = np.inf

    return Costmap(
        cost=cost,
        lethal=lethal,
        unknown=unknown,
        distance_m=dist_m,
        meta=dict(meta),
        bounds_ij=snap.get("bounds_ij"),
        config=cfg,
    )


# ── Numpy helpers ──────────────────────────────────────────────────


def _shift(arr: np.ndarray, di: int, dj: int, fill) -> np.ndarray:
    """Return arr shifted by (di, dj). Vacated cells get `fill`."""
    out = np.full_like(arr, fill)
    h, w = arr.shape
    si0 = max(0, -di); si1 = min(h, h - di)
    sj0 = max(0, -dj); sj1 = min(w, w - dj)
    di0 = max(0,  di); di1 = min(h, h + di)
    dj0 = max(0,  dj); dj1 = min(w, w + dj)
    if si1 > si0 and sj1 > sj0:
        out[di0:di1, dj0:dj1] = arr[si0:si1, sj0:sj1]
    return out


def _drop_speckle(
    mask: np.ndarray, *, min_neighbors: int = 2,
) -> np.ndarray:
    """Drop True cells whose count of True 8-neighbors is below
    min_neighbors. With min_neighbors=1, drops only fully-isolated
    specks (preserves 1-cell-thick walls). With min_neighbors=2,
    also drops sparse 2-cell pairs that would each inflate to a
    50-cell lethal disk under footprint+safety inflation. Walls
    survive either way: each cell on a line has 2 line-direction
    neighbors.
    """
    count = np.zeros(mask.shape, dtype=np.int8)
    src = mask.astype(np.int8)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            count += _shift(src, di, dj, fill=0)
    return mask & (count >= int(min_neighbors))


def _wavefront_distance(blocked: np.ndarray, *, max_cells: int) -> np.ndarray:
    """Iterative-dilation Euclidean distance transform, capped at
    `max_cells`. Cells beyond that get max_cells + 1.

    Uses 8-neighbor wavefront with weights 1.0 / √2 to approximate
    Euclidean. Error vs. exact EDT is < 4 % at the radii we care
    about — fine for inflation purposes.
    """
    SQRT2 = math.sqrt(2.0)
    INF = float(max_cells + 1)
    dist = np.where(blocked, 0.0, INF).astype(np.float32)
    if max_cells <= 0:
        return dist

    # Small precomputed neighbor list.
    neighbors = [
        (-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
        ( 0, -1, 1.0),                 ( 0, 1, 1.0),
        ( 1, -1, SQRT2), ( 1, 0, 1.0), ( 1, 1, SQRT2),
    ]
    for _ in range(max_cells):
        prev = dist
        cur = prev
        for di, dj, w in neighbors:
            shifted = _shift(prev, di, dj, fill=INF) + np.float32(w)
            cur = np.minimum(cur, shifted)
        if np.array_equal(cur, dist):
            break
        dist = cur
    return dist


# ── Visualization helper (used by WorldCostmapView) ────────────────


def costmap_to_rgb(cm: Costmap) -> np.ndarray:
    """Render a Costmap as a uint8 (nx, ny, 3) RGB image.

    Color choices, picked for operator legibility:
      lethal       → bright red
      halo (cost)  → red→orange→yellow gradient on cost magnitude
      unknown      → medium gray
      free (cost 0) → near-black green
    """
    nx, ny = cm.cost.shape
    rgb = np.zeros((nx, ny, 3), dtype=np.uint8)

    # Free baseline.
    rgb[:] = (10, 32, 16)

    # Halo: scale cost to [0, 1] over [0, halo_max].
    halo_max = float(cm.config.halo_max)
    finite = np.isfinite(cm.cost)
    halo = finite & (cm.cost > 0) & (~cm.unknown)
    if np.any(halo):
        t = np.clip(cm.cost[halo] / max(halo_max, 1e-6), 0.0, 1.0)
        # red rises with t; green stays moderate; blue stays low.
        r = (60 + 195 * t).astype(np.uint8)
        g = (90 + 110 * (1 - t)).astype(np.uint8)
        b = np.full_like(r, 30)
        rgb_halo = np.stack([r, g, b], axis=-1)
        rgb[halo] = rgb_halo

    # Unknown.
    rgb[cm.unknown] = (90, 90, 90)

    # Lethal — last so it overwrites halo edges.
    rgb[cm.lethal] = (235, 50, 50)

    return rgb

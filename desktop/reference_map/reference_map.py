"""ReferenceMap — frozen 2D occupancy grid for MCL and static planning.

Saved as ``reference_map.npz`` with precomputed likelihood and distance
fields for fast beam-model localization.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

MAP_VERSION = 1

# Log-odds defaults for mapping ray casting.
LOG_ODDS_OCC = 0.85
LOG_ODDS_FREE = -0.4
LOG_ODDS_MIN = -4.0
LOG_ODDS_MAX = 4.0

# Occupied threshold when exporting from log-odds (the hysteresis "high"
# seed threshold).
OCCUPIED_LOG_ODDS_THRESHOLD = 0.5

# Hysteresis "low" threshold. Faint cells between this and the occupied
# threshold are promoted to occupied only where 8-connected to a confident
# (seed) cell. Pairs with the range-scaled OccupancyBuilder, which leaves
# one-off far hits faint on purpose: this recovers faint-but-connected wall
# continuations while leaving isolated faint returns out.
OCCUPIED_LOG_ODDS_LOW_THRESHOLD = 0.25

# Likelihood field Gaussian σ as multiple of cell size.
LIKELIHOOD_SIGMA_CELLS = 2.0


@dataclass
class ReferenceMap:
    """Read-only world-frame 2D map for localization and planning."""

    occupancy_log_odds: np.ndarray  # float32 (nx, ny)
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    likelihood_field: np.ndarray  # float32 (nx, ny), precomputed
    distance_field_m: np.ndarray  # float32 (nx, ny), EDT to obstacle
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectory: Optional[np.ndarray] = None  # (N, 4) ts, x, y, theta
    # Operator keep-out mask (bool, nx×ny). POLICY layer for planning only:
    # folded into the costmap's lethal set, NEVER into the localization
    # likelihood/distance fields (those are built from occupancy alone).
    # None == no keep-out cells.
    nogo_mask: Optional[np.ndarray] = None

    @property
    def nx(self) -> int:
        return int(self.occupancy_log_odds.shape[0])

    @property
    def ny(self) -> int:
        return int(self.occupancy_log_odds.shape[1])

    @property
    def extent_m(self) -> float:
        return self.nx * self.resolution_m

    def occupied_mask(self, *, threshold: float = OCCUPIED_LOG_ODDS_THRESHOLD) -> np.ndarray:
        return self.occupancy_log_odds > threshold

    def driveable_int8(self) -> np.ndarray:
        return driveable_from_occupancy(self.occupancy_log_odds)

    def nogo_or_empty(self) -> np.ndarray:
        """bool (nx, ny) keep-out mask; all-False when unset."""
        if self.nogo_mask is None:
            return np.zeros((self.nx, self.ny), dtype=bool)
        return self.nogo_mask.astype(bool)

    def snapshot_for_ui(self) -> Dict[str, Any]:
        """Shape compatible with ``build_costmap`` / map views."""
        drive = self.driveable_int8()
        bounds = self._bounds_ij()
        grid = np.full(drive.shape, np.nan, dtype=np.float32)
        return {
            "grid": grid,
            "driveable": drive,
            "nogo": self.nogo_or_empty(),
            "meta": {
                "resolution_m": self.resolution_m,
                "origin_x_m": self.origin_x_m,
                "origin_y_m": self.origin_y_m,
                "nx": self.nx,
                "ny": self.ny,
                "frame": "world",
            },
            "session_id": self.session_id,
            "bounds_ij": bounds,
        }

    def _bounds_ij(self) -> Optional[Tuple[int, int, int, int]]:
        known = np.isfinite(self.occupancy_log_odds) & (
            np.abs(self.occupancy_log_odds) > 1e-6
        )
        if not np.any(known):
            return None
        ii, jj = np.where(known)
        return int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max())

    def world_to_cell(self, x_w: float, y_w: float) -> Tuple[int, int]:
        i = int(math.floor((x_w - self.origin_x_m) / self.resolution_m + 1e-9))
        j = int(math.floor((y_w - self.origin_y_m) / self.resolution_m + 1e-9))
        return i, j

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny


def driveable_from_occupancy(log_odds: np.ndarray) -> np.ndarray:
    """int8: 1=clear, 0=blocked, -1=unknown."""
    out = np.full(log_odds.shape, -1, dtype=np.int8)
    out[log_odds > OCCUPIED_LOG_ODDS_THRESHOLD] = 0
    out[log_odds < -OCCUPIED_LOG_ODDS_THRESHOLD] = 1
    return out


def build_likelihood_field(
    occupied: np.ndarray,
    *,
    resolution_m: float,
    sigma_cells: float = LIKELIHOOD_SIGMA_CELLS,
) -> np.ndarray:
    """Borenstein/Konolige-style likelihood field from occupied cells.

    Each occupied cell splats a Gaussian peak; result is normalized to
    [0, 1] for use as beam-endpoint score in MCL.
    """
    occ = occupied.astype(np.float32)
    if not np.any(occ):
        return np.zeros_like(occ)
    sigma_m = sigma_cells * resolution_m
    radius_cells = max(1, int(math.ceil(3.0 * sigma_cells)))
    size = 2 * radius_cells + 1
    yy, xx = np.mgrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    kernel = np.exp(
        -(xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2)
        * (resolution_m ** 2) / (2.0 * sigma_m ** 2)
    ).astype(np.float32)
    # Manual splat for sparse maps (scipy-free).
    nx, ny = occ.shape
    field = np.zeros((nx, ny), dtype=np.float32)
    occ_i, occ_j = np.nonzero(occ > 0.5)
    kh, kw = kernel.shape
    half_h, half_w = kh // 2, kw // 2
    for oi, oj in zip(occ_i, occ_j):
        i0 = max(0, oi - half_h)
        i1 = min(nx, oi + half_h + 1)
        j0 = max(0, oj - half_w)
        j1 = min(ny, oj + half_w + 1)
        ki0 = i0 - (oi - half_h)
        ki1 = ki0 + (i1 - i0)
        kj0 = j0 - (oj - half_w)
        kj1 = kj0 + (j1 - j0)
        field[i0:i1, j0:j1] = np.maximum(
            field[i0:i1, j0:j1],
            kernel[ki0:ki1, kj0:kj1],
        )
    mx = float(field.max())
    if mx > 0:
        field /= mx
    return field


def build_distance_field(
    occupied: np.ndarray,
    *,
    resolution_m: float,
    max_distance_m: float = 5.0,
) -> np.ndarray:
    """Euclidean distance (m) to nearest occupied cell, capped."""
    max_cells = max(2, int(math.ceil(max_distance_m / resolution_m)))
    dist_cells = _wavefront_distance(occupied, max_cells=max_cells)
    return (dist_cells * resolution_m).astype(np.float32)


def _wavefront_distance(blocked: np.ndarray, *, max_cells: int) -> np.ndarray:
    """8-connected distance transform; 0 on blocked, increasing outward."""
    h, w = blocked.shape
    INF = np.int32(max_cells + 1)
    dist = np.where(blocked, np.int32(0), INF).astype(np.int32)
    for _ in range(max_cells):
        changed = False
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                shifted = _shift_int(dist, di, dj, fill=INF)
                step = np.int32(2 if di != 0 and dj != 0 else 1)
                cand = shifted + step
                mask = cand < dist
                if np.any(mask):
                    dist[mask] = cand[mask]
                    changed = True
        if not changed:
            break
    dist[dist > max_cells] = max_cells
    return dist


def _shift_int(arr: np.ndarray, di: int, dj: int, fill: int) -> np.ndarray:
    out = np.full_like(arr, fill)
    h, w = arr.shape
    si0 = max(0, -di)
    si1 = min(h, h - di)
    sj0 = max(0, -dj)
    sj1 = min(w, w - dj)
    di0 = max(0, di)
    di1 = min(h, h + di)
    dj0 = max(0, dj)
    dj1 = min(w, w + dj)
    if si1 > si0 and sj1 > sj0:
        out[di0:di1, dj0:dj1] = arr[si0:si1, sj0:sj1]
    return out


def finalize_log_odds(
    log_odds: np.ndarray,
    *,
    denoise_min_neighbors: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Threshold log-odds, denoise occupied, build likelihood + distance."""
    occupied = log_odds > OCCUPIED_LOG_ODDS_THRESHOLD
    if denoise_min_neighbors > 0:
        occupied = _drop_speckle(occupied, min_neighbors=denoise_min_neighbors)
    resolution_m = 1.0  # caller passes via ReferenceMap; fields rebuilt below
    return occupied, np.zeros_like(log_odds), np.zeros_like(log_odds)


def _drop_speckle(blocked: np.ndarray, *, min_neighbors: int) -> np.ndarray:
    h, w = blocked.shape
    count = np.zeros((h, w), dtype=np.int32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            count += _shift_bool(blocked, di, dj)
    return blocked & (count >= min_neighbors)


def _shift_bool(arr: np.ndarray, di: int, dj: int) -> np.ndarray:
    out = np.zeros_like(arr)
    h, w = arr.shape
    si0 = max(0, -di)
    si1 = min(h, h - di)
    sj0 = max(0, -dj)
    sj1 = min(w, w - dj)
    di0 = max(0, di)
    di1 = min(h, h + di)
    dj0 = max(0, dj)
    dj1 = min(w, w + dj)
    if si1 > si0 and sj1 > sj0:
        out[di0:di1, dj0:dj1] = arr[si0:si1, sj0:sj1]
    return out


def _hysteresis_occupied(
    log_odds: np.ndarray,
    *,
    high_thresh: float,
    low_thresh: float,
) -> np.ndarray:
    """Two-threshold (Canny-style) occupied mask via morphological reconstruction.

    Cells above ``high_thresh`` are confident seeds. Faint cells above
    ``low_thresh`` are kept only where 8-connected to a seed; everything else
    is dropped. ``seed`` is a subset of the candidates because
    ``low_thresh < high_thresh``, so the seed always survives.
    """
    seed = log_odds > high_thresh
    if low_thresh >= high_thresh or not seed.any():
        return seed
    candidates = log_odds > low_thresh
    marker = seed
    while True:
        grown = marker.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                grown |= _shift_bool(marker, di, dj)
        grown &= candidates
        if np.array_equal(grown, marker):
            return marker
        marker = grown


def build_reference_map_from_log_odds(
    log_odds: np.ndarray,
    *,
    resolution_m: float,
    origin_x_m: float,
    origin_y_m: float,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trajectory: Optional[np.ndarray] = None,
    nogo_mask: Optional[np.ndarray] = None,
    denoise_min_neighbors: int = 2,
    occupied_low_log_odds: float = OCCUPIED_LOG_ODDS_LOW_THRESHOLD,
) -> ReferenceMap:
    # Hysteresis sharpening: promote faint-but-connected wall cells, drop
    # isolated faint returns. Feeds the MCL likelihood/distance fields only;
    # the raw occupancy_log_odds (and occupied_mask/driveable_int8, which
    # re-threshold it) are left untouched — same scope as _drop_speckle.
    occupied = _hysteresis_occupied(
        log_odds,
        high_thresh=OCCUPIED_LOG_ODDS_THRESHOLD,
        low_thresh=occupied_low_log_odds,
    )
    if denoise_min_neighbors > 0:
        occupied = _drop_speckle(occupied, min_neighbors=denoise_min_neighbors)
    lik = build_likelihood_field(occupied, resolution_m=resolution_m)
    dist = build_distance_field(occupied, resolution_m=resolution_m)
    meta = dict(metadata or {})
    meta.setdefault("map_version", MAP_VERSION)
    meta.setdefault("created_ts", time.time())
    return ReferenceMap(
        occupancy_log_odds=log_odds.astype(np.float32),
        resolution_m=float(resolution_m),
        origin_x_m=float(origin_x_m),
        origin_y_m=float(origin_y_m),
        likelihood_field=lik,
        distance_field_m=dist,
        session_id=session_id or uuid.uuid4().hex[:12],
        metadata=meta,
        trajectory=trajectory,
        nogo_mask=(None if nogo_mask is None else nogo_mask.astype(bool)),
    )


def save_reference_map(path: str, ref_map: ReferenceMap) -> None:
    meta_json = json.dumps(ref_map.metadata)
    kwargs: Dict[str, Any] = {
        "occupancy_log_odds": ref_map.occupancy_log_odds.astype(np.float32),
        "likelihood_field": ref_map.likelihood_field.astype(np.float32),
        "distance_field_m": ref_map.distance_field_m.astype(np.float32),
        "resolution_m": np.float32(ref_map.resolution_m),
        "origin_x_m": np.float32(ref_map.origin_x_m),
        "origin_y_m": np.float32(ref_map.origin_y_m),
        "session_id": np.array(ref_map.session_id),
        "meta_json": np.array(meta_json),
        "map_version": np.int32(MAP_VERSION),
    }
    if ref_map.trajectory is not None and ref_map.trajectory.size > 0:
        kwargs["trajectory"] = ref_map.trajectory.astype(np.float64)
    # Operator keep-out mask — stored only when non-empty (uint8 keeps
    # allow_pickle=False loads working). Absent key == no keep-out cells.
    if ref_map.nogo_mask is not None and bool(np.any(ref_map.nogo_mask)):
        kwargs["nogo_mask"] = ref_map.nogo_mask.astype(np.uint8)
    np.savez_compressed(path, **kwargs)


def load_reference_map(path: str) -> ReferenceMap:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    trajectory = None
    if "trajectory" in data and data["trajectory"].size > 0:
        trajectory = data["trajectory"].astype(np.float64)
    log_odds = data["occupancy_log_odds"].astype(np.float32)
    resolution_m = float(data["resolution_m"])
    origin_x_m = float(data["origin_x_m"])
    origin_y_m = float(data["origin_y_m"])
    if "likelihood_field" in data and "distance_field_m" in data:
        lik = data["likelihood_field"].astype(np.float32)
        dist = data["distance_field_m"].astype(np.float32)
    else:
        occupied = log_odds > OCCUPIED_LOG_ODDS_THRESHOLD
        lik = build_likelihood_field(occupied, resolution_m=resolution_m)
        dist = build_distance_field(occupied, resolution_m=resolution_m)
    nogo = data["nogo_mask"].astype(bool) if "nogo_mask" in data else None
    return ReferenceMap(
        occupancy_log_odds=log_odds,
        resolution_m=resolution_m,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        likelihood_field=lik,
        distance_field_m=dist,
        session_id=str(data.get("session_id") or meta.get("session_id") or ""),
        metadata=meta,
        trajectory=trajectory,
        nogo_mask=nogo,
    )

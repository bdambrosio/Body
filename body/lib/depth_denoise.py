"""Blob gates for the depth veto: an obstacle is a *thing*, not a pixel count.

The 10-31 veto counted raw obstacle pixels (≥ 8 of 10,800) and fired ~92 %
of the time on floor glare, mat edges and near-field junk (36 stops, 3
real). These gates run on the same body-frame points the veto scores and
keep only connected floor-plane blobs that

  * are at least ``min_range_m`` away (the camera's near field is junk),
  * span at least ``min_extent_m`` in their longest floor dimension (a
    coin-sized cluster of hundreds of points is a stereo mismatch on a
    bright spot; anything that small can be pushed anyway — a cable is thin
    but LONG, so this is the longest dimension, not the area),
  * reach ``min_height_m`` above the floor somewhere (a tilted floor patch
    or reflection is a thin slab).

Persistence under motion (the same odom cells across frames while the robot
moves) is the fourth gate and lives with the tracker, not here.

Pure NumPy; ``scripts/depth_denoise_score.py`` scores these gates on the
logged would-block frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from body.lib.depth_veto import DepthPoints, DepthVetoConfig, obstacle_mask


@dataclass(frozen=True)
class DenoiseConfig:
    enabled: bool = True
    cell_m: float = 0.04
    min_pts_per_cell: int = 2
    min_range_m: float = 0.25
    min_extent_m: float = 0.06       # longest floor dimension of the blob
    min_height_m: float = 0.06       # some point of the blob this high above ground (a shoe is 8)
    min_cells: int = 3


@dataclass
class Blob:
    n_pts: int
    n_cells: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    h_max: float
    idx: np.ndarray                  # indices into the frame's points

    @property
    def extent_m(self) -> float:
        return max(self.x_max - self.x_min, self.y_max - self.y_min)

    @property
    def range_m(self) -> float:
        return self.x_min


def _components(cells: np.ndarray) -> List[np.ndarray]:
    """8-connected components over unique integer cells (N, 2); returns a
    list of index arrays into ``cells``."""
    if len(cells) == 0:
        return []
    index = {(int(a), int(b)): k for k, (a, b) in enumerate(cells)}
    seen = np.zeros(len(cells), dtype=bool)
    out: List[np.ndarray] = []
    for k0 in range(len(cells)):
        if seen[k0]:
            continue
        stack = [k0]
        seen[k0] = True
        comp = []
        while stack:
            k = stack.pop()
            comp.append(k)
            a, b = int(cells[k, 0]), int(cells[k, 1])
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da == 0 and db == 0:
                        continue
                    j = index.get((a + da, b + db))
                    if j is not None and not seen[j]:
                        seen[j] = True
                        stack.append(j)
        out.append(np.asarray(comp))
    return out


def find_blobs(dp: DepthPoints, region: np.ndarray, vcfg: DepthVetoConfig,
               cfg: DenoiseConfig) -> List[Blob]:
    """All obstacle blobs in ``region`` (bool over dp's points), ungated."""
    obs = obstacle_mask(dp, region, vcfg)
    idx = np.nonzero(obs)[0]
    if idx.size == 0:
        return []
    x = dp.pts[idx, 0]
    y = dp.pts[idx, 1]
    h = dp.height[idx]
    ci = np.floor(x / cfg.cell_m).astype(np.int64)
    cj = np.floor(y / cfg.cell_m).astype(np.int64)
    keys = np.stack([ci, cj], axis=1)
    cells, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    keep_cell = counts >= cfg.min_pts_per_cell
    if not keep_cell.any():
        return []
    cell_ids = np.nonzero(keep_cell)[0]
    remap = {int(c): k for k, c in enumerate(cell_ids)}
    sub_cells = cells[cell_ids]
    blobs: List[Blob] = []
    for comp in _components(sub_cells):
        comp_cells = set(int(cell_ids[k]) for k in comp)
        member = np.isin(inv, list(comp_cells))
        pi = idx[member]
        blobs.append(Blob(
            n_pts=int(member.sum()), n_cells=len(comp),
            x_min=float(x[member].min()), x_max=float(x[member].max()),
            y_min=float(y[member].min()), y_max=float(y[member].max()),
            h_max=float(h[member].max()), idx=pi))
    return blobs


def gate_blobs(blobs: List[Blob], cfg: DenoiseConfig) -> List[Blob]:
    return [b for b in blobs
            if b.n_cells >= cfg.min_cells
            and b.range_m >= cfg.min_range_m
            and b.extent_m >= cfg.min_extent_m
            and b.h_max >= cfg.min_height_m]


def denoised_hits(dp: Optional[DepthPoints], region: np.ndarray, vcfg: DepthVetoConfig,
                  cfg: DenoiseConfig) -> Tuple[int, Optional[np.ndarray], List[Blob]]:
    """(hit count, index array of surviving points, surviving blobs) — the
    veto's evidence after the gates. With ``cfg.enabled`` False the raw
    obstacle mask is returned unchanged."""
    if dp is None:
        return 0, None, []
    if not cfg.enabled:
        obs = obstacle_mask(dp, region, vcfg)
        idx = np.nonzero(obs)[0]
        return int(idx.size), idx, []
    kept = gate_blobs(find_blobs(dp, region, vcfg, cfg), cfg)
    if not kept:
        return 0, np.zeros(0, dtype=np.int64), []
    idx = np.concatenate([b.idx for b in kept])
    return int(idx.size), idx, kept

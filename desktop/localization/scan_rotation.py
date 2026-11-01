"""Scan-to-scan rotation estimate: the tie-breaker when wheels and IMU disagree.

The EKF's wheel-vs-IMU guard (I2) sees "wheels turned, IMU flat" and cannot
tell a frozen IMU from wheels slipping or spinning against an obstacle. On
2026-11-01 it guessed "frozen IMU" twice in the same spot, the wheels were in
fact the ones lying, and the robot lost ~30° of heading. Two lidar scans that
bracket the disputed interval measure the rotation independently of both
sensors. This module estimates it and judges which hypothesis it supports.

Method: rasterize scan A into a dilated occupancy grid; transform scan B by
candidate (dθ, dx, dy) and score the fraction of its points that land on
occupied cells. Brute force over dθ spanning both hypotheses (plus margin)
and a small xy window (lidar offset from the rotation centre + any creep),
then refine dθ. Pure numpy; ~10–40 ms for the angles involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ScanRotationConfig:
    max_range_m: float = 6.0
    min_range_m: float = 0.12
    res_m: float = 0.03
    dilate_cells: int = 1
    xy_half_m: float = 0.20
    xy_step_m: float = 0.04
    theta_margin_rad: float = math.radians(12.0)
    theta_step_rad: float = math.radians(1.0)
    refine_step_rad: float = math.radians(0.25)
    # Score around a hypothesis = best within ±this of it (scan skew while
    # rotating, IMU/wheel ts mismatch).
    hypothesis_tol_rad: float = math.radians(3.0)
    min_points: int = 40
    # Scan B is subsampled to this many points (the cost driver).
    max_points_b: int = 240


@dataclass(frozen=True)
class ScanRotationResult:
    dtheta: float               # best rotation of B relative to A (rad, CCW +)
    score: float                # fraction of B points matched at the best pose
    hypothesis_scores: Dict[str, float]
    n_points: int


class _Grid:
    def __init__(self, pts: np.ndarray, cfg: ScanRotationConfig) -> None:
        self.res = cfg.res_m
        half = cfg.max_range_m + cfg.xy_half_m + 0.1
        self.origin = -half
        self.n = int(math.ceil(2 * half / self.res)) + 1
        occ = np.zeros((self.n, self.n), dtype=bool)
        ij = np.floor((pts - self.origin) / self.res).astype(np.int64)
        ok = (ij >= 0).all(axis=1) & (ij < self.n).all(axis=1)
        occ[ij[ok, 0], ij[ok, 1]] = True
        d = cfg.dilate_cells
        if d > 0:
            pad = np.pad(occ, d)
            out = np.zeros_like(occ)
            for di in range(-d, d + 1):
                for dj in range(-d, d + 1):
                    out |= pad[d + di:d + di + self.n, d + dj:d + dj + self.n]
            occ = out
        self.occ = occ

    def score(self, pts: np.ndarray) -> float:
        """pts: (K, N, 2) candidate point sets → max over K of matched fraction."""
        ij = np.floor((pts - self.origin) / self.res).astype(np.int64)
        ok = (ij >= 0).all(axis=-1) & (ij < self.n).all(axis=-1)
        ij = np.clip(ij, 0, self.n - 1)
        hit = self.occ[ij[..., 0], ij[..., 1]] & ok
        return hit.mean(axis=-1)


def _filter(pts: np.ndarray, cfg: ScanRotationConfig) -> np.ndarray:
    r = np.hypot(pts[:, 0], pts[:, 1])
    return pts[(r >= cfg.min_range_m) & (r <= cfg.max_range_m)]


def estimate_rotation(
    points_a: np.ndarray,
    points_b: np.ndarray,
    hypotheses: Dict[str, float],
    cfg: ScanRotationConfig = ScanRotationConfig(),
) -> Optional[ScanRotationResult]:
    """Rotation of the robot between scan A and scan B (body-frame points).

    ``hypotheses`` maps a name (e.g. "wheels", "imu") to the rotation that
    source claims over the same interval; the search spans all of them.
    Returns None when either scan is too sparse to judge.
    """
    a = _filter(np.asarray(points_a, dtype=np.float64), cfg)
    b = _filter(np.asarray(points_b, dtype=np.float64), cfg)
    if len(a) < cfg.min_points or len(b) < cfg.min_points or not hypotheses:
        return None
    if len(b) > cfg.max_points_b:
        b = b[np.linspace(0, len(b) - 1, cfg.max_points_b).astype(np.int64)]
    grid = _Grid(a, cfg)
    steps = np.arange(-cfg.xy_half_m, cfg.xy_half_m + 1e-9, cfg.xy_step_m)
    offs = np.stack(np.meshgrid(steps, steps, indexing="ij"), axis=-1).reshape(-1, 1, 2)

    def best_at(th: float) -> float:
        c, s = math.cos(th), math.sin(th)
        rot = b @ np.array([[c, s], [-s, c]])        # R(th) applied to rows
        return float(grid.score(rot[None, :, :] + offs).max())

    hv = list(hypotheses.values())
    lo, hi = min(hv) - cfg.theta_margin_rad, max(hv) + cfg.theta_margin_rad
    thetas = np.arange(lo, hi + 1e-9, cfg.theta_step_rad)
    scores = np.array([best_at(float(t)) for t in thetas])
    k = int(np.argmax(scores))
    best_th, best_sc = float(thetas[k]), float(scores[k])
    for t in np.arange(best_th - cfg.theta_step_rad,
                       best_th + cfg.theta_step_rad + 1e-9, cfg.refine_step_rad):
        sc = best_at(float(t))
        if sc > best_sc:
            best_th, best_sc = float(t), sc

    hyp_scores: Dict[str, float] = {}
    for name, h in hypotheses.items():
        near = np.abs(thetas - h) <= cfg.hypothesis_tol_rad
        hyp_scores[name] = float(scores[near].max()) if near.any() else best_at(h)
    return ScanRotationResult(best_th, best_sc, hyp_scores, len(b))


@dataclass(frozen=True)
class VerdictConfig:
    min_separation_rad: float = math.radians(8.0)
    min_score: float = 0.35
    # Winner's angular error must be under this fraction of the separation.
    max_winner_err_frac: float = 0.35
    min_score_margin: float = 0.05


def judge(
    result: Optional[ScanRotationResult],
    claims: Dict[str, float],
    cfg: VerdictConfig = VerdictConfig(),
) -> str:
    """Name of the claim the scans support, or "unknown"."""
    if result is None or len(claims) != 2:
        return "unknown"
    (n1, h1), (n2, h2) = claims.items()
    sep = abs(h1 - h2)
    if sep < cfg.min_separation_rad or result.score < cfg.min_score:
        return "unknown"
    e1, e2 = abs(result.dtheta - h1), abs(result.dtheta - h2)
    win, lose, ew = (n1, n2, e1) if e1 < e2 else (n2, n1, e2)
    if ew > cfg.max_winner_err_frac * sep:
        return "unknown"
    s = result.hypothesis_scores
    if s.get(win, 0.0) - s.get(lose, 0.0) < cfg.min_score_margin:
        return "unknown"
    return win

"""Occlusion-aware (first-hit ray-cast) scan-match scorer.

The core of the radius-limited checkpoint match (see
docs/topological_localization_design.md §6 / Phase 3). Unlike the
likelihood-field correlation in `desktop.nav.slam.scan_matcher` — which scores
each endpoint by distance to the *nearest* occupied cell and so "sees through
walls" — this scorer ray-casts each beam from the candidate pose and stops at
the **first** occupied cell. That single operation encodes both halves of a
correct match:

  * occupied *at* the hit (the endpoint), and
  * clear *up to* it (it would have stopped earlier otherwise),

so behind-wall smear is physically invisible and an all-occupied map scores
*low* (every beam blocked immediately), not high.

Per-beam scoring is **asymmetric** (the classic beam-model p_short/p_hit/p_max
split), tuned for a map that is mid-healing — topologically right but only
partially Recognized:

  * predicted ≈ measured (within tolerance) → inlier (+1),
  * predicted **shorter** than measured (the map blocks a beam reality says is
    clear — phantom / smear / all-red) → contradiction, penalized hard,
  * predicted **longer** / max-range (the map is *missing* a wall reality has —
    an un-Recognized spot or a dynamic obstacle) → neutral (0).

Aggregation is the mean per-beam value (a robust inlier-minus-contradiction
score), not a summed Gaussian, so a handful of unmodeled returns can't dominate.

Pure: numpy only, no Qt / zenoh / map classes. The occupancy grid is a bool
(nx, ny) array with i↔x, j↔y and the standard
``cell = floor((world - origin) / resolution)`` convention (matches
`ReferenceMap` / `EditorMap`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

Pose = Tuple[float, float, float]   # world (x, y, theta)


@dataclass(frozen=True)
class RaycastConfig:
    max_range_m: float = 4.0       # ignore measured/predicted beyond this
    step_m: float = 0.025          # ray-march sample step (≈ half a 0.05 cell)
    inlier_tol_m: float = 0.12     # |predicted - measured| within → inlier
    short_tol_m: float = 0.12      # predicted < measured - short_tol → contradiction
    short_penalty: float = 1.0     # per-beam penalty for a blocked-early beam


@dataclass(frozen=True)
class RaycastScore:
    score: float        # mean per-beam value in [-short_penalty, 1]; higher = better
    inlier_frac: float  # fraction of scored beams within tolerance
    short_frac: float   # fraction blocked early (contradiction)
    n: int              # valid beams scored (None if 0 → score 0)


def predicted_ranges(
    occupied: np.ndarray,
    origin_x_m: float,
    origin_y_m: float,
    resolution_m: float,
    sensor_xy: Tuple[float, float],
    bearings: np.ndarray,
    *,
    max_range_m: float,
    step_m: float,
) -> np.ndarray:
    """First-hit range (m) along each world `bearings` from `sensor_xy`,
    marching `occupied` in `step_m` increments out to `max_range_m`. Beams
    that hit nothing in range return `max_range_m`. Vectorized over beams."""
    nx, ny = occupied.shape
    sx, sy = float(sensor_xy[0]), float(sensor_xy[1])
    d = np.arange(step_m, max_range_m + 1e-9, step_m)            # (S,)
    b = np.asarray(bearings, dtype=np.float64)                   # (B,)
    if b.size == 0 or d.size == 0:
        return np.full(b.shape, max_range_m, dtype=np.float64)
    cos = np.cos(b)[:, None]
    sin = np.sin(b)[:, None]
    px = sx + d[None, :] * cos                                   # (B, S)
    py = sy + d[None, :] * sin
    ix = np.floor((px - origin_x_m) / resolution_m).astype(np.intp)
    iy = np.floor((py - origin_y_m) / resolution_m).astype(np.intp)
    inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    occ = np.zeros(ix.shape, dtype=bool)
    occ[inb] = occupied[ix[inb], iy[inb]]
    has = occ.any(axis=1)
    first = np.argmax(occ, axis=1)        # first True per beam (0 when none → guard with `has`)
    return np.where(has, d[first], max_range_m)


def score_pose(
    occupied: np.ndarray,
    origin_x_m: float,
    origin_y_m: float,
    resolution_m: float,
    pose: Pose,
    angles: Sequence[float],
    ranges: Sequence[float],
    cfg: RaycastConfig = RaycastConfig(),
) -> RaycastScore:
    """Occlusion-aware match score of `pose` against `occupied` for a scan
    given as body-frame `angles` (rad) + measured `ranges` (m). Invalid /
    out-of-range measured beams are dropped."""
    a = np.asarray(angles, dtype=np.float64)
    m = np.asarray(ranges, dtype=np.float64)
    valid = np.isfinite(m) & (m > cfg.step_m) & (m <= cfg.max_range_m)
    a = a[valid]
    m = m[valid]
    n = int(a.size)
    if n == 0:
        return RaycastScore(0.0, 0.0, 0.0, 0)
    bearings = float(pose[2]) + a
    pred = predicted_ranges(
        occupied, origin_x_m, origin_y_m, resolution_m,
        (pose[0], pose[1]), bearings,
        max_range_m=cfg.max_range_m, step_m=cfg.step_m,
    )
    resid = pred - m
    inlier = np.abs(resid) <= cfg.inlier_tol_m
    short = resid < -cfg.short_tol_m               # predicted too short = contradiction
    # Gaussian on the residual (σ = inlier tol): ≈1 at a match, with a gradient
    # so the search has a sharp peak instead of a flat within-tolerance plateau;
    # decays to ≈0 for a long residual (map missing a wall → neutral). A
    # contradiction (blocked early) overrides with a hard negative penalty.
    sigma = cfg.inlier_tol_m
    g = np.exp(-(resid * resid) / (2.0 * sigma * sigma))
    g[short] = -cfg.short_penalty
    return RaycastScore(
        score=float(g.mean()),
        inlier_frac=float(inlier.mean()),
        short_frac=float(short.mean()),
        n=n,
    )


def best_pose_in_window(
    occupied: np.ndarray,
    origin_x_m: float,
    origin_y_m: float,
    resolution_m: float,
    prior: Pose,
    angles: Sequence[float],
    ranges: Sequence[float],
    *,
    xy_half_m: float = 0.30,
    xy_step_m: float = 0.05,
    theta_half_rad: float = math.radians(15.0),
    theta_step_rad: float = math.radians(3.0),
    cfg: RaycastConfig = RaycastConfig(),
) -> Tuple[Pose, RaycastScore]:
    """Brute-force search a small (xy, θ) window around `prior` for the pose
    that maximizes `score_pose`. Intended for a radius-limited checkpoint
    match — short rays, small window, well-primed by odom. Returns
    (best_pose, best_score)."""
    px, py, pth = float(prior[0]), float(prior[1]), float(prior[2])
    n_xy = max(0, int(round(xy_half_m / xy_step_m)))
    n_th = max(0, int(round(theta_half_rad / theta_step_rad)))
    offsets_xy = [k * xy_step_m for k in range(-n_xy, n_xy + 1)]
    offsets_th = [k * theta_step_rad for k in range(-n_th, n_th + 1)]
    best_pose: Pose = prior
    best = score_pose(
        occupied, origin_x_m, origin_y_m, resolution_m, prior, angles, ranges, cfg)
    for dth in offsets_th:
        for dx in offsets_xy:
            for dy in offsets_xy:
                cand: Pose = (px + dx, py + dy, pth + dth)
                s = score_pose(
                    occupied, origin_x_m, origin_y_m, resolution_m,
                    cand, angles, ranges, cfg)
                if s.score > best.score:
                    best, best_pose = s, cand
    return best_pose, best

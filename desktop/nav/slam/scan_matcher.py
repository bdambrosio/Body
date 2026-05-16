"""Correlation-based scan-to-map matcher.

Given a 2D lidar scan, a prior pose, and an occupancy evidence grid,
search a small (dx, dy, dθ) window around the prior and pick the pose
that maximizes Σ evidence[scan_point_cell_index].

Why correlation (vs ICP):
- Bounded search makes it robust to a bad prior — unlike ICP, which
  diverges when initialization is off.
- Pure numpy; no Jacobians, no iterative convergence drama.
- Naturally handles sparse-feature or ambiguous environments because
  the score is over all scan points at once.

When IMU yaw is a tight prior, the θ search dimension collapses to a
few degrees and xy dominates the search cost. Sized below for that
regime; 500k-candidate brute force still finishes in ~50 ms.

Public API:
    matcher = ScanMatcher(config)
    result = matcher.search(scan_xy_body, prior_pose, evidence_grid, grid_meta)

See ScanMatchResult in types.py for result shape.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .types import Pose2D, ScanMatchResult, ScoreField

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanMatcherConfig:
    # Search window half-widths. Best-candidate search runs over:
    #   dx ∈ [-xy_half_m, +xy_half_m]
    #   dy ∈ [-xy_half_m, +xy_half_m]
    #   dθ ∈ [-theta_half_rad, +theta_half_rad]
    xy_half_m: float = 0.30
    theta_half_rad: float = math.radians(8.0)

    # Step sizes.
    xy_step_m: float = 0.02   # = 2 cm, half the 4 cm local_map res
    theta_step_rad: float = math.radians(1.0)

    # Score improvement gate: how much better the best candidate must
    # be than the prior pose to be accepted. Prevents "move the pose
    # to whatever tiny noise-driven peak exists in the search window."
    min_improvement: float = 5.0

    # Max scan range considered (m). Longer beams carry more geometric
    # leverage but are noisier and more likely to see unmapped space.
    max_range_m: float = 5.0

    # Min scan range — filter out near-field self-hits / clutter.
    min_range_m: float = 0.15

    # Optional: if True, the matcher crops the scan to those points
    # whose world-frame cell at the prior pose is known (score > 0 in
    # evidence). Reduces spurious scoring on scans that mostly fall in
    # unmapped cells, but costs a pre-pass lookup. Off by default.
    crop_to_known: bool = False


class ScanMatcher:
    def __init__(self, config: ScanMatcherConfig = ScanMatcherConfig()):
        self.config = config

    # ── Public ───────────────────────────────────────────────────────

    def search(
        self,
        scan_xy_body: np.ndarray,           # (N, 2) float, body frame
        prior_pose: Pose2D,
        evidence: np.ndarray,               # (nx, ny) int/float; higher=more obstacle evidence
        origin_x_m: float,                  # world x coord of cell (0, 0) lower-left corner
        origin_y_m: float,
        resolution_m: float,
        return_field: bool = False,
    ) -> ScanMatchResult:
        # When return_field=True, the result carries the full
        # (Nx, Ny, Nth) correlation score grid. The argmax-derived pose,
        # score, improvement, accepted, and search_exhausted fields are
        # bit-for-bit identical regardless of return_field — the flag
        # only controls whether the side-product field is materialized.
        cfg = self.config

        # Axes are computed up front so an empty- or filtered-out scan
        # still has well-defined field axes to return.
        dx_vals = _linspace_symmetric(cfg.xy_half_m, cfg.xy_step_m)
        dy_vals = _linspace_symmetric(cfg.xy_half_m, cfg.xy_step_m)
        dth_vals = _linspace_symmetric(cfg.theta_half_rad, cfg.theta_step_rad)

        def _empty_field() -> ScoreField:
            return ScoreField(
                field=np.zeros(
                    (dx_vals.size, dy_vals.size, dth_vals.size),
                    dtype=np.float32,
                ),
                dx_axis=dx_vals.copy(),
                dy_axis=dy_vals.copy(),
                dth_axis=dth_vals.copy(),
            )

        if scan_xy_body.size == 0:
            return ScanMatchResult(
                pose=prior_pose, score=0.0, score_prior=0.0,
                improvement=0.0, accepted=False, search_exhausted=False,
                score_field=_empty_field() if return_field else None,
            )

        points = _filter_range(scan_xy_body, cfg.min_range_m, cfg.max_range_m)
        if points.size == 0:
            return ScanMatchResult(
                pose=prior_pose, score=0.0, score_prior=0.0,
                improvement=0.0, accepted=False, search_exhausted=False,
                score_field=_empty_field() if return_field else None,
            )

        # Precompute the scored image. Cast to float32 for vectorized
        # add; evidence may be int32 (as in WorldGrid.block_votes).
        ev = evidence.astype(np.float32, copy=False)
        nx, ny = ev.shape

        # Score at prior — baseline.
        score_prior = _score_at(
            points, prior_pose, ev, origin_x_m, origin_y_m, resolution_m,
        )

        if cfg.crop_to_known:
            # Remove scan points whose prior-pose cell is zero-evidence.
            # Heuristic — skip if evidence sparse; in that case we *want*
            # them to vote on the correlation below.
            keep = _keep_known(
                points, prior_pose, ev, origin_x_m, origin_y_m, resolution_m,
            )
            if keep.sum() >= max(8, points.shape[0] // 4):
                points = points[keep]

        # Allocate the score field only when requested. Zero-init means
        # fully-out-of-bounds cells carry 0.0, matching the score-of-0.0
        # convention used elsewhere in this matcher.
        if return_field:
            field = np.zeros(
                (dx_vals.size, dy_vals.size, dth_vals.size),
                dtype=np.float32,
            )
        else:
            field = None

        best_score = -np.inf
        best_dx = 0.0
        best_dy = 0.0
        best_dth = 0.0
        best_ix = 0
        best_iy = 0
        best_ith = 0

        # Loop over θ (few values), vectorize over dx, dy inside.
        # For each θ: rotate points once, then score a grid of xy
        # translations in one vectorized pass.
        for ith, dth in enumerate(dth_vals):
            th = prior_pose.theta + dth
            c, s = math.cos(th), math.sin(th)
            # Rotated body→world points (pre-translation).
            px = c * points[:, 0] - s * points[:, 1]
            py = s * points[:, 0] + c * points[:, 1]

            # World position at prior xy + (dx, dy)
            base_x = prior_pose.x + px
            base_y = prior_pose.y + py

            # Evaluate each (dx, dy) — loop over dx outer, vectorize
            # dy inner across points. A fully-vectorized 3D scoring
            # tensor would blow RAM for the default window; this
            # nested loop is 31*31 = ~1 k iterations × point count.
            for ix, dx in enumerate(dx_vals):
                x_world = base_x + dx
                # Columns -> i index
                i = np.floor(
                    (x_world - origin_x_m) / resolution_m + 1e-9
                ).astype(np.int32)
                # We'll add dy along the j axis.
                # Actually: inner-vectorize over dy too, per point.
                for iy, dy in enumerate(dy_vals):
                    y_world = base_y + dy
                    j = np.floor(
                        (y_world - origin_y_m) / resolution_m + 1e-9
                    ).astype(np.int32)
                    in_bounds = (
                        (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
                    )
                    if not np.any(in_bounds):
                        # Field cell stays at its zero-init value.
                        continue
                    score = float(ev[i[in_bounds], j[in_bounds]].sum())
                    if field is not None:
                        field[ix, iy, ith] = score
                    if score > best_score:
                        best_score = score
                        best_dx, best_dy, best_dth = float(dx), float(dy), float(dth)
                        best_ix, best_iy, best_ith = ix, iy, ith

        # Detect "best was at window edge" — means the prior is too far
        # off and we should either widen the search or flag low trust.
        exhausted = (
            math.isclose(abs(best_dx), cfg.xy_half_m, abs_tol=cfg.xy_step_m / 2.0)
            or math.isclose(abs(best_dy), cfg.xy_half_m, abs_tol=cfg.xy_step_m / 2.0)
            or math.isclose(abs(best_dth), cfg.theta_half_rad, abs_tol=cfg.theta_step_rad / 2.0)
        )

        improvement = best_score - score_prior
        accepted = improvement >= cfg.min_improvement

        if accepted:
            best_pose = Pose2D(
                x=prior_pose.x + best_dx,
                y=prior_pose.y + best_dy,
                theta=prior_pose.theta + best_dth,
            )
        else:
            best_pose = prior_pose

        score_field = None
        if field is not None:
            score_field = ScoreField(
                field=field,
                dx_axis=dx_vals.copy(),
                dy_axis=dy_vals.copy(),
                dth_axis=dth_vals.copy(),
            )

        return ScanMatchResult(
            pose=best_pose,
            score=best_score if best_score > -np.inf else 0.0,
            score_prior=score_prior,
            improvement=improvement if best_score > -np.inf else 0.0,
            accepted=accepted,
            search_exhausted=exhausted,
            score_field=score_field,
        )


# ── Helpers ──────────────────────────────────────────────────────────

def _linspace_symmetric(half_width: float, step: float) -> np.ndarray:
    """Centered-at-zero samples spanning [-half_width, +half_width].

    Uses arange so 0 always hits exactly. Includes both endpoints if
    half_width is a multiple of step.
    """
    n = int(math.floor(half_width / step))
    # arange(-n*step, (n+1)*step, step) — inclusive at upper end.
    return np.arange(-n * step, (n + 0.5) * step, step, dtype=np.float64)


def _filter_range(
    points_xy: np.ndarray, min_r: float, max_r: float,
) -> np.ndarray:
    r = np.hypot(points_xy[:, 0], points_xy[:, 1])
    mask = (r >= min_r) & (r <= max_r)
    return points_xy[mask]


def _score_at(
    points_xy_body: np.ndarray, pose: Pose2D, evidence: np.ndarray,
    origin_x_m: float, origin_y_m: float, resolution_m: float,
) -> float:
    """Score a single candidate pose."""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    x_world = pose.x + c * points_xy_body[:, 0] - s * points_xy_body[:, 1]
    y_world = pose.y + s * points_xy_body[:, 0] + c * points_xy_body[:, 1]
    nx, ny = evidence.shape
    i = np.floor((x_world - origin_x_m) / resolution_m + 1e-9).astype(np.int32)
    j = np.floor((y_world - origin_y_m) / resolution_m + 1e-9).astype(np.int32)
    in_bounds = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    if not np.any(in_bounds):
        return 0.0
    return float(evidence[i[in_bounds], j[in_bounds]].sum())


def _keep_known(
    points_xy_body: np.ndarray, pose: Pose2D, evidence: np.ndarray,
    origin_x_m: float, origin_y_m: float, resolution_m: float,
) -> np.ndarray:
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    x_world = pose.x + c * points_xy_body[:, 0] - s * points_xy_body[:, 1]
    y_world = pose.y + s * points_xy_body[:, 0] + c * points_xy_body[:, 1]
    nx, ny = evidence.shape
    i = np.floor((x_world - origin_x_m) / resolution_m + 1e-9).astype(np.int32)
    j = np.floor((y_world - origin_y_m) / resolution_m + 1e-9).astype(np.int32)
    in_bounds = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    keep = np.zeros(points_xy_body.shape[0], dtype=bool)
    if not np.any(in_bounds):
        return keep
    vals = np.zeros(points_xy_body.shape[0], dtype=evidence.dtype)
    vals[in_bounds] = evidence[i[in_bounds], j[in_bounds]]
    keep = in_bounds & (vals > 0)
    return keep


def likelihood_at(
    dx: float, dy: float, dth: float, score_field: ScoreField,
) -> float:
    """Trilinear interpolation into a ScoreField at a delta-from-prior.

    Returns the raw correlation score (treat as log-likelihood up to an
    additive constant). Queries strictly outside the field's spanned
    window clamp to 0.0 — same value the matcher reports for fully
    out-of-bounds candidates, so particle-filter consumers don't see a
    hard cliff at the window boundary.
    """
    f = score_field.field
    ax = score_field.dx_axis
    ay = score_field.dy_axis
    ath = score_field.dth_axis
    nx, ny, nth = f.shape

    if nx == 0 or ny == 0 or nth == 0:
        return 0.0

    def _frac_index(v: float, axis: np.ndarray, n: int) -> Tuple[int, int, float]:
        # axis is uniform (np.arange-built); pick step from end points.
        if n == 1:
            return 0, 0, 0.0
        step = (axis[-1] - axis[0]) / (n - 1)
        t = (v - axis[0]) / step
        # Tolerance covers fp slop when the caller passes back an exact
        # axis value (t = n-1 + 1e-15) — without it those lattice queries
        # would fall through to the OOB branch.
        eps = 1e-9
        if t < -eps or t > (n - 1) + eps:
            return -1, -1, 0.0  # out of range
        if t < 0.0:
            t = 0.0
        if t > n - 1:
            t = float(n - 1)
        i0 = int(math.floor(t))
        if i0 >= n - 1:
            return n - 1, n - 1, 0.0
        return i0, i0 + 1, t - i0

    ix0, ix1, fx = _frac_index(dx, ax, nx)
    iy0, iy1, fy = _frac_index(dy, ay, ny)
    ith0, ith1, fth = _frac_index(dth, ath, nth)
    if ix0 < 0 or iy0 < 0 or ith0 < 0:
        return 0.0

    # Trilinear blend across the eight corners.
    c000 = float(f[ix0, iy0, ith0])
    c100 = float(f[ix1, iy0, ith0])
    c010 = float(f[ix0, iy1, ith0])
    c110 = float(f[ix1, iy1, ith0])
    c001 = float(f[ix0, iy0, ith1])
    c101 = float(f[ix1, iy0, ith1])
    c011 = float(f[ix0, iy1, ith1])
    c111 = float(f[ix1, iy1, ith1])

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    return c0 * (1 - fth) + c1 * fth


def lidar_scan_to_xy(
    ranges_m: np.ndarray, angles_rad: np.ndarray,
) -> np.ndarray:
    """Convert polar lidar scan arrays to (N, 2) cartesian in body frame.

    Invalid / out-of-range entries (NaN, inf, zero) are dropped.
    """
    r = np.asarray(ranges_m, dtype=np.float64)
    a = np.asarray(angles_rad, dtype=np.float64)
    valid = np.isfinite(r) & (r > 0)
    r = r[valid]
    a = a[valid]
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)

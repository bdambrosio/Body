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

from .types import Pose2D, ScanMatchResult

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
    ) -> ScanMatchResult:
        if scan_xy_body.size == 0:
            return ScanMatchResult(
                pose=prior_pose, score=0.0, score_prior=0.0,
                improvement=0.0, accepted=False, search_exhausted=False,
            )

        cfg = self.config
        points = _filter_range(scan_xy_body, cfg.min_range_m, cfg.max_range_m)
        if points.size == 0:
            return ScanMatchResult(
                pose=prior_pose, score=0.0, score_prior=0.0,
                improvement=0.0, accepted=False, search_exhausted=False,
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

        # Build candidate offsets.
        dx_vals = _linspace_symmetric(cfg.xy_half_m, cfg.xy_step_m)
        dy_vals = _linspace_symmetric(cfg.xy_half_m, cfg.xy_step_m)
        dth_vals = _linspace_symmetric(cfg.theta_half_rad, cfg.theta_step_rad)

        best_score = -np.inf
        best_dx = 0.0
        best_dy = 0.0
        best_dth = 0.0

        # Loop over θ (few values), vectorize over dx, dy inside.
        # For each θ: rotate points once, then score a grid of xy
        # translations in one vectorized pass.
        for dth in dth_vals:
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
            for dx in dx_vals:
                x_world = base_x + dx
                # Columns -> i index
                i = np.floor(
                    (x_world - origin_x_m) / resolution_m + 1e-9
                ).astype(np.int32)
                # We'll add dy along the j axis.
                # Actually: inner-vectorize over dy too, per point.
                for dy in dy_vals:
                    y_world = base_y + dy
                    j = np.floor(
                        (y_world - origin_y_m) / resolution_m + 1e-9
                    ).astype(np.int32)
                    in_bounds = (
                        (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
                    )
                    if not np.any(in_bounds):
                        continue
                    score = float(ev[i[in_bounds], j[in_bounds]].sum())
                    if score > best_score:
                        best_score = score
                        best_dx, best_dy, best_dth = float(dx), float(dy), float(dth)

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

        return ScanMatchResult(
            pose=best_pose,
            score=best_score if best_score > -np.inf else 0.0,
            score_prior=score_prior,
            improvement=improvement if best_score > -np.inf else 0.0,
            accepted=accepted,
            search_exhausted=exhausted,
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

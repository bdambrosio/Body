"""Unit tests for Phase 5.5 Variant A — σ-aware vote weighting.

Covers (a) the _pose_weight_scale helper that converts a 3×3 posterior
covariance into a [floor, 1] vote-weight multiplier, and (b) the
fuse_local_map integration that applies that multiplier to per-vote
contributions.

Run:
    PYTHONPATH=. python3 -m unittest desktop.world_map.test_pose_weight_scale -v
"""
from __future__ import annotations

import unittest

import numpy as np

from desktop.world_map.controller import (
    _POSE_SIGMA_NOMINAL_M,
    _POSE_WEIGHT_FLOOR,
    _pose_weight_scale,
)
from desktop.world_map.world_grid import WorldGrid


def _cov(sigma_x_m: float, sigma_y_m: float, sigma_th_rad: float = 0.01) -> np.ndarray:
    """Diagonal 3×3 cov from per-axis σ."""
    return np.diag([sigma_x_m**2, sigma_y_m**2, sigma_th_rad**2])


class TestPoseWeightScale(unittest.TestCase):
    def test_none_returns_full_weight(self):
        # Point-estimate sources return cov_at()=None — keep current behavior.
        self.assertEqual(_pose_weight_scale(None), 1.0)

    def test_below_nominal_returns_full_weight(self):
        # Healthy operation: σ_xy = 1 cm, below the 2 cm nominal floor.
        self.assertEqual(_pose_weight_scale(_cov(0.01, 0.01)), 1.0)

    def test_at_nominal_returns_full_weight(self):
        # σ exactly at nominal → still full weight.
        self.assertEqual(_pose_weight_scale(_cov(_POSE_SIGMA_NOMINAL_M, _POSE_SIGMA_NOMINAL_M)), 1.0)

    def test_2x_nominal_returns_quarter(self):
        # σ at 2× nominal → ratio 0.5 → 0.25× weight (information ∝ 1/σ²).
        w = _pose_weight_scale(_cov(2 * _POSE_SIGMA_NOMINAL_M, 2 * _POSE_SIGMA_NOMINAL_M))
        self.assertAlmostEqual(w, 0.25, places=5)

    def test_anisotropic_uses_worse_axis(self):
        # σ_x = 1 cm (healthy), σ_y = 8 cm (bad) → use y σ for the scale.
        w = _pose_weight_scale(_cov(0.01, 4 * _POSE_SIGMA_NOMINAL_M))
        self.assertAlmostEqual(w, 1 / 16.0, places=5)

    def test_extreme_clamps_to_floor(self):
        # σ_xy = 1 m → ratio² ≈ 0.0004, well below floor 0.05.
        w = _pose_weight_scale(_cov(1.0, 1.0))
        self.assertEqual(w, _POSE_WEIGHT_FLOOR)

    def test_bad_shape_returns_full_weight(self):
        # Defensive: malformed cov → don't downweight.
        self.assertEqual(_pose_weight_scale(np.eye(2)), 1.0)


class TestFuseLocalMapWithScale(unittest.TestCase):
    def _make_grid_and_local(self, scale: float):
        grid = WorldGrid(
            extent_m=4.0, resolution_m=0.08,
            vote_margin=2,
            traversal_vote_weight=3,
            footprint_radius_m=0.15,
            vote_capacity=20.0,
            clear_vote_weight=1.0,
            block_vote_weight=1.0,
        )
        # 5×5 local map, 0.08 m res, centered at origin.
        # All cells blocked, all max_height = 1.0.
        local = np.full((5, 5), 1.0, dtype=np.float32)
        driveable = np.zeros((5, 5), dtype=np.int8)  # 0 = blocked
        meta = {
            "resolution_m": 0.08,
            "origin_x_m": -0.20,
            "origin_y_m": -0.20,
            "driveable_clearance_height_m": 0.4,
        }
        return grid, local, driveable, meta

    def test_full_weight_default_unchanged(self):
        # No pose_weight_scale passed → existing behaviour (votes ×1).
        grid, lm, dr, meta = self._make_grid_and_local(1.0)
        grid.fuse_local_map(
            grid=lm, driveable=dr, meta=meta,
            pose_world=(0.0, 0.0, 0.0), capture_ts=1.0,
        )
        # 25 blocked cells planted ~1 vote each (subject to constraint cap).
        total_block = float(grid.block_votes.sum())
        self.assertGreater(total_block, 20.0)  # at least many cells got votes

    def test_quarter_weight_quarters_block_votes(self):
        grid_a, lm, dr, meta = self._make_grid_and_local(1.0)
        grid_a.fuse_local_map(
            grid=lm, driveable=dr, meta=meta,
            pose_world=(0.0, 0.0, 0.0), capture_ts=1.0,
            pose_weight_scale=1.0,
        )
        full_sum = float(grid_a.block_votes.sum())

        grid_b, lm, dr, meta = self._make_grid_and_local(0.25)
        grid_b.fuse_local_map(
            grid=lm, driveable=dr, meta=meta,
            pose_world=(0.0, 0.0, 0.0), capture_ts=1.0,
            pose_weight_scale=0.25,
        )
        quarter_sum = float(grid_b.block_votes.sum())

        # Quarter-weighted total is ~1/4 of full-weight total (modulo
        # sum-bounded constraint at capacity — but a single scan at
        # weight 1 won't saturate cells, so the ratio holds).
        self.assertAlmostEqual(quarter_sum, full_sum * 0.25, delta=0.5)

    def test_zero_weight_plants_no_votes(self):
        grid, lm, dr, meta = self._make_grid_and_local(0.0)
        grid.fuse_local_map(
            grid=lm, driveable=dr, meta=meta,
            pose_world=(0.0, 0.0, 0.0), capture_ts=1.0,
            pose_weight_scale=0.0,
        )
        # Zero-weight scan → no block votes deposited.
        self.assertEqual(float(grid.block_votes.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

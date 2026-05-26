"""Tests for MCLPoseSource scan-match observation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.localization.mcl_pose_source import MCLPoseSource, MCLPoseSourceConfig
from desktop.reference_map.reference_map import (
    ReferenceMap,
    build_reference_map_from_log_odds,
)
from desktop.world_map.particle_filter_pose import ParticleFilterConfig


def _corridor_map(*, res: float = 0.08) -> ReferenceMap:
    nx, ny = 80, 40
    log_odds = np.full((nx, ny), -0.4, dtype=np.float32)
    log_odds[:, :8] = 0.85
    log_odds[:, -8:] = 0.85
    return build_reference_map_from_log_odds(
        log_odds,
        resolution_m=res,
        origin_x_m=0.0,
        origin_y_m=0.0,
        session_id="test",
    )


class TestMCLScanMatch(unittest.TestCase):
    def test_scan_match_summary_populated(self) -> None:
        ref = _corridor_map()
        src = MCLPoseSource(
            ref,
            pf_config=ParticleFilterConfig(n_particles=200, device="cpu"),
            config=MCLPoseSourceConfig(min_evidence_cells=50),
        )
        src._mcl.seed_at(2.0, 1.6, 0.0, sigma_xy_m=0.01, sigma_theta_rad=0.01)
        angles = np.linspace(-math.pi, math.pi, 90, endpoint=False)
        ranges = np.full(90, 2.5, dtype=np.float64)
        points = src._scan_points(ranges, angles)
        self.assertIsNotNone(points)
        ok = src._apply_scan_match_observation(points)  # type: ignore[arg-type]
        self.assertTrue(ok)
        sm = src.scan_match_summary()
        self.assertIn("best_pose", sm)
        self.assertIn("prior_pose", sm)
        self.assertIn("elapsed_ms", sm)


if __name__ == "__main__":
    unittest.main()

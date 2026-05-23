"""Unit tests for MCLLocalizer beam model."""

from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.localization.mcl_localizer import MCLConfig, MCLLocalizer
from desktop.reference_map.reference_map import build_reference_map_from_log_odds
from desktop.world_map.particle_filter_pose import ParticleFilterConfig


class TestMCLBeamModel(unittest.TestCase):
    def _wall_map(self):
        log_odds = np.full((60, 60), -2.0, dtype=np.float32)
        log_odds[30, 35:55] = 3.0  # vertical wall at x index ~35-55
        return build_reference_map_from_log_odds(
            log_odds,
            resolution_m=0.05,
            origin_x_m=-1.5,
            origin_y_m=-1.5,
        )

    def test_correct_pose_scores_higher(self):
        ref = self._wall_map()
        mcl = MCLLocalizer(
            ref,
            pf_config=ParticleFilterConfig(
                n_particles=500, seed=1, device="cpu",
            ),
        )
        # Robot at origin facing +x; wall ~1.0 m ahead at y=0
        mcl.seed_at(0.0, 0.0, 0.0, sigma_xy_m=0.01, sigma_theta_rad=0.01)
        angles = np.linspace(-math.pi / 4, math.pi / 4, 90)
        ranges = np.full(90, 1.0)
        w_before = mcl.filter.normalized_weights().clone()
        mcl.observe_scan_ranges(ranges, angles)
        w_good = mcl.filter.normalized_weights().clone()

        mcl.seed_at(0.5, 0.0, 0.0, sigma_xy_m=0.01, sigma_theta_rad=0.01)
        mcl.observe_scan_ranges(ranges, angles)
        mean_good = float(w_good.max())
        mean_bad = float(mcl.filter.normalized_weights().max())
        self.assertGreater(mean_good, mean_bad * 0.5)

    def test_predict_and_imu(self):
        ref = self._wall_map()
        mcl = MCLLocalizer(
            ref,
            pf_config=ParticleFilterConfig(n_particles=200, seed=2),
        )
        mcl.seed_at(0.0, 0.0, 0.0)
        mcl.predict(0.1, 0.0)
        mcl.observe_imu_yaw(0.0)
        x, y, th = mcl.posterior_mean()
        self.assertAlmostEqual(x, 0.1, delta=0.05)


if __name__ == "__main__":
    unittest.main()

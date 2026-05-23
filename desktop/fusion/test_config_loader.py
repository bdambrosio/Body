"""Tests for load_slam_config."""

from __future__ import annotations

import unittest

from desktop.fusion.load_slam_config import load_slam_config
from desktop.world_map.particle_filter_pose import (
    ALPHA_ROT_PER_M,
    ALPHA_ROT_PER_RAD,
    ALPHA_TRANS_PER_M,
    IMU_SIGMA_PER_SAMPLE_RAD,
)


class TestLoadSlamConfig(unittest.TestCase):
    def test_defaults_match_phase0(self) -> None:
        cfg = load_slam_config()
        self.assertAlmostEqual(cfg.fusion.alpha_trans_per_m, ALPHA_TRANS_PER_M)
        self.assertAlmostEqual(cfg.fusion.alpha_rot_per_m, ALPHA_ROT_PER_M)
        self.assertAlmostEqual(cfg.fusion.alpha_rot_per_rad, ALPHA_ROT_PER_RAD)
        self.assertAlmostEqual(cfg.fusion.imu_sigma_rad, IMU_SIGMA_PER_SAMPLE_RAD)
        self.assertAlmostEqual(cfg.slam.match_hz, 2.0)
        self.assertAlmostEqual(cfg.slam.resolution_m, 0.05)


if __name__ == "__main__":
    unittest.main()

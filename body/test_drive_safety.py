"""Tests for the swept-footprint check — focus on the forward-cone behaviour."""
import math
import unittest

import numpy as np

from body.lib.drive_safety import FootprintConfig, swept_path_blocked

RES, HALF = 0.08, 2.5
N = 2 * int(math.ceil(HALF / RES))
META = {"resolution_m": RES, "origin_x_m": -HALF, "origin_y_m": -HALF,
        "nx": N, "ny": N, "frame": "body"}


def _clear():
    return np.ones((N, N), dtype=np.int8)


def _block(grid, x, y):
    i = int(math.floor((x - META["origin_x_m"]) / RES))
    j = int(math.floor((y - META["origin_y_m"]) / RES))
    grid[i, j] = 0


class TestForwardCone(unittest.TestCase):
    def setUp(self):
        # Real robot ~7.5 cm half-width; modelled at 0.11 m + ½ cell ≈ 0.15 m.
        self.foot = FootprintConfig(footprint_radius_m=0.11)  # cone default 60°

    def test_abeam_obstacle_beyond_hard_allows_forward(self):
        # Obstacle abeam but BEYOND the hard radius (~15 cm) and outside the
        # forward cone → the robot may drive past it (e.g. squeeze a gap).
        grid = _clear()
        for y in (-0.15, -0.17, -0.19):
            _block(grid, 0.0, y)
        self.assertFalse(swept_path_blocked(
            grid, META, v_mps=0.12, omega_radps=0.0, config=self.foot))

    def test_abeam_obstacle_within_hard_blocks(self):
        # Doorjamb abeam (~90°) but within the hard body radius (~7 cm) → a real
        # clip; the hard-radius check vetoes even though it's outside the cone.
        grid = _clear()
        for y in (-0.06, -0.07, -0.08):
            _block(grid, 0.0, y)
        self.assertTrue(swept_path_blocked(
            grid, META, v_mps=0.12, omega_radps=0.0, config=self.foot))

    def test_head_on_obstacle_blocks(self):
        grid = _clear()
        for y in (-0.04, 0.0, 0.04):
            _block(grid, 0.15, y)            # straight ahead, within preview
        self.assertTrue(swept_path_blocked(
            grid, META, v_mps=0.12, omega_radps=0.0, config=self.foot))

    def test_full_halfplane_still_blocks_abeam(self):
        # cone = 90° recovers the legacy forward-half-plane → abeam blocks.
        legacy = FootprintConfig(footprint_radius_m=0.11,
                                 forward_cone_rad=math.pi / 2)
        grid = _clear()
        for y in (-0.10, -0.12, -0.14):
            _block(grid, 0.0, y)
        self.assertTrue(swept_path_blocked(
            grid, META, v_mps=0.12, omega_radps=0.0, config=legacy))

    def test_rotation_always_allowed(self):
        grid = _clear()
        for y in (-0.10, -0.12):
            _block(grid, 0.0, y)
        self.assertFalse(swept_path_blocked(
            grid, META, v_mps=0.0, omega_radps=0.4, config=self.foot))


if __name__ == "__main__":
    unittest.main()

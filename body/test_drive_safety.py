"""Tests for the swept-footprint check — focus on the forward-cone behaviour."""
import math
import unittest

import numpy as np

from body.lib.drive_safety import (
    FootprintConfig, seed_clear_disk, swept_path_blocked, unknown_block_mask,
)

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


class TestUnknownExplained(unittest.TestCase):
    """T5: the raster's pre-hit unknown band beside a wall must not veto
    wall-following; genuinely unobserved near cells still do."""

    def _wall_grid(self, y_wall):
        grid = _clear()
        for x in np.arange(-1.0, 1.0, RES / 2):
            _block(grid, float(x), y_wall)
            for k in (1, 2):                      # the 2-cell pre-hit buffer
                i = int(math.floor((x - META["origin_x_m"]) / RES))
                j = int(math.floor((y_wall - k * RES - META["origin_y_m"]) / RES))
                grid[i, j] = -1
        return grid

    def test_explained_unknown_does_not_block(self):
        grid = self._wall_grid(0.17)
        legacy = FootprintConfig(footprint_radius_m=0.07, hard_radius_m=0.07)
        fixed = FootprintConfig(footprint_radius_m=0.07, hard_radius_m=0.07,
                                unknown_explained_cells=3)
        self.assertTrue(swept_path_blocked(grid, META, v_mps=0.12, omega_radps=0.0,
                                           config=legacy))
        self.assertFalse(swept_path_blocked(grid, META, v_mps=0.12, omega_radps=0.0,
                                            config=fixed))

    def test_unexplained_unknown_still_blocks(self):
        grid = _clear()
        grid[int((0.12 + HALF) / RES), int((0.0 + HALF) / RES)] = -1   # no-info cell ahead
        cfg = FootprintConfig(footprint_radius_m=0.07, unknown_explained_cells=3)
        self.assertTrue(swept_path_blocked(grid, META, v_mps=0.12, omega_radps=0.0, config=cfg))
        mask = unknown_block_mask(grid, META, (0.0, 0.0), cfg)
        self.assertEqual(int(mask.sum()), 1)

    def test_seed_clear_disk_only_touches_unknown(self):
        grid = np.full((N, N), -1, dtype=np.int8)
        _block(grid, 0.04, 0.04)
        seed_clear_disk(grid, META, (0.0, 0.0), 0.11)
        self.assertEqual(grid[int((0.04 + HALF) / RES), int((0.04 + HALF) / RES)], 0)
        self.assertEqual(grid[int((-0.04 + HALF) / RES), int((-0.04 + HALF) / RES)], 1)


class TestReachAndStartPose(unittest.TestCase):
    def setUp(self):
        self.foot = FootprintConfig(footprint_radius_m=0.07, preview_time_s=1.5)

    def test_reach_cap_lets_robot_close_on_a_subgoal(self):
        # T2: a wall ~0.35 m ahead; the goal is 0.10 m ahead. Without the cap
        # the 0.27 m preview hits the wall; capped at the path end it does not.
        grid = _clear()
        for y in np.arange(-0.5, 0.5, RES / 2):
            _block(grid, 0.34, float(y))
        self.assertTrue(swept_path_blocked(grid, META, v_mps=0.18, omega_radps=0.0,
                                           config=self.foot))
        self.assertFalse(swept_path_blocked(grid, META, v_mps=0.18, omega_radps=0.0,
                                            config=self.foot, max_reach_m=0.12))

    def test_start_pose_moves_the_arc(self):
        # Obstacle 0.2 m ahead of where the robot IS now (0.5 m past the scan
        # origin) — blocked only when the arc starts at the current pose.
        grid = _clear()
        for y in (-0.04, 0.0, 0.04):
            _block(grid, 0.70, y)
        self.assertFalse(swept_path_blocked(grid, META, v_mps=0.12, omega_radps=0.0,
                                            config=self.foot))
        self.assertTrue(swept_path_blocked(grid, META, v_mps=0.12, omega_radps=0.0,
                                           config=self.foot, start_pose=(0.5, 0.0, 0.0)))
        self.assertFalse(swept_path_blocked(grid, META, v_mps=0.12, omega_radps=0.0,
                                            config=self.foot, start_pose=(0.5, 0.0, math.pi / 2)))


if __name__ == "__main__":
    unittest.main()

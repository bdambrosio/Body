"""Tests for the body-frame swept-footprint safety check."""
import math
import unittest

import numpy as np

from desktop.nav.safety import SafetyConfig, swept_path_blocked_local


def _meta(res=0.08, n=80):
    # Body-centered local_map: origin places the body at grid center.
    return {
        "resolution_m": res,
        "origin_x_m": -(n * res) / 2.0,
        "origin_y_m": -(n * res) / 2.0,
        "nx": n,
        "ny": n,
    }


def _grid(n=80, fill=1):
    return np.full((n, n), fill, dtype=np.int8)


def _set_cell(grid, meta, x, y, val):
    res = meta["resolution_m"]
    i = int(math.floor((x - meta["origin_x_m"]) / res))
    j = int(math.floor((y - meta["origin_y_m"]) / res))
    grid[i, j] = val


class TestSweptSafety(unittest.TestCase):
    def setUp(self):
        # Generous observed area so the empty-map guard never fires
        # except in the test that targets it.
        self.cfg = SafetyConfig(footprint_radius_m=0.22)

    def test_clear_path_not_blocked(self):
        meta = _meta()
        grid = _grid(fill=1)
        self.assertFalse(
            swept_path_blocked_local(
                grid, meta, v_mps=0.2, omega_radps=0.0, config=self.cfg,
            )
        )

    def test_obstacle_dead_ahead_blocks(self):
        meta = _meta()
        grid = _grid(fill=1)
        _set_cell(grid, meta, 0.30, 0.0, 0)  # blocked cell 30 cm ahead
        self.assertTrue(
            swept_path_blocked_local(
                grid, meta, v_mps=0.2, omega_radps=0.0, config=self.cfg,
            )
        )

    def test_shoulder_obstacle_blocks_but_old_wedge_would_miss(self):
        # 18 cm off-axis at 25 cm ahead: outside a ±20° wedge
        # (tan20°·0.25 ≈ 0.09 m) but inside the 22 cm footprint.
        meta = _meta()
        grid = _grid(fill=1)
        _set_cell(grid, meta, 0.25, 0.18, 0)
        self.assertTrue(
            swept_path_blocked_local(
                grid, meta, v_mps=0.2, omega_radps=0.0, config=self.cfg,
            )
        )

    def test_pure_rotation_never_blocks(self):
        meta = _meta()
        grid = _grid(fill=1)
        _set_cell(grid, meta, 0.20, 0.0, 0)  # obstacle ahead
        # v≈0, only rotating — must be allowed (escape route).
        self.assertFalse(
            swept_path_blocked_local(
                grid, meta, v_mps=0.0, omega_radps=0.5, config=self.cfg,
            )
        )

    def test_curved_path_catches_what_straight_misses(self):
        # On a sharp left curve the body swings well off the heading
        # axis. An obstacle there (lateral offset > footprint) must
        # block on the curve but NOT on a straight path of equal reach.
        # Force a long enough reach for the curvature to matter.
        cfg = SafetyConfig(
            footprint_radius_m=0.22,
            preview_distance_m=0.6,
            preview_min_distance_m=0.6,
            preview_time_s=10.0,
        )
        meta = _meta()
        v, omega = 0.2, 1.0
        R = v / omega
        phi = 2.0  # well into the arc (reach 0.6 m, t_total 3 s)
        cx = R * math.sin(phi)
        cy = R * (1.0 - math.cos(phi))
        # Beyond the effective footprint radius (incl. half-cell inflation),
        # so a straight sweep can't reach it.
        self.assertGreater(
            cy, cfg.footprint_radius_m + 0.5 * meta["resolution_m"]
        )

        grid = _grid(fill=1)
        _set_cell(grid, meta, cx, cy, 0)
        self.assertTrue(
            swept_path_blocked_local(
                grid, meta, v_mps=v, omega_radps=omega, config=cfg,
            )
        )
        # The same obstacle is off a straight path of equal reach.
        grid2 = _grid(fill=1)
        _set_cell(grid2, meta, cx, cy, 0)
        self.assertFalse(
            swept_path_blocked_local(
                grid2, meta, v_mps=v, omega_radps=0.0, config=cfg,
            )
        )

    def test_unknown_close_blocks_when_enabled(self):
        meta = _meta()
        grid = _grid(fill=1)
        # Make a patch of unknown right in front, within block range.
        _set_cell(grid, meta, 0.15, 0.0, -1)
        cfg = SafetyConfig(
            footprint_radius_m=0.22, block_on_unknown=True,
            unknown_block_range_m=0.25,
        )
        self.assertTrue(
            swept_path_blocked_local(
                grid, meta, v_mps=0.2, omega_radps=0.0, config=cfg,
            )
        )
        # Disabled → unknown is passable.
        cfg_off = SafetyConfig(footprint_radius_m=0.22, block_on_unknown=False)
        self.assertFalse(
            swept_path_blocked_local(
                grid, meta, v_mps=0.2, omega_radps=0.0, config=cfg_off,
            )
        )

    def test_far_unknown_is_passable(self):
        meta = _meta()
        grid = _grid(fill=1)
        # Unknown well beyond unknown_block_range_m (0.25 m) but still
        # surrounded by observed clear so the empty-map guard passes.
        _set_cell(grid, meta, 0.33, 0.0, -1)
        cfg = SafetyConfig(
            footprint_radius_m=0.22, block_on_unknown=True,
            unknown_block_range_m=0.25,
        )
        self.assertFalse(
            swept_path_blocked_local(
                grid, meta, v_mps=0.2, omega_radps=0.0, config=cfg,
            )
        )

    def test_empty_map_guard_blocks(self):
        # All-unknown but fresh local_map (launcher startup race): the
        # swept region has zero observed cells → refuse to drive.
        meta = _meta()
        grid = _grid(fill=-1)
        cfg = SafetyConfig(
            footprint_radius_m=0.22, block_on_unknown=False, min_observed_cells=3,
        )
        self.assertTrue(
            swept_path_blocked_local(
                grid, meta, v_mps=0.2, omega_radps=0.0, config=cfg,
            )
        )

    def test_malformed_meta_fails_safe_when_moving(self):
        grid = _grid(fill=1)
        bad = {"resolution_m": 0.0, "origin_x_m": 0.0, "origin_y_m": 0.0}
        self.assertTrue(
            swept_path_blocked_local(
                grid, bad, v_mps=0.2, omega_radps=0.0, config=self.cfg,
            )
        )
        # But not while merely rotating.
        self.assertFalse(
            swept_path_blocked_local(
                grid, bad, v_mps=0.0, omega_radps=0.5, config=self.cfg,
            )
        )

    def test_reverse_motion_checks_behind(self):
        meta = _meta()
        grid = _grid(fill=1)
        _set_cell(grid, meta, -0.25, 0.0, 0)  # obstacle behind
        self.assertTrue(
            swept_path_blocked_local(
                grid, meta, v_mps=-0.1, omega_radps=0.0, config=self.cfg,
            )
        )
        # Obstacle behind doesn't block forward motion.
        self.assertFalse(
            swept_path_blocked_local(
                grid, meta, v_mps=0.2, omega_radps=0.0, config=self.cfg,
            )
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for the single-frame lidar scan rasterizer (Tier-3 substrate)."""
import math
import unittest

import numpy as np

from body.lib.drive_safety import FootprintConfig, swept_path_blocked
from body.lib.scan_raster import ScanRasterConfig, rasterize_scan, rasterize_scan_msg

N = 360
ANGLE_MIN = -math.pi
ANGLE_INC = 2.0 * math.pi / N


def _idx(theta: float) -> int:
    return int(round((theta - ANGLE_MIN) / ANGLE_INC)) % N


def _scan(hits=None, fill=0.0):
    """ranges list of length N; hits = {index: range_m}."""
    rs = [fill] * N
    for i, r in (hits or {}).items():
        rs[i % N] = r
    return rs


def _cell(grid, meta, x, y):
    res = meta["resolution_m"]
    i = int(math.floor((x - meta["origin_x_m"]) / res))
    j = int(math.floor((y - meta["origin_y_m"]) / res))
    return int(grid[i, j])


class TestScanRaster(unittest.TestCase):
    def setUp(self):
        self.cfg = ScanRasterConfig()

    def test_obstacle_ahead_blocks_cell(self):
        # Cluster of returns straight ahead at 1.0 m.
        hits = {i: 1.0 for i in range(_idx(0.0) - 4, _idx(0.0) + 5)}
        grid, meta = rasterize_scan(_scan(hits), ANGLE_MIN, ANGLE_INC, self.cfg)
        self.assertEqual(_cell(grid, meta, 1.0, 0.0), 0)        # blocked at hit
        self.assertEqual(_cell(grid, meta, 0.3, 0.0), 1)        # cleared before it

    def test_no_return_clears_to_horizon(self):
        # All beams no-return → open space, cleared (NOT left unknown).
        grid, meta = rasterize_scan(_scan(), ANGLE_MIN, ANGLE_INC, self.cfg)
        self.assertEqual(_cell(grid, meta, 0.5, 0.0), 1)
        self.assertEqual(_cell(grid, meta, 0.0, 0.5), 1)
        self.assertFalse(np.any(grid == 0))                     # nothing blocked

    def test_full_ring_casts_unknown_shadow(self):
        # A full wall of returns at 1.0 m → ring blocked, interior clear,
        # behind it unknown (no open beam to clear the shadow).
        grid, meta = rasterize_scan(
            _scan({i: 1.0 for i in range(N)}), ANGLE_MIN, ANGLE_INC, self.cfg)
        self.assertEqual(_cell(grid, meta, 1.0, 0.0), 0)
        self.assertEqual(_cell(grid, meta, 0.3, 0.0), 1)
        self.assertEqual(_cell(grid, meta, 1.6, 0.0), -1)

    def test_yaw_offset_rotates_hit(self):
        cfg = ScanRasterConfig(lidar_yaw_rad=math.pi / 2)
        hits = {i: 1.0 for i in range(_idx(0.0) - 2, _idx(0.0) + 3)}
        grid, meta = rasterize_scan(_scan(hits), ANGLE_MIN, ANGLE_INC, cfg)
        # Beam at sensor 0° now points to body +y (left): hit at (0, 1.0).
        self.assertEqual(_cell(grid, meta, 0.0, 1.0), 0)

    def test_empty_ranges_all_unknown(self):
        grid, meta = rasterize_scan(None, ANGLE_MIN, ANGLE_INC, self.cfg)
        self.assertTrue(np.all(grid == -1))

    def test_dtype_and_shape(self):
        grid, meta = rasterize_scan(_scan(), ANGLE_MIN, ANGLE_INC, self.cfg)
        self.assertEqual(grid.dtype, np.int8)
        self.assertEqual(grid.shape, (meta["nx"], meta["ny"]))


class TestRasterPlusSweptCheck(unittest.TestCase):
    """The rasterized scan feeds the unchanged swept-footprint check."""

    def setUp(self):
        self.cfg = ScanRasterConfig()
        self.foot = FootprintConfig(footprint_radius_m=0.22)

    def test_obstacle_ahead_blocks_motion(self):
        hits = {i: 0.3 for i in range(_idx(0.0) - 6, _idx(0.0) + 7)}
        grid, meta = rasterize_scan(_scan(hits), ANGLE_MIN, ANGLE_INC, self.cfg)
        self.assertTrue(swept_path_blocked(
            grid, meta, v_mps=0.18, omega_radps=0.0, config=self.foot))

    def test_open_scan_allows_motion(self):
        grid, meta = rasterize_scan(_scan(), ANGLE_MIN, ANGLE_INC, self.cfg)
        self.assertFalse(swept_path_blocked(
            grid, meta, v_mps=0.18, omega_radps=0.0, config=self.foot))

    def test_empty_scan_blocks_motion(self):
        # No lidar this frame → all unknown → empty-region guard blocks.
        grid, meta = rasterize_scan(None, ANGLE_MIN, ANGLE_INC, self.cfg)
        self.assertTrue(swept_path_blocked(
            grid, meta, v_mps=0.18, omega_radps=0.0, config=self.foot))


class TestUnknownBins(unittest.TestCase):
    """Beams with no information must stay unknown, never clear to the horizon."""

    def setUp(self):
        self.cfg = ScanRasterConfig()

    def test_unknown_bins_neither_clear_nor_block(self):
        # Wall ahead at 1.0 m in the unknown bins; all other beams no-return.
        fwd = list(range(_idx(0.0) - 6, _idx(0.0) + 7))
        hits = {i: 1.0 for i in fwd}
        grid, meta = rasterize_scan(_scan(hits), ANGLE_MIN, ANGLE_INC, self.cfg,
                                    unknown_bins=fwd)
        self.assertEqual(_cell(grid, meta, 1.0, 0.0), -1)   # hit not stamped
        self.assertEqual(_cell(grid, meta, 2.0, 0.0), -1)   # not cleared to horizon
        # Without unknown_bins the null beams around clear as before.
        grid2, _ = rasterize_scan(_scan(), ANGLE_MIN, ANGLE_INC, self.cfg)
        self.assertEqual(_cell(grid2, meta, 2.0, 0.0), 1)

    def test_null_unknown_bin_does_not_clear(self):
        fwd = list(range(_idx(0.0) - 6, _idx(0.0) + 7))
        grid, meta = rasterize_scan(_scan(), ANGLE_MIN, ANGLE_INC, self.cfg,
                                    unknown_bins=fwd)
        self.assertEqual(_cell(grid, meta, 2.0, 0.0), -1)
        self.assertEqual(_cell(grid, meta, -2.0, 0.0), 1)    # other beams still clear

    def test_bad_unknown_bin_entries_ignored(self):
        grid, meta = rasterize_scan(_scan(), ANGLE_MIN, ANGLE_INC, self.cfg,
                                    unknown_bins=["x", None, -1, 10_000])
        self.assertEqual(_cell(grid, meta, 2.0, 0.0), 1)

    def test_too_near_return_is_unknown(self):
        fwd = list(range(_idx(0.0) - 6, _idx(0.0) + 7))
        grid, meta = rasterize_scan(_scan({i: 0.03 for i in fwd}), ANGLE_MIN, ANGLE_INC,
                                    self.cfg)
        self.assertEqual(_cell(grid, meta, 2.0, 0.0), -1)

    def test_rasterize_scan_msg_honours_unknown_bins(self):
        n = 360
        ranges = [None] * n
        msg = {"ranges": ranges, "angle_min": 0.0, "angle_increment": 2 * math.pi / n,
               "unknown_bins": list(range(0, 8)) + list(range(353, 360))}
        grid, meta = rasterize_scan_msg(msg, self.cfg)
        self.assertEqual(_cell(grid, meta, 2.0, 0.0), -1)
        self.assertEqual(_cell(grid, meta, 0.0, 2.0), 1)
        msg.pop("unknown_bins")                       # older Pi: no field
        grid, meta = rasterize_scan_msg(msg, self.cfg)
        self.assertEqual(_cell(grid, meta, 2.0, 0.0), 1)

    def test_robot_clear_radius_seeds_own_cells(self):
        cfg = ScanRasterConfig(robot_clear_radius_m=0.12)
        everything = list(range(N))
        grid, meta = rasterize_scan(_scan(), ANGLE_MIN, ANGLE_INC, cfg, unknown_bins=everything)
        self.assertEqual(_cell(grid, meta, 0.0, 0.0), 1)
        self.assertEqual(_cell(grid, meta, 0.5, 0.0), -1)
        grid0, _ = rasterize_scan(_scan(), ANGLE_MIN, ANGLE_INC, self.cfg, unknown_bins=everything)
        self.assertEqual(_cell(grid0, meta, 0.0, 0.0), -1)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the editable reference-map model.

Run: PYTHONPATH=. python3 -m unittest desktop.map_editor.test_editor_map -v
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from desktop.map_editor import editor_map as em
from desktop.reference_map.reference_map import (
    OCCUPIED_LOG_ODDS_THRESHOLD, load_reference_map,
)


def _make_map(nx: int = 20, ny: int = 20, *, res: float = 0.05,
              ox: float = -0.5, oy: float = -0.5) -> em.EditorMap:
    return em.EditorMap(
        log_odds=np.zeros((nx, ny), dtype=np.float32),
        resolution_m=res, origin_x_m=ox, origin_y_m=oy,
        session_id="testsess0001", metadata={"map_version": 1},
        trajectory=None,
    )


class TestGeometry(unittest.TestCase):
    def test_world_cell_inverse(self):
        m = _make_map(res=0.05, ox=-0.5, oy=-0.5)
        for (i, j) in [(0, 0), (5, 7), (19, 19)]:
            x, y = m.cell_to_world(i, j)
            self.assertEqual(m.world_to_cell(x, y), (i, j))

    def test_brush_clamps_at_edge(self):
        m = _make_map(nx=20, ny=20)
        ii, jj = m.brush_cells(0, 0, radius_cells=3)
        self.assertTrue((ii >= 0).all() and (jj >= 0).all())
        self.assertTrue((ii < 20).all() and (jj < 20).all())
        self.assertTrue(np.any((ii == 0) & (jj == 0)))

    def test_brush_radius_zero_single_cell(self):
        m = _make_map()
        ii, jj = m.brush_cells(5, 5, radius_cells=0)
        self.assertEqual(list(zip(ii.tolist(), jj.tolist())), [(5, 5)])

    def test_brush_disk_excludes_corners(self):
        m = _make_map()
        ii, jj = m.brush_cells(10, 10, radius_cells=2)
        cells = set(zip(ii.tolist(), jj.tolist()))
        self.assertNotIn((12, 12), cells)
        self.assertIn((12, 10), cells)

    def test_bounds_ij_tracks_edits(self):
        m = _make_map(nx=30, ny=30)
        self.assertIsNone(m.bounds_ij())
        ii, jj = m.brush_cells(15, 12, radius_cells=1)
        m.paint(ii, jj, em.WALL)
        b = m.bounds_ij()
        self.assertIsNotNone(b)
        self.assertTrue(b[0] <= 15 <= b[1] and b[2] <= 12 <= b[3])


class TestPaint(unittest.TestCase):
    def test_wall_occupied_and_drives_blocked(self):
        m = _make_map()
        ii, jj = m.brush_cells(10, 10, radius_cells=1)
        m.paint(ii, jj, em.WALL)
        self.assertTrue(
            (m.log_odds[ii, jj] > OCCUPIED_LOG_ODDS_THRESHOLD).all())
        self.assertTrue((m.driveable_grid()[ii, jj] == 0).all())

    def test_free_clears(self):
        m = _make_map()
        ii, jj = m.brush_cells(10, 10, radius_cells=1)
        m.paint(ii, jj, em.FREE)
        self.assertTrue(
            (m.log_odds[ii, jj] < -OCCUPIED_LOG_ODDS_THRESHOLD).all())
        self.assertTrue((m.driveable_grid()[ii, jj] == 1).all())

    def test_unknown_resets(self):
        m = _make_map()
        ii, jj = m.brush_cells(10, 10, radius_cells=1)
        m.paint(ii, jj, em.WALL)
        m.paint(ii, jj, em.UNKNOWN)
        self.assertTrue((m.log_odds[ii, jj] == 0).all())
        self.assertTrue((m.driveable_grid()[ii, jj] == -1).all())

    def test_bad_kind_raises(self):
        m = _make_map()
        with self.assertRaises(ValueError):
            m.paint(np.array([1]), np.array([1]), "bogus")

    def test_undo_restore(self):
        m = _make_map()
        snap = m.snapshot_occ()
        ii, jj = m.brush_cells(10, 10, radius_cells=2)
        m.paint(ii, jj, em.WALL)
        self.assertFalse(np.array_equal(m.log_odds, snap))
        m.restore_occ(snap)
        self.assertTrue(np.array_equal(m.log_odds, snap))


class TestRoundTrip(unittest.TestCase):
    def test_save_regenerates_fields_and_persists_edits(self):
        m = _make_map(nx=40, ny=40)
        # Paint a small wall block (radius ≥1 so denoise keeps it).
        ii, jj = m.brush_cells(20, 20, radius_cells=2)
        m.paint(ii, jj, em.WALL)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reference_map.npz")
            em.save_npz(m, path, backup=False)
            rm = load_reference_map(path)
        # Edited occupancy persisted.
        self.assertTrue((rm.occupancy_log_odds[ii, jj] > 0).all())
        # Derived fields regenerated from the edit (non-empty near wall).
        self.assertGreater(float(rm.likelihood_field[20, 20]), 0.0)
        self.assertTrue(np.isfinite(rm.distance_field_m).all())
        # Geometry preserved.
        self.assertAlmostEqual(rm.resolution_m, m.resolution_m, places=6)
        self.assertAlmostEqual(rm.origin_x_m, m.origin_x_m, places=6)

    def test_save_makes_backup(self):
        m = _make_map(nx=12, ny=12)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reference_map.npz")
            em.save_npz(m, path, backup=True)          # no original yet
            self.assertFalse(os.path.exists(path + ".bak"))
            em.save_npz(m, path, backup=True)          # now backs up
            self.assertTrue(os.path.exists(path + ".bak"))

    def test_real_reference_map_round_trips(self):
        base = os.path.expanduser("~/Body/maps")
        found = None
        for root, _dirs, files in os.walk(base):
            if "reference_map.npz" in files:
                found = os.path.join(root, "reference_map.npz")
                break
        if not found:
            self.skipTest("no real reference_map.npz available")
        m = em.load_npz(found)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reference_map.npz")
            em.save_npz(m, path, backup=False)
            rm2 = load_reference_map(path)
        self.assertEqual(m.shape, rm2.occupancy_log_odds.shape)
        # Unedited occupancy round-trips byte-for-byte.
        self.assertTrue(
            np.array_equal(m.log_odds, rm2.occupancy_log_odds))


if __name__ == "__main__":
    unittest.main()

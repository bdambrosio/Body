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
        snap = m.snapshot_state()
        ii, jj = m.brush_cells(10, 10, radius_cells=2)
        m.paint(ii, jj, em.WALL)
        self.assertFalse(np.array_equal(m.log_odds, snap[0]))
        m.restore_state(snap)
        self.assertTrue(np.array_equal(m.log_odds, snap[0]))


class TestNoGo(unittest.TestCase):
    def test_nogo_paint_sets_mask_not_occupancy(self):
        m = _make_map()
        ii, jj = m.brush_cells(10, 10, radius_cells=1)
        m.paint(ii, jj, em.NOGO)
        self.assertTrue(m.nogo[ii, jj].all())
        # Occupancy (and thus localization) untouched.
        self.assertTrue((m.log_odds[ii, jj] == 0).all())
        self.assertTrue((m.driveable_grid()[ii, jj] == -1).all())

    def test_erase_nogo(self):
        m = _make_map()
        ii, jj = m.brush_cells(10, 10, radius_cells=1)
        m.paint(ii, jj, em.NOGO)
        m.paint(ii, jj, em.ERASE_NOGO)
        self.assertFalse(m.nogo[ii, jj].any())

    def test_undo_restores_both_layers(self):
        m = _make_map()
        snap = m.snapshot_state()
        ii, jj = m.brush_cells(8, 8, radius_cells=1)
        m.paint(ii, jj, em.NOGO)
        self.assertTrue(m.nogo[ii, jj].all())
        m.restore_state(snap)
        self.assertFalse(m.nogo.any())

    def test_nogo_round_trips(self):
        m = _make_map(nx=40, ny=40)
        ii, jj = m.brush_cells(20, 20, radius_cells=2)
        m.paint(ii, jj, em.NOGO)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reference_map.npz")
            em.save_npz(m, path, backup=False)
            rm = load_reference_map(path)
        self.assertIsNotNone(rm.nogo_mask)
        self.assertTrue(rm.nogo_mask[ii, jj].all())

    def test_nogo_does_not_affect_likelihood_field(self):
        # Localization isolation: an identical map with vs without a
        # no-go region must produce byte-identical likelihood/distance.
        def _saved(with_nogo):
            m = _make_map(nx=40, ny=40)
            wi, wj = m.brush_cells(10, 10, radius_cells=2)
            m.paint(wi, wj, em.WALL)  # a real wall in both
            if with_nogo:
                ni, nj = m.brush_cells(28, 28, radius_cells=3)
                m.paint(ni, nj, em.NOGO)
            d = tempfile.mkdtemp()
            path = os.path.join(d, "reference_map.npz")
            em.save_npz(m, path, backup=False)
            return load_reference_map(path)

        a, b = _saved(False), _saved(True)
        self.assertTrue(np.array_equal(a.likelihood_field, b.likelihood_field))
        self.assertTrue(np.array_equal(a.distance_field_m, b.distance_field_m))

    def test_empty_nogo_not_persisted(self):
        # An all-False mask is not written (absent key), and loads as None.
        m = _make_map(nx=12, ny=12)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reference_map.npz")
            em.save_npz(m, path, backup=False)
            rm = load_reference_map(path)
        self.assertIsNone(rm.nogo_mask)


class TestStampScan(unittest.TestCase):
    def test_in_range_hit_becomes_wall(self):
        m = _make_map(nx=40, ny=40, res=0.05, ox=-1.0, oy=-1.0)
        world = np.array([[0.5, 0.0]])             # 0.5 m ahead of (0,0)
        ii, jj = m.stamp_cells_from_scan(world, (0.0, 0.0, 0.0), max_range_m=4.0)
        self.assertEqual(len(ii), 1)               # no thickening
        m.paint(ii, jj, em.WALL)
        self.assertTrue((m.driveable_grid()[ii, jj] == 0).all())

    def test_range_gate_excludes_far(self):
        m = _make_map(nx=400, ny=400, res=0.05, ox=-1.0, oy=-1.0)
        world = np.array([[3.0, 0.0], [5.0, 0.0]])  # keep 3 m, drop 5 m
        ii, _ = m.stamp_cells_from_scan(world, (0.0, 0.0, 0.0), max_range_m=4.0)
        self.assertEqual(len(ii), 1)

    def test_skips_existing_wall(self):
        m = _make_map(nx=40, ny=40, res=0.05, ox=-1.0, oy=-1.0)
        world = np.array([[0.5, 0.0]])
        ii, jj = m.stamp_cells_from_scan(world, (0.0, 0.0, 0.0), max_range_m=4.0)
        m.paint(ii, jj, em.WALL)
        ii2, _ = m.stamp_cells_from_scan(world, (0.0, 0.0, 0.0), max_range_m=4.0)
        self.assertEqual(len(ii2), 0)              # already wall → nothing

    def test_dedup_same_cell(self):
        m = _make_map(nx=40, ny=40, res=0.05, ox=-1.0, oy=-1.0)
        world = np.array([[0.500, 0.0], [0.505, 0.0]])  # same cell
        ii, _ = m.stamp_cells_from_scan(world, (0.0, 0.0, 0.0), max_range_m=4.0)
        self.assertEqual(len(ii), 1)

    def test_empty_scan(self):
        m = _make_map()
        self.assertEqual(len(m.stamp_cells_from_scan(
            None, (0.0, 0.0, 0.0), max_range_m=4.0)[0]), 0)
        self.assertEqual(len(m.stamp_cells_from_scan(
            np.empty((0, 2)), (0.0, 0.0, 0.0), max_range_m=4.0)[0]), 0)


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


class TestCheckpointPersistence(unittest.TestCase):
    def test_checkpoint_survives_save_load(self):
        from desktop.localization.checkpoints import (
            checkpoints_from_metadata, upsert_checkpoint,
            write_checkpoints_to_metadata,
        )
        m = _make_map()
        cps, _ = upsert_checkpoint([], (0.3, -0.2, 0.5), 2.0, created_ts=42.0)
        write_checkpoints_to_metadata(m.metadata, cps)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reference_map.npz")
            em.save_npz(m, path, backup=False)
            rm = load_reference_map(path)
        back = checkpoints_from_metadata(rm.metadata)
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0].id, "cp_000")
        self.assertAlmostEqual(back[0].x_m, 0.3)
        self.assertAlmostEqual(back[0].radius_m, 2.0)


class TestRestampFromScans(unittest.TestCase):
    """Recognize's local clear-and-restamp (ray-trace, replace observed)."""

    WALL = em._PAINT_LOG_ODDS[em.WALL]    # +4
    FREE = em._PAINT_LOG_ODDS[em.FREE]    # -4

    def _grid(self):
        # 60×60 @ 0.1 m, origin 0 → world (x) = 0.1*i; sensor at (3,3)=(30,30).
        return em.EditorMap(
            log_odds=np.zeros((60, 60), dtype=np.float32),
            resolution_m=0.1, origin_x_m=0.0, origin_y_m=0.0,
        )

    def test_replace_clears_stale_wall_and_stamps_endpoint(self):
        m = self._grid()
        m.log_odds[35, 30] = self.WALL          # stale wall on the beam path
        # One east-facing beam: sensor (3,3) → endpoint (4,3) = cell (40,30).
        m.restamp_from_scans(
            [(np.array([[4.0, 3.0]]), (3.0, 3.0))],
            center_xy=(3.0, 3.0), radius_m=2.0,
        )
        self.assertEqual(m.log_odds[35, 30], self.FREE)   # stale wall cleared
        self.assertEqual(m.log_odds[40, 30], self.WALL)   # endpoint occupied
        self.assertEqual(m.log_odds[30, 30], self.FREE)   # sensor cell freed
        self.assertEqual(m.log_odds[45, 30], 0.0)         # behind endpoint untouched
        self.assertEqual(m.log_odds[30, 40], 0.0)         # off-beam untouched

    def test_endpoint_beyond_radius_not_stamped_inner_freed(self):
        m = self._grid()
        # Endpoint (5.5,3)=cell (55,30) is 25 cells from center > 20-cell radius.
        m.restamp_from_scans(
            [(np.array([[5.5, 3.0]]), (3.0, 3.0))],
            center_xy=(3.0, 3.0), radius_m=2.0,
        )
        self.assertEqual(m.log_odds[50, 30], self.FREE)   # within radius: freed
        self.assertEqual(m.log_odds[55, 30], 0.0)         # endpoint past radius: untouched

    def test_endpoint_wins_over_free_across_scans(self):
        m = self._grid()
        # Scan A puts a wall at (40,30); scan B's longer beam traverses it as free.
        m.restamp_from_scans(
            [
                (np.array([[4.0, 3.0]]), (3.0, 3.0)),
                (np.array([[4.4, 3.0]]), (3.0, 3.0)),
            ],
            center_xy=(3.0, 3.0), radius_m=2.0,
        )
        self.assertEqual(m.log_odds[40, 30], self.WALL)   # occupied wins the tie


if __name__ == "__main__":
    unittest.main()

"""ChangeDetector (explore plan B1)."""
import math
import unittest

import numpy as np

from desktop.perception.change_detector import (
    KIND_MISSING, KIND_MOVING, KIND_NEW, STATUS_APPEARED, STATUS_GONE,
    ChangeDetector,
)

RES = 0.05
META = {"resolution_m": RES, "origin_x_m": 0.0, "origin_y_m": 0.0}
N = 120   # 6 m square


def room():
    ref = np.ones((N, N), dtype=np.int8)
    ref[:3, :] = ref[-3:, :] = ref[:, :3] = ref[:, -3:] = 0     # walls
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    edge = np.minimum.reduce([ii - 2, N - 3 - ii, jj - 2, N - 3 - jj]) * RES
    dist = np.maximum(edge, 0.0).astype(np.float32)
    return ref, dist


def blob_points(cx, cy, pose=(0.0, 0.0, 0.0), r=0.05, n=6):
    """Body-frame points of a small object at world (cx, cy)."""
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        wx, wy = cx + r * math.cos(a), cy + r * math.sin(a)
        dx, dy = wx - pose[0], wy - pose[1]
        c, s = math.cos(-pose[2]), math.sin(-pose[2])
        pts.append((c * dx - s * dy, s * dx + c * dy))
    return np.array(pts)


class TestMoving(unittest.TestCase):
    def test_walker_appears_then_goes(self):
        ref, dist = room()
        d = ChangeDetector(ref, dist, META)
        pose = (3.0, 1.0, 0.0)
        ev = []
        for k in range(12):                   # walks +y at 1 m/s, 10 Hz
            ev += d.observe_scan(k * 0.1, blob_points(4.0, 1.0 + 0.1 * k, pose), pose)
        app = [e for e in ev if e.status == STATUS_APPEARED]
        self.assertEqual(len(app), 1)
        self.assertEqual(app[0].kind, KIND_MOVING)
        self.assertAlmostEqual(app[0].speed_mps, 1.0, delta=0.2)
        ev = []
        for k in range(12, 30):               # out of sight
            ev += d.observe_scan(k * 0.1, np.zeros((0, 2)), pose)
        gone = [e for e in ev if e.status == STATUS_GONE]
        self.assertEqual(len(gone), 1)
        self.assertAlmostEqual(gone[0].path_m, 1.1, delta=0.15)

    def test_stationary_object_is_not_moving(self):
        ref, dist = room()
        d = ChangeDetector(ref, dist, META)
        pose = (3.0, 3.0, 0.5)
        ev = []
        for k in range(30):
            ev += d.observe_scan(k * 0.1, blob_points(4.0, 3.5, pose), pose)
        self.assertEqual(ev, [])
        self.assertEqual(len(d.tracks()), 1)

    def test_wall_hits_are_not_residuals(self):
        ref, dist = room()
        d = ChangeDetector(ref, dist, META)
        pose = (3.0, 3.0, 0.0)
        pts = blob_points(0.12, 3.0, pose)     # inside the wall margin
        self.assertEqual(d.residual_clusters(pts, pose), [])

    def test_large_cluster_rejected(self):
        ref, dist = room()
        d = ChangeDetector(ref, dist, META)
        pose = (3.0, 3.0, 0.0)
        pts = np.array([[1.0, y] for y in np.arange(-1.0, 1.0, 0.05)])  # 2 m line
        self.assertEqual(d.residual_clusters(pts, pose), [])


class TestStatic(unittest.TestCase):
    def setUp(self):
        self.ref, self.dist = room()
        self.d = ChangeDetector(self.ref, self.dist, META)
        self.score = np.zeros((N, N), dtype=np.float32)
        self.first = np.zeros((N, N))

    def test_new_object_lifecycle(self):
        self.score[60:63, 60:63] = 2.5
        self.first[60:63, 60:63] = 1.0
        ev = self.d.sweep(2.0, self.score, self.first)          # too young
        self.assertEqual(ev, [])
        ev = self.d.sweep(10.0, self.score, self.first)
        self.assertEqual([(e.kind, e.status) for e in ev], [(KIND_NEW, STATUS_APPEARED)])
        self.assertAlmostEqual(ev[0].x, 61.5 * RES, places=6)
        self.assertEqual(self.d.sweep(12.0, self.score, self.first), [])   # no repeat
        # Unobserved (score unchanged but young again? no — just no change): stays.
        self.score[60:63, 60:63] = -1.0                         # seen cleared
        self.first[60:63, 60:63] = 0.0
        ev = self.d.sweep(20.0, self.score, self.first)
        self.assertEqual([(e.kind, e.status) for e in ev], [(KIND_NEW, STATUS_GONE)])

    def test_out_of_view_is_not_gone(self):
        self.score[60:63, 60:63] = 2.5
        self.first[60:63, 60:63] = 1.0
        self.d.sweep(10.0, self.score, self.first)
        self.score[60:63, 60:63] = 0.4      # decayed a bit, not cleared
        self.assertEqual(self.d.sweep(20.0, self.score, self.first), [])
        self.assertEqual(len(self.d.static_changes()), 1)

    def test_missing_wall_segment(self):
        self.score[0:3, 50:56] = -2.5
        ev = self.d.sweep(5.0, self.score, self.first)
        self.assertEqual([(e.kind, e.status) for e in ev], [(KIND_MISSING, STATUS_APPEARED)])
        self.score[0:3, 50:56] = 1.0
        ev = self.d.sweep(6.0, self.score, self.first)
        self.assertEqual([(e.kind, e.status) for e in ev], [(KIND_MISSING, STATUS_GONE)])

    def test_new_suppressed_near_moving_track(self):
        pose = (3.0, 1.0, 0.0)
        for k in range(8):
            self.d.observe_scan(k * 0.1, blob_points(3.0, 2.0 + 0.1 * k, pose), pose)
        self.assertTrue(any(t["moving"] for t in self.d.tracks()))
        i, j = int(3.0 / RES), int(2.7 / RES)
        self.score[i:i + 2, j:j + 2] = 2.5
        self.first[i:i + 2, j:j + 2] = 0.1
        self.assertEqual(self.d.sweep(10.0, self.score, self.first), [])

    def test_single_cell_noise_ignored(self):
        self.score[40, 40] = 3.0
        self.first[40, 40] = 1.0
        self.assertEqual(self.d.sweep(10.0, self.score, self.first), [])


if __name__ == "__main__":
    unittest.main()

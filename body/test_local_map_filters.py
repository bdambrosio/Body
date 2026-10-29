"""Tests for local_map staleness gating, odom-at-sensor-time anchoring and the
driveable-layer evidence filter."""
import math
import unittest

import numpy as np

from body.local_map import DriveableFilter, OdomRing, _sensor_key, _sensor_state


def _odom(ts, x, theta=0.0, sid="aaaa0001", seq=None):
    m = {"ts": ts, "x": x, "y": 0.0, "theta": theta, "source": "wheel_encoders",
         "session_id": sid}
    if seq is not None:
        m["seq"] = seq
    return m


class TestSensorState(unittest.TestCase):
    def test_new_same_stale(self):
        msg = {"ts": 100.0, "seq": 7}
        self.assertEqual(_sensor_state(msg, None, 100.1, 0.35), "new")
        self.assertEqual(_sensor_state(msg, _sensor_key(msg), 100.2, 0.35), "same")
        self.assertEqual(_sensor_state(msg, _sensor_key(msg), 100.5, 0.35), "stale")
        self.assertEqual(_sensor_state(None, None, 100.0, 0.35), "stale")

    def test_no_seq_uses_ts(self):
        a = {"ts": 100.0}
        b = {"ts": 100.1}
        self.assertEqual(_sensor_state(b, _sensor_key(a), 100.15, 0.35), "new")


class TestOdomRing(unittest.TestCase):
    def test_interpolates_at_sensor_time(self):
        r = OdomRing()
        for k in range(11):
            r.add(_odom(10.0 + 0.02 * k, 0.01 * k, seq=k))
        p = r.pose_at(10.05)
        self.assertTrue(p["interpolated"])
        self.assertAlmostEqual(p["x"], 0.025)
        self.assertAlmostEqual(p["odom_ts"], 10.05)

    def test_theta_wraps_through_pi(self):
        r = OdomRing()
        r.add(_odom(1.0, 0.0, theta=math.pi - 0.1))
        r.add(_odom(1.1, 0.0, theta=-math.pi + 0.1))
        p = r.pose_at(1.05)
        self.assertAlmostEqual(abs(p["theta"]), math.pi, places=6)

    def test_outside_buffer(self):
        r = OdomRing(max_extrapolate_s=0.15)
        r.add(_odom(1.0, 0.0))
        r.add(_odom(1.1, 0.1))
        p = r.pose_at(1.2)
        self.assertFalse(p["interpolated"])
        self.assertAlmostEqual(p["x"], 0.1)
        self.assertIsNone(r.pose_at(2.0))

    def test_session_change_clears(self):
        r = OdomRing()
        r.add(_odom(1.0, 5.0, sid="aaaa0001", seq=10))
        r.add(_odom(1.1, 5.1, sid="aaaa0001", seq=11))
        r.add(_odom(1.2, 0.0, sid="bbbb0002", seq=0))
        self.assertEqual(r.latest()["x"], 0.0)
        self.assertEqual(r.pose_at(1.05)["x"], 0.0)           # no blend across the reset

    def test_seq_decrease_clears(self):
        r = OdomRing()
        r.add(_odom(1.0, 5.0, seq=10))
        r.add(_odom(1.1, 0.0, seq=0))
        self.assertEqual(r.pose_at(1.0)["x"], 0.0)          # pre-reset pose gone


def _grid(cells, shape=(6, 6), dtype=bool):
    g = np.zeros(shape, dtype=dtype)
    for c in cells:
        g[c] = 1 if dtype is not bool else True
    return g


class TestDriveableFilter(unittest.TestCase):
    def setUp(self):
        self.shape = (6, 6)
        self.f = DriveableFilter(self.shape, lidar_block_frames=2, depth_block_frames=2,
                                 clear_frames=1, depth_memory_frames=3)

    def _lidar(self, state="new", cleared=(), hits=(), slab=()):
        return dict(
            lidar_state=state,
            lidar_cleared=_grid(cleared, self.shape),
            lidar_hit_cells=_grid(hits, self.shape),
            lidar_slab_count=_grid(slab, self.shape, np.int32),
        )

    def test_clear_frames_at_least_block_frames(self):
        self.assertEqual(self.f.clear_frames, 2)
        d = self.f.update(depth_state="stale", **self._lidar(cleared=[(1, 1)]))
        self.assertEqual(d[1, 1], -1)
        d = self.f.update(depth_state="stale", **self._lidar(cleared=[(1, 1)]))
        self.assertEqual(d[1, 1], 1)

    def test_same_frame_hit_cell_not_cleared(self):
        for _ in range(3):
            d = self.f.update(depth_state="stale",
                              **self._lidar(cleared=[(2, 2)], hits=[(2, 2)]))
        self.assertNotEqual(d[2, 2], 1)

    def test_reused_frame_does_not_advance_streaks(self):
        self.f.update(depth_state="stale", **self._lidar(slab=[(3, 3)]))
        for _ in range(5):
            d = self.f.update(depth_state="stale", **self._lidar(state="same", slab=[(3, 3)]))
        self.assertEqual(d[3, 3], -1)          # one real frame < 2 block frames
        d = self.f.update(depth_state="stale", **self._lidar(slab=[(3, 3)]))
        self.assertEqual(d[3, 3], 0)

    def test_stale_lidar_resets_to_unknown(self):
        for _ in range(3):
            self.f.update(depth_state="stale", **self._lidar(cleared=[(1, 1)], slab=[(4, 4)]))
        d = self.f.update(depth_state="stale", **self._lidar(state="stale"))
        self.assertEqual(d[1, 1], -1)
        self.assertEqual(d[4, 4], -1)

    def test_recent_depth_slab_hit_not_cleared_by_lidar(self):
        depth = _grid([(1, 2)], self.shape, np.int32) * 5
        self.f.update(depth_state="new", depth_slab_count=depth,
                      **self._lidar(cleared=[(1, 2)]))
        # Low obstacle leaves the depth FOV; lidar sweeps over it.
        out = []
        for _ in range(4):
            d = self.f.update(depth_state="new", depth_slab_count=np.zeros(self.shape, np.int32),
                              **self._lidar(cleared=[(1, 2)]))
            out.append(int(d[1, 2]))
        self.assertEqual(out[0], -1)
        self.assertEqual(out[1], -1)
        self.assertEqual(out[-1], 1)           # memory expired, lidar clears accumulate

    def test_depth_streak_only_on_new_depth_frames(self):
        depth = _grid([(0, 0)], self.shape, np.int32) * 5
        self.f.update(depth_state="new", depth_slab_count=depth, **self._lidar(state="same"))
        d = self.f.update(depth_state="same", depth_slab_count=depth, **self._lidar(state="same"))
        self.assertEqual(d[0, 0], -1)
        d = self.f.update(depth_state="new", depth_slab_count=depth, **self._lidar(state="same"))
        self.assertEqual(d[0, 0], 0)


if __name__ == "__main__":
    unittest.main()

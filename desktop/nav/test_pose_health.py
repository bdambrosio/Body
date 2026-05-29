"""Tests for the pose-health divergence monitor."""
import unittest

from desktop.nav.pose_health import PoseHealthConfig, PoseHealthMonitor


def _match(quality, n_points=100, valid=True):
    # score_best is the count of endpoints on occupied cells; quality is
    # score_best / n_points.
    return {
        "valid": valid,
        "score_best": quality * n_points,
        "n_points": n_points,
    }


class TestPoseHealthMonitor(unittest.TestCase):
    def setUp(self):
        self.cfg = PoseHealthConfig(
            quality_threshold=0.15, window_s=6.0, min_samples=4, min_points=30,
        )

    def test_healthy_not_lost(self):
        m = PoseHealthMonitor(self.cfg)
        t = 100.0
        for k in range(8):
            m.ingest(_match(0.6), t + k, seq=k)
        self.assertFalse(m.is_lost(t + 8))

    def test_sustained_low_quality_is_lost(self):
        m = PoseHealthMonitor(self.cfg)
        t = 100.0
        for k in range(8):
            m.ingest(_match(0.05), t + k, seq=k)
        self.assertTrue(m.is_lost(t + 8))

    def test_single_bad_match_not_lost(self):
        m = PoseHealthMonitor(self.cfg)
        t = 100.0
        for k in range(7):
            m.ingest(_match(0.6), t + k, seq=k)
        m.ingest(_match(0.0), t + 7, seq=7)  # one bad sample
        self.assertFalse(m.is_lost(t + 7))

    def test_too_few_samples_not_lost(self):
        m = PoseHealthMonitor(self.cfg)
        t = 100.0
        m.ingest(_match(0.0), t, seq=0)
        m.ingest(_match(0.0), t + 1, seq=1)
        self.assertFalse(m.is_lost(t + 1))

    def test_dedupe_by_seq(self):
        m = PoseHealthMonitor(self.cfg)
        t = 100.0
        # Same seq repeated — should add only one sample.
        for _ in range(8):
            m.ingest(_match(0.0), t, seq=5)
        self.assertFalse(m.is_lost(t + 6))  # only 1 sample retained

    def test_invalid_and_sparse_ignored(self):
        m = PoseHealthMonitor(self.cfg)
        t = 100.0
        for k in range(8):
            m.ingest(_match(0.0, valid=False), t + k, seq=k)
            m.ingest(_match(0.0, n_points=5), t + k, seq=100 + k)  # sparse
        self.assertFalse(m.is_lost(t + 8))

    def test_reset_clears_lost(self):
        m = PoseHealthMonitor(self.cfg)
        t = 100.0
        for k in range(8):
            m.ingest(_match(0.05), t + k, seq=k)
        self.assertTrue(m.is_lost(t + 8))
        m.reset()
        self.assertFalse(m.is_lost(t + 8))

    def test_window_prunes_old_samples(self):
        m = PoseHealthMonitor(self.cfg)
        t = 100.0
        # Old bad samples outside the window must not count.
        for k in range(8):
            m.ingest(_match(0.0), t + k, seq=k)
        self.assertTrue(m.is_lost(t + 8))
        # 20 s later, recover with good samples; old ones pruned.
        for k in range(8):
            m.ingest(_match(0.7), t + 30 + k, seq=100 + k)
        self.assertFalse(m.is_lost(t + 38))


if __name__ == "__main__":
    unittest.main()

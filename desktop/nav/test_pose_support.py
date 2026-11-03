"""PoseSupportMonitor (explore plan A2)."""
import unittest

from desktop.nav.pose_health import (
    SUPPORT_GOOD, SUPPORT_OFF_MAP, SUPPORT_UNKNOWN, SUPPORT_WEAK, PoseSupportMonitor,
)


def summ(q=0.5, cov=0.9, n=100, pose=(0.0, 0.0, 0.0)):
    return {"valid": True, "n_points": n, "score_best": q * n,
            "map_coverage": cov, "scan_pose_world": list(pose) if pose else None}


class TestPoseSupport(unittest.TestCase):
    def feed(self, m, k, **kw):
        for i in range(k):
            m.ingest(summ(**kw), 100.0 + 0.2 * i, seq=i)
        return 100.0 + 0.2 * (k - 1)

    def test_unknown_until_samples(self):
        m = PoseSupportMonitor()
        self.assertEqual(m.level(0.0), SUPPORT_UNKNOWN)

    def test_good(self):
        m = PoseSupportMonitor()
        self.assertEqual(m.level(self.feed(m, 5)), SUPPORT_GOOD)

    def test_off_map(self):
        m = PoseSupportMonitor()
        self.assertEqual(m.level(self.feed(m, 5, cov=0.1, q=0.05)), SUPPORT_OFF_MAP)

    def test_weak_on_poor_match(self):
        m = PoseSupportMonitor()
        self.assertEqual(m.level(self.feed(m, 5, q=0.05)), SUPPORT_WEAK)

    def test_weak_on_partial_coverage(self):
        m = PoseSupportMonitor()
        self.assertEqual(m.level(self.feed(m, 5, cov=0.5)), SUPPORT_WEAK)

    def test_duplicate_seq_ignored(self):
        m = PoseSupportMonitor()
        for _ in range(5):
            m.ingest(summ(), 100.0, seq=7)
        self.assertEqual(m.level(100.0), SUPPORT_UNKNOWN)

    def test_summary_without_coverage_ignored(self):
        m = PoseSupportMonitor()
        s = summ()
        del s["map_coverage"]
        for i in range(5):
            m.ingest(s, 100.0 + i * 0.1, seq=i)
        self.assertEqual(m.level(100.5), SUPPORT_UNKNOWN)

    def test_scan_ok(self):
        m = PoseSupportMonitor()
        self.assertTrue(m.scan_ok(summ()))
        self.assertFalse(m.scan_ok(summ(pose=None)))
        self.assertFalse(m.scan_ok(summ(cov=0.3)))
        self.assertFalse(m.scan_ok(summ(q=0.05)))
        self.assertFalse(m.scan_ok(summ(n=10)))
        self.assertFalse(m.scan_ok(None))

    def test_reset(self):
        m = PoseSupportMonitor()
        t = self.feed(m, 5)
        m.reset()
        self.assertEqual(m.level(t), SUPPORT_UNKNOWN)


class TestThresholdsFromRobotData(unittest.TestCase):
    """11-03 Hallway_loop medians: the misaligned boot (0.22 / 0.72) and the
    lost robot (0.23 / 0.65) must not read "good"; localized (0.71 / 0.90,
    and its p5 0.58 / 0.87) must."""

    def level(self, q, cov):
        m = PoseSupportMonitor()
        for i in range(5):
            m.ingest(summ(q=q, cov=cov), 100.0 + 0.2 * i, seq=i)
        return m.level(100.8)

    def test_misaligned_and_lost_are_weak(self):
        self.assertEqual(self.level(0.22, 0.72), SUPPORT_WEAK)
        self.assertEqual(self.level(0.23, 0.65), SUPPORT_WEAK)
        self.assertEqual(self.level(0.38, 0.85), SUPPORT_WEAK)

    def test_localized_is_good(self):
        self.assertEqual(self.level(0.71, 0.90), SUPPORT_GOOD)
        self.assertEqual(self.level(0.58, 0.87), SUPPORT_GOOD)


if __name__ == "__main__":
    unittest.main()

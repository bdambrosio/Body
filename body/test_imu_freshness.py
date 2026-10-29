"""Tests for BNO085 driver freshness/stale logic, settle gate and report-interval enabling.

No hardware or adafruit library needed: the pure helpers in body.imu_driver are exercised
with synthetic report counters and a fake sensor object.
"""
import unittest

from body.imu_driver import ReportFreshness, SettleGate, _enable


class TestReportFreshness(unittest.TestCase):
    def test_new_only_once_per_report(self):
        f = ReportFreshness(stale_s=0.2, first_report_timeout_s=2.0, now_wall=0.0)
        self.assertEqual(f.poll(1, 0.010, 0.011), (True, False))
        self.assertEqual(f.poll(1, 0.010, 0.013), (False, False))   # cached value: no publish
        self.assertEqual(f.poll(2, 0.020, 0.021), (True, False))

    def test_frozen_stream_goes_stale(self):
        f = ReportFreshness(stale_s=0.2, first_report_timeout_s=2.0, now_wall=0.0)
        f.poll(5, 1.000, 1.001)
        self.assertEqual(f.poll(5, 1.000, 1.150), (False, False))
        self.assertEqual(f.poll(5, 1.000, 1.250), (False, True))

    def test_first_report_grace(self):
        f = ReportFreshness(stale_s=0.2, first_report_timeout_s=2.0, now_wall=10.0)
        self.assertEqual(f.poll(None, None, 11.0), (False, False))
        self.assertEqual(f.poll(None, None, 12.5), (False, True))

    def test_reset_after_reinit_new_object_counts_from_zero(self):
        f = ReportFreshness(stale_s=0.2, first_report_timeout_s=2.0, now_wall=0.0)
        f.poll(500, 5.0, 5.0)
        f.reset(6.0)
        self.assertEqual(f.poll(1, 6.1, 6.1), (True, False))


class TestSettleGate(unittest.TestCase):
    def test_requires_time_accuracy_and_stillness(self):
        g = SettleGate(2.0, 0.18, 0.05, 3, 10.0, now_mono=0.0)
        self.assertFalse(g.update(1.0, 0.1, 0.0))          # too early
        self.assertFalse(g.update(2.1, 0.1, 0.0))          # still_count 2 < 3
        self.assertTrue(g.update(2.2, 0.1, 0.0))
        self.assertFalse(g.forced)

    def test_motion_restarts_stillness_then_max_wait_forces(self):
        g = SettleGate(2.0, 0.18, 0.05, 3, 5.0, now_mono=0.0)
        for t in (2.0, 2.5, 3.0, 4.0):
            self.assertFalse(g.update(t, 0.1, 0.5))        # rotating
        self.assertTrue(g.update(5.1, 0.1, 0.5))
        self.assertTrue(g.forced)

    def test_accuracy_gate_never_waived(self):
        g = SettleGate(2.0, 0.087, 0.05, 1, 5.0, now_mono=0.0)
        self.assertFalse(g.update(100.0, 0.2, 0.0))


class _FakeNew:
    def __init__(self):
        self.calls = []

    def enable_feature(self, fid, report_interval=None):
        self.calls.append((fid, report_interval))


class _FakeOld:
    """Old adafruit API: enable_feature(feature_id) only, with private helpers."""

    def __init__(self):
        self._readings = {}
        self.sent = []

    def enable_feature(self, fid):
        raise AssertionError("should use the private set-feature path")

    @staticmethod
    def _get_feature_enable_report(fid, report_interval=50000, sensor_specific_config=0):
        return ("set_feature", fid, report_interval)

    def _send_packet(self, channel, data):
        self.sent.append((channel, data))
        self._readings[data[1]] = (0.0, 0.0, 0.0, 1.0)

    def _process_available_packets(self, max_packets=None):
        return None


class _FakeOldNoPrivate:
    def __init__(self):
        self.calls = []

    def enable_feature(self, fid):
        self.calls.append(fid)


class TestEnableReportInterval(unittest.TestCase):
    def test_passes_interval_in_microseconds(self):
        b = _FakeNew()
        _enable(b, 0x08, "grv", 10.0)
        self.assertEqual(b.calls, [(0x08, 10000)])

    def test_legacy_library_uses_set_feature_command(self):
        b = _FakeOld()
        _enable(b, 0x08, "grv", 10.0)
        self.assertEqual(b.sent, [(2, ("set_feature", 0x08, 10000))])

    def test_legacy_without_private_helpers_falls_back_to_default(self):
        b = _FakeOldNoPrivate()
        _enable(b, 0x08, "grv", 10.0)
        self.assertEqual(b.calls, [0x08])


if __name__ == "__main__":
    unittest.main()

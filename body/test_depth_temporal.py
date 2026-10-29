"""Tests for the depth temporal smoother (bounded hold) and capture-time stamping."""
import datetime
import unittest

import numpy as np

from body.lib.depth_temporal import DepthTemporalSmoother, capture_wall_ts


class TestDepthHoldCap(unittest.TestCase):
    def test_invalid_pixel_held_at_most_one_frame(self):
        sm = DepthTemporalSmoother(hold_max_frames=1)
        sm.apply(np.array([[500.0]]))
        self.assertEqual(sm.apply(np.array([[0.0]]))[0, 0], 500.0)   # held once
        self.assertEqual(sm.apply(np.array([[0.0]]))[0, 0], 0.0)     # then dropped
        self.assertEqual(sm.apply(np.array([[0.0]]))[0, 0], 0.0)     # never resurrected

    def test_hold_zero_disables_hold(self):
        sm = DepthTemporalSmoother(hold_max_frames=0)
        sm.apply(np.array([[500.0]]))
        self.assertEqual(sm.apply(np.array([[0.0]]))[0, 0], 0.0)

    def test_hold_age_resets_after_valid_frame(self):
        sm = DepthTemporalSmoother(hold_max_frames=2)
        sm.apply(np.array([[500.0]]))
        sm.apply(np.array([[0.0]]))
        sm.apply(np.array([[0.0]]))
        self.assertEqual(sm.apply(np.array([[0.0]]))[0, 0], 0.0)     # 2 holds used
        sm.apply(np.array([[700.0]]))
        self.assertEqual(sm.apply(np.array([[0.0]]))[0, 0], 700.0)

    def test_stationary_ghost_clears(self):
        # Chair at 0.5 m, then the pixel goes permanently invalid (dark hallway).
        sm = DepthTemporalSmoother(hold_max_frames=1)
        for _ in range(5):
            sm.apply(np.full((4, 4), 500.0))
        out = None
        for _ in range(10):
            out = sm.apply(np.zeros((4, 4)))
        self.assertTrue(np.all(out == 0.0))

    def test_iir_blend_and_step_gate(self):
        sm = DepthTemporalSmoother(iir_max_delta_mm=200.0)
        sm.apply(np.array([[900.0]]))
        self.assertAlmostEqual(sm.apply(np.array([[1200.0]]))[0, 0], 1200.0)  # 300 mm step: take new
        self.assertAlmostEqual(sm.apply(np.array([[1230.0]]))[0, 0], 1220.0)  # small: blend 1/3:2/3

    def test_shape_change_resets(self):
        sm = DepthTemporalSmoother()
        sm.apply(np.full((2, 2), 500.0))
        out = sm.apply(np.zeros((3, 3)))
        self.assertTrue(np.all(out == 0.0))


class _Frame:
    def __init__(self, mono):
        self._mono = mono

    def getTimestamp(self):
        return datetime.timedelta(seconds=self._mono)


class _NoTs:
    pass


class TestCaptureTs(unittest.TestCase):
    def test_maps_device_ts_to_wall(self):
        ts = capture_wall_ts(_Frame(99.95), arrival_wall=1000.0, arrival_mono=100.0)
        self.assertAlmostEqual(ts, 999.95)

    def test_fallback_when_missing_or_implausible(self):
        self.assertEqual(capture_wall_ts(_NoTs(), 1000.0, 100.0), 1000.0)
        self.assertEqual(capture_wall_ts(_Frame(100.5), 1000.0, 100.0), 1000.0)   # future
        self.assertEqual(capture_wall_ts(_Frame(50.0), 1000.0, 100.0), 1000.0)    # too old


try:  # oakd_driver needs cv2 + depthai; skip the publisher test where absent.
    from body import oakd_driver as _oakd
except Exception:  # pragma: no cover
    _oakd = None


class _DepthFrame(_Frame):
    def __init__(self, mono, value):
        super().__init__(mono)
        self._arr = np.full((40, 60), value, dtype=np.uint16)

    def getHeight(self):
        return self._arr.shape[0]

    def getWidth(self):
        return self._arr.shape[1]

    def getFrame(self):
        return self._arr


class _Queue:
    def __init__(self):
        self.items = []

    def tryGet(self):
        return self.items.pop(0) if self.items else None


@unittest.skipIf(_oakd is None, "cv2/depthai not installed")
class TestDepthPublisher(unittest.TestCase):
    def test_seq_advances_only_on_new_frames_and_ts_is_capture_time(self):
        sent = []
        orig = _oakd.zenoh_helpers.publish_json
        _oakd.zenoh_helpers.publish_json = lambda _s, key, msg: sent.append((key, msg))
        try:
            q = _Queue()
            pub = _oakd._DepthPublisher(None, {"depth_out_width": 12, "depth_out_height": 9},
                                        0.1, q)
            import time as _t
            q.items.append(_DepthFrame(_t.monotonic() - 0.05, 800))
            pub.tick(_t.monotonic())
            pub.tick(_t.monotonic() + 0.2)          # no new frame → nothing published
            q.items.append(_DepthFrame(_t.monotonic() - 0.02, 800))
            pub.tick(_t.monotonic() + 0.3)
        finally:
            _oakd.zenoh_helpers.publish_json = orig
        self.assertEqual([m["seq"] for _, m in sent], [0, 1])
        self.assertTrue(all(k == "body/oakd/depth" for k, _ in sent))
        self.assertLess(sent[0][1]["ts"], _t.time() - 0.04)


if __name__ == "__main__":
    unittest.main()

"""OAK-D driver: wait for the device at startup; from the stub, notice it
appearing (11-03 clock-step crash left the driver on the stub for good)."""
import time
import unittest
from unittest import mock

from body import oakd_driver as od


class TestDeviceWait(unittest.TestCase):
    def test_waits_until_the_device_shows_up(self):
        seq = iter([None, None, "dev"])
        with mock.patch.object(od, "_pick_device", lambda _id: next(seq)), \
             mock.patch.object(od.time, "sleep", lambda _s: None):
            self.assertEqual(od._wait_for_device(None, [False], timeout_s=30.0), "dev")

    def test_gives_up_after_the_timeout(self):
        t = [0.0]

        def sleep(s):
            t[0] += s
        with mock.patch.object(od, "_pick_device", lambda _id: None), \
             mock.patch.object(od.time, "sleep", sleep), \
             mock.patch.object(od.time, "monotonic", lambda: t[0]):
            self.assertIsNone(od._wait_for_device(None, [False], timeout_s=5.0))
        self.assertGreaterEqual(t[0], 5.0)

    def test_enumeration_error_is_retried(self):
        calls = [0]

        def pick(_id):
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("XLink busy")
            return "dev"
        with mock.patch.object(od, "_pick_device", pick), \
             mock.patch.object(od.time, "sleep", lambda _s: None):
            self.assertEqual(od._wait_for_device(None, [False]), "dev")

    def test_probe_stops_the_stub_when_the_device_appears(self):
        stop, appeared = [False], [False]
        seq = iter([None, "dev"])
        with mock.patch.object(od, "STUB_PROBE_S", 0.01), \
             mock.patch.object(od, "_pick_device", lambda _id: next(seq)):
            th = od._start_device_probe(None, stop, appeared)
            th.join(timeout=2.0)
        self.assertTrue(stop[0])
        self.assertTrue(appeared[0])


if __name__ == "__main__":
    unittest.main()

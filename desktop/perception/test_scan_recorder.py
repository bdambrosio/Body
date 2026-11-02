"""ScanRecorder round-trip (patrol-review item 1)."""
import glob
import os
import tempfile
import unittest

import numpy as np

from desktop.perception.scan_recorder import ScanRecorder, load_scans


class TestScanRecorder(unittest.TestCase):
    def test_round_trip_in_chunks(self):
        ang = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        with tempfile.TemporaryDirectory() as d:
            rec = ScanRecorder(d, chunk=3)
            for k in range(7):
                r = np.full(360, 1.0 + k)
                r[5] = np.nan
                rec.add(10.0 + k, r, ang, None if k == 2 else (k, 0.0, 0.1),
                        ok=k % 2 == 0, quality=0.5, coverage=None)
            rec.flush()
            files = sorted(glob.glob(os.path.join(d, "scans_*.npz")))
            self.assertEqual(len(files), 3)
            rows = list(load_scans(files))
        self.assertEqual(len(rows), 7)
        ts, r, a, pose, ok, q, cov = rows[3]
        self.assertEqual(ts, 13.0)
        self.assertEqual(r[0], 4.0)
        self.assertTrue(np.isnan(r[5]))
        np.testing.assert_allclose(a, ang, atol=1e-9)
        self.assertEqual(pose, (3.0, 0.0, 0.1))
        self.assertFalse(ok)
        self.assertEqual(q, 0.5)
        self.assertTrue(np.isnan(cov))
        self.assertIsNone(rows[2][3])

    def test_disabled_without_dir(self):
        rec = ScanRecorder(None)
        rec.add(1.0, np.ones(4), np.arange(4.0), None, True, None, None)
        self.assertIsNone(rec.flush())


if __name__ == "__main__":
    unittest.main()

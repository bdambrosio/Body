"""Tests for the lidar self-occlusion mask (range-gated angular sectors)."""
import math
import unittest

from body.lidar_driver import _parse_self_mask


def _bins(parsed):
    """Union of all bin indices across parsed sectors."""
    out = set()
    for b, _ in parsed:
        out |= b
    return out


class TestParseSelfMask(unittest.TestCase):
    def test_empty_no_mask(self):
        self.assertEqual(_parse_self_mask([], 360), [])
        self.assertEqual(_parse_self_mask(None, 360), [])

    def test_simple_sector_no_range_gate(self):
        parsed = _parse_self_mask([[185, 205]], 360)
        bins, rng = parsed[0]
        self.assertEqual(bins, set(range(185, 206)))
        self.assertEqual(rng, math.inf)            # 2-element → drop whole sector

    def test_range_gated_sector(self):
        parsed = _parse_self_mask([[80, 170, 0.2]], 360)
        bins, rng = parsed[0]
        self.assertEqual(bins, set(range(80, 171)))
        self.assertEqual(rng, 0.2)

    def test_antenna_bearing_covered(self):
        # body=(-0.06, +0.10) → bearing atan2(0.10,-0.06) ≈ 121°.
        bearing = round(math.degrees(math.atan2(0.10, -0.06)))
        parsed = _parse_self_mask([[80, 170, 0.2]], 360)
        self.assertIn(bearing, parsed[0][0])

    def test_wraparound_sector(self):
        bins = _bins(_parse_self_mask([[350, 10]], 360))
        self.assertEqual(bins, set(range(350, 360)) | set(range(0, 11)))

    def test_non_360_binning(self):
        bins = _bins(_parse_self_mask([[180, 181]], 720))
        self.assertEqual(bins, {360, 361, 362})

    def test_malformed_sector_ignored(self):
        self.assertEqual(_bins(_parse_self_mask([[1], "x", [10, 12]], 360)), {10, 11, 12})


if __name__ == "__main__":
    unittest.main()

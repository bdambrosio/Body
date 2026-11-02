"""Per-waypoint speed cap (explore plan A3): schema round-trip and the
drive's leg cap = min(drive-wide sub_v_max, waypoint v_max_mps)."""
import unittest
from types import SimpleNamespace

from desktop.nav.hierarchical_drive import HierarchicalDrive
from desktop.nav.patrol import Waypoint


def _leg_cap(sub_v_max, wp):
    fake = SimpleNamespace(_cfg=SimpleNamespace(sub_v_max=sub_v_max),
                           _runner=SimpleNamespace(current_target=lambda: wp))
    return HierarchicalDrive._leg_v_max(fake)


class TestWaypointSpeed(unittest.TestCase):
    def test_round_trip(self):
        wp = Waypoint(1.0, 2.0, v_max_mps=0.1)
        self.assertEqual(Waypoint.from_dict(wp.to_dict()).v_max_mps, 0.1)

    def test_absent_stays_absent(self):
        d = Waypoint(1.0, 2.0).to_dict()
        self.assertNotIn("v_max_mps", d)
        self.assertIsNone(Waypoint.from_dict(d).v_max_mps)

    def test_leg_cap(self):
        self.assertIsNone(_leg_cap(None, Waypoint(0, 0)))
        self.assertIsNone(_leg_cap(None, None))
        self.assertEqual(_leg_cap(None, Waypoint(0, 0, v_max_mps=0.1)), 0.1)
        self.assertEqual(_leg_cap(0.15, Waypoint(0, 0)), 0.15)
        self.assertEqual(_leg_cap(0.15, Waypoint(0, 0, v_max_mps=0.1)), 0.1)
        self.assertEqual(_leg_cap(0.08, Waypoint(0, 0, v_max_mps=0.1)), 0.08)


if __name__ == "__main__":
    unittest.main()

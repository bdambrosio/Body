"""Tests for scripts/motor_friction_ramp.py against its simulated Coulomb wheel."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import motor_friction_ramp as mfr  # noqa: E402

M_PER_TICK = 2 * math.pi * 0.0325 / 1920


class TestRamp(unittest.TestCase):
    def test_recovers_static_and_kinetic(self):
        sim = mfr.SimRig(M_PER_TICK, {"left": 0.19, "right": 0.22}, {"left": 0.11, "right": 0.13})
        results = [
            mfr.run_ramp(sim, side, "fwd", step=0.005, hold_s=0.15, max_duty=0.45,
                         overshoot=0.1, m_per_tick=M_PER_TICK, sleep=sim.advance)
            for side in ("left", "right")
        ]
        self.assertAlmostEqual(results[0].static_duty, 0.19, delta=0.011)
        self.assertAlmostEqual(results[1].static_duty, 0.22, delta=0.011)
        self.assertAlmostEqual(results[0].kinetic_duty, 0.11, delta=0.011)
        summary = mfr.summarize(results, 0.9, 0.05)
        self.assertAlmostEqual(summary["right"]["coulomb_intercept_median"], 0.13, delta=0.01)
        self.assertGreater(summary["right"]["suggest_min_drive_pwm"], 0.22)

    def test_wheel_that_never_moves(self):
        sim = mfr.SimRig(M_PER_TICK, {"left": 0.9, "right": 0.9}, {"left": 0.5, "right": 0.5})
        r = mfr.run_ramp(sim, "left", "fwd", step=0.01, hold_s=0.1, max_duty=0.3,
                         overshoot=0.1, m_per_tick=M_PER_TICK, sleep=sim.advance)
        self.assertIsNone(r.static_duty)
        self.assertEqual(sim.duty["left"], 0.0)

    def test_fit_line(self):
        k, d0 = mfr.fit_line([(0.2, 0.1), (0.3, 0.2), (0.4, 0.3)])
        self.assertAlmostEqual(k, 1.0)
        self.assertAlmostEqual(d0, 0.1)
        self.assertIsNone(mfr.fit_line([(0.2, 0.1)]))


if __name__ == "__main__":
    unittest.main()

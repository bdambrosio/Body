"""Tests for the Phase 6.4.2 VPR calibration sweep orchestrator.

The motion side is exercised via a fake chassis that records every
set_cmd_vel call. The anchor side is exercised by pre-loading the
shadow driver's estimator with synthetic pairs before the sweep
runs — the sweep then scores and installs (or rejects) the result.
"""
from __future__ import annotations

import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import List, Tuple

from desktop.world_map.vpr.anchor import (
    AnchorOffsetConfig,
    AnchorOffsetEstimator,
    AnchorPair,
    CalibrationScoringConfig,
)
from desktop.world_map.vpr.calibration_sweep import (
    CalibrationSweepConfig,
    VPRCalibrationSweep,
)


# ── Fakes ─────────────────────────────────────────────────────────────


class _FakeChassis:
    def __init__(self):
        self.lock = threading.Lock()
        self.cmd_history: List[Tuple[float, float, float]] = []  # (mono, lin, ang)
        self.live_history: List[Tuple[float, bool]] = []
        self.t0 = time.monotonic()

    def set_cmd_vel(self, linear: float, angular: float) -> None:
        with self.lock:
            self.cmd_history.append((time.monotonic() - self.t0, linear, angular))

    def set_live_command(self, enable: bool) -> None:
        with self.lock:
            self.live_history.append((time.monotonic() - self.t0, enable))


class _FakeDriver:
    """Minimal stand-in for ShadowVPRDriver: holds an
    AnchorOffsetEstimator and a trace list."""

    def __init__(self, *, anchor_pairs: List[AnchorPair] = None):
        self._anchor = AnchorOffsetEstimator(AnchorOffsetConfig(
            min_pairs=99,  # don't auto-calibrate from opportunistic path
            min_spatial_spread_m=0.0,
            max_residual_m=1e9,
        ))
        if anchor_pairs:
            for p in anchor_pairs:
                self._anchor.observe(
                    bank_xy=p.bank_xy, current_xy=p.current_xy,
                    similarity=p.similarity,
                )
        self.events: List[dict] = []

    def log_event(self, record_type: str, payload: dict) -> None:
        self.events.append({"type": record_type, **payload})

    def anchor(self):
        return self._anchor


# ── Helpers ─────────────────────────────────────────────────────────


def _apply_se2(points, dx, dy, dth):
    c, s = math.cos(dth), math.sin(dth)
    return [(c * x - s * y + dx, s * x + c * y + dy) for x, y in points]


def _grid_pairs(n_side: int = 4, dx=0.3, dy=-0.2, dth=math.radians(15),
                noise: float = 0.02):
    import torch
    gen = torch.Generator().manual_seed(7)
    src = [(float(x), float(y)) for x in range(n_side) for y in range(n_side)]
    dst = _apply_se2(src, dx=dx, dy=dy, dth=dth)
    noise_t = torch.randn(len(src), 2, generator=gen) * noise
    dst_n = [(d[0] + float(noise_t[i, 0]), d[1] + float(noise_t[i, 1]))
             for i, d in enumerate(dst)]
    return [AnchorPair(bank_xy=s, current_xy=d, similarity=0.9)
            for s, d in zip(src, dst_n)]


def _fast_sweep_config(scoring=None) -> CalibrationSweepConfig:
    """Tight timings so tests run in <1 second."""
    return CalibrationSweepConfig(
        sweep_total_rad=math.radians(20),  # quick "sweep"
        sweep_speed_rad_s=math.radians(40),
        startup_delay_s=0.05,
        settle_after_motion_s=0.05,
        scoring=scoring or CalibrationScoringConfig(
            min_pairs=4, min_spatial_spread_m=0.5,
            min_unique_bank_cells=3, max_residual_rms_m=0.10,
            max_cov_xy_trace_m2=0.50,
        ),
    )


# ── Tests ───────────────────────────────────────────────────────────


class TestSweepMotion(unittest.TestCase):
    def test_issues_rotation_then_zeros(self):
        ch = _FakeChassis()
        drv = _FakeDriver(anchor_pairs=_grid_pairs())
        sweep = VPRCalibrationSweep(
            chassis=ch, vpr_driver=drv, config=_fast_sweep_config(),
        )
        sweep.start()
        self.assertTrue(sweep._done.wait(timeout=3.0))
        # Live command engaged at sweep start.
        engages = [e for t, e in ch.live_history if e]
        self.assertGreaterEqual(len(engages), 1)
        # cmd_vel was nonzero at some point, then explicitly zeroed.
        nonzero = [a for t, l, a in ch.cmd_history if a != 0.0]
        self.assertGreater(len(nonzero), 0)
        zeros = [a for t, l, a in ch.cmd_history if a == 0.0]
        self.assertGreater(len(zeros), 0)
        # Final command is zero (the stop in finally block).
        self.assertEqual(ch.cmd_history[-1][2], 0.0)

    def test_sweep_calls_log_event_at_each_phase(self):
        ch = _FakeChassis()
        drv = _FakeDriver(anchor_pairs=_grid_pairs())
        sweep = VPRCalibrationSweep(
            chassis=ch, vpr_driver=drv, config=_fast_sweep_config(),
        )
        sweep.start()
        sweep._done.wait(timeout=3.0)
        phases = [e["phase"] for e in drv.events if e["type"] == "vpr_calibration"]
        self.assertIn("starting", phases)
        self.assertIn("motion_start", phases)
        self.assertIn("complete", phases)


class TestSweepScoring(unittest.TestCase):
    def test_pass_installs_calibration(self):
        ch = _FakeChassis()
        drv = _FakeDriver(anchor_pairs=_grid_pairs())
        sweep = VPRCalibrationSweep(
            chassis=ch, vpr_driver=drv, config=_fast_sweep_config(),
        )
        sweep.start()
        sweep._done.wait(timeout=3.0)
        self.assertIsNotNone(sweep.score)
        self.assertTrue(sweep.score.passed, sweep.score.reason)
        self.assertEqual(drv.anchor().state, "calibrated")
        cal = drv.anchor().calibration
        self.assertIsNotNone(cal)
        self.assertAlmostEqual(cal.dx, 0.3, delta=0.05)

    def test_fail_does_not_install_calibration(self):
        ch = _FakeChassis()
        drv = _FakeDriver(anchor_pairs=_grid_pairs(noise=0.5))  # huge noise
        scoring = CalibrationScoringConfig(
            min_pairs=4, min_spatial_spread_m=0.5,
            min_unique_bank_cells=3, max_residual_rms_m=0.02,  # tight
            max_cov_xy_trace_m2=0.05,
        )
        sweep = VPRCalibrationSweep(
            chassis=ch, vpr_driver=drv, config=_fast_sweep_config(scoring),
        )
        sweep.start()
        sweep._done.wait(timeout=3.0)
        self.assertIsNotNone(sweep.score)
        self.assertFalse(sweep.score.passed)
        self.assertEqual(drv.anchor().state, "uncalibrated")

    def test_on_complete_fires_with_score(self):
        ch = _FakeChassis()
        drv = _FakeDriver(anchor_pairs=_grid_pairs())
        seen = []
        sweep = VPRCalibrationSweep(
            chassis=ch, vpr_driver=drv, config=_fast_sweep_config(),
            on_complete=lambda s: seen.append(s),
        )
        sweep.start()
        sweep._done.wait(timeout=3.0)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].passed)


class TestSweepCancel(unittest.TestCase):
    def test_cancel_before_motion_stops_immediately(self):
        ch = _FakeChassis()
        drv = _FakeDriver(anchor_pairs=_grid_pairs())
        cfg = _fast_sweep_config()
        # Long startup delay so we have time to cancel.
        cfg = CalibrationSweepConfig(
            sweep_total_rad=cfg.sweep_total_rad,
            sweep_speed_rad_s=cfg.sweep_speed_rad_s,
            startup_delay_s=2.0,
            settle_after_motion_s=cfg.settle_after_motion_s,
            scoring=cfg.scoring,
        )
        sweep = VPRCalibrationSweep(
            chassis=ch, vpr_driver=drv, config=cfg,
        )
        sweep.start()
        time.sleep(0.05)
        sweep.cancel()
        self.assertTrue(sweep._done.wait(timeout=1.0))
        # No motion issued.
        nonzero = [a for t, l, a in ch.cmd_history if a != 0.0]
        self.assertEqual(len(nonzero), 0)
        self.assertEqual(drv.anchor().state, "uncalibrated")


if __name__ == "__main__":
    unittest.main()

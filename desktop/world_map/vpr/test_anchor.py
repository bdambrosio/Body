"""Tests for AnchorOffsetEstimator + the closed-form SE(2) fit."""
from __future__ import annotations

import math
import unittest

import torch

from desktop.world_map.vpr.anchor import (
    AnchorOffsetConfig,
    AnchorOffsetEstimator,
    AnchorPair,
    CalibrationResult,
    CalibrationScoringConfig,
    _fit_se2,
    _max_pairwise_distance,
    bootstrap_se2_covariance,
    score_calibration,
)


def _apply_se2(
    points, dx: float, dy: float, dtheta: float,
) -> list:
    c, s = math.cos(dtheta), math.sin(dtheta)
    return [(c * x - s * y + dx, s * x + c * y + dy) for x, y in points]


class TestFitSE2(unittest.TestCase):
    def test_perfect_translation(self):
        src = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        dst = _apply_se2(src, dx=0.3, dy=-0.7, dtheta=0.0)
        tx, ty, dth, rms = _fit_se2(src, dst)
        self.assertAlmostEqual(tx, 0.3, places=6)
        self.assertAlmostEqual(ty, -0.7, places=6)
        self.assertAlmostEqual(dth, 0.0, places=6)
        self.assertLess(rms, 1e-9)

    def test_perfect_rotation_and_translation(self):
        src = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)]
        dst = _apply_se2(src, dx=2.0, dy=-1.0, dtheta=math.radians(35.0))
        tx, ty, dth, rms = _fit_se2(src, dst)
        self.assertAlmostEqual(tx, 2.0, places=6)
        self.assertAlmostEqual(ty, -1.0, places=6)
        self.assertAlmostEqual(dth, math.radians(35.0), places=6)
        self.assertLess(rms, 1e-9)

    def test_noisy_fit_returns_residual(self):
        torch.manual_seed(0)
        src = [(float(x), float(y)) for x in range(4) for y in range(4)]
        dst = _apply_se2(src, dx=0.5, dy=0.25, dtheta=math.radians(10.0))
        # Add 5 cm Gaussian noise to dst.
        gen = torch.Generator().manual_seed(1)
        noise = torch.randn(len(dst), 2, generator=gen) * 0.05
        dst_noisy = [(d[0] + float(noise[i, 0]), d[1] + float(noise[i, 1]))
                     for i, d in enumerate(dst)]
        tx, ty, dth, rms = _fit_se2(src, dst_noisy)
        self.assertAlmostEqual(tx, 0.5, delta=0.05)
        self.assertAlmostEqual(ty, 0.25, delta=0.05)
        self.assertAlmostEqual(dth, math.radians(10.0), delta=math.radians(3.0))
        self.assertGreater(rms, 0.01)  # noise visible in residual
        self.assertLess(rms, 0.1)

    def test_rejects_too_few_points(self):
        with self.assertRaises(ValueError):
            _fit_se2([(0.0, 0.0)], [(1.0, 1.0)])


class TestMaxPairwiseDistance(unittest.TestCase):
    def test_zero_for_single_point(self):
        self.assertEqual(_max_pairwise_distance([(1.0, 2.0)]), 0.0)

    def test_picks_farthest_pair(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (3.0, 4.0)]  # max = (0,0)-(3,4) = 5
        self.assertAlmostEqual(_max_pairwise_distance(pts), 5.0, places=6)


class TestEstimatorAccumulation(unittest.TestCase):
    def _cfg(self, **kw):
        return AnchorOffsetConfig(
            min_similarity=kw.pop("min_similarity", 0.85),
            min_pairs=kw.pop("min_pairs", 3),
            min_spatial_spread_m=kw.pop("min_spatial_spread_m", 0.5),
            max_residual_m=kw.pop("max_residual_m", 0.25),
        )

    def test_low_similarity_pairs_dropped(self):
        est = AnchorOffsetEstimator(self._cfg(min_similarity=0.9))
        for i in range(5):
            est.observe(bank_xy=(i, 0), current_xy=(i, 1), similarity=0.5)
        self.assertEqual(est.n_pairs_collected, 0)

    def test_needs_min_pairs_before_fit(self):
        est = AnchorOffsetEstimator(self._cfg(min_pairs=4))
        for i in range(3):
            est.observe(bank_xy=(float(i), 0.0), current_xy=(float(i)+1, 1.0),
                        similarity=0.9)
        self.assertIsNone(est.calibrate_if_ready())
        est.observe(bank_xy=(3.0, 0.0), current_xy=(4.0, 1.0), similarity=0.9)
        # Now we have 4 pairs spanning 3 m — fit should succeed.
        result = est.calibrate_if_ready()
        self.assertIsNotNone(result)
        self.assertEqual(result.n_pairs, 4)
        self.assertAlmostEqual(result.dx, 1.0, places=5)
        self.assertAlmostEqual(result.dy, 1.0, places=5)

    def test_defers_when_spatial_spread_insufficient(self):
        # All bank poses at the same point → no spatial diversity → defer.
        est = AnchorOffsetEstimator(self._cfg(min_pairs=3, min_spatial_spread_m=0.5))
        for i in range(5):
            est.observe(bank_xy=(0.0, 0.0), current_xy=(0.5, 0.25),
                        similarity=0.9)
        self.assertIsNone(est.calibrate_if_ready())
        self.assertEqual(est.state, AnchorOffsetEstimator.UNCALIBRATED)

    def test_calibration_is_idempotent(self):
        est = AnchorOffsetEstimator(self._cfg(min_pairs=3))
        for i in range(3):
            est.observe(bank_xy=(float(i), 0.0), current_xy=(float(i)+0.5, 0.5),
                        similarity=0.9)
        r1 = est.calibrate_if_ready()
        r2 = est.calibrate_if_ready()
        self.assertIs(r1, r2)
        # New observations after calibration are no-ops.
        est.observe(bank_xy=(100.0, 100.0), current_xy=(0.0, 0.0), similarity=1.0)
        self.assertEqual(est.calibration.n_pairs, 3)

    def test_apply_xy_transforms_correctly(self):
        est = AnchorOffsetEstimator(self._cfg(min_pairs=3))
        # Construct a known transformation: rotate 90° + translate (+1, +2).
        src = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        dst = _apply_se2(src, dx=1.0, dy=2.0, dtheta=math.radians(90.0))
        for s_p, d_p in zip(src, dst):
            est.observe(bank_xy=s_p, current_xy=d_p, similarity=0.9)
        est.calibrate_if_ready()
        # Now apply to bank pose (1, 0) → expected (1, 2+1) = (1, 3).
        xy = torch.tensor([[1.0, 0.0]])
        out = est.apply_xy(xy)
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(out[0, 1]), 3.0, places=5)

    def test_apply_xy_raises_when_uncalibrated(self):
        est = AnchorOffsetEstimator(self._cfg())
        with self.assertRaises(RuntimeError):
            est.apply_xy(torch.zeros(1, 2))

    def test_rejects_fit_when_residual_too_large(self):
        # Wildly inconsistent pairs → high residual → reject.
        est = AnchorOffsetEstimator(
            self._cfg(min_pairs=4, max_residual_m=0.05, min_spatial_spread_m=0.1),
        )
        est.observe(bank_xy=(0.0, 0.0),  current_xy=(0.0, 0.0),   similarity=0.9)
        est.observe(bank_xy=(1.0, 0.0),  current_xy=(1.0, 0.0),   similarity=0.9)
        est.observe(bank_xy=(0.0, 1.0),  current_xy=(0.0, 1.0),   similarity=0.9)
        est.observe(bank_xy=(1.0, 1.0),  current_xy=(5.0, -5.0),  similarity=0.9)
        self.assertIsNone(est.calibrate_if_ready())


class TestBootstrapCovariance(unittest.TestCase):
    @staticmethod
    def _pairs_with_noise(n: int, dx: float, dy: float, dth: float,
                          noise_m: float, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        src = [(float(x), float(y)) for x in range(int(n ** 0.5) + 1)
               for y in range(int(n ** 0.5) + 1)][:n]
        dst_clean = _apply_se2(src, dx=dx, dy=dy, dtheta=dth)
        noise = torch.randn(n, 2, generator=gen) * noise_m
        dst = [(d[0] + float(noise[i, 0]), d[1] + float(noise[i, 1]))
               for i, d in enumerate(dst_clean)]
        return [AnchorPair(bank_xy=s, current_xy=d, similarity=0.9)
                for s, d in zip(src, dst)]

    def test_returns_none_when_too_few_pairs(self):
        pairs = self._pairs_with_noise(2, 0, 0, 0, 0.01)
        self.assertIsNone(bootstrap_se2_covariance(pairs))

    def test_covariance_shrinks_with_more_pairs(self):
        # More pairs → tighter offset covariance (CLT-like behavior).
        p_few = self._pairs_with_noise(5, 0.5, -0.3, math.radians(10), 0.05, seed=1)
        p_many = self._pairs_with_noise(50, 0.5, -0.3, math.radians(10), 0.05, seed=1)
        cov_few = bootstrap_se2_covariance(p_few, n_resamples=200, seed=42)
        cov_many = bootstrap_se2_covariance(p_many, n_resamples=200, seed=42)
        self.assertEqual(cov_few.shape, (3, 3))
        self.assertEqual(cov_many.shape, (3, 3))
        self.assertGreater(cov_few[0, 0], cov_many[0, 0])
        self.assertGreater(cov_few[1, 1], cov_many[1, 1])

    def test_covariance_scales_with_noise(self):
        p_low = self._pairs_with_noise(20, 0.0, 0.0, 0.0, 0.01, seed=2)
        p_hi = self._pairs_with_noise(20, 0.0, 0.0, 0.0, 0.10, seed=2)
        cov_low = bootstrap_se2_covariance(p_low, n_resamples=100, seed=7)
        cov_hi = bootstrap_se2_covariance(p_hi, n_resamples=100, seed=7)
        self.assertGreater(cov_hi[0, 0], cov_low[0, 0])


class TestScoreCalibration(unittest.TestCase):
    def _good_pairs(self, n=10, noise=0.02, seed=11):
        gen = torch.Generator().manual_seed(seed)
        src = [(float(x), float(y)) for x in range(int(n**0.5)+1)
               for y in range(int(n**0.5)+1)][:n]
        dst = _apply_se2(src, dx=0.3, dy=-0.2, dtheta=math.radians(15))
        noise_t = torch.randn(n, 2, generator=gen) * noise
        dst_noisy = [(d[0] + float(noise_t[i, 0]), d[1] + float(noise_t[i, 1]))
                     for i, d in enumerate(dst)]
        return [AnchorPair(bank_xy=s, current_xy=d, similarity=0.9)
                for s, d in zip(src, dst_noisy)]

    def test_passes_on_clean_data(self):
        pairs = self._good_pairs(n=16, noise=0.02)
        score = score_calibration(pairs, CalibrationScoringConfig(
            min_pairs=5, min_unique_bank_cells=3,
            min_spatial_spread_m=0.5, max_residual_rms_m=0.10,
            max_cov_xy_trace_m2=0.10,
        ))
        self.assertTrue(score.passed, score.reason)
        self.assertEqual(score.reason, "passed")
        self.assertIsNotNone(score.offset)
        self.assertAlmostEqual(score.offset.dx, 0.3, delta=0.05)
        self.assertAlmostEqual(score.offset.dy, -0.2, delta=0.05)

    def test_fails_on_too_few_pairs(self):
        pairs = self._good_pairs(n=3)
        score = score_calibration(pairs, CalibrationScoringConfig(min_pairs=5))
        self.assertFalse(score.passed)
        self.assertEqual(score.reason, "too_few_pairs")

    def test_fails_on_insufficient_spatial_spread(self):
        # All pairs at the same physical bank location.
        pairs = [
            AnchorPair(bank_xy=(0.1, 0.2), current_xy=(0.5 * i, 0.0),
                       similarity=0.9)
            for i in range(8)
        ]
        score = score_calibration(pairs, CalibrationScoringConfig(
            min_pairs=5, min_spatial_spread_m=0.5,
        ))
        self.assertFalse(score.passed)
        self.assertEqual(score.reason, "insufficient_spatial_spread")

    def test_fails_on_residual_too_large(self):
        # Big random current_xy noise → residual will be high.
        gen = torch.Generator().manual_seed(99)
        src = [(float(x), float(y)) for x in range(4) for y in range(4)]
        noisy = torch.randn(len(src), 2, generator=gen) * 0.5
        dst = [(float(noisy[i, 0]), float(noisy[i, 1])) for i in range(len(src))]
        pairs = [AnchorPair(bank_xy=s, current_xy=d, similarity=0.9)
                 for s, d in zip(src, dst)]
        score = score_calibration(pairs, CalibrationScoringConfig(
            min_pairs=5, min_spatial_spread_m=0.5, max_residual_rms_m=0.05,
            max_cov_xy_trace_m2=1e9,  # don't fail this gate
        ))
        self.assertFalse(score.passed)
        self.assertEqual(score.reason, "residual_too_large")

    def test_fails_on_too_few_unique_cells(self):
        # Many pairs but most at the same coarse cell — sweep-in-one-spot
        # case where matches all point at one physical location.
        pairs = ([
            AnchorPair(bank_xy=(0.0, 0.0), current_xy=(0.5 * i, 0.0),
                       similarity=0.9) for i in range(8)
        ] + [
            AnchorPair(bank_xy=(2.0, 2.0), current_xy=(2.5, 2.0),
                       similarity=0.9),
        ])
        score = score_calibration(pairs, CalibrationScoringConfig(
            min_pairs=5, min_spatial_spread_m=0.5,
            min_unique_bank_cells=3, bank_cell_size_m=0.10,
        ))
        self.assertFalse(score.passed)
        self.assertEqual(score.reason, "too_few_unique_bank_cells")


class TestSetCalibration(unittest.TestCase):
    def test_set_calibration_installs_result(self):
        est = AnchorOffsetEstimator(AnchorOffsetConfig(min_pairs=99))
        # Externally compute a calibration and inject it.
        result = CalibrationResult(
            dx=1.0, dy=2.0, dtheta_rad=math.radians(30.0),
            n_pairs=10, residual_rms_m=0.05,
        )
        est.set_calibration(result)
        self.assertEqual(est.state, "calibrated")
        self.assertIs(est.calibration, result)

    def test_set_calibration_is_no_op_when_already_calibrated(self):
        est = AnchorOffsetEstimator(AnchorOffsetConfig(min_pairs=99))
        r1 = CalibrationResult(dx=0, dy=0, dtheta_rad=0,
                               n_pairs=5, residual_rms_m=0.01)
        r2 = CalibrationResult(dx=99, dy=99, dtheta_rad=1.5,
                               n_pairs=5, residual_rms_m=0.99)
        est.set_calibration(r1)
        est.set_calibration(r2)
        self.assertIs(est.calibration, r1)


if __name__ == "__main__":
    unittest.main()

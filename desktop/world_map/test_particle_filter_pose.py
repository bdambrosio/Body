"""Unit tests for ParticleFilterPose (Phase 2.1).

Coverage:
    - seed_at: cloud sits near the seed pose, weights uniform.
    - predict: per-particle slip variance grows as predicted by the
      Phase 0 noise model (sqrt-N random-walk for repeated steps).
    - predict: zero-motion ticks don't freeze the cloud (sigma floors).
    - observe_imu_yaw: tightens the yaw marginal, leaves xy alone.
    - posterior_mean: returns the seed pose within a few cm when
      observations agree.
    - n_eff: full N at uniform weights, drops on a sharp observation.

Run:
    PYTHONPATH=. python3 -m unittest desktop.world_map.test_particle_filter_pose -v
"""
from __future__ import annotations

import math
import unittest

import torch

import numpy as np

from desktop.nav.slam.scan_matcher import likelihood_at
from desktop.nav.slam.types import Pose2D, ScoreField

from .particle_filter_pose import (
    FilterDiagnostics,
    ParticleFilterConfig,
    ParticleFilterPose,
    _wrap_torch,
    interp_score_field,
)


class TestSeedAndState(unittest.TestCase):
    def test_seed_at_places_cloud_around_seed(self):
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_xy_m=0.05,
            init_sigma_theta_rad=math.radians(2.0),
            seed=42,
        ))
        pf.seed_at(1.5, -0.7, math.radians(30.0))
        s = pf.state
        self.assertEqual(s.shape, (2000, 3))
        # Mean within a few standard errors of the seed.
        self.assertAlmostEqual(float(s[:, 0].mean()), 1.5, places=2)
        self.assertAlmostEqual(float(s[:, 1].mean()), -0.7, places=2)
        # θ wraps near +30°.
        self.assertAlmostEqual(
            float(s[:, 2].mean()), math.radians(30.0), places=2,
        )

    def test_seed_at_uniform_weights(self):
        pf = ParticleFilterPose(ParticleFilterConfig(n_particles=500, seed=1))
        pf.seed_at(0.0, 0.0, 0.0)
        w = pf.normalized_weights()
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        # All equal at start.
        self.assertLess(float(w.std()), 1e-9)
        self.assertAlmostEqual(pf.n_eff(), 500.0, places=2)


class TestPredictNoiseGrowth(unittest.TestCase):
    """Validate that the motion-model variance matches Phase 0 priors."""

    def test_pure_rotation_theta_variance_grows_as_random_walk(self):
        # α_4 = 0.01 means σ_rot per step = 0.01 · |Δθ|. Over N
        # independent steps the cloud's θ-variance grows as N · σ².
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=4000,
            init_sigma_theta_rad=0.0,
            init_sigma_xy_m=0.0,
            sigma_floor_rot_rad=0.0,  # measure α_4 cleanly
            sigma_floor_trans_m=0.0,
            seed=7,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        delta = math.radians(5.0)
        n_steps = 100
        for _ in range(n_steps):
            pf.predict(0.0, delta)

        # Per-step rotation σ from priors:
        sigma_per_step = ParticleFilterConfig().alpha_rot_per_rad * abs(delta)
        expected_std = sigma_per_step * math.sqrt(n_steps)

        # Particle θ should be tightly distributed around N·Δθ.
        target = n_steps * delta
        deviations = _wrap_torch(pf.state[:, 2] - target)
        measured_std = float(deviations.std())
        # 4000 particles → standard error on σ ≈ σ/√(2(N-1)) ≈ 1.1%.
        # Allow ±10% to keep the test robust to RNG seed and to small
        # nonlinearities introduced by the midpoint heading integration.
        ratio = measured_std / expected_std
        self.assertGreater(ratio, 0.85, f"σ too tight: {ratio:.3f}")
        self.assertLess(ratio, 1.15, f"σ too loose: {ratio:.3f}")

    def test_pure_translation_xy_variance_grows_as_random_walk(self):
        # α_1 = 0.04 → translation σ = 0.04 · |Δs| per step. The
        # particle's translation step has σ along its current heading
        # (≈ 0 for this trajectory) — so it shows up in x.
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=4000,
            init_sigma_xy_m=0.0,
            init_sigma_theta_rad=0.0,
            sigma_floor_trans_m=0.0,
            sigma_floor_rot_rad=0.0,
            seed=11,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        delta_s = 0.10  # 10 cm / step
        n_steps = 60
        for _ in range(n_steps):
            pf.predict(delta_s, 0.0)

        sigma_per_step = ParticleFilterConfig().alpha_trans_per_m * delta_s
        expected_std = sigma_per_step * math.sqrt(n_steps)
        target_x = n_steps * delta_s
        deviations_x = pf.state[:, 0] - target_x
        measured_std = float(deviations_x.std())
        ratio = measured_std / expected_std
        self.assertGreater(ratio, 0.85, f"σ too tight: {ratio:.3f}")
        self.assertLess(ratio, 1.15, f"σ too loose: {ratio:.3f}")
        # And y stays put — translation is along heading=0, so the y
        # marginal stays at 0 modulo the α_3 cross-term spread on θ
        # that then deflects x→y over many steps. Modest cap.
        self.assertLess(float(pf.state[:, 1].std()), 0.05)

    def test_zero_motion_floor_keeps_cloud_alive(self):
        # Idle tick — Δs = Δθ = 0. Without the σ floor the cloud would
        # freeze. With it, particles drift a tiny amount; verify that.
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=500,
            init_sigma_xy_m=0.0,
            init_sigma_theta_rad=0.0,
            seed=13,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        s0 = pf.state.clone()
        for _ in range(50):
            pf.predict(0.0, 0.0)
        # Cloud should have moved by a few mm at most — small, but not
        # zero. This is the diversity-preservation safeguard.
        d = (pf.state - s0).norm(dim=1)
        self.assertGreater(float(d.max()), 1e-4)
        # And not exploded.
        self.assertLess(float(d.max()), 0.05)


class TestObserveImuYaw(unittest.TestCase):
    def test_observation_concentrates_theta_marginal(self):
        # Seed with broad θ spread; observe yaw = 0 with tight σ; the
        # weighted θ marginal should peak near 0.
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_xy_m=0.001,
            init_sigma_theta_rad=math.radians(15.0),
            imu_sigma_rad=math.radians(1.0),
            seed=23,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        n_eff_before = pf.n_eff()
        pf.observe_imu_yaw(world_yaw=0.0)
        n_eff_after = pf.n_eff()
        self.assertLess(n_eff_after, n_eff_before, "N_eff should drop")
        # Weighted θ mean should be near 0.
        _, _, theta_post = pf.posterior_mean()
        self.assertLess(abs(theta_post), math.radians(1.0))

    def test_observation_leaves_xy_marginal_alone(self):
        # Particle xy and θ are independent at seed time, so an
        # observation on θ should not bias the weighted xy mean. Use a
        # wider observation σ here so N_eff doesn't crash — the test
        # validates the no-bias property, not the convergence rate.
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=4000,
            init_sigma_xy_m=0.10,
            init_sigma_theta_rad=math.radians(15.0),
            imu_sigma_rad=math.radians(5.0),
            seed=29,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        pf.observe_imu_yaw(world_yaw=math.radians(5.0))
        x, y, _ = pf.posterior_mean()
        self.assertLess(abs(x), 0.01)
        self.assertLess(abs(y), 0.01)


class TestPosteriorMode(unittest.TestCase):
    def test_mode_agrees_with_mean_on_uniform_weights(self):
        # Uniform weights → argmax picks the first particle (or
        # whichever index ties to the max). For a tight Gaussian cloud,
        # any single particle is within ~σ of the mean.
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_xy_m=0.02,
            init_sigma_theta_rad=math.radians(1.0),
            seed=600,
        ))
        pf.seed_at(0.5, -0.3, 0.0)
        mean = pf.posterior_mean()
        mode = pf.posterior_mode()
        # Within 3σ for each coordinate.
        self.assertAlmostEqual(mean[0], mode[0], delta=0.06)
        self.assertAlmostEqual(mean[1], mode[1], delta=0.06)
        self.assertAlmostEqual(mean[2], mode[2], delta=math.radians(3.0))

    def test_mode_tracks_peak_after_sharp_observation(self):
        # Sharp IMU observation → one particle (near the obs) dominates.
        # Mode should sit at that particle, very close to the obs value.
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_xy_m=0.001,
            init_sigma_theta_rad=math.radians(5.0),
            imu_sigma_rad=math.radians(0.3),
            seed=601,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        pf.observe_imu_yaw(math.radians(2.0))
        _, _, theta = pf.posterior_mode()
        self.assertAlmostEqual(theta, math.radians(2.0), delta=math.radians(0.7))


class TestPosteriorMean(unittest.TestCase):
    def test_posterior_mean_uses_circular_mean_for_theta(self):
        # Seed cloud straddling +π/-π. Naive linear mean would land
        # near 0; correct circular mean stays near π.
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_xy_m=0.0,
            init_sigma_theta_rad=math.radians(3.0),
            seed=37,
        ))
        pf.seed_at(0.0, 0.0, math.pi)
        _, _, theta = pf.posterior_mean()
        # Wrap to [-π, π], then |θ| close to π.
        self.assertGreater(abs(theta), math.radians(178.0))


def _gaussian_score_field(
    peak_dx: float = 0.06,
    peak_dy: float = -0.04,
    peak_dth: float = math.radians(2.0),
    sigma_xy: float = 0.05,
    sigma_th: float = math.radians(3.0),
    amplitude: float = 1000.0,
) -> ScoreField:
    """Synthetic peaky Gaussian score field for filter tests."""
    dx_axis = np.linspace(-0.30, 0.30, 31).astype(np.float64)
    dy_axis = np.linspace(-0.30, 0.30, 31).astype(np.float64)
    dth_axis = np.linspace(
        math.radians(-15.0), math.radians(15.0), 31,
    ).astype(np.float64)
    DX, DY, DTH = np.meshgrid(dx_axis, dy_axis, dth_axis, indexing="ij")
    z2 = (
        ((DX - peak_dx) / sigma_xy) ** 2
        + ((DY - peak_dy) / sigma_xy) ** 2
        + ((DTH - peak_dth) / sigma_th) ** 2
    )
    field = (amplitude * np.exp(-0.5 * z2)).astype(np.float32)
    return ScoreField(field=field, dx_axis=dx_axis, dy_axis=dy_axis, dth_axis=dth_axis)


class TestInterpScoreField(unittest.TestCase):
    """Validate the vectorized trilinear interp matches scalar likelihood_at."""

    def test_lattice_points_match_scalar(self):
        sf = _gaussian_score_field()
        # Sample a handful of lattice points; vectorized eval must match
        # the scalar likelihood_at at every one.
        ixs = [0, 5, 15, 20, 30]
        iys = [0, 7, 15, 25, 30]
        iths = [0, 5, 15, 22, 30]
        dx = torch.tensor(sf.dx_axis[ixs])
        dy = torch.tensor(sf.dy_axis[iys])
        dth = torch.tensor(sf.dth_axis[iths])
        out = interp_score_field(sf, dx, dy, dth)
        for k, (i, j, m) in enumerate(zip(ixs, iys, iths)):
            scalar = likelihood_at(
                float(sf.dx_axis[i]), float(sf.dy_axis[j]), float(sf.dth_axis[m]), sf,
            )
            self.assertAlmostEqual(float(out[k]), scalar, places=4)

    def test_oob_returns_zero(self):
        sf = _gaussian_score_field()
        dx = torch.tensor([1.0, 0.0, 0.0])
        dy = torch.tensor([0.0, -1.0, 0.0])
        dth = torch.tensor([0.0, 0.0, math.radians(45.0)])
        out = interp_score_field(sf, dx, dy, dth)
        for v in out.tolist():
            self.assertEqual(v, 0.0)

    def test_midpoint_interpolation(self):
        sf = _gaussian_score_field()
        # Midpoint between two cells on the dx axis. Result should be
        # the linear average of the two cell values (within rounding).
        ix0, ix1 = 14, 15
        iy = 15
        ith = 15
        v0 = float(sf.field[ix0, iy, ith])
        v1 = float(sf.field[ix1, iy, ith])
        mid_dx = 0.5 * (sf.dx_axis[ix0] + sf.dx_axis[ix1])
        out = interp_score_field(
            sf,
            torch.tensor([mid_dx]),
            torch.tensor([float(sf.dy_axis[iy])]),
            torch.tensor([float(sf.dth_axis[ith])]),
        )
        self.assertAlmostEqual(float(out[0]), 0.5 * (v0 + v1), places=4)


class TestScanLikelihoodUpdate(unittest.TestCase):
    """End-to-end: scan-likelihood observation pulls particles toward the
    field's peak and tightens N_eff."""

    def test_observation_shifts_posterior_toward_peak(self):
        # Field peaks at (+0.06, -0.04, +2°). Prior pose is at origin
        # in world frame. The expected posterior pose ≈ prior + peak.
        sf = _gaussian_score_field(
            peak_dx=0.06, peak_dy=-0.04, peak_dth=math.radians(2.0),
            sigma_xy=0.04, sigma_th=math.radians(2.5),
        )
        prior = Pose2D(x=0.0, y=0.0, theta=0.0)
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=4000,
            # Spread wide enough that some particles hit the peak.
            init_sigma_xy_m=0.08,
            init_sigma_theta_rad=math.radians(4.0),
            seed=101,
        ))
        pf.seed_at(prior.x, prior.y, prior.theta)
        pf.update_from_scan_likelihood(sf, prior)

        x, y, th = pf.posterior_mean()
        # Posterior mean shifts toward peak from origin. The default
        # auto-temperature (log_ratio=5) caps the per-particle weight
        # ratio at exp(5)≈148, so a single observation pulls the mean
        # most of the way to peak but not all the way — that's the
        # whole point of the softening (cloud retains diversity for the
        # *next* observation). Tolerance widened accordingly.
        self.assertAlmostEqual(x, 0.06, delta=0.035)
        self.assertAlmostEqual(y, -0.04, delta=0.035)
        self.assertAlmostEqual(th, math.radians(2.0), delta=math.radians(1.5))
        # And the shift is in the right direction.
        self.assertGreater(x, 0.02)
        self.assertLess(y, -0.01)

    def test_flat_field_does_not_reweight(self):
        # Zero-variance field → temperature floor kicks in (max(std, 1))
        # so every particle gets the same log_lik = 0 → no reweight.
        flat = ScoreField(
            field=np.zeros((15, 15, 17), dtype=np.float32),
            dx_axis=np.linspace(-0.28, 0.28, 15).astype(np.float64),
            dy_axis=np.linspace(-0.28, 0.28, 15).astype(np.float64),
            dth_axis=np.linspace(
                math.radians(-8.0), math.radians(8.0), 17
            ).astype(np.float64),
        )
        prior = Pose2D(x=0.0, y=0.0, theta=0.0)
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=1000, init_sigma_xy_m=0.05, seed=42,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        n_eff_before = pf.n_eff()
        pf.update_from_scan_likelihood(flat, prior)
        n_eff_after = pf.n_eff()
        # Within sampling noise. N_eff for uniform weights is exactly N.
        self.assertAlmostEqual(n_eff_after, n_eff_before, places=3)

    def test_observation_drops_n_eff(self):
        # Peaky field over a wide-but-overlapping cloud: N_eff must drop
        # substantially while leaving enough survivors to keep filtering.
        # If the cloud is much wider than the field's peak σ, N_eff
        # legitimately collapses toward 1 — that's a posterior with
        # very little support, and the right response is "resample"
        # (Phase 2.3) rather than refusing to reweight.
        sf = _gaussian_score_field(
            peak_dx=0.0, peak_dy=0.0, peak_dth=0.0,
            sigma_xy=0.03, sigma_th=math.radians(2.0),
            amplitude=2000.0,
        )
        prior = Pose2D(x=0.0, y=0.0, theta=0.0)
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_xy_m=0.06,        # 2× field σ — meaningful overlap
            init_sigma_theta_rad=math.radians(4.0),
            seed=99,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        n0 = pf.n_eff()
        pf.update_from_scan_likelihood(sf, prior)
        n1 = pf.n_eff()
        self.assertLess(n1, n0 * 0.5)
        # Survivors are well above 1: a reasonable cloud-vs-peak overlap
        # keeps tens to hundreds of effective particles after one obs.
        self.assertGreater(n1, 20.0)

    def test_oob_particles_contribute_zero(self):
        # Particles whose delta-from-prior exceeds the field's window
        # should get log_lik += 0, so their relative weight is unchanged.
        sf = _gaussian_score_field()  # window ±0.30 m, ±15°
        prior = Pose2D(x=0.0, y=0.0, theta=0.0)
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=200, init_sigma_xy_m=0.0,
            init_sigma_theta_rad=0.0, seed=5,
        ))
        # Plant the seed pose well outside the field window.
        pf.seed_at(2.0, 2.0, math.radians(60.0))
        log_w_before = pf.log_weights.clone()
        pf.update_from_scan_likelihood(sf, prior)
        # OOB contributes 0 to scores; auto-temperature divides by std
        # of an all-zero score array → falls back to floor 1 → 0/1 = 0.
        # Net effect: log-weights unchanged.
        self.assertTrue(torch.allclose(pf.log_weights, log_w_before))


class TestResampling(unittest.TestCase):
    def test_uniform_resample_is_near_identity_in_distribution(self):
        # With uniform weights, systematic resampling is essentially a
        # permutation — each particle gets one expected copy. The
        # weighted mean should not move (within sampling noise).
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_xy_m=0.10,
            init_sigma_theta_rad=math.radians(5.0),
            seed=200,
        ))
        pf.seed_at(0.5, -0.3, math.radians(15.0))
        mx0, my0, mth0 = pf.posterior_mean()
        pf.resample()
        mx1, my1, mth1 = pf.posterior_mean()
        self.assertAlmostEqual(mx1, mx0, places=3)
        self.assertAlmostEqual(my1, my0, places=3)
        self.assertAlmostEqual(mth1, mth0, places=3)

    def test_resample_resets_log_weights_to_uniform(self):
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=500, init_sigma_xy_m=0.05,
            init_sigma_theta_rad=math.radians(3.0), seed=201,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        # Bias weights so resample has work to do.
        pf.observe_imu_yaw(world_yaw=math.radians(0.0), sigma_rad=math.radians(1.0))
        self.assertLess(pf.n_eff(), 500.0)
        pf.resample()
        self.assertAlmostEqual(pf.n_eff(), 500.0, places=2)
        # All log-weights equal -log(N).
        self.assertTrue(
            torch.allclose(
                pf.log_weights,
                torch.full_like(pf.log_weights, -math.log(500)),
                atol=1e-5,
            )
        )

    def test_resample_concentrates_around_observation_peak(self):
        # Drive the cloud with a sharp observation at +5°, then resample.
        # Post-resample particles should cluster around +5°.
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_xy_m=0.001,
            init_sigma_theta_rad=math.radians(10.0),
            imu_sigma_rad=math.radians(0.5),
            seed=202,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        pf.observe_imu_yaw(world_yaw=math.radians(5.0))
        pf.resample()
        # Post-resample θ std should be much smaller than the seed σ.
        std_th = float(pf.state[:, 2].std())
        self.assertLess(std_th, math.radians(2.0))
        # And the mean θ near +5°.
        _, _, theta = pf.posterior_mean()
        self.assertAlmostEqual(theta, math.radians(5.0), delta=math.radians(0.5))

    def test_maybe_resample_skips_when_n_eff_high(self):
        # Uniform-ish weights → no resample fires.
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=1000, init_sigma_xy_m=0.02, seed=203,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        fired = pf.maybe_resample()
        self.assertFalse(fired)

    def test_maybe_resample_fires_when_n_eff_low(self):
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=1000,
            init_sigma_theta_rad=math.radians(10.0),
            imu_sigma_rad=math.radians(0.5),
            seed=204,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        pf.observe_imu_yaw(world_yaw=0.0)
        # That sharp observation pushes N_eff < N/2.
        self.assertLess(pf.n_eff(), 500.0)
        fired = pf.maybe_resample()
        self.assertTrue(fired)


class TestPosteriorCov(unittest.TestCase):
    def test_cov_matches_init_sigma(self):
        # Uniform weights, isotropic Gaussian seed cloud → diagonal cov
        # with σ_x = σ_y = init_sigma, σ_θ = init_sigma_theta.
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=5000,
            init_sigma_xy_m=0.08,
            init_sigma_theta_rad=math.radians(3.0),
            seed=300,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        cov = pf.posterior_cov()
        # Standard error on σ² estimate at N=5000 ≈ σ²·sqrt(2/N) ≈ 2%.
        self.assertAlmostEqual(float(cov[0, 0].sqrt()), 0.08, delta=0.005)
        self.assertAlmostEqual(float(cov[1, 1].sqrt()), 0.08, delta=0.005)
        self.assertAlmostEqual(
            float(cov[2, 2].sqrt()), math.radians(3.0),
            delta=math.radians(0.3),
        )
        # Off-diagonals near zero — seed components are independent.
        self.assertLess(abs(float(cov[0, 1])), 5e-4)
        self.assertLess(abs(float(cov[0, 2])), 5e-4)
        self.assertLess(abs(float(cov[1, 2])), 5e-4)

    def test_cov_shrinks_after_observation(self):
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_theta_rad=math.radians(8.0),
            imu_sigma_rad=math.radians(1.0),
            seed=301,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        std0 = float(pf.posterior_cov()[2, 2].sqrt())
        pf.observe_imu_yaw(world_yaw=0.0)
        std1 = float(pf.posterior_cov()[2, 2].sqrt())
        self.assertLess(std1, std0 * 0.5)


class TestDiagnostics(unittest.TestCase):
    def test_diagnostics_uniform_weights(self):
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=1000, init_sigma_xy_m=0.05, seed=400,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        d = pf.diagnostics()
        self.assertIsInstance(d, FilterDiagnostics)
        self.assertAlmostEqual(d.n_eff, 1000.0, places=2)
        # Uniform: max_weight = 1/N, entropy = log N.
        self.assertAlmostEqual(d.max_weight, 1.0 / 1000.0, places=5)
        self.assertAlmostEqual(d.weight_entropy, math.log(1000.0), places=4)
        self.assertFalse(d.resampled)

    def test_diagnostics_after_observation(self):
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=1000,
            init_sigma_theta_rad=math.radians(8.0),
            imu_sigma_rad=math.radians(1.0),
            seed=401,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        pf.observe_imu_yaw(world_yaw=0.0)
        d = pf.diagnostics()
        # N_eff dropped; entropy below max.
        self.assertLess(d.n_eff, 1000.0)
        self.assertLess(d.weight_entropy, math.log(1000.0))
        self.assertGreater(d.max_weight, 1.0 / 1000.0)
        self.assertGreater(d.std_x, 0.0)


if __name__ == "__main__":
    unittest.main()

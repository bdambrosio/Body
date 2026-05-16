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

from .particle_filter_pose import (
    ParticleFilterConfig,
    ParticleFilterPose,
    _wrap_torch,
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


if __name__ == "__main__":
    unittest.main()

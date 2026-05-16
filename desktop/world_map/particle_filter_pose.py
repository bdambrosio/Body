"""Particle-filter pose source — Bayesian replacement for the
point-estimate ImuPlusScanMatchPose. Phase 2 of the localization
redesign (docs/bayesian_localization_redesign.md).

Phase 2.1 (this commit): bare filter — particle state, motion-model
predict step using Phase 0's α priors, and IMU-yaw Gaussian observation.
Scan-likelihood consumption (Phase 2.2), systematic resampling
(Phase 2.3), and FuserController wiring (Phase 2.4) land in follow-up
commits.

State layout
------------
- `state`: (P, 3) tensor — columns (x, y, θ_rad). World frame.
- `log_weights`: (P,) tensor — log importance weights, unnormalized.
  Consumers `softmax`-normalize when they need probabilities.

Threading
---------
Phase 2.1 has no zenoh subscriptions — all calls are synchronous and
the caller owns thread safety. Phase 2.4 introduces a lock when the
filter is wired into FuserController's subscriber callbacks.

Backend choice
--------------
PyTorch on CPU for Phase 2; Phase 4 flips `cfg.device` to `"cuda"` for
the per-particle scan-likelihood scoring. einops is used for shape
operations where named axes clarify the intent (per-particle dimension
called `p`, state-component called `d`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from einops import rearrange

# Phase 0 noise priors. See docs/noise_models.md §"Output: filter priors".
# Convention: σ_X² = α_{X→from_trans}² · Δs² + α_{X→from_rot}² · Δθ²
ALPHA_TRANS_PER_M: float = 0.04        # α_1 — translation σ from |Δs|
ALPHA_TRANS_PER_RAD: float = 0.0       # α_2 — translation σ from |Δθ|; NOT MEASURED
ALPHA_ROT_PER_M: float = 0.017         # α_3 — rotation σ from |Δs| (rad/m)
ALPHA_ROT_PER_RAD: float = 0.01        # α_4 — rotation σ from |Δθ|
IMU_SIGMA_PER_SAMPLE_RAD: float = 1.23e-3  # BNO085 game_rotation_vector


@dataclass
class ParticleFilterConfig:
    # Particle count. Plan §3 Phase 2 calls for 1000–2000 on CPU.
    n_particles: int = 1000

    # Motion-model coefficients. Defaults are Phase 0's locked priors.
    alpha_trans_per_m: float = ALPHA_TRANS_PER_M
    alpha_trans_per_rad: float = ALPHA_TRANS_PER_RAD  # α_2 unmeasured
    alpha_rot_per_m: float = ALPHA_ROT_PER_M
    alpha_rot_per_rad: float = ALPHA_ROT_PER_RAD

    # IMU yaw observation σ (per single measurement). Drift rate from
    # Phase 0 was ≈ 0; consumers can pass a wider σ at the call site if
    # they're integrating over an interval where drift is non-negligible.
    imu_sigma_rad: float = IMU_SIGMA_PER_SAMPLE_RAD

    # Per-step floor on motion-noise σ. Without this, zero-motion ticks
    # add zero noise and the cloud freezes — the "particle deprivation"
    # failure mode. Small floors keep diversity alive without distorting
    # the motion model in the regime where it matters.
    sigma_floor_trans_m: float = 1e-3
    sigma_floor_rot_rad: float = 1e-4

    # Initial cloud spread around the seed pose. Tight by default
    # because the typical seed point is "current pose at session reset"
    # which the operator has just calibrated to.
    init_sigma_xy_m: float = 0.02
    init_sigma_theta_rad: float = math.radians(1.0)

    # Torch device. CPU for Phase 2; Phase 4 flips to "cuda".
    device: str = "cpu"

    # State dtype. float64 for trajectory accumulation over long
    # sessions; float32 is fine for log_weights (only ratios matter).
    state_dtype: torch.dtype = torch.float64
    weight_dtype: torch.dtype = torch.float32

    # Deterministic seed for reproducible tests. None = nondeterministic.
    seed: Optional[int] = None


def _wrap_torch(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-π, π]. torch.remainder is the right idiom here —
    it handles negative inputs correctly, unlike a naive `% (2π) - π`."""
    return (a + math.pi).remainder(2.0 * math.pi) - math.pi


class ParticleFilterPose:
    """Per-particle SE(2) Bayesian pose tracker.

    Phase 2.1 surface: ``seed_at`` → ``predict`` (×many) interleaved
    with ``observe_imu_yaw``. Posterior mean and N_eff diagnostics are
    available throughout. Scan-likelihood consumption and resampling
    arrive in Phase 2.2 and 2.3.
    """

    def __init__(self, cfg: Optional[ParticleFilterConfig] = None):
        self.cfg = cfg or ParticleFilterConfig()
        if self.cfg.seed is not None:
            self._gen = torch.Generator(device=self.cfg.device)
            self._gen.manual_seed(self.cfg.seed)
        else:
            self._gen = None
        self._state: Optional[torch.Tensor] = None
        self._log_w: Optional[torch.Tensor] = None

    # ── Internals ─────────────────────────────────────────────────────

    def _randn(self, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randn(
            shape, dtype=self.cfg.state_dtype,
            device=self.cfg.device, generator=self._gen,
        )

    def _require_seeded(self) -> None:
        if self._state is None or self._log_w is None:
            raise RuntimeError(
                "ParticleFilterPose: call seed_at() before predict / observe."
            )

    # ── Read-only views ──────────────────────────────────────────────

    @property
    def state(self) -> torch.Tensor:
        self._require_seeded()
        return self._state  # type: ignore[return-value]

    @property
    def log_weights(self) -> torch.Tensor:
        self._require_seeded()
        return self._log_w  # type: ignore[return-value]

    def n_particles(self) -> int:
        return self.cfg.n_particles

    # ── Initialization ───────────────────────────────────────────────

    def seed_at(self, x: float, y: float, theta: float) -> None:
        """Initialize the cloud as a tight Gaussian around (x, y, θ).

        Resets log-weights to uniform. Called once at session start and
        again when the operator hits 'rebind' to anchor the world frame.
        """
        P = self.cfg.n_particles
        mean_vec = torch.tensor(
            [x, y, theta],
            dtype=self.cfg.state_dtype, device=self.cfg.device,
        )
        sigma_vec = torch.tensor(
            [
                self.cfg.init_sigma_xy_m,
                self.cfg.init_sigma_xy_m,
                self.cfg.init_sigma_theta_rad,
            ],
            dtype=self.cfg.state_dtype, device=self.cfg.device,
        )
        noise = self._randn((P, 3))
        # Broadcast (d,) → (1, d) using named axes — equivalent to
        # .unsqueeze(0) but reads as "lift to the particle dim".
        self._state = (
            rearrange(mean_vec, "d -> 1 d")
            + rearrange(sigma_vec, "d -> 1 d") * noise
        )
        self._state[:, 2] = _wrap_torch(self._state[:, 2])
        self._log_w = torch.full(
            (P,), -math.log(P),
            dtype=self.cfg.weight_dtype, device=self.cfg.device,
        )

    # ── Predict ──────────────────────────────────────────────────────

    def predict(self, delta_s: float, delta_theta: float) -> None:
        """Advance the cloud by one odom motion increment.

        Args:
            delta_s: signed body-frame translation since the last predict
                (positive = forward, negative = reverse), meters.
            delta_theta: signed body-frame rotation, radians (CCW+).

        Noise model:
            σ_trans² = α_1²·Δs² + α_2²·Δθ²  (α_2 unmeasured = 0)
            σ_rot²   = α_3²·Δs² + α_4²·Δθ²

        Each particle draws independent ε_s, ε_θ; per-particle motion
        applied with midpoint-heading integration (small-step trapezoid).
        """
        self._require_seeded()
        assert self._state is not None  # for type-checker

        s_abs = abs(delta_s)
        th_abs = abs(delta_theta)
        sigma_trans = math.hypot(
            self.cfg.alpha_trans_per_m * s_abs,
            self.cfg.alpha_trans_per_rad * th_abs,
        )
        sigma_rot = math.hypot(
            self.cfg.alpha_rot_per_m * s_abs,
            self.cfg.alpha_rot_per_rad * th_abs,
        )
        sigma_trans = max(sigma_trans, self.cfg.sigma_floor_trans_m)
        sigma_rot = max(sigma_rot, self.cfg.sigma_floor_rot_rad)

        eps = self._randn((self.cfg.n_particles, 2))
        ds = delta_s + sigma_trans * eps[:, 0]
        dth = delta_theta + sigma_rot * eps[:, 1]

        theta_curr = self._state[:, 2]
        # Midpoint heading: integrate translation at θ + Δθ/2 rather
        # than at θ or θ' alone. At 50 Hz odom and the velocities this
        # bot reaches, either-endpoint is fine; midpoint is the same
        # cost and slightly more accurate during arcs.
        theta_mid = theta_curr + 0.5 * dth
        self._state[:, 0] += ds * torch.cos(theta_mid)
        self._state[:, 1] += ds * torch.sin(theta_mid)
        self._state[:, 2] = _wrap_torch(theta_curr + dth)

    # ── IMU yaw observation ──────────────────────────────────────────

    def observe_imu_yaw(
        self, world_yaw: float, sigma_rad: Optional[float] = None,
    ) -> None:
        """Apply an IMU yaw observation as a Gaussian log-likelihood.

        Args:
            world_yaw: IMU-reported heading already mapped to the
                filter's world frame (caller subtracts any yaw_offset
                captured at rebind_world_to_current()).
            sigma_rad: observation σ. Defaults to cfg.imu_sigma_rad.
                Pass a larger σ when integrating drift over a long
                gap since the last sample.

        Updates log_weights in place. The -log(σ√2π) constant is
        omitted — it's identical across particles and cancels in
        importance-weight normalization.
        """
        self._require_seeded()
        assert self._state is not None and self._log_w is not None

        sigma = sigma_rad if sigma_rad is not None else self.cfg.imu_sigma_rad
        if sigma <= 0:
            raise ValueError(f"observe_imu_yaw: sigma must be positive, got {sigma}")
        err = _wrap_torch(self._state[:, 2] - world_yaw)
        # Cast err down to weight dtype to keep weight accumulation in
        # one precision. The 1/σ² term is small enough that float32 is
        # plenty for log-weights.
        log_lik = -0.5 * (err / sigma) ** 2
        self._log_w = self._log_w + log_lik.to(self.cfg.weight_dtype)

    # ── Diagnostics + posterior summary ──────────────────────────────

    def normalized_weights(self) -> torch.Tensor:
        """Softmax-normalized weights (P,). Stable: subtract the max
        before exponentiating."""
        self._require_seeded()
        assert self._log_w is not None
        return torch.softmax(self._log_w, dim=0)

    def n_eff(self) -> float:
        """Effective sample size (Kong 1992).

        N_eff = 1 / Σ wᵢ². Equals N for uniform weights; collapses to
        ~1 when one particle dominates. Resampling gate in Phase 2.3
        will fire when this drops below N/2.
        """
        w = self.normalized_weights()
        return float(1.0 / (w * w).sum())

    def posterior_mean(self) -> Tuple[float, float, float]:
        """Weighted mean of the particle cloud as (x, y, θ).

        x, y are linear weighted means. θ uses the standard circular
        mean (atan2 of weighted unit-vector sum) so the wrap at ±π
        doesn't bias the answer.
        """
        self._require_seeded()
        assert self._state is not None
        w = self.normalized_weights().to(self.cfg.state_dtype)
        x = float((w * self._state[:, 0]).sum())
        y = float((w * self._state[:, 1]).sum())
        c = (w * torch.cos(self._state[:, 2])).sum()
        s = (w * torch.sin(self._state[:, 2])).sum()
        theta = float(torch.atan2(s, c))
        return (x, y, theta)

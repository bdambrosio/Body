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
from einops import rearrange, reduce

from desktop.nav.slam.types import Pose2D, ScoreField

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


def _frac_indices(
    v: torch.Tensor, axis: torch.Tensor, n: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized fractional index along a uniform axis.

    Returns (i0, frac, oob_mask). i0 is the lower cell index (clamped
    so i0+1 is valid), frac ∈ [0, 1] is the offset within the cell, and
    oob_mask is True where the query was strictly outside the axis
    extent (modulo a 1e-9 fp tolerance so exact-lattice queries don't
    fall off the upper edge).
    """
    if n == 1:
        zeros_long = torch.zeros_like(v, dtype=torch.int64)
        zeros = torch.zeros_like(v)
        return zeros_long, zeros, torch.zeros_like(v, dtype=torch.bool)
    step = (axis[-1] - axis[0]) / (n - 1)
    t = (v - axis[0]) / step
    eps = 1e-9
    oob = (t < -eps) | (t > (n - 1 + eps))
    t = t.clamp(min=0.0, max=float(n - 1))
    i0 = t.floor().to(torch.int64).clamp(max=n - 2)
    frac = t - i0.to(t.dtype)
    return i0, frac, oob


def interp_score_field(
    score_field: ScoreField,
    dx: torch.Tensor,
    dy: torch.Tensor,
    dth: torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Vectorized trilinear interpolation into a ScoreField.

    All of dx, dy, dth share the same (P,) shape — the per-particle
    delta-from-prior. Returns (P,) scores. Out-of-window queries return
    0.0, matching the scalar ``likelihood_at`` convention.

    einops `rearrange` stacks the 2×2×2 corner block per particle and
    the 2×2×2 weight block as named-axis tensors, then `reduce` collapses
    them — much closer to the math than manual `.expand()` / `.view()`.
    """
    field = torch.as_tensor(score_field.field, device=device, dtype=dtype)
    ax = torch.as_tensor(score_field.dx_axis, device=device, dtype=dtype)
    ay = torch.as_tensor(score_field.dy_axis, device=device, dtype=dtype)
    ath = torch.as_tensor(score_field.dth_axis, device=device, dtype=dtype)
    nx, ny, nth = field.shape

    dx = dx.to(device=device, dtype=dtype)
    dy = dy.to(device=device, dtype=dtype)
    dth = dth.to(device=device, dtype=dtype)

    ix0, fx, oob_x = _frac_indices(dx, ax, nx)
    iy0, fy, oob_y = _frac_indices(dy, ay, ny)
    ith0, fth, oob_th = _frac_indices(dth, ath, nth)
    oob = oob_x | oob_y | oob_th

    # Pair indices and fractional weights along each axis. Shape (P, 2)
    # where the trailing dim is [lower, upper].
    ix_pair = torch.stack([ix0, ix0 + 1], dim=-1)
    iy_pair = torch.stack([iy0, iy0 + 1], dim=-1)
    ith_pair = torch.stack([ith0, ith0 + 1], dim=-1)
    wx = torch.stack([1.0 - fx, fx], dim=-1)
    wy = torch.stack([1.0 - fy, fy], dim=-1)
    wth = torch.stack([1.0 - fth, fth], dim=-1)

    # Broadcast indices to (P, 2, 2, 2) — one entry per cube corner.
    # The named axes i, j, k correspond to (x, y, θ) endpoint choice.
    P = dx.shape[0]
    ix_g = rearrange(ix_pair, "p i -> p i 1 1").expand(P, 2, 2, 2)
    iy_g = rearrange(iy_pair, "p j -> p 1 j 1").expand(P, 2, 2, 2)
    ith_g = rearrange(ith_pair, "p k -> p 1 1 k").expand(P, 2, 2, 2)
    corners = field[ix_g, iy_g, ith_g]  # (P, 2, 2, 2)

    # Outer-product the three (P, 2) weight vectors into the matching
    # (P, 2, 2, 2) blend tensor — exactly the trilinear weight identity.
    weights = (
        rearrange(wx, "p i -> p i 1 1")
        * rearrange(wy, "p j -> p 1 j 1")
        * rearrange(wth, "p k -> p 1 1 k")
    )

    # Trilinear blend = Σ corners · weights over the corner cube.
    out = reduce(corners * weights, "p i j k -> p", "sum")
    out = torch.where(oob, torch.zeros_like(out), out)
    return out


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

    # ── Scan-likelihood observation ──────────────────────────────────

    def update_from_scan_likelihood(
        self,
        score_field: ScoreField,
        prior_pose: Pose2D,
        temperature: Optional[float] = None,
    ) -> None:
        """Apply a scan-match log-likelihood term to log_weights.

        For each particle, interpolate its (dx, dy, dθ)-from-prior into
        the score field (trilinear) and add ``score / temperature`` to
        log_weights. Particles outside the field's window contribute
        zero (matches the scalar likelihood_at convention).

        Args:
            score_field: from ``ScanMatcher.search(..., return_field=True)``.
            prior_pose: the world-frame prior the field was computed
                at — same Pose2D that was passed to ``search()``.
            temperature: divides raw scores before they enter the log-
                weight. Phase 1 left score normalization open: scores
                are raw correlation sums in evidence units, no absolute
                scale. The temperature converts them to log-likelihood
                units. Default = max(field.std(), 1.0) so a 1σ score
                difference equals 1 nat of log-weight; auto-scales to
                each scan's information content (flat scan → flat
                contribution).
        """
        self._require_seeded()
        assert self._state is not None and self._log_w is not None

        dx = self._state[:, 0] - prior_pose.x
        dy = self._state[:, 1] - prior_pose.y
        dth = _wrap_torch(self._state[:, 2] - prior_pose.theta)

        scores = interp_score_field(
            score_field, dx, dy, dth,
            device=self.cfg.device, dtype=self.cfg.state_dtype,
        )

        if temperature is None:
            std = float(scores.std())
            T = max(std, 1.0)
        else:
            T = float(temperature)
        if T <= 0:
            raise ValueError(f"update_from_scan_likelihood: T must be > 0, got {T}")

        log_lik = scores / T
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

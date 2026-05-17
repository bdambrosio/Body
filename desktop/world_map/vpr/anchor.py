"""Phase 6.4 — bank↔session frame alignment via Procrustes SE(2) fit.

Each nav session calls ``rebind_world_to_current`` at startup, so the
world frame's origin is "wherever the bot was when the session
started." A bank built in a previous session lives in a different
world frame; without correction, even a perfect VPR match would pull
the filter toward the wrong world coordinates.

This module estimates the SE(2) offset (Δx, Δy, Δθ) such that
applying it to bank poses lands them in the current session's frame.
Estimator state machine:

- ``UNCALIBRATED``: collect (bank_pose, current_pose) pairs from
  high-similarity matches. Apply no offset yet (VPR observations are
  suppressed when uncalibrated — would only inject noise).
- ``CALIBRATED``: a one-shot Procrustes fit was performed and the
  offset is locked in. ``apply_xy`` transforms bank positions.

V1 calibration is one-shot: once we accept the fit, we don't update
it. Re-calibration after kidnapping or seed-rebind is a future
extension (probably Phase 7 territory).
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class AnchorOffsetConfig:
    # Minimum cosine similarity to use a match for offset estimation.
    # The bank's pose is trusted only when DINOv2 says the bot is
    # really at that location.
    min_similarity: float = 0.85

    # Number of accepted pairs before fitting. 5 is enough for a
    # stable closed-form SE(2) fit with measurement noise; 3 is the
    # mathematical minimum (over-determined past 2 for SE(2)).
    min_pairs: int = 5

    # Pairwise max distance between accepted *bank* poses, in meters.
    # If all accepted matches come from one physical spot (e.g. the
    # bot sat still), rotation is unobservable and the fit reduces to
    # pure translation. This guard ensures we have meaningful
    # geometric diversity before locking the offset.
    min_spatial_spread_m: float = 0.5

    # Maximum residual after the fit (RMS in meters); above this we
    # reject the calibration as likely noisy and keep collecting.
    # 0.25 m is generous — VPR poses are sparse and the bot pose at
    # query time is the particle filter's MMSE, not ground truth.
    max_residual_m: float = 0.25

    # Phase 6.4.3 — bootstrap covariance gate for opportunistic mode.
    # Looser than the sweep's 0.10 because opportunistic typically locks
    # at K≈5–8 pairs where bootstrap variance is large by construction
    # (any pair missing from a resample swings the fit). 0.5 m² ≈ 0.7 m
    # 1-σ in each axis — generous enough that small N doesn't auto-fail
    # but tight enough to catch genuinely uninformative data.
    max_cov_xy_trace_m2: float = 0.5

    # Quantize bank XY to this cell size when counting unique cells —
    # avoids the degenerate "all matches at the same point" case where
    # rotation is unobservable.
    bank_cell_size_m: float = 0.10
    min_unique_bank_cells: int = 3

    # Bootstrap params for the covariance estimate.
    bootstrap_n_resamples: int = 100
    bootstrap_seed: int = 0


@dataclass
class AnchorPair:
    """One observation usable for fitting the bank→session offset."""
    bank_xy: Tuple[float, float]
    current_xy: Tuple[float, float]
    similarity: float


@dataclass
class CalibrationResult:
    dx: float
    dy: float
    dtheta_rad: float
    n_pairs: int
    residual_rms_m: float

    def apply_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """Transform an (N, 2) tensor of bank XY into session frame.

        Out = R(dθ) · xy + (dx, dy).
        """
        c, s = math.cos(self.dtheta_rad), math.sin(self.dtheta_rad)
        R = xy.new_tensor([[c, -s], [s, c]])
        return xy @ R.T + xy.new_tensor([self.dx, self.dy])


class AnchorOffsetEstimator:
    """Thread-safe accumulator + one-shot fitter. The driver calls
    ``observe(...)`` on every VPR result with sim ≥ floor; once
    ``calibrate_if_ready()`` succeeds, subsequent calls do nothing and
    ``apply_xy`` is the only relevant method."""

    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"

    def __init__(self, config: Optional[AnchorOffsetConfig] = None) -> None:
        self._cfg = config or AnchorOffsetConfig()
        self._lock = threading.Lock()
        self._pairs: List[AnchorPair] = []
        self._result: Optional[CalibrationResult] = None

    # ── State ────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        with self._lock:
            return self.CALIBRATED if self._result is not None else self.UNCALIBRATED

    @property
    def calibration(self) -> Optional[CalibrationResult]:
        with self._lock:
            return self._result

    @property
    def n_pairs_collected(self) -> int:
        with self._lock:
            return len(self._pairs)

    def snapshot_pairs(self) -> List[AnchorPair]:
        """Return a copy of the current pair list (for diagnostics +
        bootstrap scoring). Safe to call from any thread."""
        with self._lock:
            return list(self._pairs)

    # ── Accumulate ───────────────────────────────────────────────────

    def observe(
        self,
        *,
        bank_xy: Tuple[float, float],
        current_xy: Tuple[float, float],
        similarity: float,
    ) -> None:
        """Record a candidate pair. No-op once calibrated."""
        with self._lock:
            if self._result is not None:
                return
            if similarity < self._cfg.min_similarity:
                return
            self._pairs.append(AnchorPair(
                bank_xy=bank_xy, current_xy=current_xy, similarity=similarity,
            ))

    # ── Fit ──────────────────────────────────────────────────────────

    def calibrate_if_ready(
        self,
        *,
        on_attempt: Optional[Any] = None,
    ) -> Optional[CalibrationResult]:
        """Attempt a fit; return the result if accepted (also stored),
        ``None`` if not enough data or fit was rejected.

        Phase 6.4.3 — internally delegates to ``score_calibration`` so
        the opportunistic path gets bootstrap covariance + unique-cells
        checking + all the structured failure reasons that the sweep
        scoring already had. The estimator's own ``min_pairs`` /
        ``min_spatial_spread_m`` / ``max_residual_m`` thresholds are
        passed through to ``CalibrationScoringConfig``.

        Args:
            on_attempt: optional callback fired with the full
                ``CalibrationScore`` on every attempt (pass or fail).
                Lets the shadow driver log a ``vpr_calibration`` trace
                event from the opportunistic path, matching the schema
                the sweep produces.

        Idempotent once calibrated — returns the stored result without
        re-running the math.
        """
        with self._lock:
            if self._result is not None:
                return self._result
            if len(self._pairs) < self._cfg.min_pairs:
                return None
            pairs_snapshot = list(self._pairs)

        # Score off-lock — bootstrap covariance does N fits and can
        # take a few ms; no reason to hold the estimator lock during it.
        scoring_cfg = CalibrationScoringConfig(
            min_pairs=self._cfg.min_pairs,
            min_unique_bank_cells=self._cfg.min_unique_bank_cells,
            bank_cell_size_m=self._cfg.bank_cell_size_m,
            min_spatial_spread_m=self._cfg.min_spatial_spread_m,
            max_residual_rms_m=self._cfg.max_residual_m,
            max_cov_xy_trace_m2=self._cfg.max_cov_xy_trace_m2,
            bootstrap_n_resamples=self._cfg.bootstrap_n_resamples,
            bootstrap_seed=self._cfg.bootstrap_seed,
        )
        score = score_calibration(pairs_snapshot, scoring_cfg)

        if on_attempt is not None:
            try:
                on_attempt(score)
            except Exception:
                logger.exception("anchor: on_attempt callback raised")

        if not score.passed or score.offset is None:
            if score.reason not in ("too_few_pairs", "insufficient_spatial_spread",
                                    "too_few_unique_bank_cells"):
                logger.warning(
                    "anchor: %d pairs fit but %s — rms=%.3f cov_xy_trace=%.4f "
                    "spread=%.2fm cells=%d. Will retry with more data.",
                    score.n_pairs, score.reason, score.residual_rms_m,
                    score.cov_xy_trace_m2, score.spatial_spread_m,
                    score.n_unique_bank_cells,
                )
            else:
                logger.debug(
                    "anchor: %d pairs not yet calibratable (%s).",
                    score.n_pairs, score.reason,
                )
            return None

        with self._lock:
            # Recheck — another thread might have set this while we
            # were off-lock running the bootstrap.
            if self._result is not None:
                return self._result
            self._result = score.offset
        logger.info(
            "anchor: calibrated from %d pairs — Δ=(%+.3f m, %+.3f m, "
            "%+.2f°) residual_rms=%.3f m cov_xy_trace=%.4f m²",
            score.n_pairs, score.offset.dx, score.offset.dy,
            math.degrees(score.offset.dtheta_rad),
            score.residual_rms_m, score.cov_xy_trace_m2,
        )
        return self._result

    def set_calibration(self, result: CalibrationResult) -> None:
        """Force-install a calibration result (e.g. from a sweep-driven
        scoring pass). No-op if already calibrated."""
        with self._lock:
            if self._result is not None:
                return
            self._result = result
            logger.info(
                "anchor: externally calibrated — Δ=(%+.3f m, %+.3f m, "
                "%+.2f°) residual_rms=%.3f m (n_pairs=%d)",
                result.dx, result.dy, math.degrees(result.dtheta_rad),
                result.residual_rms_m, result.n_pairs,
            )

    # ── Apply ────────────────────────────────────────────────────────

    def apply_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """Transform bank XY → session frame. Raises if uncalibrated."""
        with self._lock:
            r = self._result
        if r is None:
            raise RuntimeError(
                "AnchorOffsetEstimator: apply_xy called before calibration"
            )
        return r.apply_xy(xy)


# ── Geometry helpers ────────────────────────────────────────────────


def _max_pairwise_distance(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    best = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d = math.hypot(dx, dy)
            if d > best:
                best = d
    return best


def _fit_se2(
    src: List[Tuple[float, float]],
    dst: List[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """Closed-form Procrustes SE(2): solve for (dx, dy, dθ) minimizing
    Σ |R(dθ)·src_i + (dx, dy) − dst_i|².

    Returns (dx, dy, dθ, rms_residual).

    See Umeyama 1991 — the 2D case admits a closed form for rotation
    via atan2 of the cross-product / dot-product sums; translation is
    then the centroid difference under that rotation.
    """
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError("_fit_se2 needs ≥2 paired points")
    n = len(src)
    sx = sum(p[0] for p in src) / n
    sy = sum(p[1] for p in src) / n
    dx_ = sum(p[0] for p in dst) / n
    dy_ = sum(p[1] for p in dst) / n
    num = 0.0
    den = 0.0
    for s_p, d_p in zip(src, dst):
        a, b = s_p[0] - sx, s_p[1] - sy
        c, d = d_p[0] - dx_, d_p[1] - dy_
        num += a * d - b * c
        den += a * c + b * d
    dtheta = math.atan2(num, den)
    cos_t, sin_t = math.cos(dtheta), math.sin(dtheta)
    tx = dx_ - (cos_t * sx - sin_t * sy)
    ty = dy_ - (sin_t * sx + cos_t * sy)
    # Residual RMS.
    sq = 0.0
    for s_p, d_p in zip(src, dst):
        rx = cos_t * s_p[0] - sin_t * s_p[1] + tx - d_p[0]
        ry = sin_t * s_p[0] + cos_t * s_p[1] + ty - d_p[1]
        sq += rx * rx + ry * ry
    rms = math.sqrt(sq / n)
    return tx, ty, dtheta, rms


# ── Phase 6.4.2 — bootstrap covariance + structured scoring ──────────


@dataclass(frozen=True)
class CalibrationScore:
    """Diagnostic snapshot of a calibration attempt. Always logged to
    the VPR trace (under record type ``vpr_calibration``); ``passed``
    governs whether the offset gets locked into the estimator."""

    passed: bool
    reason: str            # "passed" or first failed check
    n_pairs: int
    n_unique_bank_cells: int
    spatial_spread_m: float
    residual_rms_m: float
    cov_xy_trace_m2: float
    cov_theta_var_rad2: float
    offset: Optional[CalibrationResult]


@dataclass
class CalibrationScoringConfig:
    """Thresholds applied to a sweep-driven calibration attempt.

    Stricter than the opportunistic-mode defaults in AnchorOffsetConfig
    because the sweep procedure can collect enough pairs to actually
    demand a high-confidence fit before locking the anchor."""

    min_pairs: int = 5
    min_unique_bank_cells: int = 3      # avoid pure-rotation degeneracy
    bank_cell_size_m: float = 0.10      # for the unique-cells count
    min_spatial_spread_m: float = 0.5
    max_residual_rms_m: float = 0.15
    max_cov_xy_trace_m2: float = 0.10
    bootstrap_n_resamples: int = 100
    bootstrap_seed: int = 0


def bootstrap_se2_covariance(
    pairs: List[AnchorPair],
    n_resamples: int = 100,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Empirical 3×3 covariance of (Δx, Δy, Δθ) via paired bootstrap.

    Each resample draws ``len(pairs)`` indices with replacement, refits
    SE(2), and collects the (dx, dy, dθ) estimate. The sample covariance
    over those estimates is the bootstrap covariance. Returns ``None``
    if too few pairs to fit (need ≥ 3).
    """
    if len(pairs) < 3:
        return None
    rng = np.random.default_rng(seed)
    n = len(pairs)
    samples = np.empty((n_resamples, 3), dtype=np.float64)
    pair_arr = np.array(
        [[p.bank_xy[0], p.bank_xy[1], p.current_xy[0], p.current_xy[1]]
         for p in pairs], dtype=np.float64,
    )
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sub = pair_arr[idx]
        src = [(float(r[0]), float(r[1])) for r in sub]
        dst = [(float(r[2]), float(r[3])) for r in sub]
        dx, dy, dth, _ = _fit_se2(src, dst)
        samples[i] = (dx, dy, dth)
    # rowvar=False: each row is one observation. ddof=1 for unbiased.
    return np.cov(samples, rowvar=False, ddof=1)


def score_calibration(
    pairs: List[AnchorPair],
    config: Optional[CalibrationScoringConfig] = None,
) -> CalibrationScore:
    """Attempt a calibration fit on the given pairs and score it.
    Does not mutate any state — the caller can decide whether to
    accept the fit (typically by injecting it into the estimator
    via ``AnchorOffsetEstimator.set_calibration``)."""
    cfg = config or CalibrationScoringConfig()
    n = len(pairs)
    if n < cfg.min_pairs:
        return CalibrationScore(
            passed=False, reason="too_few_pairs",
            n_pairs=n, n_unique_bank_cells=0, spatial_spread_m=0.0,
            residual_rms_m=float("nan"),
            cov_xy_trace_m2=float("nan"), cov_theta_var_rad2=float("nan"),
            offset=None,
        )
    bank_xy = [p.bank_xy for p in pairs]
    current_xy = [p.current_xy for p in pairs]
    spread = _max_pairwise_distance(bank_xy)
    # Quantize bank XY to a coarse cell size and count unique cells —
    # tells us how many physically distinct bank locations the matches
    # cover. A 360° sweep at one spot might match 10 bank frames but
    # all at the same physical point → poor rotation observability.
    cell = max(1e-6, cfg.bank_cell_size_m)
    unique_cells = {
        (round(x / cell), round(y / cell)) for x, y in bank_xy
    }
    n_unique = len(unique_cells)
    if spread < cfg.min_spatial_spread_m:
        return CalibrationScore(
            passed=False, reason="insufficient_spatial_spread",
            n_pairs=n, n_unique_bank_cells=n_unique,
            spatial_spread_m=spread,
            residual_rms_m=float("nan"),
            cov_xy_trace_m2=float("nan"), cov_theta_var_rad2=float("nan"),
            offset=None,
        )
    if n_unique < cfg.min_unique_bank_cells:
        return CalibrationScore(
            passed=False, reason="too_few_unique_bank_cells",
            n_pairs=n, n_unique_bank_cells=n_unique,
            spatial_spread_m=spread,
            residual_rms_m=float("nan"),
            cov_xy_trace_m2=float("nan"), cov_theta_var_rad2=float("nan"),
            offset=None,
        )
    dx, dy, dth, rms = _fit_se2(bank_xy, current_xy)
    cov = bootstrap_se2_covariance(
        pairs, n_resamples=cfg.bootstrap_n_resamples, seed=cfg.bootstrap_seed,
    )
    cov_xy_trace = (
        float(cov[0, 0] + cov[1, 1]) if cov is not None else float("nan")
    )
    cov_theta_var = float(cov[2, 2]) if cov is not None else float("nan")
    offset = CalibrationResult(
        dx=dx, dy=dy, dtheta_rad=dth, n_pairs=n, residual_rms_m=rms,
    )
    if rms > cfg.max_residual_rms_m:
        return CalibrationScore(
            passed=False, reason="residual_too_large",
            n_pairs=n, n_unique_bank_cells=n_unique,
            spatial_spread_m=spread,
            residual_rms_m=rms,
            cov_xy_trace_m2=cov_xy_trace, cov_theta_var_rad2=cov_theta_var,
            offset=offset,
        )
    if cov is not None and cov_xy_trace > cfg.max_cov_xy_trace_m2:
        return CalibrationScore(
            passed=False, reason="offset_covariance_too_large",
            n_pairs=n, n_unique_bank_cells=n_unique,
            spatial_spread_m=spread,
            residual_rms_m=rms,
            cov_xy_trace_m2=cov_xy_trace, cov_theta_var_rad2=cov_theta_var,
            offset=offset,
        )
    return CalibrationScore(
        passed=True, reason="passed",
        n_pairs=n, n_unique_bank_cells=n_unique,
        spatial_spread_m=spread,
        residual_rms_m=rms,
        cov_xy_trace_m2=cov_xy_trace, cov_theta_var_rad2=cov_theta_var,
        offset=offset,
    )


__all__ = [
    "AnchorOffsetConfig",
    "AnchorOffsetEstimator",
    "AnchorPair",
    "CalibrationResult",
    "CalibrationScore",
    "CalibrationScoringConfig",
    "bootstrap_se2_covariance",
    "score_calibration",
]

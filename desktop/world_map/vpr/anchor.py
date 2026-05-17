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

    def calibrate_if_ready(self) -> Optional[CalibrationResult]:
        """Attempt a fit; return the result if accepted (also stored),
        ``None`` if not enough data or fit was rejected.

        Idempotent once calibrated — returns the stored result without
        re-running the math.
        """
        with self._lock:
            if self._result is not None:
                return self._result
            if len(self._pairs) < self._cfg.min_pairs:
                return None
            bank = [(p.bank_xy[0], p.bank_xy[1]) for p in self._pairs]
            curr = [(p.current_xy[0], p.current_xy[1]) for p in self._pairs]
            # Spatial-spread guard on bank side.
            spread = _max_pairwise_distance(bank)
            if spread < self._cfg.min_spatial_spread_m:
                logger.debug(
                    "anchor: %d pairs collected but bank spread %.2f m < %.2f m "
                    "threshold; deferring fit.",
                    len(self._pairs), spread, self._cfg.min_spatial_spread_m,
                )
                return None
            dx, dy, dth, rms = _fit_se2(bank, curr)
            if rms > self._cfg.max_residual_m:
                logger.warning(
                    "anchor: %d pairs fit but residual RMS %.3f m > %.3f m "
                    "threshold; rejecting fit, will retry with more data.",
                    len(self._pairs), rms, self._cfg.max_residual_m,
                )
                return None
            self._result = CalibrationResult(
                dx=dx, dy=dy, dtheta_rad=dth,
                n_pairs=len(self._pairs), residual_rms_m=rms,
            )
            logger.info(
                "anchor: calibrated from %d pairs — Δ=(%+.3f m, %+.3f m, "
                "%+.2f°) residual_rms=%.3f m",
                self._result.n_pairs, dx, dy, math.degrees(dth), rms,
            )
            return self._result

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


__all__ = [
    "AnchorOffsetConfig",
    "AnchorOffsetEstimator",
    "AnchorPair",
    "CalibrationResult",
]

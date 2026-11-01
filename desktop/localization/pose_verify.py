"""Is the pose still right after a fault? Lidar first, camera as fallback.

Used by the recovery supervisor (desktop/nav/recovery.py) before it resumes a
drive on its own — after a WiFi drop (SUSPENDED pose_lost) or before an
escalation rung of a blocked leg. It answers ``ok`` / ``bad`` / ``unknown``.

Lidar: the pose source already scan-matches every scan against the reference
map. Quality is the fraction of scan endpoints that land on mapped walls at
the best alignment (``score_best / n_points``, the same signal as
desktop/nav/pose_health.py). Absolute values depend on the map and the room,
so the bar is relative: the median quality just before the fault is the
baseline. The best alignment's offset from the prior says whether the match is
pulling the pose somewhere else.

Camera (fallback, only when lidar says unknown): an injected callable compares
the newest frame with the last pre-fault frame (see image_check.py).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

OK = "ok"
BAD = "bad"
UNKNOWN = "unknown"
PENDING = "pending"          # not enough fresh evidence yet


@dataclass(frozen=True)
class PoseVerifyConfig:
    min_fresh_matches: int = 3
    min_points: int = 30
    ok_quality_frac: float = 0.7      # of baseline
    bad_quality_frac: float = 0.4
    ok_shift_m: float = 0.15
    ok_dtheta_rad: float = math.radians(8.0)
    bad_shift_m: float = 0.30
    bad_dtheta_rad: float = math.radians(15.0)
    # Used only when no pre-fault baseline exists (fault right after start).
    fallback_quality: float = 0.15


@dataclass(frozen=True)
class Verdict:
    kind: str                 # ok | bad | unknown | pending
    detail: str
    source: str = "lidar"     # lidar | image | none


@dataclass
class _Window:
    baseline: Optional[float]
    last_seq: Optional[int]
    samples: List[Tuple[float, float, float]] = field(default_factory=list)
    # (quality, shift_m, |dtheta|)


def match_quality(summary: Dict[str, Any]) -> Optional[float]:
    n = int(summary.get("n_points", 0) or 0)
    if not summary.get("valid") or n <= 0:
        return None
    return float(summary.get("score_best", 0.0)) / n


class PoseVerifier:
    """Collects fresh scan-match evidence after ``mark_fault`` and judges it.

    ``summary_fn`` returns the pose source's match summary (needs ``valid``,
    ``score_best``, ``n_points``, ``shift_m``, ``dtheta_rad``, ``scan_obs_run``).
    ``baseline_fn`` returns the pre-fault median quality (or None).
    ``image_check`` (optional) returns a Verdict from the camera.
    """

    def __init__(
        self,
        summary_fn: Callable[[], Optional[Dict[str, Any]]],
        baseline_fn: Callable[[], Optional[float]],
        *,
        image_check: Optional[Callable[[], Verdict]] = None,
        cfg: PoseVerifyConfig = PoseVerifyConfig(),
    ) -> None:
        self._summary_fn = summary_fn
        self._baseline_fn = baseline_fn
        self._image_check = image_check
        self._cfg = cfg
        self._win: Optional[_Window] = None

    def mark_fault(self) -> None:
        """Snapshot the baseline (pre-fault quality) — call when the fault
        starts, before stale/post-fault matches dilute it."""
        s = self._safe_summary()
        self._win = _Window(baseline=self._baseline_fn(),
                            last_seq=_seq(s) if s else None)

    def restart_window(self) -> None:
        """Discard collected evidence (e.g. after a re-localize) but keep the
        baseline; only matches after this call count."""
        if self._win is None:
            self.mark_fault()
            return
        s = self._safe_summary()
        self._win.samples.clear()
        self._win.last_seq = _seq(s) if s else None

    def baseline(self) -> Optional[float]:
        return self._win.baseline if self._win else None

    def poll(self) -> None:
        """Ingest the latest match if it is new. Call every tick while
        verifying (scans arrive at ~10 Hz)."""
        if self._win is None:
            self.mark_fault()
        s = self._safe_summary()
        if not s or not s.get("valid"):
            return
        seq = _seq(s)
        if seq is not None and seq == self._win.last_seq:
            return
        self._win.last_seq = seq
        if int(s.get("n_points", 0) or 0) < self._cfg.min_points:
            return
        q = match_quality(s)
        if q is None:
            return
        self._win.samples.append(
            (q, float(s.get("shift_m", 0.0)), abs(float(s.get("dtheta_rad", 0.0)))))

    def verdict(self, *, final: bool = False) -> Verdict:
        """Lidar verdict from the collected window; ``pending`` until enough
        fresh matches arrived (``final=True`` turns pending into unknown and
        consults the camera)."""
        lid = self._lidar_verdict()
        if lid.kind == PENDING and not final:
            return lid
        if lid.kind in (OK, BAD):
            return lid
        if self._image_check is not None:
            try:
                img = self._image_check()
            except Exception as e:  # never let the fallback kill recovery
                img = Verdict(UNKNOWN, f"image check failed: {e}", "image")
            if img.kind in (OK, BAD):
                return Verdict(img.kind, f"{lid.detail}; camera: {img.detail}", "image")
            return Verdict(UNKNOWN, f"{lid.detail}; camera: {img.detail}", "none")
        return Verdict(UNKNOWN, lid.detail, "none")

    def _lidar_verdict(self) -> Verdict:
        cfg = self._cfg
        w = self._win
        if w is None or len(w.samples) < cfg.min_fresh_matches:
            n = 0 if w is None else len(w.samples)
            return Verdict(PENDING, f"lidar: {n}/{cfg.min_fresh_matches} fresh matches")
        q = statistics.median(s[0] for s in w.samples)
        sh = statistics.median(s[1] for s in w.samples)
        dth = statistics.median(s[2] for s in w.samples)
        base = w.baseline
        if base is not None and base > 0:
            rel = q / base
            qtxt = f"quality {q:.2f} ({rel:.0%} of pre-fault {base:.2f})"
            q_ok, q_bad = rel >= cfg.ok_quality_frac, rel < cfg.bad_quality_frac
        else:
            qtxt = f"quality {q:.2f} (no pre-fault baseline)"
            q_ok, q_bad = q >= cfg.fallback_quality, False
        mtxt = f"match offset {sh:.2f} m / {math.degrees(dth):.0f}°"
        detail = f"lidar: {qtxt}, {mtxt}, n={len(w.samples)}"
        if q_bad or sh > cfg.bad_shift_m or dth > cfg.bad_dtheta_rad:
            return Verdict(BAD, detail)
        if q_ok and sh <= cfg.ok_shift_m and dth <= cfg.ok_dtheta_rad:
            return Verdict(OK, detail)
        return Verdict(UNKNOWN, detail)

    def _safe_summary(self) -> Optional[Dict[str, Any]]:
        try:
            return self._summary_fn()
        except Exception:
            return None


def _seq(s: Dict[str, Any]) -> Optional[int]:
    v = s.get("scan_obs_run")
    return int(v) if v is not None else None

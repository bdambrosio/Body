"""Pose-health monitor: detect localization divergence from scan-match quality.

The particle filter never raises a "lost" signal — `n_eff` and scan-match
scores are recorded but nothing thresholds them. A confidently-wrong pose
(fresh timestamp, bad value) therefore sails past the mission's pose-age
gate, and the only thing between the robot and a collision is the per-tick
local_map veto. This monitor closes that gap: it watches the quality of the
live scan-to-map alignment and flags a sustained collapse as "pose lost",
so the caller can stop and force a relocate before the robot drives where
the (wrong) map says is clear.

Signal: ``match_quality = score_best / n_points`` — the fraction of scan
endpoints that land on an occupied map cell at the best alignment the
matcher found (the matcher score is a count of endpoints on occupied
cells). Scale-free in [0, 1]: when well-localized a large fraction of
endpoints align with mapped walls; when lost, few do, because no pose fits
the scan to the map.

False-positive guards:
  * Only *valid* matches with enough points are ingested (a sparse scan
    says nothing about localization quality).
  * A "lost" verdict needs the rolling-window MEDIAN to sit below the
    threshold across a sustained time window AND a minimum sample count —
    a single bad match (someone walks through the lidar, a momentary
    featureless view) never trips it.

Sparse open areas legitimately produce low quality even when localized, so
the threshold is conservative and the caller's action (force a relocate,
which either re-snaps or fails harmlessly) is reversible — not a hard fault.
This is pure logic: no threading or I/O. The caller owns when to ingest and
what to do on a lost verdict.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple


@dataclass
class PoseHealthConfig:
    # Median per-point match quality below which the window counts as
    # "lost". 0.0 = no endpoints align; 1.0 = every endpoint on a wall.
    # Conservative default — tune against live traces (match_summary's
    # score_best / n_points for your map) before trusting auto-action.
    quality_threshold: float = 0.15
    # A lost verdict needs at least `min_samples` valid matches spanning
    # at least half of `window_s`, so brief dropouts can't trip it.
    window_s: float = 6.0
    min_samples: int = 4
    # Ignore matches with fewer points than this — too sparse to judge.
    min_points: int = 30
    # Cap retained samples (window_s also prunes by time).
    max_samples: int = 64


class PoseHealthMonitor:
    """Rolling-window detector for localization divergence.

    Feed it scan-match summaries via `ingest`; query `is_lost`. Call
    `reset` after a successful relocate / set-location so a fresh fix
    isn't immediately re-flagged by stale samples.
    """

    def __init__(self, config: Optional[PoseHealthConfig] = None):
        self.config = config or PoseHealthConfig()
        self._samples: Deque[Tuple[float, float]] = deque()  # (ts, quality)
        self._last_seq: Optional[int] = None

    def ingest(
        self,
        summary: Optional[Dict[str, Any]],
        now: float,
        *,
        seq: Optional[int] = None,
    ) -> None:
        """Add one scan-match result. `summary` is a `match_summary` dict
        (needs `valid`, `score_best`/`score`, `n_points`). `seq`, when
        given, dedupes repeat reads of the same match (pass scan_obs_run).
        """
        if not summary or not summary.get("valid"):
            return
        if seq is not None:
            if seq == self._last_seq:
                return
            self._last_seq = seq
        n_points = int(summary.get("n_points", 0))
        if n_points < self.config.min_points:
            return
        score = float(summary.get("score_best", summary.get("score", 0.0)))
        quality = score / n_points if n_points > 0 else 0.0
        self._samples.append((now, quality))
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.config.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        while len(self._samples) > self.config.max_samples:
            self._samples.popleft()

    def _median(self) -> Optional[float]:
        if not self._samples:
            return None
        qualities = sorted(q for _, q in self._samples)
        n = len(qualities)
        if n % 2:
            return qualities[n // 2]
        return 0.5 * (qualities[n // 2 - 1] + qualities[n // 2])

    def is_lost(self, now: float) -> bool:
        self._prune(now)
        cfg = self.config
        if len(self._samples) < cfg.min_samples:
            return False
        span = self._samples[-1][0] - self._samples[0][0]
        if span < cfg.window_s * 0.5:
            # Samples present but don't yet span a meaningful window.
            return False
        median = self._median()
        return median is not None and median < cfg.quality_threshold

    def median_quality(self, now: float) -> Optional[float]:
        self._prune(now)
        return self._median()

    def reset(self) -> None:
        self._samples.clear()
        self._last_seq = None


# ── Pose support (explore plan A2) ───────────────────────────────────

SUPPORT_UNKNOWN = "unknown"
SUPPORT_GOOD = "good"
SUPPORT_WEAK = "weak"
SUPPORT_OFF_MAP = "off_map"


@dataclass
class PoseSupportConfig:
    # Per-scan gate for writing observations into the world (session grid,
    # change detector): the scan must sit on the map and match it.
    scan_min_coverage: float = 0.6
    scan_min_quality: float = 0.15
    # Rolling-window levels.
    window_s: float = 3.0
    min_samples: int = 3
    off_map_coverage: float = 0.35   # median coverage below → off_map
    good_coverage: float = 0.6       # median coverage below → weak
    good_quality: float = 0.15       # median quality below → weak
    min_points: int = 30
    max_samples: int = 64


def _summary_quality(summary: Dict[str, Any]) -> Optional[float]:
    n = int(summary.get("n_points", 0) or 0)
    if n <= 0:
        return None
    return float(summary.get("score_best", summary.get("score", 0.0))) / n


class PoseSupportMonitor:
    """How much the reference map supports the current pose.

    ``good``    — the scan sits on mapped space and matches it;
    ``weak``    — on the map but matching poorly (crowd, moved furniture,
                  or a pose going wrong), or only partly on the map;
    ``off_map`` — most of what the lidar sees is space the map never saw:
                  the PF is dead-reckoning, whatever its timestamps say.

    Fed with scan-match summaries that carry ``map_coverage`` (MCLPoseSource
    adds it). Pure logic, no I/O; the caller owns threading.
    """

    def __init__(self, config: Optional[PoseSupportConfig] = None):
        self.config = config or PoseSupportConfig()
        self._samples: Deque[Tuple[float, float, float]] = deque()  # ts, quality, coverage
        self._last_seq: Optional[int] = None

    def scan_ok(self, summary: Optional[Dict[str, Any]]) -> bool:
        """Per-scan gate: may this scan be written into the world model?"""
        if not summary or not summary.get("valid"):
            return False
        if summary.get("scan_pose_world") is None:
            return False
        if int(summary.get("n_points", 0) or 0) < self.config.min_points:
            return False
        q = _summary_quality(summary)
        cov = summary.get("map_coverage")
        return (q is not None and cov is not None
                and q >= self.config.scan_min_quality
                and float(cov) >= self.config.scan_min_coverage)

    def ingest(self, summary: Optional[Dict[str, Any]], now: float, *,
               seq: Optional[int] = None) -> None:
        if not summary or not summary.get("valid") or summary.get("map_coverage") is None:
            return
        if seq is not None:
            if seq == self._last_seq:
                return
            self._last_seq = seq
        if int(summary.get("n_points", 0) or 0) < self.config.min_points:
            return
        q = _summary_quality(summary)
        if q is None:
            return
        self._samples.append((now, q, float(summary["map_coverage"])))
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.config.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        while len(self._samples) > self.config.max_samples:
            self._samples.popleft()

    @staticmethod
    def _median(vals) -> float:
        v = sorted(vals)
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

    def medians(self, now: float) -> Optional[Tuple[float, float]]:
        """(median quality, median coverage) over the window, or None."""
        self._prune(now)
        if not self._samples:
            return None
        return (self._median(s[1] for s in self._samples),
                self._median(s[2] for s in self._samples))

    def level(self, now: float) -> str:
        self._prune(now)
        cfg = self.config
        if len(self._samples) < cfg.min_samples:
            return SUPPORT_UNKNOWN
        q, cov = self.medians(now)
        if cov < cfg.off_map_coverage:
            return SUPPORT_OFF_MAP
        if cov < cfg.good_coverage or q < cfg.good_quality:
            return SUPPORT_WEAK
        return SUPPORT_GOOD

    def reset(self) -> None:
        self._samples.clear()
        self._last_seq = None

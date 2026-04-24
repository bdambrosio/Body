"""Shadow-mode SLAM driver.

Consumes body/imu and body/lidar/scan through the fuser's existing
Zenoh session, feeds samples into ImuYawTracker + ScanMatcher, and
logs each candidate pose correction. **Does not write to any
PoseSource.** The fuser's output is unaffected; this module is purely
observational — a dry run of the SLAM wire integration against live
Pi data before anything in the fuser depends on its output.

Purpose:
1. Surface real-data shape surprises (units, frame conventions,
   missing fields) before they can break the fuser.
2. Build confidence that ScanMatcher's `improvement` and
   `search_exhausted` signals behave sensibly against real lidar +
   the live world grid.
3. Give an offline-analyzable log of shadow decisions so
   `min_improvement` and search-window sizes can be tuned from data.

Threading:
- Zenoh subscribe callbacks arrive on Zenoh threads.
- Scan-match is run synchronously in the scan callback. At default
  rate-limit (2 Hz) and default window (≈ 16k candidates) it's well
  under the 100 ms scan period. Promote to a worker thread only if
  profiling says so.
- ImuYawTracker has internal locking; ScanMatcher is stateless.

When the shadow numbers look right (accepted rate reasonable,
search_exhausted rare, improvement vs noise distribution clean), the
same tracker + matcher gets promoted into an ImuPlusScanMatchPose
implementation that replaces OdomPose in the fuser. That replacement
is a separate, reviewable change.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .imu_yaw import ImuYawTracker
from .scan_matcher import ScanMatcher, ScanMatcherConfig, lidar_scan_to_xy
from .types import ImuReading, Pose2D

logger = logging.getLogger(__name__)


@dataclass
class ShadowDriverConfig:
    # Rate-limit scan-match invocations. 10 Hz lidar → 2 Hz match =
    # log every 5th scan. Plenty for tuning observation.
    match_hz: float = 2.0

    # If the grid has fewer block-votes than this, skip the match:
    # correlation against an almost-empty evidence field is garbage
    # in / garbage out.
    min_grid_evidence_cells: int = 200

    # If the IMU tracker hasn't settled, skip the match and log why.
    # (Prior-pose yaw stays whatever OdomPose says — which is what
    # the current fuser is already using for real output.)
    require_imu_settle: bool = True


class ShadowSlamDriver:
    """Observational SLAM driver. No side effects on fuser state."""

    IMU_TOPIC = "body/imu"
    SCAN_TOPIC = "body/lidar/scan"

    def __init__(
        self,
        *,
        session: Any,                     # zenoh session (from fuser)
        grid: Any,                        # WorldGrid (fuser-owned)
        pose_source: Any,                 # PoseSource (fuser-owned)
        matcher: Optional[ScanMatcher] = None,
        tracker: Optional[ImuYawTracker] = None,
        config: ShadowDriverConfig = ShadowDriverConfig(),
    ) -> None:
        self._session = session
        self._grid = grid
        self._pose_source = pose_source
        self._matcher = matcher if matcher is not None else ScanMatcher(
            ScanMatcherConfig(),
        )
        self._tracker = tracker if tracker is not None else ImuYawTracker()
        self._config = config

        self._subs: List[Any] = []
        self._last_match_wall: float = 0.0
        self._match_lock = threading.Lock()

        # Counters for a periodic summary log.
        self._counters: Dict[str, int] = {
            "imu_received": 0,
            "imu_malformed": 0,
            "scan_received": 0,
            "scan_malformed": 0,
            "matches_run": 0,
            "matches_accepted": 0,
            "matches_exhausted": 0,
            "matches_skipped_rate_limit": 0,
            "matches_skipped_imu_unsettled": 0,
            "matches_skipped_no_pose": 0,
            "matches_skipped_sparse_grid": 0,
        }

    # ── Lifecycle ────────────────────────────────────────────────────

    def connect(self) -> None:
        """Install subscribers on the supplied Zenoh session."""
        if self._subs:
            return
        self._subs.append(
            self._session.declare_subscriber(self.IMU_TOPIC, self._on_imu),
        )
        self._subs.append(
            self._session.declare_subscriber(self.SCAN_TOPIC, self._on_scan),
        )
        logger.info(
            "shadow_slam: subscribed to %s + %s (match_hz=%.1f)",
            self.IMU_TOPIC, self.SCAN_TOPIC, self._config.match_hz,
        )

    def disconnect(self) -> None:
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                pass
        self._subs.clear()
        logger.info(
            "shadow_slam: disconnected. counters=%s", self._counters,
        )

    # ── Subscriber callbacks ─────────────────────────────────────────

    def _payload_bytes(self, sample: Any) -> bytes:
        try:
            return bytes(sample.payload.to_bytes())
        except AttributeError:
            return bytes(sample.payload)

    def _on_imu(self, sample: Any) -> None:
        self._counters["imu_received"] += 1
        try:
            msg = json.loads(self._payload_bytes(sample).decode("utf-8"))
        except Exception:
            self._counters["imu_malformed"] += 1
            return
        reading = ImuReading.from_payload(msg)
        if reading is None:
            self._counters["imu_malformed"] += 1
            return
        self._tracker.update(reading)

    def _on_scan(self, sample: Any) -> None:
        self._counters["scan_received"] += 1
        try:
            msg = json.loads(self._payload_bytes(sample).decode("utf-8"))
        except Exception:
            self._counters["scan_malformed"] += 1
            return

        scan_ts = float(msg.get("ts") or time.time())
        ranges = msg.get("ranges")
        angle_min = float(msg.get("angle_min", 0.0))
        angle_inc = float(msg.get("angle_increment", 0.0))
        if not isinstance(ranges, list) or angle_inc <= 0:
            self._counters["scan_malformed"] += 1
            return

        # Rate-limit match invocations.
        with self._match_lock:
            now_wall = time.monotonic()
            min_period = 1.0 / max(0.1, self._config.match_hz)
            if now_wall - self._last_match_wall < min_period:
                self._counters["matches_skipped_rate_limit"] += 1
                return
            self._last_match_wall = now_wall

        # Precondition checks: IMU settled, prior pose available,
        # grid has enough evidence to score against.
        if self._config.require_imu_settle and not self._tracker.is_settled():
            self._counters["matches_skipped_imu_unsettled"] += 1
            return

        prior_latest = self._pose_source.latest_pose()
        if prior_latest is None:
            self._counters["matches_skipped_no_pose"] += 1
            return
        prior_xytheta, _prior_ts = prior_latest

        block_votes = self._grid.snapshot_block_votes()
        evidence_count = int((block_votes > 0).sum())
        if evidence_count < self._config.min_grid_evidence_cells:
            self._counters["matches_skipped_sparse_grid"] += 1
            return

        # Override prior yaw with IMU-derived yaw when available.
        yaw_sigma: Optional[float] = None
        imu_yaw = self._tracker.yaw_at(scan_ts)
        if imu_yaw is not None:
            yaw, yaw_sigma = imu_yaw
            # Wrap to [-π, π] to stay consistent with PoseSource output.
            yaw_wrapped = (yaw + math.pi) % (2.0 * math.pi) - math.pi
            prior_pose = Pose2D(
                x=prior_xytheta[0], y=prior_xytheta[1], theta=yaw_wrapped,
            )
        else:
            prior_pose = Pose2D(*prior_xytheta)

        # Convert scan.
        angles = np.arange(len(ranges), dtype=np.float64) * angle_inc + angle_min
        ranges_arr = np.asarray(
            [r if isinstance(r, (int, float)) else np.nan for r in ranges],
            dtype=np.float64,
        )
        points_xy = lidar_scan_to_xy(ranges_arr, angles)
        if points_xy.shape[0] < 10:
            self._counters["scan_malformed"] += 1
            return

        # Run match.
        t0 = time.monotonic()
        result = self._matcher.search(
            points_xy,
            prior_pose,
            block_votes,
            self._grid.origin_x_m,
            self._grid.origin_y_m,
            self._grid.resolution_m,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        self._counters["matches_run"] += 1
        if result.accepted:
            self._counters["matches_accepted"] += 1
        if result.search_exhausted:
            self._counters["matches_exhausted"] += 1

        logger.info(
            "shadow_slam: "
            "pts=%d ev=%d "
            "prior=(%+.2f %+.2f %+.1f°) best=(%+.2f %+.2f %+.1f°) "
            "Δ=(%+.3f %+.3f %+.2f°) "
            "score=%.0f/%.0f imp=%+.1f "
            "acc=%s exh=%s imu=%s(σ=%s) el=%.1fms",
            points_xy.shape[0], evidence_count,
            prior_pose.x, prior_pose.y, math.degrees(prior_pose.theta),
            result.pose.x, result.pose.y, math.degrees(result.pose.theta),
            result.pose.x - prior_pose.x,
            result.pose.y - prior_pose.y,
            math.degrees(result.pose.theta - prior_pose.theta),
            result.score, result.score_prior, result.improvement,
            result.accepted, result.search_exhausted,
            self._tracker.fusion_mode().value,
            f"{yaw_sigma:.3f}" if yaw_sigma is not None else "—",
            elapsed_ms,
        )

    # ── Introspection ────────────────────────────────────────────────

    def counters(self) -> Dict[str, int]:
        """Snapshot of lifetime counters. Safe to call from any thread."""
        return dict(self._counters)

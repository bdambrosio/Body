"""ImuPlusScanMatchPose — pose source fusing encoder translation,
BNO085 yaw, and lidar scan-match against the world grid.

Drops in for OdomPose when the --slam flag is on. Subscribes to
body/imu and body/lidar/scan on the fuser's session via a connect()
call from FuserController; exposes the same PoseSource interface so
fusion / publish / UI paths don't change.

Pose model
----------
- Translation: from body/odom (encoder integration on Pi) via an
  internal OdomPose, with the same world-frame offset semantics.
- Yaw: from body/imu via an ImuYawTracker, with a yaw_offset captured
  at rebind_world_to_current() so the world frame's +x is the robot's
  heading at reset (regardless of GAME_ROTATION_VECTOR's arbitrary
  boot heading).
- Corrections: every ~match_hz scans, run ScanMatcher against the
  WorldGrid evidence, starting from the IMU-aided prior. If the
  acceptance gates pass, slew-limit and apply the correction by
  rewriting both the underlying OdomPose offset and yaw_offset so
  future queries at any timestamp return the corrected lineage.

Threading
---------
- update(): odom callback thread (existing FuserController wiring).
- _on_imu(): zenoh subscriber thread.
- _on_scan(): zenoh subscriber thread; runs synchronous scan-match
  (~200 ms at default cfg, well below the 500 ms match period).
- pose_at() / latest_pose() / rebind_world_to_current(): fuser
  thread + UI thread.

All public methods take an internal lock; underlying OdomPose and
ImuYawTracker have their own.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from desktop.nav.slam.imu_yaw import ImuYawTracker
from desktop.nav.slam.scan_matcher import (
    ScanMatcher, ScanMatcherConfig, lidar_scan_to_xy,
)
from desktop.nav.slam.types import ImuReading, Pose2D

from .pose_source import OdomPose, Pose, PoseSource

logger = logging.getLogger(__name__)


@dataclass
class ImuScanPoseConfig:
    # Run scan-match at most this often. 2 Hz against a 10 Hz lidar
    # = match every 5th scan. Default search at this rate is well
    # under the scan period.
    match_hz: float = 2.0

    # Skip matching against an almost-empty grid — correlation gives
    # garbage from sparse evidence, and during the first second or
    # two of mapping the grid hasn't accumulated enough yet.
    min_grid_evidence_cells: int = 200

    # Acceptance: the matcher's improvement (best_score − prior_score)
    # must exceed this. Default tuned conservative for GAME_RV from
    # shadow logs (smallest accepted improvement was ~+15).
    min_improvement: float = 10.0

    # Slew limits: corrections larger than these caps are rejected.
    # Bad matches that escape the search-bounds check should still
    # not produce visible map seams. Caps are chosen larger than
    # plausible per-step honest corrections, smaller than disasters.
    max_translation_correction_m: float = 0.30
    max_rotation_correction_rad: float = math.radians(8.0)


class ImuPlusScanMatchPose(PoseSource):
    IMU_TOPIC = "body/imu"
    SCAN_TOPIC = "body/lidar/scan"

    def __init__(
        self,
        *,
        config: Optional[ImuScanPoseConfig] = None,
        scan_matcher_config: Optional[ScanMatcherConfig] = None,
    ) -> None:
        self.config = config or ImuScanPoseConfig()
        self._lock = threading.RLock()

        self._odom = OdomPose()
        self._imu_yaw = ImuYawTracker()
        self._matcher = ScanMatcher(scan_matcher_config or ScanMatcherConfig())

        # World yaw = wrap(imu_yaw - yaw_offset). Captured at
        # rebind_world_to_current(); zero until the first reset.
        self._yaw_offset = 0.0

        # Zenoh wiring (filled by connect()).
        self._session: Optional[Any] = None
        self._grid: Optional[Any] = None
        self._subs: list = []

        # Match-rate limiter (monotonic clock).
        self._last_match_mono: float = 0.0

        # Telemetry counters surfaced via match_summary().
        self._n_match_attempted = 0
        self._n_match_accepted = 0
        self._n_match_exhausted = 0
        self._n_match_rejected_low_imp = 0
        self._n_match_rejected_too_large = 0
        self._n_skipped_sparse_grid = 0
        self._n_skipped_no_prior = 0
        self._n_skipped_imu_unsettled = 0
        # Cumulative correction magnitudes since last session reset.
        # Surfaced via correction_summary() and the operator status
        # strip; reset in rebind_world_to_current().
        self._correction_total_m = 0.0
        self._correction_total_rad = 0.0
        self._correction_n_applied = 0

    # ── PoseSource interface ─────────────────────────────────────────

    def update(self, ts: float, x: float, y: float, theta: float) -> None:
        self._odom.update(ts, x, y, theta)

    def latest_pose(self) -> Optional[Tuple[Pose, float]]:
        latest = self._odom.latest_pose()
        if latest is None:
            return None
        (x_w, y_w, theta_enc), ts = latest
        yaw = self._yaw_at_world(ts)
        if yaw is None:
            return ((x_w, y_w, theta_enc), ts)
        return ((x_w, y_w, yaw), ts)

    def pose_at(self, ts: float) -> Optional[Pose]:
        odom_pose = self._odom.pose_at(ts)
        if odom_pose is None:
            return None
        x_w, y_w, theta_enc = odom_pose
        yaw = self._yaw_at_world(ts)
        if yaw is None:
            return (x_w, y_w, theta_enc)
        return (x_w, y_w, yaw)

    def rebind_world_to_current(self) -> Optional[Pose]:
        """Anchor world frame at the current robot pose.
        - OdomPose offset: encoder pose at this moment becomes world
          origin (translation transform).
        - yaw_offset: IMU yaw at this moment becomes world heading 0,
          so +x = "forward at reset" regardless of GAME_RV's boot yaw.
        Pre-settle resets capture yaw_offset = 0; queries will fall
        back to encoder yaw until the tracker settles, then start
        returning offset-aligned IMU yaw.
        """
        with self._lock:
            self._odom.rebind_world_to_current()
            latest_imu = self._imu_yaw.latest()
            if latest_imu is not None:
                _ts, yaw, _sigma = latest_imu
                self._yaw_offset = yaw
            else:
                self._yaw_offset = 0.0
            # Reset accumulated correction so the displayed total
            # tracks scan-match work *this* session, not all-time.
            self._correction_total_m = 0.0
            self._correction_total_rad = 0.0
            self._correction_n_applied = 0
        latest = self._odom.latest_pose()
        return latest[0] if latest is not None else None

    def to_world(self, x_o: float, y_o: float, th_o: float) -> Pose:
        """Translation transform via OdomPose; theta passed through
        (consumers reading world theta should use pose_at, which
        substitutes IMU yaw)."""
        return self._odom.to_world(x_o, y_o, th_o)

    def source_name(self) -> str:
        return "imu+scan_match"

    def buffer_span(self):
        return self._odom.buffer_span()

    # ── Connection ──────────────────────────────────────────────────

    def connect(self, session: Any, grid: Any) -> None:
        """Wire to the live zenoh session + WorldGrid. Called by
        FuserController after open_session; safe to call once."""
        if self._subs:
            return
        self._session = session
        self._grid = grid
        self._subs.append(session.declare_subscriber(self.IMU_TOPIC, self._on_imu))
        self._subs.append(session.declare_subscriber(self.SCAN_TOPIC, self._on_scan))
        logger.info(
            f"imu_scan_pose: subscribed to {self.IMU_TOPIC} + "
            f"{self.SCAN_TOPIC} (match_hz={self.config.match_hz})"
        )

    def disconnect(self) -> None:
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                logger.debug("subscriber undeclare failed", exc_info=True)
        self._subs.clear()
        self._session = None
        self._grid = None
        logger.info(f"imu_scan_pose: disconnected. {self.match_summary()}")

    def match_summary(self) -> dict:
        return {
            "attempted": self._n_match_attempted,
            "accepted": self._n_match_accepted,
            "exhausted": self._n_match_exhausted,
            "rejected_low_imp": self._n_match_rejected_low_imp,
            "rejected_too_large": self._n_match_rejected_too_large,
            "skipped_sparse_grid": self._n_skipped_sparse_grid,
            "skipped_no_prior": self._n_skipped_no_prior,
            "skipped_imu_unsettled": self._n_skipped_imu_unsettled,
        }

    def correction_summary(self) -> dict:
        with self._lock:
            return {
                "total_m": float(self._correction_total_m),
                "total_rad": float(self._correction_total_rad),
                "n_applied": int(self._correction_n_applied),
            }

    # ── Subscriber callbacks ────────────────────────────────────────

    def _on_imu(self, sample: Any) -> None:
        try:
            msg = json.loads(_payload_bytes(sample).decode("utf-8"))
        except Exception:
            logger.debug("imu decode failed", exc_info=True)
            return
        reading = ImuReading.from_payload(msg)
        if reading is None:
            return
        self._imu_yaw.update(reading)

    def _on_scan(self, sample: Any) -> None:
        # Rate-limit before doing any decode work.
        now_mono = time.monotonic()
        period = 1.0 / max(0.1, self.config.match_hz)
        if now_mono - self._last_match_mono < period:
            return
        self._last_match_mono = now_mono

        try:
            msg = json.loads(_payload_bytes(sample).decode("utf-8"))
        except Exception:
            logger.debug("scan decode failed", exc_info=True)
            return

        ts = float(msg.get("ts") or time.time())
        ranges = msg.get("ranges")
        angle_min = float(msg.get("angle_min", 0.0))
        angle_inc = float(msg.get("angle_increment", 0.0))
        if not isinstance(ranges, list) or angle_inc <= 0:
            return

        if self._grid is None:
            return
        block_votes = self._grid.snapshot_block_votes()
        evidence_count = int((block_votes > 0).sum())
        if evidence_count < self.config.min_grid_evidence_cells:
            self._n_skipped_sparse_grid += 1
            return

        # Prior: world-frame pose at scan ts (uses IMU yaw if settled).
        prior_xytheta = self.pose_at(ts)
        if prior_xytheta is None:
            self._n_skipped_no_prior += 1
            return
        if not self._imu_yaw.is_settled():
            # Don't run scan-match yet — we'd be matching with a
            # drifting encoder yaw and pulling the map out of sync.
            self._n_skipped_imu_unsettled += 1
            return
        prior_pose = Pose2D(*prior_xytheta)

        angles = np.arange(len(ranges), dtype=np.float64) * angle_inc + angle_min
        ranges_arr = np.asarray(
            [r if isinstance(r, (int, float)) else np.nan for r in ranges],
            dtype=np.float64,
        )
        try:
            points_xy = lidar_scan_to_xy(ranges_arr, angles)
        except Exception:
            logger.debug("scan→xy failed", exc_info=True)
            return
        if points_xy.shape[0] < 10:
            return

        try:
            result = self._matcher.search(
                points_xy,
                prior_pose,
                block_votes,
                self._grid.origin_x_m,
                self._grid.origin_y_m,
                self._grid.resolution_m,
            )
        except Exception:
            logger.exception("scan_match crashed; continuing")
            return

        self._n_match_attempted += 1
        if result.search_exhausted:
            self._n_match_exhausted += 1
            return
        if result.improvement < self.config.min_improvement:
            self._n_match_rejected_low_imp += 1
            return

        best = result.pose
        dx = best.x - prior_pose.x
        dy = best.y - prior_pose.y
        dth = _wrap(best.theta - prior_pose.theta)
        if (math.hypot(dx, dy) > self.config.max_translation_correction_m
                or abs(dth) > self.config.max_rotation_correction_rad):
            self._n_match_rejected_too_large += 1
            logger.info(
                f"imu_scan_pose: rejected jump trans={math.hypot(dx, dy):.2f} m "
                f"dθ={math.degrees(dth):+.1f}° "
                f"(caps {self.config.max_translation_correction_m:.2f} m, "
                f"{math.degrees(self.config.max_rotation_correction_rad):.1f}°)"
            )
            return

        self._apply_correction((best.x, best.y, best.theta), ts)
        self._n_match_accepted += 1
        with self._lock:
            self._correction_total_m += math.hypot(dx, dy)
            self._correction_total_rad += abs(dth)
            self._correction_n_applied += 1

    # ── Correction application ──────────────────────────────────────

    def _apply_correction(self, corrected: Pose, ts: float) -> None:
        """Rewrite OdomPose's offset and the yaw offset so that future
        queries at *any* timestamp return poses consistent with the
        corrected pose at scan ts.

        Translation: solve for (off_x, off_y) such that
            to_world(odom_at_ts) = (corrected.x, corrected.y, _)
        keeping off_theta unchanged (encoder θ governs the rotation
        of the translation transform; world θ comes from IMU yaw
        independently, so adjusting off_theta would just double-correct
        when we re-pin yaw_offset below).

        Yaw: re-pin yaw_offset so wrap(imu_yaw_at_ts - offset) ==
        corrected.theta.
        """
        x_target, y_target, theta_target = corrected
        odom_in_odom = self._odom.pose_at_in_odom_frame(ts)
        if odom_in_odom is None:
            # No interpolatable odom sample at scan ts — skip; the
            # next match will pick this up. Counted as accepted by
            # the caller; the correction just had no anchor to apply
            # against. Rare in practice (odom is 50 Hz vs scan 2 Hz).
            return
        x_o, y_o, _th_o = odom_in_odom

        with self._lock:
            with self._odom._lock:
                off_theta = self._odom._off_theta
                # OdomPose.to_world: world = R(-off_theta) · (odom - off).
                # Solving for off given world target:
                #   odom - off = R(off_theta) · target
                #   off = odom - R(off_theta) · target
                co, so = math.cos(off_theta), math.sin(off_theta)
                self._odom._off_x = x_o - (co * x_target - so * y_target)
                self._odom._off_y = y_o - (so * x_target + co * y_target)
                # off_theta unchanged.

            yaw_at_ts = self._imu_yaw.yaw_at(ts)
            if yaw_at_ts is not None:
                yaw_imu, _sigma = yaw_at_ts
                self._yaw_offset = yaw_imu - theta_target

    # ── Internals ───────────────────────────────────────────────────

    def _yaw_at_world(self, ts: float) -> Optional[float]:
        result = self._imu_yaw.yaw_at(ts)
        if result is None:
            return None
        yaw_imu, _sigma = result
        return _wrap(yaw_imu - self._yaw_offset)


def _payload_bytes(sample: Any) -> bytes:
    try:
        return bytes(sample.payload.to_bytes())
    except AttributeError:
        return bytes(sample.payload)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

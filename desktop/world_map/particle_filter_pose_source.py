"""ParticleFilterPoseSource — Phase 8 promotion of ParticleFilterPose
to the production PoseSource. Drop-in for OdomPose / ImuPlusScanMatchPose.

This is the production-mode wrapper. The pure filter math lives in
``particle_filter_pose.py``; this file adds the PoseSource interface
plus the zenoh wiring (body/imu, body/lidar/scan) that ``shadow_pf_driver``
duplicates for shadow mode. The shadow driver remains the right tool
for diagnostic side-by-side comparison; this class is what
``FuserController`` instantiates when ``pose_source_type == "particle"``.

Frame handling mirrors OdomPose: the filter lives in a world frame
anchored at the most recent ``rebind_world_to_current``. At seed time
we capture (odom_pose_at_seed, imu_yaw_at_seed) so that ``to_world``
can convert any odom-frame Pi message anchor to world frame using
the same offset OdomPose uses.

What ``pose_at(ts)`` returns
----------------------------
The filter integrates incrementally and doesn't keep a temporal
buffer of past posteriors. For the typical fuser usage (pose lookup
at a recent scan timestamp), returning ``posterior_mean()`` is fine —
the time skew is bounded by the odom inter-arrival (~20 ms) plus the
caller's own buffering. If/when sub-ms accuracy at past timestamps
matters (it doesn't today), add a small (ts, pose) ring buffer here.

Threading
---------
All public methods take ``_lock`` (RLock). The lock is also held
during predict / observe inside the IMU and odom paths so
``pose_at`` queries can't race with filter mutation. Held for
microseconds; never across network I/O or scan-match.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from desktop.nav.slam.imu_yaw import ImuYawTracker
from desktop.nav.slam.scan_matcher import (
    ScanMatcher, ScanMatcherConfig, lidar_scan_to_xy,
)
from desktop.nav.slam.types import ImuReading, Pose2D

from .particle_filter_pose import (
    ParticleFilterConfig, ParticleFilterPose,
)
from .pose_source import Pose, PoseSource

logger = logging.getLogger(__name__)


@dataclass
class ParticleFilterPoseSourceConfig:
    # Scan-likelihood rate cap. 2 Hz matches production's
    # ImuPlusScanMatchPose; vectorized scan_matcher fits inside the
    # 200 ms heartbeat period at this rate.
    scan_hz: float = 2.0

    # Skip scan-likelihood if grid evidence is too sparse — same
    # convention as ImuPlusScanMatchPose / shadow driver.
    min_grid_evidence_cells: int = 200

    # Teleport thresholds — guard against pathological odom jumps.
    teleport_distance_m: float = 0.5
    teleport_rotation_rad: float = math.radians(45.0)


class ParticleFilterPoseSource(PoseSource):
    """Particle-filter pose source for production use."""

    IMU_TOPIC = "body/imu"
    SCAN_TOPIC = "body/lidar/scan"

    def __init__(
        self,
        *,
        pf_config: Optional[ParticleFilterConfig] = None,
        scan_matcher_config: Optional[ScanMatcherConfig] = None,
        config: Optional[ParticleFilterPoseSourceConfig] = None,
    ) -> None:
        self._pf = ParticleFilterPose(pf_config)
        # Wider θ search (±12°) than production's classical scan-match
        # default (±8°) — empirically eliminated scan_exhausted hits
        # during rotation in shadow-mode validation. Vectorization
        # absorbs the ~40% cost increase.
        if scan_matcher_config is None:
            scan_matcher_config = ScanMatcherConfig(
                theta_half_rad=math.radians(12.0),
            )
        self._matcher = ScanMatcher(scan_matcher_config)
        self._imu_tracker = ImuYawTracker()
        self._config = config or ParticleFilterPoseSourceConfig()

        self._lock = threading.RLock()
        self._scan_rate_lock = threading.Lock()

        # Seed bookkeeping. Filter only operates after we have an
        # odom sample and a settled IMU reading at the same ts.
        self._seeded: bool = False
        # (ts, x_o, y_o, θ_o) — most recent raw odom in odom frame.
        self._last_odom: Optional[Tuple[float, float, float, float]] = None
        # Offset captured at seed/rebind: world_pose = inv(odom_offset) · odom_pose.
        # Stored as the odom pose that maps to world (0, 0, 0).
        self._off_x: float = 0.0
        self._off_y: float = 0.0
        self._off_theta: float = 0.0
        # yaw_offset = imu_yaw_at_seed - world_yaw_at_seed
        self._yaw_offset: float = 0.0

        # Zenoh handles
        self._session: Optional[Any] = None
        self._grid: Optional[Any] = None
        self._subs: List[Any] = []
        self._last_scan_mono: float = 0.0

        # Counters surfaced via match_summary().
        self._counters: Dict[str, int] = {
            "odom_seen": 0,
            "odom_skipped_no_imu": 0,
            "predicts_run": 0,
            "teleports": 0,
            "imu_received": 0,
            "imu_malformed": 0,
            "scan_received": 0,
            "scan_skipped_rate_limit": 0,
            "scan_skipped_sparse_grid": 0,
            "scan_obs_run": 0,
            "resamples_fired": 0,
        }
        # Cumulative scan-match correction magnitudes since last reset
        # (informational only; the filter applies updates as Bayesian
        # reweights rather than hard pose-deltas, but we surface
        # something equivalent so the operator UI's "correction
        # accumulated" readout keeps working).
        self._correction_total_m: float = 0.0
        self._correction_total_rad: float = 0.0
        self._correction_n_applied: int = 0

    # ── PoseSource interface ─────────────────────────────────────────

    def update(self, ts: float, x: float, y: float, theta: float) -> None:
        """Called by FuserController on every body/odom sample.

        Drives the filter's predict step + IMU observation. Seeds on
        the first sample where IMU has data, defers otherwise.
        """
        self._counters["odom_seen"] += 1
        with self._lock:
            if not self._seeded:
                imu = self._imu_tracker.yaw_at(ts)
                if imu is None:
                    self._counters["odom_skipped_no_imu"] += 1
                    return
                imu_yaw, _ = imu
                # World (0, 0, 0) at seed time. The filter operates
                # in that world frame from here on.
                self._pf.seed_at(0.0, 0.0, 0.0)
                self._yaw_offset = _wrap(imu_yaw - 0.0)
                self._off_x, self._off_y, self._off_theta = x, y, theta
                self._last_odom = (ts, x, y, theta)
                self._seeded = True
                logger.info(
                    "particle_pose: seeded at odom (%+.2f, %+.2f, %+.1f°), "
                    "yaw_offset=%+.1f°",
                    x, y, math.degrees(theta), math.degrees(self._yaw_offset),
                )
                return

            assert self._last_odom is not None
            _, last_xo, last_yo, last_tho = self._last_odom
            dx = x - last_xo
            dy = y - last_yo
            dth = _wrap(theta - last_tho)

            if (math.hypot(dx, dy) > self._config.teleport_distance_m
                    or abs(dth) > self._config.teleport_rotation_rad):
                logger.info(
                    "particle_pose: odom teleport detected "
                    "dist=%.2f m dθ=%.1f° → skip step",
                    math.hypot(dx, dy), math.degrees(dth),
                )
                self._counters["teleports"] += 1
                self._last_odom = (ts, x, y, theta)
                return

            # Body-frame forward displacement (frame-invariant under
            # constant rigid offsets between odom and world).
            th_mid = last_tho + 0.5 * dth
            ds = dx * math.cos(th_mid) + dy * math.sin(th_mid)
            self._pf.predict(ds, dth)

            imu = self._imu_tracker.yaw_at(ts)
            if imu is not None:
                imu_yaw, _imu_sigma = imu
                world_yaw = _wrap(imu_yaw - self._yaw_offset)
                # cfg.imu_sigma_rad (5 mrad default) — see commit 6d8dc2d.
                self._pf.observe_imu_yaw(world_yaw)

            self._counters["predicts_run"] += 1
            self._last_odom = (ts, x, y, theta)

    def latest_pose(self) -> Optional[Tuple[Pose, float]]:
        with self._lock:
            if not self._seeded or self._last_odom is None:
                return None
            x, y, theta = self._pf.posterior_mean()
            return ((x, y, theta), self._last_odom[0])

    def pose_at(self, ts: float) -> Optional[Pose]:
        latest = self.latest_pose()
        if latest is None:
            return None
        # No temporal buffer of past posteriors. Return the current
        # posterior; ts is usually within a frame of latest_pose's ts
        # and the difference is below filter noise.
        return latest[0]

    def rebind_world_to_current(self) -> Optional[Pose]:
        with self._lock:
            if not self._seeded or self._last_odom is None:
                return None
            ts, x, y, theta = self._last_odom
            imu = self._imu_tracker.yaw_at(ts)
            self._pf.seed_at(0.0, 0.0, 0.0)
            self._off_x, self._off_y, self._off_theta = x, y, theta
            if imu is not None:
                imu_yaw, _ = imu
                self._yaw_offset = _wrap(imu_yaw)
            self._correction_total_m = 0.0
            self._correction_total_rad = 0.0
            self._correction_n_applied = 0
            logger.info(
                "particle_pose: rebound to current — new offset "
                "odom=(%+.2f, %+.2f, %+.1f°)",
                x, y, math.degrees(theta),
            )
            return (x, y, theta)

    def to_world(self, x_o: float, y_o: float, th_o: float) -> Pose:
        """Same logic as OdomPose: world = R(-off_theta) · (odom - off)."""
        dx = x_o - self._off_x
        dy = y_o - self._off_y
        c, s = math.cos(-self._off_theta), math.sin(-self._off_theta)
        x_w = c * dx - s * dy
        y_w = s * dx + c * dy
        th_w = _wrap(th_o - self._off_theta)
        return (x_w, y_w, th_w)

    def cov_at(self, ts: float) -> Optional[np.ndarray]:
        with self._lock:
            if not self._seeded:
                return None
            cov_t = self._pf.posterior_cov()
            return cov_t.detach().cpu().numpy()

    def source_name(self) -> str:
        return "particle"

    def match_summary(self) -> dict:
        return dict(self._counters)

    def correction_summary(self) -> dict:
        with self._lock:
            return {
                "total_m": float(self._correction_total_m),
                "total_rad": float(self._correction_total_rad),
                "n_applied": int(self._correction_n_applied),
            }

    # ── Zenoh wiring ─────────────────────────────────────────────────

    def connect(self, session: Any, grid: Any) -> None:
        """Wire to the live zenoh session + WorldGrid. Called by
        FuserController after open_session."""
        if self._subs:
            return
        self._session = session
        self._grid = grid
        self._subs.append(
            session.declare_subscriber(self.IMU_TOPIC, self._on_imu),
        )
        self._subs.append(
            session.declare_subscriber(self.SCAN_TOPIC, self._on_scan),
        )
        logger.info(
            "particle_pose: subscribed to %s + %s (scan_hz=%.2f)",
            self.IMU_TOPIC, self.SCAN_TOPIC, self._config.scan_hz,
        )

    def disconnect(self) -> None:
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                logger.debug("particle_pose: sub undeclare failed", exc_info=True)
        self._subs.clear()
        self._session = None
        self._grid = None
        logger.info("particle_pose: disconnected. counters=%s", self._counters)

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
        self._imu_tracker.update(reading)

    def _on_scan(self, sample: Any) -> None:
        self._counters["scan_received"] += 1
        with self._scan_rate_lock:
            now_mono = time.monotonic()
            min_period = 1.0 / max(0.1, self._config.scan_hz)
            if now_mono - self._last_scan_mono < min_period:
                self._counters["scan_skipped_rate_limit"] += 1
                return
            self._last_scan_mono = now_mono

        try:
            msg = json.loads(self._payload_bytes(sample).decode("utf-8"))
        except Exception:
            return
        ranges = msg.get("ranges")
        angle_min = float(msg.get("angle_min", 0.0))
        angle_inc = float(msg.get("angle_increment", 0.0))
        if not isinstance(ranges, list) or angle_inc <= 0:
            return

        with self._lock:
            if not self._seeded:
                return
            posterior = self._pf.posterior_mean()
        prior_pose = Pose2D(*posterior)

        if self._grid is None:
            return
        block_votes = self._grid.snapshot_block_votes()
        evidence_count = int((block_votes > 0).sum())
        if evidence_count < self._config.min_grid_evidence_cells:
            self._counters["scan_skipped_sparse_grid"] += 1
            return

        angles = np.arange(len(ranges), dtype=np.float64) * angle_inc + angle_min
        ranges_arr = np.asarray(
            [r if isinstance(r, (int, float)) else np.nan for r in ranges],
            dtype=np.float64,
        )
        try:
            points_xy = lidar_scan_to_xy(ranges_arr, angles)
        except Exception:
            return
        if points_xy.shape[0] < 10:
            return

        try:
            result = self._matcher.search(
                points_xy, prior_pose, block_votes,
                self._grid.origin_x_m, self._grid.origin_y_m,
                self._grid.resolution_m,
                return_field=True,
            )
        except Exception:
            logger.exception("particle_pose: scan_match crashed; continuing")
            return

        with self._lock:
            if result.score_field is not None:
                self._pf.update_from_scan_likelihood(
                    result.score_field, prior_pose,
                )
            resampled = self._pf.maybe_resample()
            if resampled:
                self._counters["resamples_fired"] += 1
            # Track the magnitude of the argmax shift for the operator
            # readout. This isn't the actual filter update (which is
            # a Bayesian reweight, not a snap), but it's a reasonable
            # proxy for "scan-match correction magnitude" — what the
            # legacy ImuPlusScanMatchPose's correction_summary exposes.
            dx = result.pose.x - prior_pose.x
            dy = result.pose.y - prior_pose.y
            dth = _wrap(result.pose.theta - prior_pose.theta)
            self._correction_total_m += math.hypot(dx, dy)
            self._correction_total_rad += abs(dth)
            self._correction_n_applied += 1
        self._counters["scan_obs_run"] += 1


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

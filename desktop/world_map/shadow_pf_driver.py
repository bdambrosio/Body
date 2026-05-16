"""Phase 2.4 — Shadow-mode driver for ParticleFilterPose.

Subscribes to body/odom and body/lidar/scan on the fuser's Zenoh
session, runs the particle filter in parallel with the production
pose source, and writes a JSONL trace comparing legacy pose to filter
posterior on every scan tick. **No side effects on the fuser** —
purely observational, like ``ShadowSlamDriver``.

Frame handling
--------------
The filter lives in production's world frame. On each odom sample we
read ``pose_source.pose_at(ts)`` to get the IMU-corrected world-frame
pose, integrate the increment as ``(Δs, Δθ)`` for the motion model,
and apply the production yaw as an IMU-style observation. Scan
observations score against the live ``WorldGrid`` block_votes via
``ScanMatcher.search(return_field=True)``.

Teleport handling
-----------------
A "rebind world to current" in production teleports the world pose to
the origin. We detect that via a per-step displacement threshold and
re-seed the filter rather than try to absorb it through the motion
model.

Threading
---------
Zenoh callbacks fire on the session's threads. We hold ``_pf_lock``
across every filter mutation (predict, observe, resample, posterior
reads for the trace). The lock is per-instance and never held during
network I/O. Scan-match (~50 ms) is the most expensive step; well
under the 500 ms 2 Hz scan period.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np

from desktop.nav.slam.imu_yaw import ImuYawTracker
from desktop.nav.slam.scan_matcher import (
    ScanMatcher, ScanMatcherConfig, lidar_scan_to_xy,
)
from desktop.nav.slam.types import ImuReading, Pose2D

from .particle_filter_pose import (
    ParticleFilterConfig, ParticleFilterPose,
)

logger = logging.getLogger(__name__)


@dataclass
class ShadowPfConfig:
    # Scan-likelihood rate cap. Dropped from 2.0 → 0.5 after the first
    # live trace showed CPU saturation: production scan-match runs every
    # 500 ms and ours did too (~400 ms each on CPU), saturating one
    # core and starving the chassis heartbeat thread → Pi watchdog →
    # bot e-stop. 0.5 Hz halves shadow's cost; the filter has plenty
    # of information at 1 obs / 2 s and is largely driven by predict +
    # IMU between scan ticks anyway.
    scan_hz: float = 0.5

    # Skip scan-likelihood if grid evidence is too sparse — same logic
    # as the production matcher.
    min_grid_evidence_cells: int = 200

    # Teleport thresholds for rebind detection. Anything beyond these
    # in a single odom step almost certainly came from a frame reset,
    # not honest motion at 50 Hz odom.
    teleport_distance_m: float = 0.5
    teleport_rotation_rad: float = math.radians(45.0)

    # JSONL trace flush cadence. Buffered writes amortize the syscall
    # cost; flushed on disconnect or every N scan records.
    trace_flush_every: int = 20


class ShadowParticleFilterDriver:
    """Observational driver — particle filter running parallel to the
    production pose source. Writes a JSONL trace; the production
    fuser is untouched.
    """

    ODOM_TOPIC = "body/odom"
    IMU_TOPIC = "body/imu"
    SCAN_TOPIC = "body/lidar/scan"

    def __init__(
        self,
        *,
        session: Any,
        grid: Any,
        pose_source: Any,
        trace_path: Path,
        pf_config: Optional[ParticleFilterConfig] = None,
        scan_matcher_config: Optional[ScanMatcherConfig] = None,
        config: Optional[ShadowPfConfig] = None,
    ) -> None:
        self._session = session
        self._grid = grid
        self._pose_source = pose_source
        self._trace_path = Path(trace_path)
        self._config = config or ShadowPfConfig()

        self._pf = ParticleFilterPose(pf_config)
        self._matcher = ScanMatcher(scan_matcher_config or ScanMatcherConfig())
        # Own IMU tracker — decouples us from production's pose_source
        # buffer. The first live trace showed 30% of odom callbacks
        # racing ahead of the fuser's _on_odom, returning None from
        # pose_at(ts), and skipping predicts. Subscribing to body/imu
        # ourselves removes the race; we feed the tracker on _on_imu
        # and query it directly on each predict.
        self._imu_tracker = ImuYawTracker()

        self._pf_lock = threading.RLock()
        # Lock around the scan-rate gate. Without it, multiple zenoh
        # threads can race past the "elapsed >= period" check while
        # the previous callback is still inside the 400 ms scan-match.
        # The first live trace logged scan records 100 ms apart for
        # this reason.
        self._scan_rate_lock = threading.Lock()
        self._subs: List[Any] = []
        self._trace_fp: Optional[TextIO] = None
        self._trace_pending = 0
        # (ts, x_o, y_o, θ_o) — last raw odom-frame sample. We integrate
        # in odom frame and apply the seed-time world offset; this avoids
        # any dependency on production's pose buffer being current.
        self._last_odom: Optional[Tuple[float, float, float, float]] = None
        # Seed bookkeeping. The seed needs both a world-frame pose
        # snapshot (from production at seed-ts) and an IMU yaw reading
        # (from our tracker at seed-ts) so the filter is anchored
        # consistently in world frame.
        self._seeded = False
        self._yaw_offset: Optional[float] = None  # imu_yaw_seed - world_yaw_seed
        self._last_scan_mono: float = 0.0

        self._counters: Dict[str, int] = {
            "odom_received": 0,
            "odom_malformed": 0,
            "predicts_run": 0,
            "predicts_skipped_no_seed": 0,
            "imu_received": 0,
            "imu_malformed": 0,
            "teleports_detected": 0,
            "scan_received": 0,
            "scan_malformed": 0,
            "scan_obs_run": 0,
            "scan_obs_skipped_rate_limit": 0,
            "scan_obs_skipped_sparse_grid": 0,
            "scan_obs_skipped_no_seed": 0,
            "resamples_fired": 0,
            "trace_records_written": 0,
        }

    # ── Lifecycle ────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._subs:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_fp = self._trace_path.open("a", buffering=1)  # line-buffered
        # Header record — lets a downstream notebook tell where a new
        # run starts in a long-running trace file.
        self._write_trace({
            "type": "session_start",
            "ts": time.time(),
            "n_particles": self._pf.n_particles(),
            "scan_hz": self._config.scan_hz,
        })

        self._subs.append(
            self._session.declare_subscriber(self.ODOM_TOPIC, self._on_odom),
        )
        self._subs.append(
            self._session.declare_subscriber(self.IMU_TOPIC, self._on_imu),
        )
        self._subs.append(
            self._session.declare_subscriber(self.SCAN_TOPIC, self._on_scan),
        )
        logger.info(
            "shadow_pf: subscribed to %s + %s + %s, trace=%s, scan_hz=%.2f",
            self.ODOM_TOPIC, self.IMU_TOPIC, self.SCAN_TOPIC,
            self._trace_path, self._config.scan_hz,
        )

    def disconnect(self) -> None:
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                logger.debug("shadow_pf: subscriber undeclare failed", exc_info=True)
        self._subs.clear()
        if self._trace_fp is not None:
            self._write_trace({"type": "session_end", "ts": time.time()})
            try:
                self._trace_fp.flush()
                self._trace_fp.close()
            except Exception:
                logger.debug("shadow_pf: trace close failed", exc_info=True)
            self._trace_fp = None
        logger.info("shadow_pf: disconnected. counters=%s", self._counters)

    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

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

    def _on_odom(self, sample: Any) -> None:
        self._counters["odom_received"] += 1
        try:
            msg = json.loads(self._payload_bytes(sample).decode("utf-8"))
        except Exception:
            self._counters["odom_malformed"] += 1
            return

        ts = float(msg.get("ts") or time.time())
        try:
            x_o = float(msg["x"])
            y_o = float(msg["y"])
            th_o = float(msg["theta"])
        except (KeyError, TypeError, ValueError):
            self._counters["odom_malformed"] += 1
            return

        with self._pf_lock:
            if not self._seeded:
                # Seeding requires three things at the same ts:
                #   (a) a raw odom sample — have it
                #   (b) production's world pose at ts (for the world
                #       offset and the comparison anchor)
                #   (c) IMU yaw at ts (to capture yaw_offset)
                # If (b) or (c) aren't ready yet (subscriber buffers
                # still warming up, or IMU tracker still settling),
                # defer the seed. The fuser doesn't read this driver's
                # output, so deferring is harmless.
                world = self._pose_source.pose_at(ts)
                imu = self._imu_tracker.yaw_at(ts)
                if world is None or imu is None:
                    self._counters["predicts_skipped_no_seed"] += 1
                    return
                x_w, y_w, th_w = world
                imu_yaw, _imu_sigma = imu
                self._yaw_offset = _wrap(imu_yaw - th_w)
                self._pf.seed_at(x_w, y_w, th_w)
                self._last_odom = (ts, x_o, y_o, th_o)
                self._seeded = True
                logger.info(
                    "shadow_pf: seeded at world (%+.2f, %+.2f, %+.1f°), "
                    "yaw_offset=%+.1f°",
                    x_w, y_w, math.degrees(th_w),
                    math.degrees(self._yaw_offset),
                )
                return

            assert self._last_odom is not None  # for the type-checker
            _last_ts, last_xo, last_yo, last_tho = self._last_odom
            dx = x_o - last_xo
            dy = y_o - last_yo
            dth = _wrap(th_o - last_tho)

            if (math.hypot(dx, dy) > self._config.teleport_distance_m
                    or abs(dth) > self._config.teleport_rotation_rad):
                # The Pi's odom integration shouldn't produce jumps this
                # large at 50 Hz. Treat as a glitch — log and skip.
                logger.info(
                    "shadow_pf: odom teleport detected dist=%.2f m dθ=%.1f° "
                    "→ skip this step (cloud unchanged)",
                    math.hypot(dx, dy), math.degrees(dth),
                )
                self._counters["teleports_detected"] += 1
                self._last_odom = (ts, x_o, y_o, th_o)
                return

            # Δs is the signed body-frame forward displacement, frame-
            # invariant — same in odom and world frames since they
            # differ by a constant rigid transform. Projected on the
            # midpoint heading in odom frame.
            th_mid = last_tho + 0.5 * dth
            ds = dx * math.cos(th_mid) + dy * math.sin(th_mid)

            self._pf.predict(ds, dth)
            # IMU observation: convert raw IMU yaw to world frame via
            # the seed-time offset. tracker.yaw_at returns (yaw, σ);
            # use σ as the observation σ floor so the obs auto-relaxes
            # when the IMU tracker reports lower confidence.
            imu = self._imu_tracker.yaw_at(ts)
            if imu is not None and self._yaw_offset is not None:
                imu_yaw, imu_sigma = imu
                world_yaw = _wrap(imu_yaw - self._yaw_offset)
                self._pf.observe_imu_yaw(
                    world_yaw, sigma_rad=max(imu_sigma, 1e-4),
                )

            self._counters["predicts_run"] += 1
            self._last_odom = (ts, x_o, y_o, th_o)

    def _on_scan(self, sample: Any) -> None:
        self._counters["scan_received"] += 1

        # Rate-limit before decode. Locked so simultaneous zenoh threads
        # can't both pass the gate while a previous callback is still
        # inside the 400 ms scan-match — the first live trace logged
        # records 100 ms apart for exactly this reason.
        with self._scan_rate_lock:
            now_mono = time.monotonic()
            min_period = 1.0 / max(0.1, self._config.scan_hz)
            if now_mono - self._last_scan_mono < min_period:
                self._counters["scan_obs_skipped_rate_limit"] += 1
                return
            self._last_scan_mono = now_mono

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

        with self._pf_lock:
            if not self._seeded:
                # Wait for the first odom to seed us.
                self._counters["scan_obs_skipped_no_seed"] += 1
                return
            posterior = self._pf.posterior_mean()
        prior_pose = Pose2D(*posterior)

        block_votes = self._grid.snapshot_block_votes()
        evidence_count = int((block_votes > 0).sum())
        if evidence_count < self._config.min_grid_evidence_cells:
            self._counters["scan_obs_skipped_sparse_grid"] += 1
            return

        angles = np.arange(len(ranges), dtype=np.float64) * angle_inc + angle_min
        ranges_arr = np.asarray(
            [r if isinstance(r, (int, float)) else np.nan for r in ranges],
            dtype=np.float64,
        )
        try:
            points_xy = lidar_scan_to_xy(ranges_arr, angles)
        except Exception:
            self._counters["scan_malformed"] += 1
            return
        if points_xy.shape[0] < 10:
            self._counters["scan_malformed"] += 1
            return

        t0 = time.monotonic()
        try:
            result = self._matcher.search(
                points_xy, prior_pose, block_votes,
                self._grid.origin_x_m, self._grid.origin_y_m,
                self._grid.resolution_m,
                return_field=True,
            )
        except Exception:
            logger.exception("shadow_pf: scan_match crashed; continuing")
            return
        match_elapsed_ms = (time.monotonic() - t0) * 1000.0

        with self._pf_lock:
            if result.score_field is not None:
                self._pf.update_from_scan_likelihood(
                    result.score_field, prior_pose,
                )
            resampled = self._pf.maybe_resample()
            if resampled:
                self._counters["resamples_fired"] += 1
            diagnostics = self._pf.diagnostics(resampled=resampled)
            filter_mean = self._pf.posterior_mean()
            cov = self._pf.posterior_cov()

        self._counters["scan_obs_run"] += 1

        legacy_pose = self._pose_source.pose_at(scan_ts)
        record: Dict[str, Any] = {
            "type": "scan_obs",
            "ts": scan_ts,
            "legacy_pose": (
                [float(legacy_pose[0]), float(legacy_pose[1]), float(legacy_pose[2])]
                if legacy_pose is not None else None
            ),
            "filter_mean": [float(filter_mean[0]), float(filter_mean[1]), float(filter_mean[2])],
            "filter_cov_diag": [
                float(cov[0, 0]), float(cov[1, 1]), float(cov[2, 2]),
            ],
            "n_eff": diagnostics.n_eff,
            "max_weight": diagnostics.max_weight,
            "weight_entropy": diagnostics.weight_entropy,
            "std_x": diagnostics.std_x,
            "std_y": diagnostics.std_y,
            "std_theta": diagnostics.std_theta,
            "resampled": diagnostics.resampled,
            "scan_score": float(result.score),
            "scan_score_prior": float(result.score_prior),
            "scan_improvement": float(result.improvement),
            "scan_exhausted": bool(result.search_exhausted),
            "scan_points_n": int(points_xy.shape[0]),
            "evidence_cells": evidence_count,
            "match_elapsed_ms": match_elapsed_ms,
        }
        self._write_trace(record)

    # ── Trace I/O ────────────────────────────────────────────────────

    def _write_trace(self, record: Dict[str, Any]) -> None:
        if self._trace_fp is None:
            return
        try:
            self._trace_fp.write(json.dumps(record) + "\n")
        except Exception:
            logger.debug("shadow_pf: trace write failed", exc_info=True)
            return
        self._trace_pending += 1
        self._counters["trace_records_written"] += 1
        if self._trace_pending >= self._config.trace_flush_every:
            try:
                self._trace_fp.flush()
            except Exception:
                logger.debug("shadow_pf: trace flush failed", exc_info=True)
            self._trace_pending = 0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

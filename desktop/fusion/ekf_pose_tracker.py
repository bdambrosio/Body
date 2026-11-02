"""2D EKF pose tracker — IMU predict, encoder odom + IMU yaw update.

When ``mag_idle_fusion_enabled`` (off by default, I5), applies an additional
θ update from ``body/imu.mag`` while the Pi marks ``mag.valid`` (motors-off
snapshots).

Robustness (fragility review S1/S6/X4/I2/I4):
- Encoder ``ds`` is NEVER dropped. When no IMU yaw covers the odom ts (WiFi
  jitter, IMU stale or restarting) the step is projected along the wheel-
  propagated heading (θ + ½·dθ_enc) and θ advances by dθ_enc; the next IMU
  yaw update pulls θ back. Counted in ``yaw_at_misses`` / ``wheel_heading_steps``.
- Odom stream resets (``session_id`` change / ``seq`` decrease / backward
  clock step) re-baseline instead of integrating the jump.
- IMU stream resets (ImuYawTracker epoch) re-bind the yaw offset to the
  current θ instead of stepping.
- Wheel-vs-IMU contradiction guard: a sliding window comparing wheel and IMU
  rotation; a frozen IMU (wheels turned, IMU flat) is distrusted — heading
  from wheels, with the missed rotation restored at trip time — until the
  IMU visibly moves again, then the offset is re-bound.
  The trip also opens a *contest*: wheels slipping or spinning against an
  obstacle look identical, so the pose source compares two lidar scans
  bracketing the window (desktop/localization/scan_rotation.py) and calls
  resolve_rotation_contest(); an "imu" verdict reverts θ to the IMU heading.
- State interpolation is wrap-safe in θ.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from desktop.fusion.load_slam_config import FusionNoiseConfig
from desktop.nav.slam.imu_yaw import ImuYawTracker
from desktop.nav.slam.types import ImuReading, quaternion_to_yaw
from desktop.utils.stream_epoch import StreamEpoch
from desktop.world_map.pose_source import OdomPose, Pose

logger = logging.getLogger(__name__)

PoseTuple = Tuple[float, float, float]
_OdomSample = Tuple[float, float, float, float]
_StateSample = Tuple[float, float, float, float, float, float, float, float, float, float]
# ts, x, y, theta, P00, P01, P02, P11, P12, P22


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class EkfPoseTrackerConfig:
    noise: FusionNoiseConfig
    buffer_seconds: float = 2.0


class EkfPoseTracker:
    """Fuse IMU yaw with encoder forward motion (Thrun diff-drive noise).

    The IMU owns heading; encoder contributes forward ``ds`` only, projected
    with IMU world yaw at each odom tick. Maintains a 3×3 covariance on
    (x, y, θ) for diagnostics and SLAM edge information.
    """

    def __init__(
        self,
        *,
        config: Optional[EkfPoseTrackerConfig] = None,
        noise: Optional[FusionNoiseConfig] = None,
        buffer_seconds: float = 2.0,
    ) -> None:
        if config is not None:
            self._noise = config.noise
            self._buf_seconds = config.buffer_seconds
        else:
            self._noise = noise or FusionNoiseConfig()
            self._buf_seconds = buffer_seconds

        self._lock = threading.RLock()
        self._odom = OdomPose(buffer_seconds=buffer_seconds)
        self._imu = ImuYawTracker(
            min_settle_samples=max(20, int(self._noise.imu_settle_time_s * 100)),
        )
        self._yaw_offset = 0.0
        self._mag_yaw_offset: Optional[float] = None
        self._last_mag_update_ts = -1.0
        self._last_mag_valid = False
        self._seeded = False
        self._yaw_at_misses = 0
        self._last_heading_source = "none"
        self._last_raw_odom: Optional[_OdomSample] = None
        self._state_buf: Deque[_StateSample] = deque(maxlen=512)

        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._P = np.eye(3, dtype=np.float64) * 1e-4

        self._predict_count = 0
        self._imu_update_count = 0
        self._mag_update_count = 0

        # Odom epoch (motor_controller restart / clock step) detection.
        self._odom_epoch = StreamEpoch()
        self._odom_resets = 0
        self._wheel_heading_steps = 0
        # IMU stream epoch last bound against (ImuYawTracker.epoch()).
        self._imu_epoch_seen = 0
        self._imu_rebind_pending = False
        # Wheel-vs-IMU contradiction guard state.
        self._guard_win: Deque[Tuple[float, float, float]] = deque()  # (ts, dθ_enc, dθ_imu)
        self._last_imu_world_yaw: Optional[float] = None
        self._imu_distrusted = False
        self._imu_guard_trips = 0
        # Bumped whenever the IMU→world yaw mapping is re-bound (IMU stream
        # reset, guard revival). Consumers that keep their own offset against
        # the raw IMU yaw (MCL) re-bind when it changes.
        self._imu_rebind_epoch = 0
        # Cumulative (ts, wheel yaw, IMU yaw) for rotation_between(); kept
        # independently of the odom/IMU buffers so a contest's window
        # (~1 s before the trip + scan lag) is always covered.
        self._rot_hist: Deque[Tuple[float, float, float]] = deque()
        self._rot_wheel_cum = 0.0
        self._rot_imu_cum = 0.0
        # Open wheel-vs-IMU contest awaiting a scan-to-scan verdict:
        # {"t0", "t1", "offset", "opened_ts"} or None.
        self._contest: Optional[Dict[str, float]] = None
        self._contest_counts = {"imu": 0, "wheels": 0, "unknown": 0, "expired": 0}

    def update_imu(self, reading: ImuReading) -> None:
        self._last_mag_valid = reading.mag_valid
        self._imu.update(reading)
        self._maybe_update_mag(reading)

    def update_odom(
        self, ts: float, x: float, y: float, theta: float, *,
        session_id: Optional[str] = None, seq: Optional[int] = None,
    ) -> bool:
        """Ingest one body/odom sample. ``session_id``/``seq`` are the
        optional odom epoch fields. Returns True when this sample started a
        new odom epoch (the EKF re-baselined instead of integrating the
        jump)."""
        reason = self._odom_epoch.check(session_id, seq, ts)
        if reason is not None:
            self._on_odom_reset(reason)
        self._odom.update(ts, x, y, theta, reset=reason is not None)
        if not self._seeded and self._imu.is_settled() and self._odom.latest_pose() is not None:
            self.rebind_world_to_current()
            return reason is not None
        if not self._seeded:
            return reason is not None
        self._integrate_odom(ts, x, y, theta)
        return reason is not None

    def _on_odom_reset(self, reason: str) -> None:
        with self._lock:
            self._odom_resets += 1
            # The next sample is the new baseline (no ds across the reset).
            self._last_raw_odom = None
            self._guard_win.clear()
            self._contest = None
            if reason == "clock_step":
                # Timestamps before/after the step are incomparable.
                self._state_buf.clear()
        logger.warning("ekf: odom stream reset (%s) — re-baselined, not integrated",
                       reason)

    def imu_trusted(self) -> bool:
        """False while the wheel-vs-IMU guard has the IMU distrusted."""
        with self._lock:
            return not self._imu_distrusted

    def imu_rebind_epoch(self) -> int:
        """Changes whenever the raw-IMU→world yaw mapping was re-bound."""
        with self._lock:
            return self._imu_rebind_epoch

    def rebind_world_to_current(self) -> Optional[Pose]:
        with self._lock:
            self._odom.rebind_world_to_current()
            latest_imu = self._imu.latest()
            if latest_imu is not None:
                _ts, yaw, _sigma = latest_imu
                self._yaw_offset = yaw
            else:
                self._yaw_offset = 0.0
            self._mag_yaw_offset = None
            self._last_mag_update_ts = -1.0
            self._last_mag_valid = False
            self._seeded = True
            self._yaw_at_misses = 0
            self._last_heading_source = "none"
            self._state_buf.clear()
            self._last_raw_odom = None
            self._guard_win.clear()
            self._last_imu_world_yaw = None
            self._imu_distrusted = False
            self._contest = None
            self._imu_epoch_seen = self._imu.epoch()
            self._imu_rebind_pending = False
            self._x = 0.0
            self._y = 0.0
            self._theta = 0.0
            self._P = np.eye(3, dtype=np.float64) * 1e-4

        latest = self._odom.latest_pose()
        if latest is None:
            return None
        _pose, ots = latest
        raw = self._odom.pose_at_in_odom_frame(ots)
        if raw is not None:
            yaw = self._yaw_at_world(ots)
            if yaw is not None:
                with self._lock:
                    self._theta = yaw
                    self._last_raw_odom = (ots, raw[0], raw[1], raw[2])
                    self._append_state(ots)
        return (0.0, 0.0, 0.0)

    def rebind_world_to(self, pose: PoseTuple) -> Optional[Pose]:
        """Like ``rebind_world_to_current`` but the robot's current odom
        pose becomes world ``pose`` instead of the origin — a mapping run
        started at a known spot of an older map comes out roughly in that
        map's frame (patrols/checkpoints still need re-authoring)."""
        out = self.rebind_world_to_current()
        x, y, theta = float(pose[0]), float(pose[1]), _wrap(float(pose[2]))
        with self._lock:
            self._x, self._y, self._theta = x, y, theta
            latest_imu = self._imu.latest()
            if latest_imu is not None:
                # theta = raw_yaw - offset  ->  offset = raw_yaw - theta
                self._yaw_offset = _wrap(latest_imu[1] - theta)
            if self._state_buf:
                ts = self._state_buf[-1][0]
                self._state_buf.clear()
                self._append_state(ts)
        return out if out is None else (x, y, theta)

    def is_ready(self) -> bool:
        with self._lock:
            return self._seeded and self._imu.is_settled()

    def pose_at(self, ts: float) -> Optional[Pose]:
        if not self._seeded:
            return None
        with self._lock:
            sample = self._interp_state(ts)
            if sample is None:
                return None
            return (sample[1], sample[2], sample[3])

    def cov_at(self, ts: float) -> Optional[np.ndarray]:
        if not self._seeded:
            return None
        with self._lock:
            sample = self._interp_state(ts)
            if sample is None:
                return None
            return np.array(
                [
                    [sample[4], sample[5], sample[6]],
                    [sample[5], sample[7], sample[8]],
                    [sample[6], sample[8], sample[9]],
                ],
                dtype=np.float64,
            )

    def latest_state_ts(self) -> Optional[float]:
        """Pi-clock ts of the newest fused state, or None."""
        with self._lock:
            return self._state_buf[-1][0] if self._state_buf else None

    def pose(self) -> PoseTuple:
        latest = self._odom.latest_pose()
        if latest is None:
            return (0.0, 0.0, 0.0)
        (_x, _y, _th), ts = latest
        at = self.pose_at(ts)
        return at if at is not None else (self._x, self._y, self._theta)

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            p = self._P.copy()
            return {
                "imu_settled": self._imu.is_settled(),
                "seeded": self._seeded,
                "heading_source": self._last_heading_source,
                "yaw_at_misses": self._yaw_at_misses,
                "translation_source": "ekf_imu_projected_odom_ds",
                "cov_trace_xy": float(p[0, 0] + p[1, 1]),
                "cov_trace": float(np.trace(p)),
                "predict_count": self._predict_count,
                "imu_update_count": self._imu_update_count,
                "mag_update_count": self._mag_update_count,
                "wheel_heading_steps": self._wheel_heading_steps,
                "odom_resets": self._odom_resets,
                "imu_distrusted": self._imu_distrusted,
                "imu_guard_trips": self._imu_guard_trips,
                "rotation_contest_open": self._contest is not None,
                "rotation_contest_verdicts": dict(self._contest_counts),
                "imu_epoch": self._imu.epoch(),
            }

    def _maybe_update_mag(self, reading: ImuReading) -> None:
        if not self._noise.mag_idle_fusion_enabled:
            return
        if not self._seeded or not reading.mag_valid or reading.mag_quat_wxyz is None:
            return
        if (
            self._last_mag_update_ts >= 0.0
            and reading.ts - self._last_mag_update_ts
            < self._noise.mag_update_min_interval_s
        ):
            return

        reported = reading.mag_accuracy_rad
        if reported is None or reported <= 0.0:
            sigma = self._noise.mag_sigma_rad
        else:
            sigma = max(reported, self._noise.mag_sigma_rad)

        raw_mag = quaternion_to_yaw(reading.mag_quat_wxyz)
        with self._lock:
            if self._mag_yaw_offset is None:
                self._mag_yaw_offset = _wrap(raw_mag - self._theta)
            yaw = _wrap(raw_mag - self._mag_yaw_offset)
            self._update_yaw_observation(yaw, sigma)
            self._append_state(reading.ts)
            self._last_mag_update_ts = reading.ts
            self._mag_update_count += 1
            self._last_heading_source = "mag_idle"

    def _integrate_odom(
        self, ts: float, x: float, y: float, theta: float,
    ) -> None:
        yaw = self._yaw_at_world(ts)
        with self._lock:
            # IMU stream reset (driver restart): its yaw origin moved. Re-bind
            # the offset so the IMU's current yaw maps to the current θ.
            ep = self._imu.epoch()
            if ep != self._imu_epoch_seen:
                self._imu_epoch_seen = ep
                self._imu_rebind_pending = True
                self._guard_win.clear()
                self._contest = None
                self._last_imu_world_yaw = None
            if yaw is not None and self._imu_rebind_pending:
                yaw = self._rebind_yaw_offset_locked(ts, yaw)
                self._imu_rebind_pending = False

            if self._last_raw_odom is None:
                self._last_raw_odom = (ts, x, y, theta)
                if yaw is not None and not self._imu_distrusted:
                    self._theta = yaw
                self._last_imu_world_yaw = yaw
                self._append_state(ts)
                return

            _pts, lx, ly, lth = self._last_raw_odom
            dx = x - lx
            dy = y - ly
            dth_enc = _wrap(theta - lth)
            th_mid = lth + 0.5 * dth_enc
            ds = dx * math.cos(th_mid) + dy * math.sin(th_mid)

            if yaw is not None:
                yaw = self._imu_guard_locked(ts, _pts, dth_enc, yaw)

            if yaw is None:
                # No usable IMU yaw at this ts (stale, not yet arrived, or
                # distrusted). Never drop ds (S1): project along the wheel-
                # propagated heading and advance θ by the wheel rotation.
                self._yaw_at_misses += 1
                self._wheel_heading_steps += 1
                self._last_heading_source = (
                    "wheel_imu_distrusted" if self._imu_distrusted else "wheel")
                self._predict_motion(ds, dth_enc, None)
            else:
                use_game_yaw = not (
                    self._noise.mag_idle_fusion_enabled and self._last_mag_valid
                )
                self._predict_motion(ds, dth_enc, yaw, adopt_imu_yaw=use_game_yaw)
                if use_game_yaw:
                    self._update_yaw_observation(yaw, self._noise.imu_sigma_rad)
                    self._imu_update_count += 1
                    self._last_heading_source = "imu"
                else:
                    self._last_heading_source = "mag_idle"
            self._append_state(ts)
            self._last_raw_odom = (ts, x, y, theta)
            self._trim_state_buf()

    def _rebind_yaw_offset_locked(self, ts: float, yaw_world: float) -> float:
        """Re-bind ``_yaw_offset`` so the IMU's yaw at ``ts`` maps to the
        current θ. Returns the re-mapped world yaw (== θ)."""
        self._yaw_offset = _wrap(self._yaw_offset + (yaw_world - self._theta))
        self._imu_rebind_epoch += 1
        self._last_imu_world_yaw = self._theta
        return self._theta

    # ---- wheel-vs-IMU contest (scan-to-scan tie-breaker) -------------------

    _ROT_HIST_S = 5.0
    _CONTEST_TTL_S = 3.0

    def _record_rotation_locked(
        self, ts: float, prev_ts: float, dth_enc: float, dimu: float,
    ) -> None:
        if not self._rot_hist or self._rot_hist[-1][0] < prev_ts - 0.5:
            # Start (or restart after a gap) at the step's own start time.
            self._rot_hist.append((prev_ts, self._rot_wheel_cum, self._rot_imu_cum))
        self._rot_wheel_cum += dth_enc
        self._rot_imu_cum += dimu
        self._rot_hist.append((ts, self._rot_wheel_cum, self._rot_imu_cum))
        cutoff = ts - self._ROT_HIST_S
        while len(self._rot_hist) > 2 and self._rot_hist[0][0] < cutoff:
            self._rot_hist.popleft()

    def _expire_contest_locked(self, ts: float) -> None:
        c = self._contest
        if c is not None and ts - c["opened_ts"] > self._CONTEST_TTL_S:
            self._contest = None
            self._contest_counts["expired"] += 1
            logger.warning("ekf: wheel/IMU contest got no scan verdict in %.0fs — "
                           "keeping wheel heading", self._CONTEST_TTL_S)

    def rotation_contest(self) -> Optional[Tuple[float, float]]:
        """(t0, t1) of an open wheel-vs-IMU dispute awaiting a scan verdict:
        compare a scan at/before t0 with one at/after t1. None when none."""
        with self._lock:
            c = self._contest
            return None if c is None else (c["t0"], c["t1"])

    def rotation_between(self, ta: float, tb: float) -> Optional[Tuple[float, float]]:
        """(wheel, IMU) net rotation over [ta, tb] from the guard history, or
        None when the history doesn't cover it."""
        with self._lock:
            h = list(self._rot_hist)
        # Small tolerance at both ends: scan and odom ticks don't align.
        if len(h) < 2 or ta < h[0][0] - 0.15 or tb > h[-1][0] + 0.2 or tb <= ta:
            return None

        def at(t: float) -> Tuple[float, float]:
            if t >= h[-1][0]:
                return h[-1][1], h[-1][2]
            for (t0, w0, i0), (t1, w1, i1) in zip(h, h[1:]):
                if t0 <= t <= t1:
                    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                    return w0 + a * (w1 - w0), i0 + a * (i1 - i0)
            return h[0][1], h[0][2]

        wa, ia = at(ta)
        wb, ib = at(tb)
        return wb - wa, ib - ia

    def resolve_rotation_contest(self, verdict: str) -> bool:
        """Apply a scan verdict ("imu" | "wheels" | "unknown") to the open
        contest. "imu": the wheels lied (slip / stalled spin) — trust the IMU
        again and snap θ back to its heading under the pre-trip offset (undoing
        any revival re-bind). Others keep the wheel heading. Returns True when
        θ was changed."""
        with self._lock:
            c = self._contest
            if c is None:
                return False
            self._contest = None
            key = verdict if verdict in self._contest_counts else "unknown"
            self._contest_counts[key] += 1
            if verdict != "imu":
                return False
            latest = self._imu.latest()
            if latest is None:
                return False
            raw_yaw = latest[1]
            self._yaw_offset = c["offset"]
            new_theta = _wrap(raw_yaw - self._yaw_offset)
            dth = _wrap(new_theta - self._theta)
            self._theta = new_theta
            self._imu_distrusted = False
            self._guard_win.clear()
            self._last_imu_world_yaw = None
            self._imu_rebind_epoch += 1
            if self._last_raw_odom is not None:
                self._append_state(self._last_raw_odom[0])
        logger.warning("ekf: scans side with the IMU (wheels slipped) — heading "
                       "corrected %+.0f°", math.degrees(dth))
        return True

    def _imu_guard_locked(
        self, ts: float, prev_ts: float, dth_enc: float, yaw: float,
    ) -> Optional[float]:
        """Sliding-window wheel-vs-IMU contradiction guard (I2). Returns the
        yaw to fuse, or None while the IMU is distrusted."""
        n = self._noise
        prev_imu = self._last_imu_world_yaw
        self._last_imu_world_yaw = yaw
        if not n.imu_guard_enabled or prev_imu is None:
            return None if self._imu_distrusted else yaw
        dimu = _wrap(yaw - prev_imu)
        self._record_rotation_locked(ts, prev_ts, dth_enc, dimu)
        self._expire_contest_locked(ts)
        dt = max(ts - prev_ts, 1e-3)
        # A wheel-yaw step faster than any real chassis rotation is a glitch
        # (or a reset the epoch fields didn't flag), not evidence.
        enc_ev = dth_enc if abs(dth_enc) / dt <= 4.0 else 0.0
        self._guard_win.append((ts, enc_ev, dimu))
        cutoff = ts - n.imu_guard_window_s
        while self._guard_win and self._guard_win[0][0] < cutoff:
            self._guard_win.popleft()
        enc_net = sum(e for _t, e, _i in self._guard_win)
        imu_net = sum(i for _t, _e, i in self._guard_win)
        frozen = abs(enc_net) >= n.imu_guard_rot_rad and abs(imu_net) < n.imu_guard_still_rad
        unmatched = (n.imu_guard_distrust_unmatched_imu_rotation
                     and abs(imu_net) >= n.imu_guard_rot_rad
                     and abs(enc_net) < n.imu_guard_still_rad)
        if not self._imu_distrusted:
            if frozen or unmatched:
                self._imu_distrusted = True
                self._imu_guard_trips += 1
                if frozen:
                    # Wheels slipping / spinning against an obstacle look
                    # exactly like this too. Provisionally follow the wheels,
                    # but open a contest: a scan-to-scan rotation over the
                    # window decides, and an "imu" verdict reverts to the
                    # IMU heading via the offset saved here.
                    t0 = self._guard_win[0][0] if self._guard_win else ts
                    self._contest = {
                        "t0": t0 - n.imu_guard_window_s / 10.0,
                        "t1": ts,
                        "offset": self._yaw_offset,
                        "opened_ts": ts,
                    }
                    # Restore the rotation the frozen IMU hid during the
                    # window's earlier ticks (θ followed the flat IMU yaw
                    # until now); this tick's dθ_enc is applied by the
                    # wheel-heading predict.
                    self._theta = _wrap(
                        self._theta + (enc_net - enc_ev) - (imu_net - dimu))
                logger.error(
                    "ekf: IMU yaw %s — wheels %.0f° vs IMU %.0f° over %.1fs; "
                    "heading from wheel yaw until the IMU agrees again",
                    "FROZEN" if frozen else "rotating unseen by wheels",
                    math.degrees(enc_net), math.degrees(imu_net),
                    n.imu_guard_window_s)
                self._guard_win.clear()
                return None
            return yaw
        # Distrusted: revive once the IMU visibly moves in agreement with
        # the wheels (alive again), then re-bind its offset to the current θ.
        alive = abs(imu_net) >= 0.5 * n.imu_guard_rot_rad
        agrees = abs(imu_net - enc_net) <= 0.5 * abs(imu_net)
        if alive and agrees:
            self._imu_distrusted = False
            self._guard_win.clear()
            logger.info("ekf: IMU yaw moving with the wheels again — trusting it")
            # θ has advanced by dθ_enc for this tick only once predicted;
            # bind so the IMU maps to θ + dθ_enc (this tick's end heading).
            self._yaw_offset = _wrap(self._yaw_offset + (yaw - (self._theta + dth_enc)))
            self._imu_rebind_epoch += 1
            yaw_mapped = _wrap(self._theta + dth_enc)
            self._last_imu_world_yaw = yaw_mapped
            return yaw_mapped
        return None

    def _predict_motion(
        self, ds: float, dth_enc: float, yaw: Optional[float], *,
        adopt_imu_yaw: bool = True,
    ) -> None:
        """Propagate (x,y) with ds along heading; grow covariance.

        ``yaw=None`` → wheel heading: project along θ + ½·dθ_enc and advance
        θ by dθ_enc."""
        if yaw is None:
            heading = self._theta + 0.5 * dth_enc
        else:
            heading = yaw if adopt_imu_yaw else self._theta
        c = math.cos(heading)
        s = math.sin(heading)
        self._x += ds * c
        self._y += ds * s
        if yaw is None:
            self._theta = _wrap(self._theta + dth_enc)
        elif adopt_imu_yaw:
            self._theta = yaw

        n = self._noise
        sigma_trans = n.alpha_trans_per_m * abs(ds) + n.alpha_rot_per_m * abs(dth_enc)
        sigma_rot = n.alpha_rot_per_m * abs(ds) + n.alpha_rot_per_rad * abs(dth_enc)
        q_xy = max(sigma_trans ** 2, 1e-8)
        q_th = max(sigma_rot ** 2, 1e-8)

        G = np.array([[1.0, 0.0, -ds * s], [0.0, 1.0, ds * c], [0.0, 0.0, 1.0]])
        Q = np.diag([q_xy, q_xy, q_th])
        self._P = G @ self._P @ G.T + Q
        self._predict_count += 1

    def _update_yaw_observation(self, yaw: float, sigma_rad: float) -> None:
        """Kalman update: θ observation (IMU game RV or idle mag)."""
        z = yaw
        H = np.array([[0.0, 0.0, 1.0]])
        R = np.array([[max(sigma_rad, 1e-6) ** 2]])
        innov = _wrap(z - self._theta)
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        delta = K @ np.array([[innov]])
        self._x += float(delta[0, 0])
        self._y += float(delta[1, 0])
        self._theta = _wrap(self._theta + float(delta[2, 0]))
        I = np.eye(3)
        self._P = (I - K @ H) @ self._P

    def _append_state(self, ts: float) -> None:
        p = self._P
        self._state_buf.append(
            (
                ts,
                self._x,
                self._y,
                self._theta,
                float(p[0, 0]),
                float(p[0, 1]),
                float(p[0, 2]),
                float(p[1, 1]),
                float(p[1, 2]),
                float(p[2, 2]),
            ),
        )

    def _trim_state_buf(self) -> None:
        if len(self._state_buf) < 2:
            return
        cutoff = self._state_buf[-1][0] - self._buf_seconds
        while len(self._state_buf) > 2 and self._state_buf[0][0] < cutoff:
            self._state_buf.popleft()

    def _interp_state(self, ts: float) -> Optional[_StateSample]:
        n = len(self._state_buf)
        if n == 0:
            return None
        if n == 1:
            return self._state_buf[0]

        first_ts = self._state_buf[0][0]
        last_ts = self._state_buf[-1][0]
        if ts <= first_ts:
            grace = (last_ts - first_ts) / max(1, n - 1) * 0.5
            if first_ts - ts <= grace:
                return self._state_buf[0]
            return None
        if ts >= last_ts:
            grace = (last_ts - first_ts) / max(1, n - 1) * 0.5
            if ts - last_ts <= grace:
                return self._state_buf[-1]
            return None

        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._state_buf[mid][0] <= ts:
                lo = mid
            else:
                hi = mid
        s0 = self._state_buf[lo]
        s1 = self._state_buf[hi]
        t0, t1 = s0[0], s1[0]
        if t1 == t0:
            return s1
        alpha = (ts - t0) / (t1 - t0)
        out = list(s0)
        out[0] = ts
        for i in range(1, 10):
            out[i] = s0[i] + alpha * (s1[i] - s0[i])
        # θ wrap-safe (S6): interpolate along the short arc. Linear on the raw
        # values between 3.13 and -3.13 would give ~0 (pointing backwards).
        out[3] = _wrap(s0[3] + alpha * _wrap(s1[3] - s0[3]))
        return tuple(out)  # type: ignore[return-value]

    def _yaw_at_world(self, ts: float) -> Optional[float]:
        if not self._imu.is_settled():
            return None
        result = self._imu.yaw_at(ts)
        if result is None:
            return None
        yaw_imu, _sigma = result
        return _wrap(yaw_imu - self._yaw_offset)

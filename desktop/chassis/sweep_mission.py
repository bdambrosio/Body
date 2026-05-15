"""Sweep-360 mission worker. Implements docs/sweep360_spec.md.

Drives the robot through a stepped 360° rotation, fuses the post-settle
local_2p5d frames into a sweep-frame accumulator, and reports a loop-
closure residual at completion.

Threading:
- One QThread (`SweepMission`) running `run()`.
- Drives motion via `controller.set_cmd_vel` + forced `live_command`,
  letting the existing `_cmd_loop` handle the wire publish at cmd_vel_hz.
- Publishes `body/sweep/status` and `body/map/sweep_360` directly through
  the controller's shared Zenoh session.
- Emits Qt signals for the dock to update UI (auto-queued across threads).
"""
from __future__ import annotations

import enum
import json
import logging
import math
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from desktop.nav.slam.imu_yaw import ImuYawTracker
from desktop.nav.slam.types import ImuReading

from .yaw_estimator import estimate_lidar_corr

logger = logging.getLogger(__name__)


class SweepState(str, enum.Enum):
    IDLE = "idle"
    PRECHECK = "precheck"
    ROTATING = "rotating"
    SETTLING = "settling"
    FUSING = "fusing"
    DONE = "done"
    ABORTED = "aborted"
    ESTOP = "estop"
    ERROR = "error"


_TERMINAL = (SweepState.DONE, SweepState.ABORTED, SweepState.ESTOP, SweepState.ERROR)


DEFAULT_PARAMS: Dict[str, Any] = {
    "step_deg": 30.0,
    "total_deg": 390.0,
    "angular_rate_dps": 30.0,
    "settle_ms": 1500,
    "direction": "ccw",
}


class SweepMission(QThread):
    """One-shot sweep mission. Reusable: call `start_mission()` again
    after a previous run terminated.
    """

    # Signals (auto-queued across threads when receiver is in GUI thread)
    state_changed = pyqtSignal(str)         # SweepState.value
    step_complete = pyqtSignal(dict)         # last_step debug info
    mission_done = pyqtSignal(dict)          # final accumulator dict
    accumulator_updated = pyqtSignal()       # UI repaint hook

    STATUS_TOPIC = "body/sweep/status"
    MAP_TOPIC = "body/map/sweep_360"
    IMU_TOPIC = "body/imu"

    SCAN_MATCH_MIN_CONFIDENCE = 0.35
    FUSE_VOTE_MARGIN = 1

    # Closed-loop rotation gate: stop commanding ω when the IMU-measured
    # progress reaches (step_deg - tolerance). Tolerance is rate-scaled
    # to anticipate motor coast — the gate fires *before* the target
    # so the bot can coast through settling and land near step_deg.
    #
    # Coast model fit to 2026-05-15 sweep data (Bruce's robot):
    #   30 dps → 18° coast/step; 15 dps → 6° coast/step.
    #   Linear+quadratic in dps passes through both points exactly.
    #   tolerance = max(MIN_DEG, LINEAR·ω + QUADRATIC·ω²)
    # MIN floor keeps the gate above BNO085 gyro noise at very low rates.
    # Recalibrate the coefficients if motor PWM / battery / floor friction
    # changes significantly (different surface, swapped tires, etc.).
    ROTATE_TOLERANCE_MIN_DEG = 2.0
    ROTATE_COAST_LINEAR_DEG_PER_DPS = 0.224
    ROTATE_COAST_QUADRATIC_DEG_PER_DPS2 = 0.0128
    # Multiplier on the open-loop step duration. If IMU feedback hasn't
    # closed the gate by `rotate_max_time_s = budget * t_step`, fall
    # back to time-based termination (matches pre-IMU behavior on this
    # step). 3× absorbs reasonable slip; bigger means a stuck-wheel
    # step doesn't burn the whole mission.
    ROTATE_TIME_BUDGET = 3.0
    # Per-step IMU prior window for scan-match. Per-step rotations are
    # tightly constrained by step_deg + coast, so a narrow window suffices
    # and reliably defeats the 90°/180° flip ambiguity in symmetric rooms.
    SCAN_MATCH_PRIOR_WINDOW_PER_STEP_DEG = 15.0
    # Loop-closure prior window. Wider because yaw_accum has accumulated
    # 13 steps of fusion noise, but still tight enough to reject the
    # symmetric-room flip alignments.
    SCAN_MATCH_PRIOR_WINDOW_LOOP_DEG = 30.0
    # IMU↔lidar agreement gate: if both signals are present and the
    # absolute difference exceeds this many degrees, the step is
    # flagged in last_step_info and a logger warning is emitted.
    # 10° handles the typical scan-match-locked-at-0° failure mode
    # (commanded 60° / lidar 0° → Δ = 60°) without firing on
    # ordinary scan-match resolution noise (~2–3° per step).
    IMU_LIDAR_DISAGREEMENT_DEG = 10.0

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._params = dict(DEFAULT_PARAMS)
        self._request_id: Optional[str] = None
        self._abort_requested = False
        self._state = SweepState.IDLE
        # Publishers (declared on entering run, undeclared on exit)
        self._status_pub: Optional[Any] = None
        self._map_pub: Optional[Any] = None
        # IMU subscription for per-step yaw measurement. Tracker buffer
        # is short (default 2 s) — we only need the latest sample at
        # each step boundary, not historical lookup.
        self._imu_yaw = ImuYawTracker()
        self._imu_sub: Optional[Any] = None
        # Yaw snapshot at the start of the current rotating step.
        # Captured in _do_step before motion begins; compared to the
        # post-settle yaw to produce per-step imu_deg.
        self._yaw_pre_step_rad: Optional[float] = None
        # Mission-local state
        self._anchor_pose: Dict[str, float] = {"x_m": 0.0, "y_m": 0.0, "theta_rad": 0.0}
        self._first_scan: Optional[Dict[str, Any]] = None
        self._last_scan: Optional[Dict[str, Any]] = None
        self._yaw_accum_deg = 0.0
        self._step_index = 0
        self._step_count = 0
        self._loop_closure_deg: Optional[float] = None
        self._last_step_info: Dict[str, Any] = {}
        self._reason: Optional[str] = None
        # Accumulator (allocated on first local_map fusion)
        self._acc_max_h: Optional[np.ndarray] = None
        self._acc_clear_votes: Optional[np.ndarray] = None
        self._acc_block_votes: Optional[np.ndarray] = None
        self._acc_resolution_m = 0.0
        self._acc_extent_m = 0.0
        self._clearance_m: Optional[float] = None

    # ── Public API ───────────────────────────────────────────────────

    def state(self) -> SweepState:
        return self._state

    def is_active(self) -> bool:
        return self._state not in (SweepState.IDLE,) + _TERMINAL

    def start_mission(
        self,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        """Begin a new sweep. Returns False if a sweep is already active."""
        if self.is_active() or self.isRunning():
            logger.warning("SweepMission.start_mission called while active")
            return False
        merged = dict(DEFAULT_PARAMS)
        if params:
            for k in DEFAULT_PARAMS:
                if k in params and params[k] is not None:
                    merged[k] = params[k]
        if merged["direction"] not in ("ccw", "cw"):
            merged["direction"] = "ccw"
        merged["step_deg"] = max(1.0, float(merged["step_deg"]))
        merged["total_deg"] = max(merged["step_deg"], float(merged["total_deg"]))
        merged["angular_rate_dps"] = max(1.0, float(merged["angular_rate_dps"]))
        merged["settle_ms"] = max(100, int(merged["settle_ms"]))
        self._params = merged
        self._request_id = request_id or str(uuid.uuid4())
        self._abort_requested = False
        self._reason = None
        self._first_scan = None
        self._last_scan = None
        self._yaw_accum_deg = 0.0
        self._step_index = 0
        self._step_count = int(math.ceil(merged["total_deg"] / merged["step_deg"]))
        self._loop_closure_deg = None
        self._last_step_info = {}
        self._acc_max_h = None
        self._acc_clear_votes = None
        self._acc_block_votes = None
        self._clearance_m = None
        self.start()  # QThread.start() → run()
        return True

    def request_abort(self) -> None:
        self._abort_requested = True

    def snapshot_accumulator(self) -> Optional[Dict[str, Any]]:
        """Return a render-ready snapshot of the accumulator, or None
        if nothing has been fused yet. Shape matches local_2p5d so
        existing LocalMapView/DriveableView widgets render it directly.
        """
        if self._acc_max_h is None:
            return None
        n = self._acc_max_h.shape[0]
        meta: Dict[str, Any] = {
            "resolution_m": self._acc_resolution_m,
            "origin_x_m": -self._acc_extent_m,
            "origin_y_m": -self._acc_extent_m,
            "nx": n,
            "ny": n,
            "frame": "sweep",
        }
        if self._clearance_m is not None:
            meta["driveable_clearance_height_m"] = self._clearance_m
        drive = self._driveable_from_votes()
        return {
            "grid": self._acc_max_h.copy(),
            "meta": meta,
            "driveable": drive,
        }

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self._declare_publishers()
            # Take ownership of the cmd channel before announcing
            # PRECHECK. The chassis/nav _on_sweep_active handlers wrap
            # their UI resets in QSignalBlocker so the toggled signals
            # don't race against our per-step set_live_command(True) —
            # but that also means they no longer call stop_all() /
            # set_cmd_mode("cmd_vel") on our behalf. Without an explicit
            # zero+switch+settle here, any residual cmd_vel still
            # decelerating from the user's last manual command (or
            # stale cmd_direct values from a previously-engaged
            # motor_dock) keeps being published by _cmd_loop through
            # _take_anchor / first_scan capture — and that pre-step
            # motion lands inside step 0's imu_deg measurement as a
            # phantom "large initial rotation."
            #
            # Order matters:
            #   1. set_cmd_mode publishes a one-shot zero on both
            #      topics via supersede() if prev was cmd_direct,
            #      defeating any stale cmd_direct PWM.
            #   2. set_cmd_vel zeroes the cmd_vel state so _cmd_loop's
            #      next tick publishes a clean zero on top of it.
            #   3. set_live_command(True) arms _cmd_loop so step 1's
            #      zero actually propagates (if live was False, the
            #      loop skips publishing entirely and the zero is just
            #      state-internal).
            #   4. Brief sleep gives _cmd_loop (10 Hz default) at least
            #      one tick to push the zero out and the Pi's velocity
            #      loop time to actively brake any residual motion.
            #      Without this, step 0's `pre = imu.latest()` lands
            #      while the bot is still decelerating.
            self.controller.set_cmd_mode("cmd_vel")
            self.controller.set_cmd_vel(0.0, 0.0)
            self.controller.set_live_command(True)
            time.sleep(0.3)
            self._set_state(SweepState.PRECHECK)
            if not self._precheck():
                return

            self._take_anchor()

            # Capture the very first scan for loop-closure at the end.
            self._first_scan, _ = self.controller.state.snapshot_lidar()
            if self._first_scan is None:
                # Wait briefly for one to land
                self._first_scan = self._wait_for_scan(time.time() - 1.0, timeout_s=2.0)
            if self._first_scan is None:
                self._fail("stale_lidar_at_start")
                return
            pre_step_scan = self._first_scan

            for i in range(self._step_count):
                if self._abort_or_estop_check():
                    return
                self._step_index = i
                pre_step_scan = self._do_step(i, pre_step_scan)
                if pre_step_scan is None:
                    return  # step set the terminal state
                self._last_scan = pre_step_scan

            # Loop closure: compare first scan to last post-settle scan.
            # Pass yaw_accum (wrapped) as a prior so a symmetric room can't
            # snap scan-match to the 90°/180° decoy alignment — without
            # this guard, conf~0.4 results regularly land 150°+ off truth.
            if self._first_scan is not None and self._last_scan is not None:
                prior_deg = ((self._yaw_accum_deg + 180.0) % 360.0) - 180.0
                deg, _ = estimate_lidar_corr(
                    self._first_scan, self._last_scan,
                    prior_deg=prior_deg,
                    prior_window_deg=self.SCAN_MATCH_PRIOR_WINDOW_LOOP_DEG,
                )
                if deg is not None:
                    residual = deg - self._yaw_accum_deg
                    residual = ((residual + 180.0) % 360.0) - 180.0
                    self._loop_closure_deg = residual

            self._publish_map(final=True)
            self._set_state(SweepState.DONE)
            self.mission_done.emit(self._build_map_dict())
        except Exception as e:
            logger.exception("SweepMission.run crashed")
            self._fail(f"{type(e).__name__}: {e}")
        finally:
            self._stop_motion()
            self._undeclare_publishers()

    # ── State machine helpers ────────────────────────────────────────

    def _set_state(self, new_state: SweepState, *, reason: Optional[str] = None) -> None:
        self._state = new_state
        if reason is not None:
            self._reason = reason
        self.state_changed.emit(new_state.value)
        # Spec §4.2: silent in idle. Publish on every transition otherwise.
        if new_state != SweepState.IDLE:
            self._publish_status()

    def _fail(self, reason: str) -> None:
        self._set_state(SweepState.ERROR, reason=reason)

    def _precheck(self) -> bool:
        with self.controller.state.lock:
            connected = self.controller.state.connected
        if not connected:
            self._fail("not_connected")
            return False
        if self._estop_active():
            self._set_state(SweepState.ESTOP, reason="estop_at_start")
            return False
        return True

    def _estop_active(self) -> bool:
        with self.controller.state.lock:
            es = self.controller.state.emergency_stop
            st = self.controller.state.status or {}
            ms = self.controller.state.motor_state or {}
        if isinstance(es, dict) and (es.get("active") or es.get("e_stop_active")):
            return True
        if st.get("e_stop_active"):
            return True
        if ms.get("e_stop_active"):
            return True
        return False

    def _abort_or_estop_check(self) -> bool:
        if self._abort_requested:
            self._set_state(SweepState.ABORTED, reason="abort_requested")
            return True
        if self._estop_active():
            self._set_state(SweepState.ESTOP, reason="estop")
            return True
        return False

    def _take_anchor(self) -> None:
        with self.controller.state.lock:
            odom = self.controller.state.odom
        if isinstance(odom, dict):
            self._anchor_pose = {
                "x_m": float(odom.get("x", 0.0) or 0.0),
                "y_m": float(odom.get("y", 0.0) or 0.0),
                "theta_rad": float(odom.get("theta", 0.0) or 0.0),
            }
        else:
            self._anchor_pose = {"x_m": 0.0, "y_m": 0.0, "theta_rad": 0.0}

    # ── Per-step logic ───────────────────────────────────────────────

    def _do_step(
        self, i: int, pre_step_scan: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        params = self._params
        rate_rad_s = math.radians(params["angular_rate_dps"])
        sign = 1.0 if params["direction"] == "ccw" else -1.0
        t_step = math.radians(params["step_deg"]) / rate_rad_s

        # ── ROTATING ────────────────────────────────────────────────
        self._set_state(SweepState.ROTATING)
        # Snapshot IMU yaw at step start. ImuYawTracker.latest() returns
        # the newest unwrapped sample (no historical lookup needed) —
        # post-step yaw is read from latest() again after the scan_post
        # wait completes, and the difference is imu_deg for this step.
        # None until the tracker has settled (~0.2 s after the first
        # body/imu sample); the rotating loop falls back to pure time
        # in that case (pre-IMU behavior).
        pre = self._imu_yaw.latest()
        self._yaw_pre_step_rad = pre[1] if pre is not None else None
        # Make sure we're commanding the twist topic, not direct.
        self.controller.set_cmd_mode("cmd_vel")
        self.controller.set_cmd_vel(0.0, sign * rate_rad_s)
        self.controller.set_live_command(True)
        # Closed-loop ROTATING: end the step when the IMU-measured
        # rotation in the commanded direction reaches step_deg (slip-
        # immune), or when the time budget runs out (fallback for
        # unsettled IMU / stuck wheel). Direction-aware: `progress =
        # sign * measured_deg` is always positive when rotation is
        # going the commanded way, so an overshoot past step_deg
        # stays inside the gate (progress only grows) rather than
        # vanishing into "residual flipped sign and the bot kept
        # spinning a full loop."
        step_deg = float(params["step_deg"])
        rate_dps = float(params["angular_rate_dps"])
        tolerance_deg = max(
            self.ROTATE_TOLERANCE_MIN_DEG,
            self.ROTATE_COAST_LINEAR_DEG_PER_DPS * rate_dps
            + self.ROTATE_COAST_QUADRATIC_DEG_PER_DPS2 * rate_dps * rate_dps,
        )
        time_budget_s = self.ROTATE_TIME_BUDGET * t_step
        deadline = time.monotonic() + time_budget_s
        last_log_t = 0.0
        # 25 ms tick → 40 Hz feedback. At rate=30 dps, one tick is
        # 0.75° of motion — well inside the 2° tolerance even if the
        # IMU sample lands at the worst phase.
        tick_s = 0.025
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            if self._abort_or_estop_check():
                return None
            # IMU feedback gate. Sample latest yaw; if both pre and
            # current snapshots exist, project onto the commanded
            # direction and break when progress clears step_deg
            # (minus tolerance).
            cur = self._imu_yaw.latest()
            if (
                self._yaw_pre_step_rad is not None
                and cur is not None
            ):
                measured_deg = math.degrees(cur[1] - self._yaw_pre_step_rad)
                progress = sign * measured_deg
                if progress >= step_deg - tolerance_deg:
                    break
                # Light tracing — once per second at most — so a
                # debugger sees the closed-loop progressing without
                # flooding the log on a healthy run.
                if now - last_log_t >= 1.0:
                    last_log_t = now
                    logger.debug(
                        f"sweep step {i}: imu_progress={progress:.1f}/"
                        f"{step_deg:.1f}°"
                    )
            self._publish_status()
            time.sleep(tick_s)
        # Stop motion; keep live_command so cmd_loop emits zeros (and so
        # the watchdog stays happy) through settling and fusing.
        self.controller.set_cmd_vel(0.0, 0.0)

        # ── SETTLING ────────────────────────────────────────────────
        self._set_state(SweepState.SETTLING)
        scan_period_s = self._scan_period_s()
        local_map_period_s = self._local_map_period_s_or(default=1.0)
        spec_min_s = scan_period_s + local_map_period_s
        settle_s = max(params["settle_ms"] / 1000.0, spec_min_s)
        settle_started_wall = time.time()
        deadline = time.monotonic() + settle_s
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            if self._abort_or_estop_check():
                return None
            self._publish_status()
            time.sleep(min(0.1, deadline - now))

        # ── FUSING ──────────────────────────────────────────────────
        self._set_state(SweepState.FUSING)
        scan_threshold_ts = settle_started_wall + scan_period_s
        scan_post = self._wait_for_scan(scan_threshold_ts, timeout_s=2.0)
        if scan_post is None:
            self._fail("stale_lidar")
            return None
        # local_map threshold uses its own source.lidar_ts when available,
        # falling back to receipt ts.
        local_map_post = self._wait_for_local_map(scan_threshold_ts, timeout_s=4.0)
        if local_map_post is None:
            self._fail("stale_local_map")
            return None

        commanded_deg = sign * params["step_deg"]
        # IMU per-step yaw delta. BNO085 publishes body/imu at 100 Hz;
        # we sampled latest() before motion started and sample it again
        # here after scan_post arrived. Either snapshot can be None if
        # the tracker wasn't settled at that boundary — fall through to
        # the existing lidar/cmd fusion in that case.
        post = self._imu_yaw.latest()
        if self._yaw_pre_step_rad is not None and post is not None:
            imu_deg = math.degrees(post[1] - self._yaw_pre_step_rad)
        else:
            imu_deg = None
        cmd_deg = commanded_deg

        # Compute lidar scan-match with IMU (preferred) or commanded ω·Δt
        # as the prior. The window is narrow enough to defeat
        # 90°/180° symmetric-room flip ambiguity but loose enough to
        # absorb coast variance + IMU noise.
        scan_prior_deg = imu_deg if imu_deg is not None else commanded_deg
        lidar_deg, lidar_conf = estimate_lidar_corr(
            pre_step_scan, scan_post,
            prior_deg=scan_prior_deg,
            prior_window_deg=self.SCAN_MATCH_PRIOR_WINDOW_PER_STEP_DEG,
        )

        # Fusion preference: IMU > lidar > cmd. The BNO085 gyro is
        # slip-immune and drifts ~0.05° per 3 s step in
        # game_rotation_vector mode — substantially more accurate than
        # scan-match in featureless rooms (where lidar reliably reports
        # 0° at 0.70 conf because successive scans look identical, even
        # when the robot rotated). Lidar remains the fallback when the
        # tracker is unsettled; cmd is the last resort.
        if imu_deg is not None:
            fused = imu_deg
        elif lidar_deg is not None and lidar_conf >= self.SCAN_MATCH_MIN_CONFIDENCE:
            fused = lidar_deg
        else:
            fused = cmd_deg
        self._yaw_accum_deg += fused

        self._fuse_local_map(local_map_post, self._yaw_accum_deg)

        # IMU↔lidar agreement check. Only meaningful when both signals
        # are present AND lidar has the confidence we'd ordinarily
        # trust — a low-conf lidar value disagreeing with IMU is
        # uninteresting (lidar is just unreliable). Disagreement value
        # is lidar - imu so the sign tells you which way lidar is off.
        disagreement_deg: Optional[float] = None
        if (
            imu_deg is not None
            and lidar_deg is not None
            and lidar_conf >= self.SCAN_MATCH_MIN_CONFIDENCE
        ):
            disagreement_deg = float(lidar_deg - imu_deg)
            if abs(disagreement_deg) > self.IMU_LIDAR_DISAGREEMENT_DEG:
                logger.warning(
                    f"sweep step {i}: imu↔lidar disagreement "
                    f"Δ={disagreement_deg:+.1f}° (lidar={lidar_deg:+.1f}° "
                    f"conf={lidar_conf:.2f}, imu={imu_deg:+.1f}°, "
                    f"cmd={cmd_deg:+.1f}°)"
                )

        self._last_step_info = {
            "commanded_deg": commanded_deg,
            "yaw_sources": {"lidar": lidar_deg, "imu": imu_deg, "cmd": cmd_deg},
            "fused_deg": fused,
            "residual_xy_m": [0.0, 0.0],  # correlation gives no translation
            "settle_ms": int(settle_s * 1000),
            "lidar_confidence": lidar_conf,
            "imu_lidar_disagreement_deg": disagreement_deg,
        }
        # Per-step trace on stderr (INFO level) so a calibration sweep
        # produces a readable transcript without needing the dock open.
        # One line per step, fixed columns for grep-friendliness.
        def _fmt(v: Optional[float]) -> str:
            return "  none " if v is None else f"{v:+7.2f}"
        logger.info(
            f"sweep step {i+1:2d}/{self._step_count:2d}: "
            f"cmd={_fmt(cmd_deg)}° "
            f"imu={_fmt(imu_deg)}° "
            f"lidar={_fmt(lidar_deg)}°(conf={lidar_conf:.2f}) "
            f"fused={_fmt(fused)}° "
            f"yaw_accum={self._yaw_accum_deg:+8.2f}°"
        )
        self.step_complete.emit(dict(self._last_step_info))
        self.accumulator_updated.emit()

        self._publish_map(final=False)
        self._publish_status()
        return scan_post

    # ── Sensor waits ─────────────────────────────────────────────────

    def _scan_period_s(self) -> float:
        with self.controller.state.lock:
            scan = self.controller.state.lidar_scan
        if isinstance(scan, dict):
            stm = scan.get("scan_time_ms")
            if isinstance(stm, (int, float)) and stm > 0:
                return float(stm) / 1000.0
        return 0.2  # safe default ~5 Hz

    def _local_map_period_s_or(self, *, default: float) -> float:
        period = self.controller.state.local_map_period_s()
        if period is None or period <= 0:
            return default
        return period

    def _scan_msg_ts(self, scan: Dict[str, Any], recv_ts: float) -> float:
        ts = scan.get("ts")
        if isinstance(ts, (int, float)) and ts > 0:
            return float(ts)
        return recv_ts

    def _wait_for_scan(
        self, threshold_ts: float, timeout_s: float,
    ) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._abort_or_estop_check():
                return None
            scan, recv_ts = self.controller.state.snapshot_lidar()
            if scan is not None:
                if self._scan_msg_ts(scan, recv_ts) > threshold_ts:
                    return scan
            time.sleep(0.05)
        return None

    def _wait_for_local_map(
        self, threshold_ts: float, timeout_s: float,
    ) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._abort_or_estop_check():
                return None
            with self.controller.state.lock:
                grid = self.controller.state.local_map_grid
                meta = self.controller.state.local_map_meta
                drive = self.controller.state.local_map_driveable
                map_recv_ts = self.controller.state.local_map_ts
            if grid is not None and meta is not None:
                # local_map carries sources.lidar_ts / depth_ts; honor whichever
                # the Pi has marked as freshest, fall back to recv ts.
                src = meta.get("sources") or {}
                src_ts = 0.0
                for k in ("lidar_ts", "depth_ts"):
                    v = src.get(k)
                    if isinstance(v, (int, float)) and v > src_ts:
                        src_ts = float(v)
                eff_ts = src_ts if src_ts > 0 else map_recv_ts
                if eff_ts > threshold_ts:
                    return {
                        "grid": grid.copy(),
                        "meta": dict(meta),
                        "driveable": drive.copy() if drive is not None else None,
                    }
            time.sleep(0.05)
        return None

    # ── Grid fusion ──────────────────────────────────────────────────

    def _fuse_local_map(
        self, local_map: Dict[str, Any], theta_accum_deg: float,
    ) -> None:
        meta = local_map["meta"]
        grid = local_map["grid"]            # (nx, ny) max_height_m, NaN unknown
        drive = local_map.get("driveable")  # (nx, ny) int8 or None
        res = float(meta.get("resolution_m", 0.0))
        if res <= 0:
            return
        origin_x = float(meta.get("origin_x_m", 0.0))
        origin_y = float(meta.get("origin_y_m", 0.0))
        nx_b, ny_b = grid.shape

        if self._acc_max_h is None:
            # Allocate a square accumulator big enough to hold the source
            # extent rotated through any yaw. Take the worst-case half-extent
            # (max distance from anchor any source-cell center can reach).
            corners = [
                (origin_x, origin_y),
                (origin_x + nx_b * res, origin_y),
                (origin_x, origin_y + ny_b * res),
                (origin_x + nx_b * res, origin_y + ny_b * res),
            ]
            half = max(math.hypot(x, y) for x, y in corners)
            n_side = 2 * int(math.ceil(half / res))
            self._acc_max_h = np.full((n_side, n_side), np.nan, dtype=np.float32)
            self._acc_clear_votes = np.zeros((n_side, n_side), dtype=np.int32)
            self._acc_block_votes = np.zeros((n_side, n_side), dtype=np.int32)
            self._acc_resolution_m = res
            self._acc_extent_m = (n_side / 2.0) * res

        if abs(res - self._acc_resolution_m) > 1e-6:
            logger.warning(
                f"local_map resolution changed mid-sweep: {res} vs {self._acc_resolution_m}"
            )
            return

        clr = meta.get("driveable_clearance_height_m")
        if isinstance(clr, (int, float)):
            self._clearance_m = float(clr)

        n = self._acc_max_h.shape[0]
        # Map every accumulator cell back into the body frame of this
        # local_map by rotating by -theta_accum, then look up the source
        # cell. Vectorized.
        theta_rad = math.radians(theta_accum_deg)
        c, s = math.cos(theta_rad), math.sin(theta_rad)
        # float64 throughout the index math; float32 lets `floor((0.20)/0.1)`
        # land at 1.999… → wrong cell. Tiny epsilon nudges boundary cells
        # consistently into the next cell, matching the local_map convention.
        ii = np.arange(n, dtype=np.float64)
        jj = np.arange(n, dtype=np.float64)
        x_s = -self._acc_extent_m + (ii + 0.5) * res
        y_s = -self._acc_extent_m + (jj + 0.5) * res
        Xs, Ys = np.meshgrid(x_s, y_s, indexing="ij")
        Xb = c * Xs + s * Ys
        Yb = -s * Xs + c * Ys
        ib = np.floor((Xb - origin_x) / res + 1e-9).astype(np.int32)
        jb = np.floor((Yb - origin_y) / res + 1e-9).astype(np.int32)
        valid = (ib >= 0) & (ib < nx_b) & (jb >= 0) & (jb < ny_b)
        if not np.any(valid):
            return

        ib_v = ib[valid]
        jb_v = jb[valid]
        src_h = grid[ib_v, jb_v]
        target_h = self._acc_max_h[valid]
        src_has = ~np.isnan(src_h)
        # Fold in source heights: max if both, source if target NaN, target otherwise.
        merged = np.where(np.isnan(target_h), src_h, np.maximum(target_h, src_h))
        merged = np.where(src_has, merged, target_h)
        self._acc_max_h[valid] = merged

        if drive is not None:
            src_d = drive[ib_v, jb_v]
            self._acc_clear_votes[valid] += (src_d == 1).astype(np.int32)
            self._acc_block_votes[valid] += (src_d == 0).astype(np.int32)

    def _driveable_from_votes(self) -> np.ndarray:
        n = self._acc_max_h.shape[0]
        out = np.full((n, n), -1, dtype=np.int8)
        if self._acc_clear_votes is None or self._acc_block_votes is None:
            return out
        margin = self.FUSE_VOTE_MARGIN
        out[self._acc_clear_votes > self._acc_block_votes + margin] = 1
        out[self._acc_block_votes > self._acc_clear_votes + margin] = 0
        return out

    # ── Output building ──────────────────────────────────────────────

    def _build_map_dict(self) -> Dict[str, Any]:
        if self._acc_max_h is None:
            return {}
        n = self._acc_max_h.shape[0]
        drive = self._driveable_from_votes()
        max_h_rows = [
            [None if math.isnan(v) else float(v) for v in row]
            for row in self._acc_max_h
        ]
        drive_rows = [
            [True if v == 1 else (False if v == 0 else None) for v in row]
            for row in drive
        ]
        out: Dict[str, Any] = {
            "ts": time.time(),
            "frame": "sweep",
            "kind": "max_height_grid",
            "resolution_m": self._acc_resolution_m,
            "origin_x_m": -self._acc_extent_m,
            "origin_y_m": -self._acc_extent_m,
            "nx": n,
            "ny": n,
            "max_height_m": max_h_rows,
            "driveable": drive_rows,
            "anchor_pose": dict(self._anchor_pose),
            "step_count": self._step_count,
            "loop_closure_deg": self._loop_closure_deg,
            "request_id": self._request_id,
        }
        if self._clearance_m is not None:
            out["driveable_clearance_height_m"] = self._clearance_m
        return out

    def _publish_map(self, *, final: bool) -> None:
        if self._map_pub is None or self._acc_max_h is None:
            return
        try:
            self._map_pub.put(
                json.dumps(self._build_map_dict()).encode("utf-8")
            )
        except Exception:
            logger.exception("sweep_360 publish failed")

    def _publish_status(self) -> None:
        if self._status_pub is None or self._state == SweepState.IDLE:
            return
        payload = {
            "ts": time.time(),
            "state": self._state.value,
            "request_id": self._request_id,
            "step_index": self._step_index,
            "step_count": self._step_count,
            "yaw_accum_deg": self._yaw_accum_deg,
            "coverage_deg": self._yaw_accum_deg,
            "last_step": self._last_step_info or None,
            "loop_closure_deg": self._loop_closure_deg,
            "reason": self._reason,
        }
        try:
            self._status_pub.put(json.dumps(payload).encode("utf-8"))
        except Exception:
            logger.exception("sweep status publish failed")

    # ── Publisher lifecycle ──────────────────────────────────────────

    def _declare_publishers(self) -> None:
        sess = self.controller._session
        if sess is None:
            return
        try:
            self._status_pub = sess.declare_publisher(self.STATUS_TOPIC)
            self._map_pub = sess.declare_publisher(self.MAP_TOPIC)
        except Exception:
            logger.exception("sweep publishers declare failed")
            self._status_pub = None
            self._map_pub = None
        try:
            self._imu_sub = sess.declare_subscriber(
                self.IMU_TOPIC, self._on_imu,
            )
        except Exception:
            logger.exception("sweep imu subscribe failed")
            self._imu_sub = None

    def _undeclare_publishers(self) -> None:
        for pub in (self._status_pub, self._map_pub):
            if pub is not None:
                try:
                    pub.undeclare()
                except Exception:
                    pass
        self._status_pub = None
        self._map_pub = None
        if self._imu_sub is not None:
            try:
                self._imu_sub.undeclare()
            except Exception:
                pass
            self._imu_sub = None

    def _on_imu(self, sample: Any) -> None:
        """Zenoh callback for body/imu. Feeds ImuReading samples into
        the tracker. Best-effort: any parse failure is silently dropped
        so a single bad payload can't tear down the consumer.
        """
        try:
            msg = json.loads(bytes(sample.payload))
        except Exception:
            return
        reading = ImuReading.from_payload(msg)
        if reading is not None:
            self._imu_yaw.update(reading)

    # ── Motion shutdown ──────────────────────────────────────────────

    def _stop_motion(self) -> None:
        try:
            self.controller.set_cmd_vel(0.0, 0.0)
            self.controller.set_live_command(False)
        except Exception:
            logger.exception("sweep stop_motion raised")

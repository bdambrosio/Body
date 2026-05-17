"""Phase 6.4.2 — VPR startup calibration via sweep-in-place.

Replaces opportunistic anchor accumulation with an explicit preflight:
commands the chassis to rotate ~360° in place at a safe rate, lets
the VPR observer accumulate pairs throughout, then attempts a scored
SE(2) fit. On pass, the anchor is locked and live VPR proceeds; on
fail, the sweep falls through to the current 6.4 opportunistic mode
(continue accumulating during normal driving).

Design (locked from this-session Q&A):
- Trigger: auto on launch when ``--vpr`` is set.
- Failure mode: degrade to opportunistic accumulation (continue running).
- No explicit "skip-calibration" CLI flag — operators who can't sweep
  use ``--no-vpr`` or wait for the opportunistic fit.

Threading
---------
Runs in its own daemon thread so the Qt main loop (run_app) isn't
blocked. The chassis controller's _cmd_loop is the wire-publishing
thread; we just call set_cmd_vel / set_live_command. The shadow VPR
driver continues to accumulate pairs from its Zenoh RGB callbacks
during the sweep — we never touch its internals beyond reading
``anchor()`` after motion stops.

Safety
------
The sweep is a low-rate rotation in place (default 25 °/s); the Pi
applies its own watchdog timeouts on cmd_vel. If anything goes wrong
(estop, SIGINT, exception), the sweep zeroes the command in its
``finally`` block.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from .anchor import (
    CalibrationScore,
    CalibrationScoringConfig,
    score_calibration,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSweepConfig:
    # Total rotation in radians (default 2π + a little margin so we
    # actually close the loop).
    sweep_total_rad: float = 2.0 * math.pi + math.radians(20.0)

    # Angular speed of the rotation (rad/s). 25 °/s is gentle enough
    # to keep RGB capture sharp and let the IMU track without lag.
    sweep_speed_rad_s: float = math.radians(25.0)

    # Delay before issuing motion — gives the PF time to seed at the
    # session origin and the VPR driver to subscribe.
    startup_delay_s: float = 3.0

    # Settle time after motion stops, before evaluating calibration.
    # The last few RGB frames in the sweep might still be in flight.
    settle_after_motion_s: float = 1.0

    # Scoring thresholds applied to the post-sweep pairs.
    scoring: CalibrationScoringConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.scoring is None:
            object.__setattr__(self, "scoring", CalibrationScoringConfig())


class VPRCalibrationSweep:
    """Run a single sweep-then-score attempt, then exit. Construct
    once per session; call ``start()`` to spawn the worker thread."""

    def __init__(
        self,
        *,
        chassis: Any,                 # StubController (has set_cmd_vel, set_live_command)
        vpr_driver: Any,              # ShadowVPRDriver
        config: Optional[CalibrationSweepConfig] = None,
        on_complete: Optional[callable] = None,
    ) -> None:
        self._chassis = chassis
        self._driver = vpr_driver
        self._cfg = config or CalibrationSweepConfig()
        self._on_complete = on_complete
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._score: Optional[CalibrationScore] = None
        self._done = threading.Event()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def score(self) -> Optional[CalibrationScore]:
        return self._score

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="vpr-calibration-sweep", daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        """Best-effort early termination. The motion stop in the
        ``finally`` block still executes."""
        self._stop.set()

    # ── Worker ───────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._driver.log_event("vpr_calibration", {
                "phase": "starting",
                "sweep_total_rad": self._cfg.sweep_total_rad,
                "sweep_speed_rad_s": self._cfg.sweep_speed_rad_s,
                "startup_delay_s": self._cfg.startup_delay_s,
            })
            if self._stop.wait(self._cfg.startup_delay_s):
                self._fail("cancelled_before_motion")
                return

            duration_s = self._cfg.sweep_total_rad / max(
                1e-3, self._cfg.sweep_speed_rad_s,
            )
            pairs_before = self._driver.anchor().n_pairs_collected
            logger.info(
                "vpr_calibration: starting %.2f rad sweep at %.2f rad/s "
                "(%.1f s); %d pairs already collected",
                self._cfg.sweep_total_rad, self._cfg.sweep_speed_rad_s,
                duration_s, pairs_before,
            )
            self._driver.log_event("vpr_calibration", {
                "phase": "motion_start",
                "expected_duration_s": duration_s,
                "pairs_before": pairs_before,
            })

            # Drive the rotation. Re-set the command periodically in
            # case any external override clears it (e.g. operator
            # twitch on joystick → cmd_vel zero → we'd stop).
            self._chassis.set_cmd_vel(0.0, self._cfg.sweep_speed_rad_s)
            self._chassis.set_live_command(True)
            sweep_end_mono = time.monotonic() + duration_s
            try:
                while not self._stop.is_set():
                    now = time.monotonic()
                    if now >= sweep_end_mono:
                        break
                    # Re-affirm command each tick so a stray
                    # set_cmd_vel(0, 0) elsewhere doesn't silently
                    # halt the sweep.
                    self._chassis.set_cmd_vel(
                        0.0, self._cfg.sweep_speed_rad_s,
                    )
                    if self._stop.wait(0.1):
                        break
            finally:
                self._chassis.set_cmd_vel(0.0, 0.0)
                # Leave live_command armed — the operator may take
                # over immediately, and toggling live_command can
                # race with their input.

            if self._stop.is_set():
                self._fail("cancelled_during_motion")
                return

            # Let the last few RGB frames in flight settle into
            # the anchor estimator.
            if self._stop.wait(self._cfg.settle_after_motion_s):
                self._fail("cancelled_during_settle")
                return

            self._score_and_install()
        except Exception:
            logger.exception("vpr_calibration: sweep crashed")
            self._driver.log_event("vpr_calibration", {
                "phase": "crashed",
            })
            self._done.set()
            if self._on_complete is not None:
                try:
                    self._on_complete(None)
                except Exception:
                    logger.exception(
                        "vpr_calibration: on_complete raised",
                    )

    def _fail(self, reason: str) -> None:
        logger.warning("vpr_calibration: sweep aborted (%s)", reason)
        self._driver.log_event("vpr_calibration", {
            "phase": "aborted", "reason": reason,
        })
        self._done.set()
        if self._on_complete is not None:
            try:
                self._on_complete(None)
            except Exception:
                logger.exception("vpr_calibration: on_complete raised")

    def _score_and_install(self) -> None:
        anchor = self._driver.anchor()
        pairs = anchor.snapshot_pairs()
        score = score_calibration(pairs, self._cfg.scoring)
        self._score = score
        if score.passed and score.offset is not None and anchor.calibration is None:
            anchor.set_calibration(score.offset)
            logger.info(
                "vpr_calibration: PASSED — n=%d, rms=%.3f m, "
                "cov_xy_trace=%.4f m², Δ=(%+.3f, %+.3f, %+.2f°)",
                score.n_pairs, score.residual_rms_m,
                score.cov_xy_trace_m2,
                score.offset.dx, score.offset.dy,
                math.degrees(score.offset.dtheta_rad),
            )
        else:
            logger.warning(
                "vpr_calibration: FAILED (%s) — n=%d rms=%.3f cov=%.4f "
                "spread=%.2fm cells=%d. Falling through to opportunistic "
                "accumulation.",
                score.reason, score.n_pairs, score.residual_rms_m,
                score.cov_xy_trace_m2, score.spatial_spread_m,
                score.n_unique_bank_cells,
            )
        self._driver.log_event("vpr_calibration", {
            "phase": "complete",
            "passed": score.passed,
            "reason": score.reason,
            "n_pairs": score.n_pairs,
            "n_unique_bank_cells": score.n_unique_bank_cells,
            "spatial_spread_m": score.spatial_spread_m,
            "residual_rms_m": score.residual_rms_m,
            "cov_xy_trace_m2": score.cov_xy_trace_m2,
            "cov_theta_var_rad2": score.cov_theta_var_rad2,
            "offset": (
                {
                    "dx": score.offset.dx, "dy": score.offset.dy,
                    "dtheta_rad": score.offset.dtheta_rad,
                    "n_pairs": score.offset.n_pairs,
                    "residual_rms_m": score.offset.residual_rms_m,
                } if score.offset is not None else None
            ),
        })
        self._done.set()
        if self._on_complete is not None:
            try:
                self._on_complete(score)
            except Exception:
                logger.exception("vpr_calibration: on_complete raised")


__all__ = [
    "CalibrationSweepConfig",
    "VPRCalibrationSweep",
]

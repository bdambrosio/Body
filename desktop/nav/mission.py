"""Mission state machine.

Tracks the high-level "are we currently driving autonomously?" state.
The state transitions are deliberately small and explicit so the
operator (and the UI) always knows what hitting Go or Cancel will
actually do.

States:

    IDLE        no autonomous drive is active. May or may not have a
                plan loaded. Pressing Go transitions to FOLLOWING if
                preconditions hold.
    FOLLOWING   actively driving. Each tick, main_window pushes the
                follower's cmd_vel to chassis.set_cmd_vel(). Stays
                here until the follower reports arrival, the operator
                cancels, the mission pauses, or a fault is detected.
    PAUSED      cmd_vel zeroed, waiting for a transient condition to
                clear. `pause_reason` carries the why
                (e.g. "no_pose", "no_path:goal_in_unknown"). Resumes
                automatically when the condition clears, OR escalates
                to RECOVERING if the recovery policy elects to act.
    RECOVERING  running a recovery primitive (e.g. 360-rotate, back-up)
                to clear the paused condition. cmd_vel is driven by
                the primitive, NOT the follower. `recovery_action`
                names the primitive. Returns to FOLLOWING on success,
                back to PAUSED on failure (next tick the policy may
                pick a different action), or transitions to FAILED if
                the policy exhausts its options.
    ARRIVED     follower reached the goal. cmd_vel zeroed.
    CANCELED    operator pressed Cancel. cmd_vel zeroed.
    FAILED      mission ended on an error (no plan, chassis disconnect,
                Live cmd dropped, recovery exhausted, ...).
                `failure_reason` holds a human-readable explanation.

Terminal states (ARRIVED / CANCELED / FAILED) are reset back to IDLE
on the next start() — i.e. pressing Go again after a finish.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .tracing import Tracer


class MissionState:
    IDLE = "IDLE"
    FOLLOWING = "FOLLOWING"
    PAUSED = "PAUSED"
    RECOVERING = "RECOVERING"
    # Patrols only: between FOLLOWING (arrived at wp[i]) and the next
    # FOLLOWING (toward wp[i+1]). cmd_vel during this state comes from
    # a RotateToHeading primitive, not the follower.
    ROTATING_TO_NEXT = "ROTATING_TO_NEXT"
    ARRIVED = "ARRIVED"
    CANCELED = "CANCELED"
    FAILED = "FAILED"


_TERMINAL = {MissionState.ARRIVED, MissionState.CANCELED, MissionState.FAILED}
_ACTIVE = {
    MissionState.FOLLOWING, MissionState.PAUSED, MissionState.RECOVERING,
    MissionState.ROTATING_TO_NEXT,
}


@dataclass
class MissionConfig:
    # How stale the latest pose can be before mission pauses for "no_pose".
    # Odom publishes at ~10 Hz on the Pi; 0.50 s = 5 missed ticks. Generous
    # but tight enough to catch a real failure quickly.
    pose_age_threshold_s: float = 0.50

    # Time spent in PAUSED before recovery may escalate. Lets transient
    # conditions (someone walks across, scan-match catches up) clear on
    # their own without spinning the robot in their face.
    pause_grace_s: float = 1.5

    # Cap on consecutive recovery attempts within a single mission before
    # we give up and FAIL. Counted across reasons — three swings at it
    # then call it a day.
    max_recovery_attempts: int = 3

    # Hard ceiling on PAUSED("no_pose") duration. Beyond this the mission
    # transitions to FAILED rather than waiting forever for the pose
    # source to come back. 30 s comfortably absorbs a brief Pi-side
    # local_map stall but won't wedge the operator on a dead Pi.
    no_pose_timeout_s: float = 30.0


@dataclass
class Mission:
    state: str = MissionState.IDLE
    failure_reason: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    # Pause / recovery telemetry. Reset on start().
    pause_reason: str = ""
    pause_started_at: float = 0.0
    recovery_action: str = ""
    recovery_attempts: int = 0

    # Patrol-only: index of the waypoint the robot is currently
    # targeting and the number of laps completed since start. Single-
    # goal missions leave these at zero (a single goal is conceptually
    # a Patrol of length 1, loop=False, laps=1).
    waypoint_index: int = 0
    lap_index: int = 0
    # During ROTATING_TO_NEXT, the world-frame heading the robot is
    # spinning toward. Informational only — the primitive owns the
    # actual rotation; this is for tracing / UI labels.
    rotate_target_rad: float = 0.0

    # Optional Tracer for emitting state transitions. Wired by
    # main_window; tests construct Mission() with no tracer. None means
    # "transitions still run normally, just no trace output."
    tracer: Optional["Tracer"] = None

    def _emit(self, category: str, event: str, **data: Any) -> None:
        t = self.tracer
        if t is None:
            return
        try:
            t.emit(category, event, data)
        except Exception:
            # Tracing must never crash the mission tick.
            pass

    def is_active(self) -> bool:
        """True when the mission is in any state that wants the follower
        to be running and the chassis to be live (FOLLOWING / PAUSED /
        RECOVERING)."""
        return self.state in _ACTIVE

    def is_following(self) -> bool:
        return self.state == MissionState.FOLLOWING

    def is_paused(self) -> bool:
        return self.state == MissionState.PAUSED

    def is_recovering(self) -> bool:
        return self.state == MissionState.RECOVERING

    def is_rotating_to_next(self) -> bool:
        return self.state == MissionState.ROTATING_TO_NEXT

    def is_terminal(self) -> bool:
        return self.state in _TERMINAL

    def can_start(self) -> bool:
        """A start is allowed from IDLE or any terminal state — the
        latter so the operator can re-run after arriving / canceling
        / failing without an extra "clear" click.
        """
        return self.state == MissionState.IDLE or self.is_terminal()

    def can_cancel(self) -> bool:
        return self.is_active()

    # ── Transitions ─────────────────────────────────────────────────

    def start(self) -> None:
        self.state = MissionState.FOLLOWING
        self.failure_reason = ""
        self.started_at = time.time()
        self.finished_at = 0.0
        self.pause_reason = ""
        self.pause_started_at = 0.0
        self.recovery_action = ""
        self.recovery_attempts = 0
        self.waypoint_index = 0
        self.lap_index = 0
        self.rotate_target_rad = 0.0
        self._emit("mission", "start")

    def arrive(self) -> None:
        if self.is_active():
            self.state = MissionState.ARRIVED
            self.finished_at = time.time()
            self.pause_reason = ""
            self.recovery_action = ""
            self._emit("mission", "arrive")

    def cancel(self) -> None:
        # Cancel is benign on terminal states (no-op); allowed from any
        # active state.
        if self.is_active():
            self.state = MissionState.CANCELED
            self.finished_at = time.time()
            self.pause_reason = ""
            self.recovery_action = ""
            self._emit("mission", "cancel")

    def fail(self, reason: str) -> None:
        self.state = MissionState.FAILED
        self.failure_reason = reason
        self.finished_at = time.time()
        self.pause_reason = ""
        self.recovery_action = ""
        self._emit("mission", "fail", reason=reason)

    def pause(self, reason: str) -> None:
        """Enter PAUSED with the given reason. If already PAUSED with
        the same reason, leave the timer alone — successive ticks
        re-asserting the condition shouldn't reset the grace clock.
        Re-entering with a different reason resets the clock so the new
        reason gets its full grace window.
        """
        now = time.time()
        # Idempotent re-entry: same reason, no emit so the trace stays
        # edge-triggered.
        if self.state == MissionState.PAUSED and self.pause_reason == reason:
            return
        # PAUSED → PAUSED with new reason, FOLLOWING/RECOVERING → PAUSED.
        # Do nothing from terminal states; pause should only be invoked
        # while the mission is active.
        if not self.is_active():
            return
        prev_state = self.state
        self.state = MissionState.PAUSED
        self.pause_reason = reason
        self.pause_started_at = now
        self.recovery_action = ""
        self._emit("mission", "pause", reason=reason, from_state=prev_state)

    def resume(self) -> None:
        """PAUSED or RECOVERING → FOLLOWING. The condition that caused
        the pause has cleared (fresh pose arrived, or recovery action
        succeeded)."""
        if self.state in (MissionState.PAUSED, MissionState.RECOVERING):
            prev_state = self.state
            prev_reason = self.pause_reason
            self.state = MissionState.FOLLOWING
            self.pause_reason = ""
            self.recovery_action = ""
            self._emit(
                "mission", "resume",
                from_state=prev_state, prev_reason=prev_reason,
            )

    def begin_recovery(self, action: str) -> None:
        """PAUSED → RECOVERING. The policy has chosen to run a
        primitive (e.g. 360-rotate). The primitive drives cmd_vel until
        it reports done; main_window then calls end_recovery().
        """
        if self.state != MissionState.PAUSED:
            return
        reason = self.pause_reason
        self.state = MissionState.RECOVERING
        self.recovery_action = action
        self.recovery_attempts += 1
        self._emit(
            "recovery", "begin",
            action=action,
            reason=reason,
            attempt=self.recovery_attempts,
        )

    def end_recovery(self, success: bool) -> None:
        """RECOVERING → FOLLOWING (success) or PAUSED (failure, allowing
        the next tick's policy to pick a different action or escalate to
        FAIL)."""
        if self.state != MissionState.RECOVERING:
            return
        prior_action = self.recovery_action
        attempts = self.recovery_attempts
        if success:
            self.state = MissionState.FOLLOWING
            self.pause_reason = ""
            self.recovery_action = ""
            self._emit(
                "recovery", "end",
                action=prior_action, success=True, attempt=attempts,
            )
        else:
            # Drop back to PAUSED with an explanatory reason so the
            # policy / operator can see why we re-paused.
            self.state = MissionState.PAUSED
            self.pause_reason = f"recovery_failed:{prior_action}"
            self.pause_started_at = time.time()
            self.recovery_action = ""
            self._emit(
                "recovery", "end",
                action=prior_action, success=False, attempt=attempts,
            )

    def reset(self) -> None:
        """Force back to IDLE. Used when the goal is cleared, so the
        terminal state from the previous run doesn't linger and
        confuse the next planning attempt."""
        self.state = MissionState.IDLE
        self.failure_reason = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self.pause_reason = ""
        self.pause_started_at = 0.0
        self.recovery_action = ""
        self.recovery_attempts = 0
        self.waypoint_index = 0
        self.lap_index = 0
        self.rotate_target_rad = 0.0

    # ── Patrol transitions ─────────────────────────────────────────

    def begin_rotation_to_next(
        self,
        target_theta_rad: float,
        *,
        to_wp_index: Optional[int] = None,
    ) -> None:
        """FOLLOWING → ROTATING_TO_NEXT. Called after arriving at a
        non-final waypoint of a patrol, before driving the next leg.
        Emits `patrol.rotating` with the target heading; the actual
        spin is driven by a RotateToHeading primitive in main_window.

        `to_wp_index` (optional) is the index the patrol is rotating
        TOWARD — captured in the trace so a reviewer can read intent
        without inferring it from waypoint geometry. Omitted in
        non-patrol contexts (no current caller, but reserved).
        """
        if self.state != MissionState.FOLLOWING:
            return
        self.state = MissionState.ROTATING_TO_NEXT
        self.rotate_target_rad = float(target_theta_rad)
        data = {
            "from_wp_index": self.waypoint_index,
            "lap_index": self.lap_index,
            "target_theta_rad": self.rotate_target_rad,
        }
        if to_wp_index is not None:
            data["to_wp_index"] = int(to_wp_index)
        t = self.tracer
        if t is not None:
            try:
                t.emit("patrol", "rotating", data)
            except Exception:
                pass

    def complete_rotation_to_next(
        self,
        new_wp_index: int,
        new_lap_index: int,
        lap_completed: bool,
    ) -> None:
        """ROTATING_TO_NEXT → FOLLOWING. Update wp/lap indices to the
        new leg's target. Emits `patrol.lap_complete` if this advance
        closed a lap, and always emits `patrol.advance` with the new
        index pair.
        """
        if self.state != MissionState.ROTATING_TO_NEXT:
            return
        prev_wp = self.waypoint_index
        prev_lap = self.lap_index
        self.waypoint_index = int(new_wp_index)
        self.lap_index = int(new_lap_index)
        self.state = MissionState.FOLLOWING
        self.rotate_target_rad = 0.0
        if lap_completed:
            self._emit(
                "patrol", "lap_complete",
                lap_index=self.lap_index,
                prev_lap_index=prev_lap,
            )
        self._emit(
            "patrol", "advance",
            wp_index=self.waypoint_index,
            prev_wp_index=prev_wp,
            lap_index=self.lap_index,
        )

    def abort_rotation_to_next(self) -> None:
        """ROTATING_TO_NEXT → FOLLOWING with no advance. Used when the
        primitive is canceled (operator stop, pose loss, etc.)."""
        if self.state != MissionState.ROTATING_TO_NEXT:
            return
        self.state = MissionState.FOLLOWING
        self.rotate_target_rad = 0.0
        self._emit("patrol", "rotation_aborted")

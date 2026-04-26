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


class MissionState:
    IDLE = "IDLE"
    FOLLOWING = "FOLLOWING"
    PAUSED = "PAUSED"
    RECOVERING = "RECOVERING"
    ARRIVED = "ARRIVED"
    CANCELED = "CANCELED"
    FAILED = "FAILED"


_TERMINAL = {MissionState.ARRIVED, MissionState.CANCELED, MissionState.FAILED}
_ACTIVE = {MissionState.FOLLOWING, MissionState.PAUSED, MissionState.RECOVERING}


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

    def arrive(self) -> None:
        if self.is_active():
            self.state = MissionState.ARRIVED
            self.finished_at = time.time()
            self.pause_reason = ""
            self.recovery_action = ""

    def cancel(self) -> None:
        # Cancel is benign on terminal states (no-op); allowed from any
        # active state.
        if self.is_active():
            self.state = MissionState.CANCELED
            self.finished_at = time.time()
            self.pause_reason = ""
            self.recovery_action = ""

    def fail(self, reason: str) -> None:
        self.state = MissionState.FAILED
        self.failure_reason = reason
        self.finished_at = time.time()
        self.pause_reason = ""
        self.recovery_action = ""

    def pause(self, reason: str) -> None:
        """Enter PAUSED with the given reason. If already PAUSED with
        the same reason, leave the timer alone — successive ticks
        re-asserting the condition shouldn't reset the grace clock.
        Re-entering with a different reason resets the clock so the new
        reason gets its full grace window.
        """
        now = time.time()
        if self.state == MissionState.PAUSED and self.pause_reason == reason:
            return
        # PAUSED → PAUSED with new reason, FOLLOWING/RECOVERING → PAUSED.
        # Do nothing from terminal states; pause should only be invoked
        # while the mission is active.
        if not self.is_active():
            return
        self.state = MissionState.PAUSED
        self.pause_reason = reason
        self.pause_started_at = now
        self.recovery_action = ""

    def resume(self) -> None:
        """PAUSED or RECOVERING → FOLLOWING. The condition that caused
        the pause has cleared (fresh pose arrived, or recovery action
        succeeded)."""
        if self.state in (MissionState.PAUSED, MissionState.RECOVERING):
            self.state = MissionState.FOLLOWING
            self.pause_reason = ""
            self.recovery_action = ""

    def begin_recovery(self, action: str) -> None:
        """PAUSED → RECOVERING. The policy has chosen to run a
        primitive (e.g. 360-rotate). The primitive drives cmd_vel until
        it reports done; main_window then calls end_recovery().
        """
        if self.state != MissionState.PAUSED:
            return
        self.state = MissionState.RECOVERING
        self.recovery_action = action
        self.recovery_attempts += 1

    def end_recovery(self, success: bool) -> None:
        """RECOVERING → FOLLOWING (success) or PAUSED (failure, allowing
        the next tick's policy to pick a different action or escalate to
        FAIL)."""
        if self.state != MissionState.RECOVERING:
            return
        if success:
            self.state = MissionState.FOLLOWING
            self.pause_reason = ""
            self.recovery_action = ""
        else:
            # Drop back to PAUSED with an explanatory reason so the
            # policy / operator can see why we re-paused.
            prior_action = self.recovery_action
            self.state = MissionState.PAUSED
            self.pause_reason = f"recovery_failed:{prior_action}"
            self.pause_started_at = time.time()
            self.recovery_action = ""

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

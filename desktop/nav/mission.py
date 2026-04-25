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
                cancels, or a fault is detected.
    ARRIVED     follower reached the goal. cmd_vel zeroed.
    CANCELED    operator pressed Cancel. cmd_vel zeroed.
    FAILED      mission ended on an error (no plan, no pose, chassis
                disconnect, Live cmd dropped, ...). `failure_reason`
                holds a human-readable explanation.

The terminal states (ARRIVED / CANCELED / FAILED) are reset back to
IDLE on the next start() — i.e. pressing Go again after a finish.
This keeps the visible state until the operator acknowledges it by
choosing what to do next.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


class MissionState:
    IDLE = "IDLE"
    FOLLOWING = "FOLLOWING"
    ARRIVED = "ARRIVED"
    CANCELED = "CANCELED"
    FAILED = "FAILED"


_TERMINAL = {MissionState.ARRIVED, MissionState.CANCELED, MissionState.FAILED}


@dataclass
class Mission:
    state: str = MissionState.IDLE
    failure_reason: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    def is_active(self) -> bool:
        return self.state == MissionState.FOLLOWING

    def is_terminal(self) -> bool:
        return self.state in _TERMINAL

    def can_start(self) -> bool:
        """A start is allowed from IDLE or any terminal state — the
        latter so the operator can re-run after arriving / canceling
        / failing without an extra "clear" click.
        """
        return self.state == MissionState.IDLE or self.is_terminal()

    def can_cancel(self) -> bool:
        return self.state == MissionState.FOLLOWING

    # ── Transitions ─────────────────────────────────────────────────

    def start(self) -> None:
        self.state = MissionState.FOLLOWING
        self.failure_reason = ""
        self.started_at = time.time()
        self.finished_at = 0.0

    def arrive(self) -> None:
        if self.state == MissionState.FOLLOWING:
            self.state = MissionState.ARRIVED
            self.finished_at = time.time()

    def cancel(self) -> None:
        # Cancel is benign on terminal states (no-op); only forbidden
        # on IDLE.
        if self.state == MissionState.FOLLOWING:
            self.state = MissionState.CANCELED
            self.finished_at = time.time()

    def fail(self, reason: str) -> None:
        self.state = MissionState.FAILED
        self.failure_reason = reason
        self.finished_at = time.time()

    def reset(self) -> None:
        """Force back to IDLE. Used when the goal is cleared, so the
        terminal state from the previous run doesn't linger and
        confuse the next planning attempt."""
        self.state = MissionState.IDLE
        self.failure_reason = ""
        self.started_at = 0.0
        self.finished_at = 0.0

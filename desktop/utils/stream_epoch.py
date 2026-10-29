"""Stream-epoch detection for Pi telemetry (body/odom, body/imu).

The Pi producers stamp each message with ``session_id`` (random per process
start) and ``seq`` (+1 per publish). A process restart rebases its frame —
odom restarts at (0, 0, 0), the IMU's game-rotation-vector yaw origin is
arbitrary — so consumers must re-baseline instead of integrating the jump
(X4, I4). Both fields are optional on the wire (older Pi): absent fields
never trigger a reset.

A backward step of the Pi wall clock (``ts`` going back by more than
``clock_step_s``) is also reported, since ts-ordered buffers would otherwise
reject every later sample (S7).
"""
from __future__ import annotations

from typing import Optional


class StreamEpoch:
    """Detect a producer restart / reset from (session_id, seq, ts)."""

    def __init__(self, clock_step_s: float = 1.0) -> None:
        self._session: Optional[str] = None
        self._seq: Optional[int] = None
        self._ts: Optional[float] = None
        self._clock_step_s = float(clock_step_s)
        self.resets = 0
        self.last_reason: Optional[str] = None

    def check(
        self,
        session_id: Optional[str] = None,
        seq: Optional[int] = None,
        ts: Optional[float] = None,
    ) -> Optional[str]:
        """Feed one message's identity. Returns a reason string
        (``"session"``, ``"seq"``, ``"clock_step"``) when the stream was reset
        since the previous message, else None. The first message never
        resets."""
        reason: Optional[str] = None
        if (session_id is not None and self._session is not None
                and session_id != self._session):
            reason = "session"
        elif (seq is not None and self._seq is not None and seq < self._seq
                and (session_id is None or session_id == self._session)):
            reason = "seq"
        elif (ts is not None and self._ts is not None
                and ts < self._ts - self._clock_step_s):
            reason = "clock_step"
        if session_id is not None:
            self._session = session_id
        if seq is not None:
            self._seq = seq
        if ts is not None:
            self._ts = ts if (reason is not None or self._ts is None) else max(self._ts, ts)
        if reason is not None:
            self.resets += 1
            self.last_reason = reason
        return reason

    def is_duplicate_seq(self, session_id: Optional[str], seq: Optional[int]) -> bool:
        """True when ``seq`` equals the last seen seq of the same session —
        a re-publish of a report already ingested (not new data)."""
        return (seq is not None and self._seq is not None and seq == self._seq
                and (session_id is None or session_id == self._session))


def epoch_fields(msg: dict) -> tuple:
    """(session_id, seq) from a Pi message, tolerating absent/garbage fields
    (older Pi) → (None, None)."""
    sid = msg.get("session_id")
    session_id = str(sid) if sid is not None else None
    seq: Optional[int]
    try:
        seq = int(msg["seq"]) if msg.get("seq") is not None else None
    except (TypeError, ValueError):
        seq = None
    return session_id, seq

"""What the nav app says out loud, and when (published to body/voice/say).

The desktop voice service (``python -m desktop.voice``) turns the requests into
speech on the robot's speaker; if it isn't running, the requests go nowhere.
``VoiceAnnouncer`` is transport-free: it hands schemas.voice_say dicts to
``send``. Rate limits live in two places: per-event edges here (one
announcement per detection / block episode) and per-key cooldowns in the
voice service's queue.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from body.lib import schemas

logger = logging.getLogger(__name__)

SIGHTING_PHRASES = {
    "person": "Hello! I see a person.",
    "pet": "Oh, I see a pet!",
    "small stuffed animal": "I found a stuffed toy.",
    "box or bag": "There's a box here.",
}
SIGHTING_COOLDOWN_S = 60.0

STUCK_PHRASES = {
    "depth_block": "Something is in my way.",
    "swept_block": "Something is in my way.",
    "path_blocked": "My path is blocked.",
    "path_blocked_live": "My path is blocked.",
    "no_route": "I can't find a way there.",
    "terminal_unreachable": "I can't reach that spot.",
    "leg_timeout": "I'm not making progress.",
}
STUCK_AFTER_S = 4.0        # a block shorter than this clears on its own; don't announce it
STUCK_COOLDOWN_S = 20.0
STARTUP_QUIET_S = 10.0     # an e-stop already latched when nav starts is old news


def sighting_phrase(label: Optional[str]) -> Optional[str]:
    return SIGHTING_PHRASES.get(label or "")


def stuck_phrase(reason: Optional[str]) -> str:
    detail = STUCK_PHRASES.get(reason or "")
    return f"I'm stuck. {detail}" if detail else "I'm stuck."


class VoiceAnnouncer:
    def __init__(self, send: Callable[[dict], None]) -> None:
        self._send = send
        self.muted = False
        self._announced_changes: set = set()
        self._blocked_since: Optional[float] = None
        self._block_announced = False
        self._prev_state: Optional[str] = None
        self._prev_estop = False
        self._first_tick: Optional[float] = None

    def say(self, text: str, *, kind: str = "info", key: Optional[str] = None,
            cooldown_s: float = 0.0) -> None:
        if self.muted:
            return
        try:
            self._send(schemas.voice_say(text, kind=kind, key=key, cooldown_s=cooldown_s))
        except Exception:
            logger.exception("voice say failed")

    # ── events ──────────────────────────────────────────────────────

    def on_detection(self, d: Any) -> None:
        """A Detection row was updated (desktop/nav/detections.py)."""
        if d is None or d.change_id in self._announced_changes:
            return
        if getattr(d, "verdict", None) in ("expected", "candidate"):
            return  # routine for this place
        phrase = sighting_phrase(getattr(d, "label", None))
        if phrase is None:
            return
        self._announced_changes.add(d.change_id)
        self.say(phrase, kind="info", key=f"sight:{d.label}", cooldown_s=SIGHTING_COOLDOWN_S)

    def on_tick(self, state: Optional[str], block_reason: Optional[str],
                fail_reason: Optional[str], estop: bool, now: float) -> None:
        """Called every hier tick with the drive's state name (None = no drive)."""
        if self._first_tick is None:
            self._first_tick = now
        if estop and not self._prev_estop and now - self._first_tick >= STARTUP_QUIET_S:
            self.say("Emergency stop.", kind="alert", key="estop", cooldown_s=5.0)
        self._prev_estop = estop

        if state == "BLOCKED":
            if self._blocked_since is None:
                self._blocked_since = now
                self._block_announced = False
            elif not self._block_announced and now - self._blocked_since >= STUCK_AFTER_S:
                self._block_announced = True
                self.say(stuck_phrase(block_reason), kind="alert", key="stuck",
                         cooldown_s=STUCK_COOLDOWN_S)
        else:
            self._blocked_since = None

        if state == "FAILED" and self._prev_state != "FAILED":
            text = "I give up. I couldn't get there." if fail_reason == "abandoned" \
                else "I couldn't get there."
            self.say(text, kind="alert", key="failed", cooldown_s=10.0)
        self._prev_state = state

    def on_task_finished(self, why: str) -> None:
        text = "Exploration finished." if why in ("done", "complete", "finished") \
            else f"Exploration {why}."
        self.say(text, kind="info", key="explore_end", cooldown_s=5.0)

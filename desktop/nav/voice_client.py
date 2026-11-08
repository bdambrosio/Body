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
from typing import Any, Callable, Iterable, Optional

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

# ── Explore ─────────────────────────────────────────────────────────
# Spoken names for the change labels (desktop/perception/change_report.py CHANGE_CLASSES).
NOUNS = {
    "person": "person", "pet": "pet", "small stuffed animal": "stuffed toy",
    "box or bag": "box", "furniture": "piece of furniture",
}
# Labels that never make a "found something new" announcement on their own.
NOT_A_FIND = ("door", "nothing there")
FIND_MIN_CONFIDENCE = 0.6
FIND_COOLDOWN_S = 20.0     # several finds in a row: say the first, skip the burst
INVESTIGATE_RESULT_PHRASES = {
    "confirmed": "Yes, there really is {a_noun} there.",
    "gone": "Whatever it was, it's gone now.",
    "dismissed": "False alarm. There's nothing there.",
    "unreachable": "I couldn't get close enough to look.",
}


def _a(noun: str) -> str:
    noun = noun.strip()
    low = noun.lower()
    if low.startswith(("a ", "an ", "the ", "some ", "two ", "three ", "several ", "my ", "your ")):
        return noun
    return ("an " if low[:1] in "aeiou" else "a ") + noun


def thing_name(label: Optional[str], what: Optional[str]) -> str:
    """'stuffed toy' for a known label; the VLM's few words for 'other'; else 'something'."""
    if label in NOUNS:
        return NOUNS[label]
    w = " ".join((what or "").replace(".", " ").split())
    return w.lower() if 2 <= len(w) <= 40 else "thing"


def _plural(noun: str, n: int) -> str:
    if n == 1:
        return _a(noun)
    if noun == "piece of furniture":
        return f"{n} pieces of furniture"
    if noun == "person":
        return f"{n} people"
    return f"{n} {noun}es" if noun.endswith(("x", "s", "ch", "sh")) else f"{n} {noun}s"


def _and_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def explore_summary_phrase(why: str, coverage: Optional[float], finds: Iterable[Any]) -> str:
    """'Exploration finished. I looked over 72 percent of the house and found a box and 2 stuffed toys.'"""
    if why not in ("done", "complete", "finished"):
        return f"Exploration {why}."
    counts: dict[str, int] = {}
    for d in finds:
        n = thing_name(getattr(d, "label", None), getattr(d, "what", None))
        counts[n] = counts.get(n, 0) + 1
    parts = ["Exploration finished."]
    cov = f"I looked over {round(coverage * 100)} percent of the house" if coverage else "I"
    if counts:
        found = _and_list([_plural(n, c) for n, c in counts.items()][:4])
        parts.append(f"{cov} and found {found}." if coverage else f"I found {found}.")
    else:
        parts.append(f"{cov}, and found nothing new." if coverage else "I found nothing new.")
    return " ".join(parts)


def is_find(d: Any) -> bool:
    """A detection worth naming in the explore summary / a 'found something' line."""
    if d is None or getattr(d, "verdict", None) in ("expected", "candidate"):
        return False
    if getattr(d, "investigated", None) in ("gone", "dismissed"):
        return False
    label = getattr(d, "label", None)
    if label in SIGHTING_PHRASES:
        return True
    if label is None or label in NOT_A_FIND:
        return False
    conf = getattr(d, "confidence", None)
    return getattr(d, "kind", None) in ("new", "moving") and conf is not None \
        and conf >= FIND_MIN_CONFIDENCE


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
        self.exploring = False
        self._announced_results: set = set()
        self._announced_looks: set = set()

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
        if d is None:
            return
        if self.exploring:
            self._explore_result(d)
        if d.change_id in self._announced_changes:
            return
        if getattr(d, "verdict", None) in ("expected", "candidate"):
            return  # routine for this place
        phrase = sighting_phrase(getattr(d, "label", None))
        if phrase is not None:
            self._announced_changes.add(d.change_id)
            self.say(phrase, kind="info", key=f"sight:{d.label}", cooldown_s=SIGHTING_COOLDOWN_S)
        elif self.exploring and is_find(d):
            self._announced_changes.add(d.change_id)
            name = thing_name(d.label, getattr(d, "what", None))
            self.say(f"I found something new: {_a(name)}.", kind="info", key="explore_find",
                     cooldown_s=FIND_COOLDOWN_S)

    def _explore_result(self, d: Any) -> None:
        res = getattr(d, "investigated", None)
        if res not in INVESTIGATE_RESULT_PHRASES or d.change_id in self._announced_results:
            return
        self._announced_results.add(d.change_id)
        name = thing_name(d.label, getattr(d, "what", None)) if d.label else "something"
        a_noun = _a(name) if d.label else "something"
        self.say(INVESTIGATE_RESULT_PHRASES[res].format(a_noun=a_noun), kind="info",
                 key=f"look_result:{d.change_id}")

    # ── explore task ────────────────────────────────────────────────

    def on_explore_start(self) -> None:
        self.exploring = True
        self._announced_results.clear()
        self._announced_looks.clear()
        self.say("Okay, I'm going exploring.", kind="info", key="explore_start", cooldown_s=5.0)

    def on_explore_leg(self, purpose: str, change: Any = None) -> None:
        """An explore leg starts: purpose viewpoint | investigate | home."""
        if purpose == "investigate":
            cid = getattr(change, "change_id", None)
            if cid in self._announced_looks:
                return
            self._announced_looks.add(cid)
            label = getattr(change, "label", None)
            text = (f"I'm going to take a closer look at that {thing_name(label, getattr(change, 'what', None))}."
                    if label and label not in NOT_A_FIND
                    else "Something changed over there. Let me take a closer look.")
            self.say(text, kind="info", key=f"look:{cid}")
        elif purpose == "home":
            self.say("Heading home now.", kind="info", key="explore_home", cooldown_s=30.0)

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

    def on_task_finished(self, why: str, coverage: Optional[float] = None,
                         finds: Iterable[Any] = ()) -> None:
        self.exploring = False
        self.say(explore_summary_phrase(why, coverage, finds), kind="info", key="explore_end",
                 cooldown_s=5.0)

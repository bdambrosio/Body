"""Fuzzy wake-word match on a Whisper transcript.

Whisper spells a short, made-up name many ways ("Inky", "Inkie", "Inkey",
"in key", "Hinky"), so the name is matched by similarity, not equality, and
only near the start of the utterance ("Inky, ...", "Hey Inky ...", "OK, Inky").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_WORD = re.compile(r"[a-z0-9']+")
GREETINGS = {"hey", "hi", "hello", "ok", "okay", "oh", "yo", "um", "uh", "so", "and"}


@dataclass(frozen=True)
class WakeMatch:
    matched: bool
    rest: str = ""        # what was said after the name ("" = just the name)
    heard_as: str = ""    # the word(s) taken as the name
    score: float = 0.0


_Y_END = re.compile(r"(ie|ey|ee|ea|i|e)$")


def _norm(word: str) -> str:
    """Lowercase; fold the spellings of a final 'ee' sound to 'y' (inkie/inkey -> inky)."""
    w = word.lower().replace("'", "")
    return _Y_END.sub("y", w) if len(w) > 2 else w


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_wake(text: str, name: str, *, threshold: float = 0.75, max_lead: int = 2) -> WakeMatch:
    """Does ``text`` start with (a greeting and) ``name``?

    Up to ``max_lead`` leading words may come before the name, but only
    greetings/fillers ("hey", "ok", ...). One word or two words joined
    ("in key" -> "inkey") may stand for the name.
    """
    target = _norm(name)
    spans = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text.lower())]
    best = WakeMatch(False)
    for i in range(min(len(spans), max_lead + 1)):
        if i > 0 and spans[i - 1][0].replace("'", "") not in GREETINGS:
            break
        for n in (1, 2):
            if i + n > len(spans):
                break
            cand = "".join(_norm(w) for w, _, _ in spans[i:i + n])
            if len(cand) < 3:
                continue
            score = _sim(cand, target)
            if score >= threshold and score > best.score:
                end = spans[i + n - 1][2]
                rest = text[end:].lstrip(" ,.!?;:-").strip()
                best = WakeMatch(True, rest, text[spans[i][1]:end], round(score, 3))
    return best

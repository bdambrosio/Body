"""What the robot says next: priorities, pre-emption, cooldowns, staleness.

Pure policy (no I/O, clock passed in) so it can be unit-tested.

Kinds, highest first:
  alert — stuck, e-stop: interrupts anything that isn't an alert.
  chat  — replies in a conversation (Phase 2).
  info  — sightings, run summaries: dropped if not started within
          ``info_max_age_s`` (a sighting announced late is wrong).

``key`` + ``cooldown_s``: a request whose key was accepted less than
``cooldown_s`` ago is rejected, so "I see a person" isn't repeated every pass.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

KINDS = ("alert", "chat", "info")
_RANK = {k: i for i, k in enumerate(KINDS)}


@dataclass
class SpeechRequest:
    text: str
    kind: str = "info"
    key: Optional[str] = None
    cooldown_s: float = 0.0
    t_rx: float = 0.0
    order: int = field(default=0, compare=False)

    @property
    def rank(self) -> int:
        return _RANK.get(self.kind, _RANK["info"])


@dataclass
class Offer:
    accepted: bool
    reason: str = ""
    preempt: bool = False  # caller should cut the utterance now playing


class SpeechQueue:
    def __init__(self, *, max_len: int = 6, info_max_age_s: float = 10.0) -> None:
        self.max_len = max_len
        self.info_max_age_s = info_max_age_s
        self._items: list[SpeechRequest] = []
        self._last_key_t: dict[str, float] = {}
        self._seq = itertools.count()
        self.current: Optional[SpeechRequest] = None

    def __len__(self) -> int:
        return len(self._items)

    def offer(self, req: SpeechRequest, now: float) -> Offer:
        text = (req.text or "").strip()
        if not text:
            return Offer(False, "empty")
        if req.kind not in _RANK:
            req.kind = "info"
        req.text = text
        req.t_rx = now
        if req.key and req.cooldown_s > 0:
            last = self._last_key_t.get(req.key)
            if last is not None and now - last < req.cooldown_s:
                return Offer(False, "cooldown")
        if len(self._items) >= self.max_len:
            worst = max(self._items, key=lambda r: (r.rank, r.order))
            if worst.rank < req.rank or (worst.rank == req.rank):
                # queue full of equal-or-better items: the newcomer loses
                return Offer(False, "queue_full")
            self._items.remove(worst)
        req.order = next(self._seq)
        self._items.append(req)
        if req.key:
            self._last_key_t[req.key] = now
        preempt = (
            req.kind == "alert"
            and self.current is not None
            and self.current.kind != "alert"
        )
        return Offer(True, preempt=preempt)

    def pop(self, now: float) -> Optional[SpeechRequest]:
        """Next request to speak (sets ``current``), dropping stale info items."""
        self._items = [
            r for r in self._items
            if not (r.kind == "info" and now - r.t_rx > self.info_max_age_s)
        ]
        if not self._items:
            return None
        nxt = min(self._items, key=lambda r: (r.rank, r.order))
        self._items.remove(nxt)
        self.current = nxt
        return nxt

    def done(self) -> None:
        self.current = None

    def clear(self) -> None:
        self._items.clear()

"""Persistent change store (explore Phase C): what usually differs from the map.

11-03 phantom analysis: 54–92 % of "new" / "missing" reports recur at the
same places patrol after patrol — chair legs the mapper despeckled away,
furniture moved since mapping, doors, walls swept at a glancing angle. They
are real differences from the reference map but not news. This store keeps
a short, auditable list of such places per map and gives every report a
verdict:

* ``novel``     — no matching entry: report it.
* ``candidate`` — matches an entry still being learned: report it.
* ``expected``  — matches an entry the owner approved as background: log it
  quietly (still written to events.jsonl).
* ``flagged``   — matches an entry the owner rejected ("always tell me").

Drift guards (owner, 11-03): nothing becomes background silently. An entry
seen in ``propose_after_sessions`` patrols is *proposed*; only the owner's
approval makes it background (``scripts/change_store.py``). The reference map
stays the anchor — the store is a list of exceptions, and a long list means
re-map, not absorb. Entries the robot looked at (passed within
``observe_m``) but did not see for ``expire_after_misses`` patrols are
dropped. A match must also agree in size and, once labelled, in label: a box
in a door's swing zone is not the door (the 11-03 box test).

Only ``new`` and ``missing`` are stored; ``moving`` is always novel.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from desktop.perception.change_detector import KIND_MISSING, KIND_NEW, STATUS_APPEARED, ChangeEvent

logger = logging.getLogger(__name__)

NOVEL, CANDIDATE, EXPECTED, FLAGGED = "novel", "candidate", "expected", "flagged"
ST_CANDIDATE, ST_PROPOSED, ST_BACKGROUND, ST_REJECTED = (
    "candidate", "proposed", "background", "rejected")
# Labels that say nothing about what is there.
UNINFORMATIVE = (None, "", "unknown", "other", "nothing there")


@dataclass(frozen=True)
class ChangeStoreConfig:
    match_m: float = 0.40                 # same place
    extent_ratio: float = 2.0             # same size: within this factor …
    extent_slack_m: float = 0.15          # … plus this much
    propose_after_sessions: int = 3       # seen in this many patrols → proposed
    expire_after_misses: int = 5          # looked at, not seen, this many patrols → dropped
    observe_m: float = 2.5                # the robot passed this close = "looked at"
    warn_background: int = 40             # more approved entries than this → re-map hint
    max_sessions_kept: int = 20


@dataclass
class StoreEntry:
    eid: int
    kind: str
    x: float
    y: float
    extent_m: float
    state: str = ST_CANDIDATE
    sessions: List[str] = field(default_factory=list)
    n_reports: int = 0
    labels: Dict[str, int] = field(default_factory=dict)
    misses: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def label(self) -> Optional[str]:
        good = {k: v for k, v in self.labels.items() if k not in UNINFORMATIVE}
        return Counter(good).most_common(1)[0][0] if good else None


class ChangeStore:
    def __init__(self, path: Optional[str], config: Optional[ChangeStoreConfig] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.cfg = config or ChangeStoreConfig()
        self._clock = clock
        self._lock = threading.Lock()
        self.entries: Dict[int, StoreEntry] = {}
        self._next = 1
        self.session: Optional[str] = None
        self._seen_now: set = set()            # entries matched this session
        self._path_xy: List[Tuple[float, float]] = []
        self._by_change: Dict[int, Tuple[str, Optional[int]]] = {}   # change_id → (verdict, eid)
        self.load()

    # ── persistence ──────────────────────────────────────────────────
    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self.entries = {int(e["eid"]): StoreEntry(**e) for e in d.get("entries", [])}
            self._next = max([0] + list(self.entries)) + 1
        except (OSError, ValueError, TypeError, KeyError):
            logger.exception("change store: could not read %s — starting empty", self.path)
            self.entries = {}

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            d = {"version": 1, "saved": self._clock(),
                 "entries": [asdict(e) for e in sorted(self.entries.values(), key=lambda e: e.eid)]}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(d, f, indent=1)
            os.replace(tmp, self.path)
        except OSError:
            logger.exception("change store: save failed")

    # ── session ──────────────────────────────────────────────────────
    def begin_session(self, session_id: str) -> None:
        with self._lock:
            self.session = session_id
            self._seen_now = set()
            self._path_xy = []
            self._by_change = {}

    def note_pose(self, x: float, y: float) -> None:
        """Robot path (decimated to 0.25 m) — which entries were looked at."""
        with self._lock:
            if not self._path_xy or math.hypot(x - self._path_xy[-1][0],
                                               y - self._path_xy[-1][1]) >= 0.25:
                self._path_xy.append((float(x), float(y)))

    def end_session(self) -> List[StoreEntry]:
        """Count misses, expire, propose; save. Returns newly proposed entries."""
        cfg = self.cfg
        proposed: List[StoreEntry] = []
        with self._lock:
            path = list(self._path_xy)
            for e in list(self.entries.values()):
                if e.eid in self._seen_now:
                    e.misses = 0
                    if (e.state == ST_CANDIDATE
                            and len(e.sessions) >= cfg.propose_after_sessions):
                        e.state = ST_PROPOSED
                        proposed.append(e)
                    continue
                looked = any(math.hypot(px - e.x, py - e.y) <= cfg.observe_m for px, py in path)
                if looked:
                    e.misses += 1
                    if e.misses >= cfg.expire_after_misses and e.state != ST_REJECTED:
                        logger.info("change store: entry %d (%s %s at %.2f, %.2f) not seen in "
                                    "%d patrols — dropped", e.eid, e.kind, e.label or "", e.x,
                                    e.y, e.misses)
                        del self.entries[e.eid]
            n_bg = sum(e.state == ST_BACKGROUND for e in self.entries.values())
        for e in proposed:
            logger.info("change store: PROPOSE entry %d — %s %s at (%.2f, %.2f), %.2f m, seen in "
                        "%d patrols (approve: scripts/change_store.py approve %d)", e.eid, e.kind,
                        e.label or "", e.x, e.y, e.extent_m, len(e.sessions), e.eid)
        if n_bg > cfg.warn_background:
            logger.warning("change store: %d background entries — the map may be out of date; "
                           "consider re-mapping instead of approving more", n_bg)
        self.save()
        return proposed

    # ── verdicts ─────────────────────────────────────────────────────
    def _compatible(self, e: StoreEntry, ev: ChangeEvent) -> bool:
        cfg = self.cfg
        if e.kind != ev.kind or math.hypot(e.x - ev.x, e.y - ev.y) > cfg.match_m:
            return False
        lo, hi = sorted((max(e.extent_m, 0.05), max(ev.extent_m, 0.05)))
        return hi <= lo * cfg.extent_ratio + cfg.extent_slack_m

    def classify(self, ev: ChangeEvent) -> Tuple[str, Optional[int]]:
        """Verdict for one report (and learn from it). ``gone`` reports and
        ``moving`` changes take their appearance's verdict / are novel."""
        if ev.kind not in (KIND_NEW, KIND_MISSING):
            return NOVEL, None
        with self._lock:
            if ev.status != STATUS_APPEARED:
                return self._by_change.get(ev.change_id, (NOVEL, None))
            cands = [e for e in self.entries.values() if self._compatible(e, ev)]
            e = min(cands, key=lambda c: math.hypot(c.x - ev.x, c.y - ev.y)) if cands else None
            now = self._clock()
            if e is None:
                e = StoreEntry(self._next, ev.kind, ev.x, ev.y, ev.extent_m, first_seen=now)
                self.entries[e.eid] = e
                self._next += 1
                verdict = NOVEL
            else:
                verdict = {ST_BACKGROUND: EXPECTED, ST_REJECTED: FLAGGED}.get(e.state, CANDIDATE)
                k = min(e.n_reports, 9)                    # running mean, recent-weighted
                e.x = (e.x * k + ev.x) / (k + 1)
                e.y = (e.y * k + ev.y) / (k + 1)
                e.extent_m = (e.extent_m * k + ev.extent_m) / (k + 1)
            e.n_reports += 1
            e.last_seen = now
            if self.session and self.session not in e.sessions:
                e.sessions = (e.sessions + [self.session])[-self.cfg.max_sessions_kept:]
            self._seen_now.add(e.eid)
            self._by_change[ev.change_id] = (verdict, e.eid)
            return verdict, e.eid

    def on_label(self, change_id: int, label: str) -> Optional[str]:
        """Record a label. Returns NOVEL when it shows an *expected* report is
        something else (a box where a door usually is) — report it after all."""
        with self._lock:
            verdict, eid = self._by_change.get(change_id, (None, None))
            e = self.entries.get(eid) if eid is not None else None
            if e is None:
                return None
            known = e.label
            if label not in UNINFORMATIVE:
                e.labels[label] = e.labels.get(label, 0) + 1
            if (verdict == EXPECTED and known is not None and label not in UNINFORMATIVE
                    and label != known):
                self._by_change[change_id] = (NOVEL, eid)
                return NOVEL
            return None

    # ── owner actions ────────────────────────────────────────────────
    def set_state(self, eid: int, state: str) -> bool:
        with self._lock:
            e = self.entries.get(eid)
            if e is None:
                return False
            e.state = state
            e.misses = 0
        self.save()
        return True

    def forget(self, eid: int) -> bool:
        with self._lock:
            ok = self.entries.pop(eid, None) is not None
        self.save()
        return ok

    def summary(self) -> Dict[str, int]:
        with self._lock:
            return dict(Counter(e.state for e in self.entries.values()))

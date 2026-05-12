"""Mission tracing — JSONL artifact for post-hoc review.

The audience for this trace is Claude reviewing/debugging a patrol or
mission run after the fact, NOT a live operator. The artifact is the
trace file; there's no live UI consumer in v1.

Layout
------

One trace file per mission, opened on `Mission.start` and closed on
the terminal transition (ARRIVED/CANCELED/FAILED).

    ~/Body/sessions/<session_id>/trace_<UTC ts>.jsonl

First line is a single header record (kind="header") with the run
context (patrol definition, frozen configs, git sha, snapshot bundle
path at start, etc.). Subsequent lines are `TraceEvent` dicts —
edge-triggered where possible to keep the file small and readable.

    {"kind": "header", "ts": ..., "session_id": ..., "configs": {...},
     "patrol": {...|null}, "git_sha": "...", "snapshot_at_start": "..."}
    {"ts": ..., "category": "mission", "level": "info",
     "event": "start", "data": {"pose": [x, y, theta], ...}}

Every event carries `pose: [x, y, theta]` in its `data` field (stamped
at emit time via an attached pose sampler) so a reviewer doesn't have
to walk back through the file to know where the robot was at the
moment of the event.

Auto-snapshot
-------------

A small set of (category, event) tuples in `AUTO_SNAP_EVENTS` triggers
the registered snapshot callback when emitted. The bundle path the
callback returns is stamped into the triggering event's `data` as
`auto_snapshot`. This gives a reviewer a costmap + map layers at the
moment of trouble without needing the operator to click Save Snapshot.

Threading
---------

`emit()` is thread-safe. Mission transitions happen on the UI thread;
the liveness watcher also runs on the UI thread; future producers
(SLAM corrections, recovery primitives running off-thread) can call
emit() from any thread.

The file handle is line-buffered, so a `shutil.copy` of an active
trace into a snapshot bundle yields a well-formed prefix.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Category / level constants ──────────────────────────────────────


CAT_MISSION = "mission"
CAT_PLAN = "plan"
CAT_FOLLOW = "follow"
CAT_RECOVERY = "recovery"
CAT_SAFETY = "safety"
CAT_PI = "pi"
CAT_PATROL = "patrol"

LEVEL_INFO = "info"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"


# (category, event) pairs that fire the registered snapshot callback.
# Keep this set small and high-signal — every entry is one bundle of
# disk I/O per occurrence.
AUTO_SNAP_EVENTS = frozenset({
    (CAT_MISSION, "fail"),
    (CAT_RECOVERY, "begin"),
    (CAT_SAFETY, "sustained_block"),
})


# ── Types ───────────────────────────────────────────────────────────


@dataclass
class TraceEvent:
    ts: float
    category: str
    level: str
    event: str
    data: Dict[str, Any]


PoseSampler = Callable[[], Optional[Tuple[float, float, float]]]
SnapshotCallback = Callable[[str], Optional[str]]


@dataclass
class TracerConfig:
    # In-memory ring buffer cap. Big enough to hold pre-mission events
    # (liveness edges before Go) plus one long mission's edges.
    ring_capacity: int = 2048

    # Trace file root. Files go under `<base_dir>/<session_id>/`.
    base_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/Body/sessions"),
    )


# ── Tracer ──────────────────────────────────────────────────────────


class Tracer:
    """JSONL event sink + in-memory ring.

    Lifecycle:

        open(...)   — write header, open the per-mission file.
        emit(...)   — append a TraceEvent line; safe before open
                      (ring-buffered only).
        close()     — finalize the file. Idempotent.

    Auto-snap fires before the triggering event line is written, so
    the stamped `auto_snapshot` path can be embedded in `data`. Side
    effect: a snapshot bundle copied from the *active* trace file
    won't contain its own triggering event yet. The live trace file
    outside the bundle is the canonical record.
    """

    def __init__(self, config: Optional[TracerConfig] = None):
        self.config = config or TracerConfig()
        self._lock = threading.Lock()
        self._fh: Optional[Any] = None
        self._path: Optional[str] = None
        self._ring: Deque[TraceEvent] = deque(
            maxlen=self.config.ring_capacity,
        )
        self._pose_sampler: Optional[PoseSampler] = None
        self._snapshot_cb: Optional[SnapshotCallback] = None

    # ── Wiring ──────────────────────────────────────────────────────

    def attach_pose_sampler(self, sampler: PoseSampler) -> None:
        self._pose_sampler = sampler

    def attach_snapshot_cb(self, cb: SnapshotCallback) -> None:
        self._snapshot_cb = cb

    # ── Lifecycle ───────────────────────────────────────────────────

    def is_open(self) -> bool:
        return self._fh is not None

    def current_path(self) -> Optional[str]:
        return self._path

    def open(
        self,
        *,
        session_id: str,
        configs: Dict[str, Any],
        patrol: Optional[Dict[str, Any]] = None,
        git_sha: Optional[str] = None,
        snapshot_at_start: Optional[str] = None,
    ) -> str:
        """Open a new trace file. Closes any previously open one.
        Returns the absolute path."""
        with self._lock:
            self._close_locked()
            sid = session_id or "unknown"
            sid_dir = os.path.join(self.config.base_dir, sid)
            os.makedirs(sid_dir, exist_ok=True)
            ts_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            path = os.path.join(sid_dir, f"trace_{ts_str}.jsonl")
            # Line-buffered so external readers (and shutil.copy into a
            # snapshot bundle) see well-formed lines without an fsync.
            self._fh = open(path, "w", buffering=1, encoding="utf-8")
            self._path = path
            header = {
                "kind": "header",
                "ts": time.time(),
                "session_id": sid,
                "git_sha": git_sha,
                "patrol": patrol,
                "configs": configs,
                "snapshot_at_start": snapshot_at_start,
            }
            try:
                self._fh.write(
                    json.dumps(header, default=_json_default) + "\n"
                )
            except Exception:
                logger.exception("trace header write failed")
        logger.info(f"trace opened: {path}")
        return path

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.close()
        except Exception:
            logger.exception("trace close raised")
        self._fh = None
        self._path = None

    # ── Emission ────────────────────────────────────────────────────

    def emit(
        self,
        category: str,
        event: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        level: str = LEVEL_INFO,
    ) -> None:
        """Append a TraceEvent. Pose is stamped automatically from the
        attached sampler (when one is set and a `pose` key isn't
        already present). Errors during emission are swallowed —
        tracing must never crash the mission loop.
        """
        try:
            d = dict(data or {})
            if "pose" not in d:
                pose = self._sample_pose()
                if pose is not None:
                    d["pose"] = [
                        float(pose[0]), float(pose[1]), float(pose[2]),
                    ]

            # Fire auto-snap callbacks BEFORE writing the event line so
            # the bundle path can be embedded in this event's data.
            if (category, event) in AUTO_SNAP_EVENTS:
                cb = self._snapshot_cb
                if cb is not None:
                    try:
                        snap_path = cb(f"{category}.{event}")
                    except Exception:
                        logger.exception("auto-snapshot callback raised")
                        snap_path = None
                    if snap_path:
                        d["auto_snapshot"] = snap_path

            ev = TraceEvent(
                ts=time.time(),
                category=category,
                level=level,
                event=event,
                data=d,
            )
            with self._lock:
                self._ring.append(ev)
                if self._fh is not None:
                    try:
                        self._fh.write(
                            json.dumps(asdict(ev), default=_json_default)
                            + "\n"
                        )
                    except Exception:
                        logger.exception("trace write failed")
        except Exception:
            # Outer guard — emit() must never propagate.
            logger.exception("tracer.emit raised; swallowed")

    def _sample_pose(self) -> Optional[Tuple[float, float, float]]:
        sampler = self._pose_sampler
        if sampler is None:
            return None
        try:
            return sampler()
        except Exception:
            return None

    # ── Introspection (mainly for tests) ────────────────────────────

    def recent(self, n: int = 50) -> list:
        """Most recent `n` events from the ring buffer (oldest first)."""
        with self._lock:
            r = list(self._ring)
        return r[-n:]


# ── Helpers ─────────────────────────────────────────────────────────


_GIT_SHA_CACHE: Optional[str] = None
_GIT_SHA_RESOLVED: bool = False


def git_sha() -> Optional[str]:
    """Short SHA of the repo HEAD, or None if not in a git tree.
    Cached for the process — repo state changes mid-run are rare and
    not worth re-shelling every mission start.
    """
    global _GIT_SHA_CACHE, _GIT_SHA_RESOLVED
    if _GIT_SHA_RESOLVED:
        return _GIT_SHA_CACHE
    _GIT_SHA_RESOLVED = True
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        _GIT_SHA_CACHE = out.decode().strip() or None
    except Exception:
        _GIT_SHA_CACHE = None
    return _GIT_SHA_CACHE


def _json_default(o: Any) -> Any:
    # numpy scalars, paths, dataclasses-by-reference, etc.
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if hasattr(o, "__dict__"):
        return str(o)
    return str(o)

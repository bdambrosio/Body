"""Depth-block adjudicator: ask the VLM what tripped a ``depth_block``.

Tier-3 stops on a depth_block within ~0.2 s, long before any VLM can answer,
so the verdict never touches the stop itself. What it can change is what the
hierarchical drive does next, on the 1–10 s scale of its retry window:

  * person / pet / door (things that move on their own) → ``hold``: no retry
    gotos for ``hold_s``, so the robot doesn't nudge into someone;
  * a static obstacle in the path → ``note`` (``pause`` only with
    ``pause_on_static``): the planners route around it via Tier-3's depth
    memory, so the retries continue — waiting for the operator is reserved
    for the retry window running out;
  * floor transition, rug edge, shadow, nothing → ``note`` only: the robot
    keeps its normal retries, and the label tells the operator (and the log)
    that the veto may have been a false positive.

Every verdict, with its camera frame, is written to
``<log_dir>/verdicts.jsonl`` + one JPEG per block, so the depth veto can be
tuned against what the camera actually saw.

Threading: ``poll()`` runs on the UI thread (the 20 Hz hier tick) and is the
only place that touches the drive. Frames arrive on the Zenoh thread and
answers on the VLM worker thread; both just hand results back through a
locked queue.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from desktop.nav.hierarchical_drive import BlockAdvisory, HierState

from .frames import FrameGrabber, Pose, PosedFrame
from .vlm_worker import VlmJob, VlmResult, VlmWorker

logger = logging.getLogger(__name__)

MOVING = ("person", "pet", "door")
STATIC = ("chair", "table", "furniture", "box", "wall", "cable", "glass",
          "other_obstacle")
ARTIFACT = ("floor_transition", "rug_edge", "shadow_or_reflection", "nothing")
WHAT = MOVING + STATIC + ARTIFACT

# Yes/no + one category, server-enforced (vLLM guided decoding). The benchmark
# found free-form obstacle lists flaky and near/far useless; a single blocked
# flag plus a closed category set is what this model answers reliably.
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "blocked": {"type": "boolean"},
        "what": {"type": "string", "enum": list(WHAT)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["blocked", "what", "confidence"],
}

PROMPT = (
    "This image is from the forward camera of a small ground robot whose "
    "body is about 20 cm tall and 30 cm wide. Its depth sensor just stopped "
    "it because something seemed to be in its path within about half a "
    "metre. Look at the floor directly ahead, in the lower-middle of the "
    "image. Is there really something there the robot would hit? "
    "Answer blocked=true only for a physical obstacle in its path. Set "
    "`what` to the thing in its path; if the path is clear, set it to what "
    "probably fooled the sensor (floor_transition, rug_edge, "
    "shadow_or_reflection) or nothing."
)


@dataclass(frozen=True)
class AdjudicatorConfig:
    reasons: Tuple[str, ...] = ("depth_block",)   # block reasons to query
    act: bool = True              # False = log + show only, never advise
    hold_s: float = 5.0
    min_conf_hold: float = 0.6
    min_conf_pause: float = 0.7
    # A confident static verdict (box, chair, …) used to PAUSE the drive for
    # the operator. Off by default since the depth memory + Tier-2 fan-out
    # (10-31): a static obstacle is something the planners route around, and
    # a patrol must not park itself waiting for a human (13:11 run: two such
    # pauses, one of them the stall). The verdict is still logged and shown.
    pause_on_static: bool = False
    max_age_s: float = 3.0        # skip frames older than this at query time
    max_tokens: int = 60


def advisory_from_verdict(verdict: Any, cfg: AdjudicatorConfig) -> Optional[BlockAdvisory]:
    """Map a VLM verdict to a caution-only advisory; None if unusable."""
    if not isinstance(verdict, dict):
        return None
    what = verdict.get("what")
    blocked = verdict.get("blocked")
    try:
        conf = float(verdict.get("confidence"))
    except (TypeError, ValueError):
        return None
    if what not in WHAT or not isinstance(blocked, bool):
        return None
    if blocked and what in MOVING and conf >= cfg.min_conf_hold:
        return BlockAdvisory("hold", what, hold_s=cfg.hold_s)
    if blocked and what in STATIC and conf >= cfg.min_conf_pause and cfg.pause_on_static:
        return BlockAdvisory("pause", what)
    label = what if blocked else f"clear? ({what})"
    return BlockAdvisory("note", label)


def default_log_dir() -> str:
    return os.path.join(os.path.expanduser("~/body-logs/vlm"),
                        time.strftime("%Y%m%d_%H%M%S"))


@dataclass
class _Query:
    drive_id: int
    run: int
    reason: str
    status: Dict[str, Any]
    asked_ts: float


class BlockAdjudicator:
    """Watches a HierarchicalDrive; queries the VLM once per block run.

    query_fn(jpeg) → verdict dict; defaults to ``vision_service.chat_json``
    with SCHEMA/PROMPT. emit(category, event, data) is an optional tracer
    hook. log_dir=None disables the on-disk record.
    """

    def __init__(self, controller: Any, worker: VlmWorker, *,
                 cfg: Optional[AdjudicatorConfig] = None,
                 query_fn: Optional[Callable[[bytes], Any]] = None,
                 emit: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
                 log_dir: Optional[str] = None,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg or AdjudicatorConfig()
        self._worker = worker
        self._query_fn = query_fn or self._default_query
        self._emit = emit
        self._log_dir = log_dir
        self._clock = clock
        self._pose: Optional[Pose] = None           # refreshed by poll()
        self._grabber = FrameGrabber(controller, lambda: self._pose, clock=clock)
        self._asked: Tuple[int, int] = (-1, -1)     # (drive id, run) last queried
        self._lock = threading.Lock()
        self._done: Deque[Tuple[_Query, VlmResult, Optional[str]]] = deque()
        self._last: Optional[Dict[str, Any]] = None
        self._log_lock = threading.Lock()

    # ── UI-thread API ────────────────────────────────────────────────

    def poll(self, drive: Any, status: Optional[Dict[str, Any]],
             pose: Optional[Pose]) -> None:
        """Call every hier tick. Starts a query on a new qualifying block run
        and applies finished verdicts to the drive they were asked about."""
        self._pose = pose
        if drive is not None:
            self._maybe_ask(drive, status)
        with self._lock:
            done = list(self._done)
            self._done.clear()
        for q, res, jpeg_path in done:
            self._finish(drive, q, res, jpeg_path)

    def last_verdict(self) -> Optional[Dict[str, Any]]:
        """Most recent finished record (for the status label)."""
        return self._last

    # ── Internals ────────────────────────────────────────────────────

    def _maybe_ask(self, drive: Any, status: Optional[Dict[str, Any]]) -> None:
        if drive.state() is not HierState.BLOCKED:
            return
        reason = drive.block_reason()
        if reason not in self.cfg.reasons:
            return
        key = (id(drive), drive.block_run())
        if key == self._asked:
            return
        self._asked = key
        q = _Query(drive_id=key[0], run=key[1], reason=reason,
                   status=dict(status or {}), asked_ts=self._clock())
        rid = self._grabber.grab(lambda f: self._on_frame(q, f),
                                 on_fail=lambda r, e: self._on_capture_fail(q, r, e))
        if rid is None:
            logger.info("block adjudicator: no RGB capture (not connected)")

    def _on_frame(self, q: _Query, frame: PosedFrame) -> None:
        self._worker.submit(VlmJob(
            frame=frame, query=self._query_fn, kind="depth_block",
            max_age_s=self.cfg.max_age_s,
            on_result=lambda res: self._on_result(q, res)))

    def _on_capture_fail(self, q: _Query, rid: str, err: str) -> None:
        res = VlmResult(kind="depth_block", request_id=rid, frame_ts=0.0,
                        pose=self._pose, error=f"capture: {err}")
        with self._lock:
            self._done.append((q, res, None))

    def _on_result(self, q: _Query, res: VlmResult) -> None:
        # Worker thread: do the disk write here, off the UI tick.
        jpeg_path = self._save_jpeg(q, res)
        with self._lock:
            self._done.append((q, res, jpeg_path))

    def _finish(self, drive: Any, q: _Query, res: VlmResult,
                jpeg_path: Optional[str]) -> None:
        adv = advisory_from_verdict(res.verdict, self.cfg)
        applied = False
        if (adv is not None and self.cfg.act and drive is not None
                and id(drive) == q.drive_id):
            applied = drive.apply_block_advisory(q.run, adv)
        rec = {
            "ts": self._clock(),
            "run": q.run,
            "block_reason": q.reason,
            "depth_hits": q.status.get("depth_hits"),
            "depth_seq": q.status.get("depth_seq"),
            "cmd_id": q.status.get("cmd_id"),
            "request_id": res.request_id,
            "frame_ts": res.frame_ts,
            "pose": list(res.pose) if res.pose is not None else None,
            "answer_s": round(self._clock() - q.asked_ts, 3),
            "vlm_s": round(res.latency_s, 3),
            "verdict": res.verdict,
            "error": res.error,
            "advisory": (None if adv is None else
                         {"action": adv.action, "label": adv.label,
                          "hold_s": adv.hold_s}),
            "applied": applied,
            "jpeg": jpeg_path,
        }
        self._last = rec
        logger.info("block adjudicator: run %d %s → %s%s", q.run, q.reason,
                    rec["advisory"] or rec["error"],
                    " (applied)" if applied else "")
        self._append_log(rec)
        if self._emit is not None:
            try:
                self._emit("safety", "depth_block_verdict", rec)
            except Exception:
                logger.exception("block adjudicator: emit raised")

    def _default_query(self, jpeg: bytes) -> Any:
        import vision_service as vs
        return vs.chat_json([{"role": "user", "content": PROMPT}], SCHEMA,
                            images=[jpeg], max_tokens=self.cfg.max_tokens,
                            name="depth_block")

    def _save_jpeg(self, q: _Query, res: VlmResult) -> Optional[str]:
        if self._log_dir is None or res.frame is None:
            return None
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            path = os.path.join(self._log_dir,
                                f"run{q.run:04d}_{res.request_id[:8]}.jpg")
            with open(path, "wb") as fh:
                fh.write(res.frame.jpeg)
            return path
        except OSError:
            logger.exception("block adjudicator: could not save frame")
            return None

    def _append_log(self, rec: Dict[str, Any]) -> None:
        if self._log_dir is None:
            return
        try:
            with self._log_lock:
                os.makedirs(self._log_dir, exist_ok=True)
                with open(os.path.join(self._log_dir, "verdicts.jsonl"), "a",
                          encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
        except OSError:
            logger.exception("block adjudicator: could not write log")


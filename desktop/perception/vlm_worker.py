"""Background worker that runs VLM queries on pose-tagged frames.

One thread, one request at a time (the GPU serves several, but a robot
query stream gains nothing from racing itself and the latest frame is what
matters). The queue is bounded: when full, the oldest job is dropped. A job
whose frame is older than ``max_age_s`` when its turn comes is skipped
instead of spending a second of GPU on a scene the robot has left.

Every submitted job gets exactly one ``on_result`` call — answered, failed,
stale or dropped — so callers holding "query pending" state can always clear
it. ``on_result`` runs on the worker thread; Qt callers marshal it with a
queued signal.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional

from .frames import Pose, PosedFrame

logger = logging.getLogger(__name__)


@dataclass
class VlmResult:
    kind: str
    request_id: str
    frame_ts: float             # PosedFrame.recv_ts
    pose: Optional[Pose]        # pose when the frame arrived
    verdict: Any = None         # query's return value; None on error
    latency_s: float = 0.0      # query wall time (0 when not run)
    error: Optional[str] = None  # None | "stale" | "dropped" | "stopped" | exception text
    frame: Optional[PosedFrame] = None


@dataclass
class VlmJob:
    frame: PosedFrame
    # Runs the actual model call, e.g.
    #   lambda jpeg: vision_service.chat_json(msgs, schema, images=[jpeg])
    query: Callable[[bytes], Any]
    on_result: Callable[[VlmResult], None]
    kind: str = ""
    max_age_s: float = 3.0


@dataclass
class _Stats:
    submitted: int = 0
    answered: int = 0
    failed: int = 0
    stale: int = 0
    dropped: int = 0
    last_latency_s: float = 0.0


class VlmWorker:
    def __init__(self, max_queue: int = 4,
                 clock: Callable[[], float] = time.time):
        self._max_queue = max(1, int(max_queue))
        self._clock = clock
        self._q: Deque[VlmJob] = deque()
        self._cv = threading.Condition()
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._stats = _Stats()

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        with self._cv:
            if self._thread is not None:
                return
            self._stop = False
            self._thread = threading.Thread(
                target=self._run, name="vlm-worker", daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the thread; queued jobs are reported with error="stopped"."""
        with self._cv:
            self._stop = True
            pending = list(self._q)
            self._q.clear()
            self._cv.notify_all()
            t, self._thread = self._thread, None
        for job in pending:
            self._report(job, error="stopped")
        if t is not None:
            t.join(timeout_s)

    # ── Submit ───────────────────────────────────────────────────────

    def submit(self, job: VlmJob) -> bool:
        """Queue a job. Returns False only if the worker is stopped (the job
        is then reported as "stopped")."""
        evicted: Optional[VlmJob] = None
        with self._cv:
            if self._stop or self._thread is None:
                accepted = False
            else:
                accepted = True
                self._stats.submitted += 1
                if len(self._q) >= self._max_queue:
                    evicted = self._q.popleft()
                    self._stats.dropped += 1
                self._q.append(job)
                self._cv.notify()
        if not accepted:
            self._report(job, error="stopped")
        if evicted is not None:
            self._report(evicted, error="dropped")
        return accepted

    def stats(self) -> Dict[str, Any]:
        with self._cv:
            s = self._stats
            return {
                "queued": len(self._q), "submitted": s.submitted,
                "answered": s.answered, "failed": s.failed,
                "stale": s.stale, "dropped": s.dropped,
                "last_latency_s": s.last_latency_s,
            }

    # ── Thread ───────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job = self._q.popleft()
            age = self._clock() - job.frame.recv_ts
            if age > job.max_age_s:
                with self._cv:
                    self._stats.stale += 1
                self._report(job, error="stale")
                continue
            t0 = time.monotonic()
            try:
                verdict = job.query(job.frame.jpeg)
                err = None
            except Exception as e:
                verdict, err = None, f"{type(e).__name__}: {e}"
            latency = time.monotonic() - t0
            with self._cv:
                if err is None:
                    self._stats.answered += 1
                else:
                    self._stats.failed += 1
                self._stats.last_latency_s = latency
            if err is not None:
                logger.warning("vlm_worker: %s query failed: %s", job.kind, err)
            self._report(job, verdict=verdict, latency_s=latency, error=err)

    def _report(self, job: VlmJob, verdict: Any = None,
                latency_s: float = 0.0, error: Optional[str] = None) -> None:
        res = VlmResult(
            kind=job.kind, request_id=job.frame.request_id,
            frame_ts=job.frame.recv_ts, pose=job.frame.pose,
            verdict=verdict, latency_s=latency_s, error=error,
            frame=job.frame,
        )
        try:
            job.on_result(res)
        except Exception:
            logger.exception("vlm_worker: on_result raised")

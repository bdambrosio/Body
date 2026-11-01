"""Pose-tagged RGB capture for VLM queries.

The shared ``state.rgb_*`` slot is the operator preview: streaming overwrites
it at 2 Hz with 320 px q60 frames, and it carries no pose. A VLM answer lands
seconds later, past the ~2 s pose buffer, so the pose has to be captured when
the frame arrives, not when the answer does.

``FrameGrabber`` sends a tagged ``capture_rgb`` (its reply bypasses the shared
slot) and stamps the reply with the world pose at arrival. The Pi's ``ts`` is
the frame's capture time on the Pi clock (older Pi builds sent reply-build
time). Arrival time and pose stay the reference for staleness, since Pi and
desktop clocks are only as close as chrony keeps them; ``recv_ts - pi_ts`` is
the capture-to-arrival latency when they agree.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

Pose = Tuple[float, float, float]   # world (x, y, theta)


@dataclass(frozen=True)
class PosedFrame:
    request_id: str
    jpeg: bytes
    width: int
    height: int
    pi_ts: float             # Pi capture time (Pi clock); 0.0 if absent
    recv_ts: float           # desktop wall clock at arrival
    pose: Optional[Pose]     # world pose at recv_ts; None if unavailable


class FrameGrabber:
    """Request one RGB frame and deliver it with the pose at arrival.

    controller: a StubController (needs ``request_rgb_tagged``).
    pose_fn: returns the current world pose or None — e.g. a nav
        ``PoseProvider.world_pose``. Called on the Zenoh thread, so it must be
        cheap and thread-safe.
    """

    def __init__(self, controller: Any,
                 pose_fn: Callable[[], Optional[Pose]],
                 clock: Callable[[], float] = time.time):
        self._ctl = controller
        self._pose_fn = pose_fn
        self._clock = clock

    def grab(self, on_frame: Callable[[PosedFrame], None],
             on_fail: Optional[Callable[[str, str], None]] = None,
             max_width: Optional[int] = None,
             jpeg_quality: Optional[int] = None) -> Optional[str]:
        """Send the request; returns its request_id, or None if not sent.

        on_frame(frame) or on_fail(request_id, error) runs on the Zenoh
        thread. Neither fires if the Pi never answers (the controller expires
        the request). Default size is the Pi's full preview (640x400 q85).
        """
        def _reply(msg: dict) -> None:
            rid = str(msg.get("request_id", ""))
            if not msg.get("ok") or not msg.get("jpeg"):
                if on_fail is not None:
                    on_fail(rid, str(msg.get("error") or "capture failed"))
                return
            recv_ts = float(msg.get("recv_ts") or self._clock())
            try:
                pose = self._pose_fn()
            except Exception:
                logger.exception("frame grabber: pose_fn raised")
                pose = None
            on_frame(PosedFrame(
                request_id=rid,
                jpeg=msg["jpeg"],
                width=int(msg.get("width") or 0),
                height=int(msg.get("height") or 0),
                pi_ts=float(msg.get("ts") or 0.0),
                recv_ts=recv_ts,
                pose=tuple(pose) if pose is not None else None,
            ))

        return self._ctl.request_rgb_tagged(
            _reply, max_width=max_width, jpeg_quality=jpeg_quality)


class SnapshotRing:
    """The last few pose-tagged frames, kept while a drive is active.

    ``tick(now, active)`` requests one frame every ``period_s`` (non-blocking;
    replies land on the Zenoh thread). Used by the recovery supervisor: the
    camera fallback of the post-fault pose check compares the last frame from
    before the fault with a fresh one, and diagnostic dumps save the ring.
    The 2 Hz operator preview carries no pose, so it is not reused.
    """

    def __init__(self, grabber: "FrameGrabber", *, size: int = 3,
                 period_s: float = 1.0, max_width: int = 320,
                 jpeg_quality: int = 70) -> None:
        import threading
        from collections import deque
        self._grabber = grabber
        self._period = float(period_s)
        self._max_width = max_width
        self._q = jpeg_quality
        self._frames: "deque[PosedFrame]" = deque(maxlen=size)
        self._lock = threading.Lock()
        self._last_req = -1e9

    def tick(self, now: float, active: bool) -> None:
        if not active or now - self._last_req < self._period:
            return
        self._last_req = now
        try:
            self._grabber.grab(self._add, max_width=self._max_width,
                               jpeg_quality=self._q)
        except Exception:
            logger.exception("snapshot ring: grab failed")

    def _add(self, frame: PosedFrame) -> None:
        if frame.pose is None:
            return
        with self._lock:
            self._frames.append(frame)

    def frames(self) -> list:
        with self._lock:
            return list(self._frames)

    def latest(self, *, after: float = -1e18) -> Optional[PosedFrame]:
        """Newest frame that arrived after ``after`` (desktop wall clock)."""
        with self._lock:
            for f in reversed(self._frames):
                if f.recv_ts > after:
                    return f
        return None

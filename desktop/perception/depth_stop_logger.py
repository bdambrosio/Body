"""Record what the depth veto fires on — single frame plus a burst.

Tier-3 scores every depth frame. With ``local_drive.depth_veto.act=false``
its status carries ``depth_would_block`` + hits / bbox / body points; with
act on it stops with ``state=BLOCKED, blocked_reason=depth_block`` and the
same evidence. This logger saves, on each rising edge of either (rate
limited), under ``<log_dir>/``:

* ``would_block_*`` / ``depth_block_*`` ``.npz`` (+ ``.jpg`` when an RGB frame
  is fresh): the chassis' latest depth frame, the status evidence and the
  pose — the single-frame dataset ``scripts/depth_denoise_score.py`` reads;
* ``<same base>_burst.npz``: every distinct depth frame from ``pre_s`` before
  the edge to ``post_s`` after it, each with its pose and commanded v/ω
  (``depth`` (N, H, W), ``depth_ts``, ``pose`` (N, 3), ``v_mps``,
  ``omega_radps``, ``trigger_index``). That is the dataset for a persistence
  gate (same obstacle across frames while the robot moves), which a single
  frame cannot score.

The burst is written by a later ``observe`` call once ``post_s`` has passed
(or by ``flush``).
"""
from __future__ import annotations

import collections
import logging
import os
import time
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _fired(status: Optional[Dict[str, Any]]) -> Optional[str]:
    """'would_block' (observe-only veto), 'depth_block' (real stop) or None."""
    if not isinstance(status, dict):
        return None
    if status.get("depth_would_block"):
        return "would_block"
    if status.get("state") == "BLOCKED" and status.get("blocked_reason") == "depth_block":
        return "depth_block"
    return None


class DepthStopLogger:
    def __init__(self, log_dir: Optional[str], *, min_interval_s: float = 2.0,
                 rgb_max_age_s: float = 1.0, pre_s: float = 2.0, post_s: float = 1.0,
                 clock: Callable[[], float] = time.time) -> None:
        self._dir = log_dir
        self._min_interval = float(min_interval_s)
        self._rgb_max_age = float(rgb_max_age_s)
        self._pre_s = float(pre_s)
        self._post_s = float(post_s)
        self._clock = clock
        self._prev = False
        self._last_saved = -1e9
        self.count = 0
        # Ring of distinct depth frames: (wall_t, depth_ts, depth, pose, v, omega).
        self._ring: Deque[tuple] = collections.deque()
        self._last_depth_ts: Optional[float] = None
        # Pending burst: base path, trigger time, frames, trigger index.
        self._burst: Optional[Dict[str, Any]] = None

    # ── ring ────────────────────────────────────────────────────────

    def _push(self, now: float, status: Optional[Dict[str, Any]],
              depth: Optional[Dict[str, Any]], pose: Optional[tuple]) -> None:
        if not depth or depth.get("depth") is None:
            return
        dts = float(depth.get("ts") or 0.0)
        if self._last_depth_ts is not None and dts == self._last_depth_ts:
            return                                  # same frame, re-rendered
        self._last_depth_ts = dts
        st = status if isinstance(status, dict) else {}
        rec = (now, dts, np.array(depth["depth"], copy=True),
               tuple(pose) if pose is not None else (np.nan,) * 3,
               float(st.get("v_mps") or 0.0), float(st.get("omega_radps") or 0.0))
        self._ring.append(rec)
        while self._ring and now - self._ring[0][0] > self._pre_s:
            self._ring.popleft()
        if self._burst is not None:
            self._burst["frames"].append(rec)

    # ── public ──────────────────────────────────────────────────────

    def observe(self, status: Optional[Dict[str, Any]], *,
                depth: Optional[Dict[str, Any]], rgb: Optional[Dict[str, Any]],
                pose: Optional[tuple]) -> Optional[str]:
        """Call on every status / redraw. Returns the saved base path on a
        new event (the burst file follows ``post_s`` later)."""
        now = self._clock()
        if self._dir is not None:
            self._push(now, status, depth, pose)
            if self._burst is not None and now - self._burst["t0"] >= self._post_s:
                self.flush()
        kind = _fired(status)
        on = kind is not None
        rising = on and not self._prev
        self._prev = on
        if not rising or self._dir is None:
            return None
        if now - self._last_saved < self._min_interval:
            return None
        self._last_saved = now
        try:
            os.makedirs(self._dir, exist_ok=True)
            self.count += 1
            base = os.path.join(self._dir, f"{kind}_{int(now)}_{self.count:04d}")
            st = status or {}
            np.savez_compressed(
                base + ".npz",
                depth=(np.asarray(depth["depth"]) if depth and depth.get("depth") is not None
                       else np.zeros((0, 0), np.uint16)),
                depth_ts=float((depth or {}).get("ts") or 0.0),
                status_ts=float(st.get("ts") or 0.0),
                depth_hits=int(st.get("depth_hits") or 0),
                depth_bbox_px=np.asarray(st.get("depth_bbox_px") or [], dtype=float),
                depth_hit_body=np.asarray(st.get("depth_hit_body") or [], dtype=float).reshape(-1, 2),
                pose=np.asarray(pose if pose is not None else (np.nan,) * 3, dtype=float),
                v_mps=float(st.get("v_mps") or 0.0), omega_radps=float(st.get("omega_radps") or 0.0),
                acted=(kind == "depth_block"))
            if rgb and rgb.get("jpeg") and now - float(rgb.get("ts") or 0.0) <= self._rgb_max_age:
                with open(base + ".jpg", "wb") as fh:
                    fh.write(rgb["jpeg"])
            logger.info("depth %s #%d: %s hits -> %s", kind, self.count,
                        st.get("depth_hits"), base)
            if self._burst is not None:          # previous burst still open: close it now
                self.flush()
            frames = list(self._ring)
            self._burst = {"base": base, "t0": now, "frames": frames,
                           "trigger_index": max(0, len(frames) - 1),
                           "status_ts": float(st.get("ts") or 0.0)}
            return base
        except Exception:
            logger.exception("depth stop logger: save failed")
            return None

    def flush(self) -> Optional[str]:
        """Write the pending burst (if any) now. Returns its path."""
        b, self._burst = self._burst, None
        if b is None or not b["frames"]:
            return None
        frames: List[tuple] = b["frames"]
        try:
            shapes = {f[2].shape for f in frames}
            if len(shapes) != 1:                   # resolution changed mid-burst: keep the majority
                common = collections.Counter(f[2].shape for f in frames).most_common(1)[0][0]
                frames = [f for f in frames if f[2].shape == common]
            path = b["base"] + "_burst.npz"
            np.savez_compressed(
                path,
                depth=np.stack([f[2] for f in frames]),
                wall_t=np.array([f[0] for f in frames], dtype=float),
                depth_ts=np.array([f[1] for f in frames], dtype=float),
                pose=np.array([f[3] for f in frames], dtype=float).reshape(-1, 3),
                v_mps=np.array([f[4] for f in frames], dtype=float),
                omega_radps=np.array([f[5] for f in frames], dtype=float),
                trigger_index=int(min(b["trigger_index"], len(frames) - 1)),
                status_ts=b["status_ts"])
            logger.info("depth burst: %d frames -> %s", len(frames), path)
            return path
        except Exception:
            logger.exception("depth stop logger: burst save failed")
            return None

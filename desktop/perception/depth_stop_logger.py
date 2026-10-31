"""Record what the observe-only depth veto fires on.

With ``local_drive.depth_veto.act=false`` (10-31) Tier-3 scores every depth
frame but never stops; its status carries ``depth_would_block`` + hits /
bbox / body points instead. This logger saves, on each rising edge (rate
limited), the chassis' latest depth frame, the status evidence, the pose and
— when one is fresh — the latest RGB JPEG, under ``<log_dir>/would_block_*``.
That is the dataset for a denoiser: ~92 % of the day's real stops were
floor glare / transitions / nothing, and nothing filtered them.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DepthStopLogger:
    def __init__(self, log_dir: Optional[str], *, min_interval_s: float = 2.0,
                 rgb_max_age_s: float = 1.0, clock: Callable[[], float] = time.time) -> None:
        self._dir = log_dir
        self._min_interval = float(min_interval_s)
        self._rgb_max_age = float(rgb_max_age_s)
        self._clock = clock
        self._prev = False
        self._last_saved = -1e9
        self.count = 0

    def observe(self, status: Optional[Dict[str, Any]], *,
                depth: Optional[Dict[str, Any]], rgb: Optional[Dict[str, Any]],
                pose: Optional[tuple]) -> Optional[str]:
        """Call on every status / redraw. Returns the saved path on a new event."""
        on = bool(isinstance(status, dict) and status.get("depth_would_block"))
        rising = on and not self._prev
        self._prev = on
        if not rising or self._dir is None:
            return None
        now = self._clock()
        if now - self._last_saved < self._min_interval:
            return None
        self._last_saved = now
        try:
            os.makedirs(self._dir, exist_ok=True)
            self.count += 1
            base = os.path.join(self._dir, f"would_block_{int(now)}_{self.count:04d}")
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
                v_mps=float(st.get("v_mps") or 0.0), omega_radps=float(st.get("omega_radps") or 0.0))
            if rgb and rgb.get("jpeg") and now - float(rgb.get("ts") or 0.0) <= self._rgb_max_age:
                with open(base + ".jpg", "wb") as fh:
                    fh.write(rgb["jpeg"])
            logger.info("depth would-block #%d: %s hits -> %s", self.count,
                        st.get("depth_hits"), base)
            return base
        except Exception:
            logger.exception("depth stop logger: save failed")
            return None

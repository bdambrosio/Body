"""Diagnostic dumps for recovery events (suspend, verdicts, re-arm, resume).

Each event writes ``<log_dir>/<unix_ts>_<n>_<tag>/`` with ``event.json``
(tag, detail, pose, recovery status), the snapshot ring's JPEGs with their
poses, and the latest lidar scan. That is what was missing on 2026-11-01:
the morning's slip had to be reconstructed from logs alone.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def default_recovery_dir() -> str:
    return os.path.join(os.path.expanduser("~/body-logs/recovery"),
                        time.strftime("%Y%m%d_%H%M%S"))


class RecoveryLogger:
    # Phase "verify" fires every attempt and carries nothing new.
    SKIP_TAGS = ("verify",)

    def __init__(self, log_dir: Optional[str], *, min_interval_s: float = 2.0,
                 clock: Callable[[], float] = time.time) -> None:
        self._dir = log_dir
        self._min_interval = float(min_interval_s)
        self._clock = clock
        self._last: Dict[str, float] = {}
        self.count = 0

    def event(self, tag: str, detail: str, *, frames: List[Any] = (),
              scan: Optional[Dict[str, Any]] = None, pose: Optional[tuple] = None,
              status: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if self._dir is None or tag in self.SKIP_TAGS:
            return None
        now = self._clock()
        if now - self._last.get(tag, -1e9) < self._min_interval:
            return None
        self._last[tag] = now
        self.count += 1
        path = os.path.join(self._dir, f"{int(now)}_{self.count:03d}_{tag}")
        try:
            os.makedirs(path, exist_ok=True)
            meta = {
                "ts": now, "tag": tag, "detail": detail,
                "pose": list(pose) if pose is not None else None,
                "recovery": status, "frames": [],
            }
            for i, f in enumerate(frames or ()):
                name = f"frame_{i}.jpg"
                with open(os.path.join(path, name), "wb") as fh:
                    fh.write(f.jpeg)
                meta["frames"].append({
                    "file": name, "recv_ts": f.recv_ts, "pi_ts": f.pi_ts,
                    "pose": list(f.pose) if f.pose is not None else None,
                })
            if scan and isinstance(scan.get("ranges"), list):
                np.savez_compressed(
                    os.path.join(path, "scan.npz"),
                    ranges=np.asarray([r if isinstance(r, (int, float)) else np.nan
                                       for r in scan["ranges"]], dtype=np.float32),
                    angle_min=float(scan.get("angle_min", 0.0)),
                    angle_increment=float(scan.get("angle_increment", 0.0)),
                    ts=float(scan.get("ts") or 0.0),
                )
            with open(os.path.join(path, "event.json"), "w") as fh:
                json.dump(meta, fh, indent=1, default=str)
        except Exception:
            logger.exception("recovery log: write failed (%s)", path)
            return None
        return path

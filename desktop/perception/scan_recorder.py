"""Record every scan the MCL observed, with its pose verdict, so session-grid
and change-detector changes can be replayed and scored offline
(``scripts/replay_changes.py``) instead of needing another patrol.

Chunks of ``chunk`` scans go to ``<out_dir>/scans_<nnnn>.npz``:
  ts (S,), ranges (S, B) float32, angle_min (S,), angle_inc (S,),
  pose (S, 3) (NaN when the scan had no scan-time pose), ok (S,) bool —
  whether the live run wrote it into the world — quality (S,), coverage (S,).
A 360-bin scan is ~1.4 kB, so a 10-minute patrol at 10 Hz is ~9 MB.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class ScanRecorder:
    def __init__(self, out_dir: Optional[str], *, chunk: int = 600) -> None:
        self.out_dir = out_dir
        self.chunk = int(chunk)
        self._lock = threading.Lock()
        self._rows: List[tuple] = []
        self._n_chunks = 0
        self.n_recorded = 0

    def add(self, ts: float, ranges: np.ndarray, angles: np.ndarray,
            pose: Optional[Sequence[float]], ok: bool,
            quality: Optional[float], coverage: Optional[float]) -> None:
        if not self.out_dir or len(angles) < 2:
            return
        row = (float(ts), np.asarray(ranges, dtype=np.float32).copy(),
               float(angles[0]), float(angles[1] - angles[0]),
               tuple(float(v) for v in pose) if pose is not None else (np.nan,) * 3,
               bool(ok),
               np.nan if quality is None else float(quality),
               np.nan if coverage is None else float(coverage))
        with self._lock:
            self._rows.append(row)
            self.n_recorded += 1
            flush = len(self._rows) >= self.chunk
        if flush:
            self.flush()

    def flush(self) -> Optional[str]:
        with self._lock:
            rows, self._rows = self._rows, []
            idx = self._n_chunks
            if rows:
                self._n_chunks += 1
        if not rows:
            return None
        nb = max(len(r[1]) for r in rows)
        ranges = np.full((len(rows), nb), np.nan, dtype=np.float32)
        for k, r in enumerate(rows):
            ranges[k, :len(r[1])] = r[1]
        path = os.path.join(self.out_dir, f"scans_{idx:04d}.npz")
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            np.savez_compressed(
                path,
                ts=np.array([r[0] for r in rows]),
                ranges=ranges,
                angle_min=np.array([r[2] for r in rows]),
                angle_inc=np.array([r[3] for r in rows]),
                pose=np.array([r[4] for r in rows]),
                ok=np.array([r[5] for r in rows]),
                quality=np.array([r[6] for r in rows]),
                coverage=np.array([r[7] for r in rows]),
            )
        except OSError:
            logger.exception("scan recorder: write failed (%s)", path)
            return None
        return path


def load_scans(paths: Sequence[str]):
    """Yield (ts, ranges, angles, pose|None, ok, quality, coverage) in order."""
    for path in sorted(paths):
        with np.load(path) as z:
            for k in range(z["ts"].shape[0]):
                r = z["ranges"][k]
                angles = z["angle_min"][k] + np.arange(r.shape[0]) * z["angle_inc"][k]
                pose = z["pose"][k]
                yield (float(z["ts"][k]), r.astype(np.float64), angles,
                       None if np.isnan(pose).any() else tuple(float(v) for v in pose),
                       bool(z["ok"][k]), float(z["quality"][k]), float(z["coverage"][k]))

"""Raw recording of a mapping drive ("bag") and its reader.

The bag is the real product of a mapping run: every later mapping backend
(submaps, GTSAM, multi-session — Stage 1+ of the mapping plan) reprocesses
it offline with ``scripts/replay_mapping.py`` instead of driving again.

Layout under ``<dir>/``:
  odom.jsonl      one body/odom message per line (as received)
  imu.jsonl       one body/imu message per line
  scans_NNNN.npz  chunks of scans: ts, angle_min, angle_inc, ranges (S,B)
                  float32 NaN = no return, intensities (S,B) uint16,
                  unknown (S,B) bool (bins with no information at all)
  frames.jsonl    RGB keyframes: ts, pose, path (jpg next to it)
  bag.json        session id, config snapshot, start time

~1.5 kB/scan: a 20-minute drive at 10 Hz is ~20 MB.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BagRecorder:
    def __init__(self, out_dir: Optional[str], *, chunk: int = 600,
                 info: Optional[Dict[str, Any]] = None) -> None:
        self.out_dir = out_dir
        self.chunk = int(chunk)
        self._lock = threading.Lock()
        self._scans: List[Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]] = []
        self._n_chunks = 0
        self._files: Dict[str, Any] = {}
        self.n_scans = 0
        self.n_frames = 0
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "bag.json"), "w") as f:
                json.dump({"started_wall_ts": time.time(), **(info or {})}, f, indent=1)

    @property
    def enabled(self) -> bool:
        return bool(self.out_dir)

    def _jsonl(self, name: str):
        f = self._files.get(name)
        if f is None and self.out_dir:
            f = open(os.path.join(self.out_dir, name), "a")
            self._files[name] = f
        return f

    def add_json(self, name: str, msg: Dict[str, Any]) -> None:
        if not self.out_dir:
            return
        with self._lock:
            f = self._jsonl(name)
            f.write(json.dumps(msg, separators=(",", ":")) + "\n")
            # Low-rate files (frames, events) are flushed per line so a
            # killed process keeps them; odom/IMU are high-rate and rely on
            # the OS buffer + close().
            if name not in ("odom.jsonl", "imu.jsonl"):
                f.flush()

    def add_odom(self, msg: Dict[str, Any]) -> None:
        self.add_json("odom.jsonl", msg)

    def add_imu(self, msg: Dict[str, Any]) -> None:
        self.add_json("imu.jsonl", msg)

    def add_scan(self, msg: Dict[str, Any]) -> None:
        """``msg``: the body/lidar/scan message (ranges list with None for
        no-return, optional intensities, optional unknown_bins)."""
        if not self.out_dir:
            return
        ranges = msg.get("ranges")
        if not isinstance(ranges, list):
            return
        n = len(ranges)
        r = np.array([v if isinstance(v, (int, float)) else np.nan for v in ranges],
                     dtype=np.float32)
        inten = msg.get("intensities")
        it = (np.array(inten, dtype=np.uint16) if isinstance(inten, list) and len(inten) == n
              else np.zeros(n, dtype=np.uint16))
        unk = np.zeros(n, dtype=bool)
        for b in msg.get("unknown_bins") or []:
            if 0 <= int(b) < n:
                unk[int(b)] = True
        row = (float(msg.get("ts") or time.time()), float(msg.get("angle_min", 0.0)),
               float(msg.get("angle_increment", 0.0)), r, it, unk)
        with self._lock:
            self._scans.append(row)
            self.n_scans += 1
            flush = len(self._scans) >= self.chunk
        if flush:
            self.flush_scans()

    def add_frame(self, ts: float, pose: Optional[Tuple[float, float, float]],
                  jpeg: bytes, *, pi_ts: float = 0.0) -> Optional[str]:
        if not self.out_dir:
            return None
        with self._lock:
            k = self.n_frames
            self.n_frames += 1
        name = f"frame_{k:05d}.jpg"
        try:
            with open(os.path.join(self.out_dir, name), "wb") as f:
                f.write(jpeg)
        except OSError:
            logger.exception("bag: frame write failed")
            return None
        self.add_json("frames.jsonl", {"ts": ts, "pi_ts": pi_ts, "path": name,
                                       "pose": list(pose) if pose is not None else None})
        return name

    def flush_scans(self) -> Optional[str]:
        with self._lock:
            rows, self._scans = self._scans, []
            idx = self._n_chunks
            if rows:
                self._n_chunks += 1
        if not rows or not self.out_dir:
            return None
        nb = max(len(r[3]) for r in rows)
        ranges = np.full((len(rows), nb), np.nan, dtype=np.float32)
        inten = np.zeros((len(rows), nb), dtype=np.uint16)
        unk = np.zeros((len(rows), nb), dtype=bool)
        for k, r in enumerate(rows):
            m = len(r[3])
            ranges[k, :m], inten[k, :m], unk[k, :m] = r[3], r[4], r[5]
        path = os.path.join(self.out_dir, f"scans_{idx:04d}.npz")
        try:
            np.savez_compressed(path, ts=np.array([r[0] for r in rows]),
                                angle_min=np.array([r[1] for r in rows]),
                                angle_inc=np.array([r[2] for r in rows]),
                                ranges=ranges, intensities=inten, unknown=unk)
        except OSError:
            logger.exception("bag: scan chunk write failed")
            return None
        return path

    def close(self) -> None:
        self.flush_scans()
        with self._lock:
            for f in self._files.values():
                try:
                    f.close()
                except OSError:
                    pass
            self._files.clear()


# ── reader ────────────────────────────────────────────────────────────

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def iter_bag(bag_dir: str, *, scan_delay_s: float = 0.2) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ("odom" | "imu" | "scan", message) merged in arrival order,
    each scan message in the body/lidar/scan wire shape (ranges with None
    for no return, intensities, unknown_bins).

    A scan is stamped at mid-sweep but reaches the desktop ~0.1–0.2 s later,
    after the odometry for that instant: it is delivered ``scan_delay_s``
    after its ts, as live. Delivered at its own ts, the EKF has no pose for
    it yet and the mapper skips it (the 11-02 replay lost 43 % of nodes)."""
    streams: List[Tuple[float, int, str, Dict[str, Any]]] = []
    for kind, name in (("odom", "odom.jsonl"), ("imu", "imu.jsonl")):
        for k, m in enumerate(_read_jsonl(os.path.join(bag_dir, name))):
            streams.append((float(m.get("ts") or 0.0), k, kind, m))
    k = 0
    for path in sorted(glob.glob(os.path.join(bag_dir, "scans_*.npz"))):
        with np.load(path) as z:
            for i in range(z["ts"].shape[0]):
                r = z["ranges"][i]
                unk = z["unknown"][i]
                msg = {
                    "ts": float(z["ts"][i]),
                    "angle_min": float(z["angle_min"][i]),
                    "angle_increment": float(z["angle_inc"][i]),
                    "ranges": [None if np.isnan(v) else float(v) for v in r],
                    "intensities": [int(v) for v in z["intensities"][i]],
                    "unknown_bins": [int(b) for b in np.nonzero(unk)[0]],
                }
                streams.append((msg["ts"] + scan_delay_s, k, "scan", msg))
                k += 1
    # Stable by (ts, arrival) so odom/imu at the same ts precede the scan.
    order = {"odom": 0, "imu": 1, "scan": 2}
    streams.sort(key=lambda t: (t[0], order[t[2]], t[1]))
    for _ts, _k, kind, msg in streams:
        yield kind, msg


def bag_frames(bag_dir: str) -> List[Dict[str, Any]]:
    return _read_jsonl(os.path.join(bag_dir, "frames.jsonl"))

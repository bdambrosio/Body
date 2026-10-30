"""Where did I last see <object>? Session memory of world-placed sightings.

Sightings of the same label within ``merge_radius_m`` are fused into one
entry (confidence-weighted mean position); entries older than ``max_age_s``
are forgotten. Saved as JSON so a restart keeps the last known places.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from .locate import Sighting


@dataclass
class Remembered:
    label: str
    x: float
    y: float
    confidence: float
    last_seen_ts: float
    count: int = 1


class ObjectMemory:
    def __init__(self, path: Optional[str] = None, merge_radius_m: float = 0.5,
                 max_age_s: float = 1800.0) -> None:
        self.path = path
        self.merge_radius_m = merge_radius_m
        self.max_age_s = max_age_s
        self._items: Dict[str, List[Remembered]] = {}
        self._lock = threading.Lock()

    def add(self, s: Sighting) -> Optional[Remembered]:
        if not s.present or s.world_xy is None:
            return None
        x, y = s.world_xy
        w = max(1e-3, s.confidence)
        with self._lock:
            items = self._items.setdefault(s.label, [])
            for r in items:
                if math.hypot(r.x - x, r.y - y) <= self.merge_radius_m:
                    tot = r.confidence * r.count + w
                    r.x = (r.x * r.confidence * r.count + x * w) / tot
                    r.y = (r.y * r.confidence * r.count + y * w) / tot
                    r.confidence = max(r.confidence, s.confidence)
                    r.last_seen_ts = max(r.last_seen_ts, s.ts)
                    r.count += 1
                    return r
            r = Remembered(s.label, x, y, s.confidence, s.ts)
            items.append(r)
            return r

    def best(self, label: str, now: float) -> Optional[Remembered]:
        """Most recently seen live entry for ``label``."""
        live = [r for r in self.all(label, now)]
        return max(live, key=lambda r: r.last_seen_ts) if live else None

    def all(self, label: Optional[str], now: float) -> List[Remembered]:
        with self._lock:
            labels = [label] if label is not None else list(self._items)
            out = []
            for lb in labels:
                keep = [r for r in self._items.get(lb, [])
                        if now - r.last_seen_ts <= self.max_age_s]
                if lb in self._items:
                    self._items[lb] = keep
                out.extend(keep)
            return out

    def forget(self, label: str) -> None:
        with self._lock:
            self._items.pop(label, None)

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            data = {lb: [asdict(r) for r in rs] for lb, rs in self._items.items()}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, self.path)

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        with self._lock:
            self._items = {lb: [Remembered(**r) for r in rs]
                           for lb, rs in data.items()}

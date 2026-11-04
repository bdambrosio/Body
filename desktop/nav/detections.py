"""What the robot noticed, folded for the UI (Detections dock, map pins,
end-of-run review).

``ChangeReporter`` emits three record types per change id — ``change``
(appeared / gone), ``label`` (VLM class of the crop) and ``investigated``
(explore's second look). ``DetectionModel`` folds them into one row per
change and decides what is worth showing: on a typical run most reports are
places the change store already knows recur (11-04 explore: 46 of ~70
were "candidate"), so the default view is only

* novel / flagged ``new`` and ``moving`` changes,
* anything explore investigated,
* anything labelled as something that matters (person, pet, toy, box).

Pure Python (no Qt) so it is unit-testable; the GUI feeds it on the GUI
thread.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Labels worth an operator's attention on their own.
INTERESTING_LABELS = ("person", "pet", "small stuffed animal", "box or bag")
# Labels that pop an alert as they happen.
ALERT_LABELS = ("person", "pet")
# Store verdicts that mean "this place routinely differs from the map".
ROUTINE_VERDICTS = ("candidate", "expected")


@dataclass
class Detection:
    change_id: int
    kind: str                         # new | moving | missing
    x: float
    y: float
    extent_m: float = 0.0
    first_ts: float = 0.0             # wall clock
    last_ts: float = 0.0
    present: bool = True
    verdict: Optional[str] = None     # store verdict: novel | candidate | expected | flagged
    store_entry: Optional[int] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    what: Optional[str] = None
    jpeg_path: Optional[str] = None
    investigated: Optional[str] = None   # confirmed | gone | dismissed | unreachable
    speed_mps: Optional[float] = None

    @property
    def notable(self) -> bool:
        if self.verdict == "expected":
            return False
        if self.investigated:
            return True
        if self.label in INTERESTING_LABELS:
            return True
        return self.kind in ("new", "moving") and self.verdict not in ROUTINE_VERDICTS

    @property
    def alert(self) -> bool:
        """Worth flashing as it happens."""
        return self.investigated == "confirmed" or self.label in ALERT_LABELS

    def title(self) -> str:
        name = self.label or self.kind
        conf = f" {self.confidence:.2f}" if self.confidence is not None else ""
        return f"#{self.change_id} {name}{conf}"

    def detail(self) -> str:
        bits = [self.kind]
        if self.investigated:
            bits.append(self.investigated)
        elif not self.present:
            bits.append("gone")
        if self.verdict and self.verdict != "novel":
            bits.append(self.verdict)
        t = time.strftime("%H:%M:%S", time.localtime(self.first_ts)) if self.first_ts else ""
        return f"{t}  ({self.x:.2f}, {self.y:.2f})  " + " · ".join(bits)


class DetectionModel:
    def __init__(self) -> None:
        self._rows: Dict[int, Detection] = {}
        self.version = 0

    def clear(self) -> None:
        self._rows.clear()
        self.version += 1

    def get(self, change_id: int) -> Optional[Detection]:
        return self._rows.get(change_id)

    def ingest(self, rec: Dict[str, Any]) -> Optional[Detection]:
        """Fold one reporter record; returns the row it touched (None when
        the record is for a change this model never saw)."""
        cid = rec.get("change_id")
        if cid is None:
            return None
        cid = int(cid)
        typ = rec.get("type")
        ts = float(rec.get("wall_ts") or time.time())
        d = self._rows.get(cid)
        if typ == "change":
            if d is None:
                if rec.get("kind") is None or rec.get("x") is None:
                    # label-only stub (reporter's "novel at a background place")
                    return None
                d = Detection(cid, str(rec["kind"]), float(rec["x"]), float(rec["y"]),
                              extent_m=float(rec.get("extent_m") or 0.0),
                              first_ts=ts, last_ts=ts)
                self._rows[cid] = d
            d.last_ts = ts
            if rec.get("x") is not None:
                d.x, d.y = float(rec["x"]), float(rec["y"])
            d.present = rec.get("status") != "gone"
            if rec.get("verdict") is not None:
                d.verdict = rec["verdict"]
            if rec.get("store_entry") is not None:
                d.store_entry = int(rec["store_entry"])
            if rec.get("speed_mps") is not None:
                d.speed_mps = float(rec["speed_mps"])
        elif d is None:
            return None
        elif typ == "label":
            d.label = rec.get("label")
            if rec.get("confidence") is not None:
                d.confidence = float(rec["confidence"])
            d.what = rec.get("what") or d.what
            d.jpeg_path = rec.get("jpeg_path") or d.jpeg_path
            if rec.get("verdict") == "novel":
                d.verdict = "novel"
            d.last_ts = ts
        elif typ == "investigated":
            d.investigated = rec.get("outcome")
            if rec.get("label") and not d.label:
                d.label = rec["label"]
            d.last_ts = ts
        else:
            return None
        self.version += 1
        return d

    def rows(self, *, show_all: bool = False, since: float = 0.0) -> List[Detection]:
        """Newest first."""
        out = [d for d in self._rows.values()
               if d.last_ts >= since and (show_all or d.notable)]
        out.sort(key=lambda d: d.first_ts, reverse=True)
        return out

    def counts(self, since: float = 0.0) -> Dict[str, int]:
        rows = [d for d in self._rows.values() if d.last_ts >= since]
        return {"total": len(rows), "notable": sum(d.notable for d in rows),
                "present": sum(d.present and d.notable for d in rows)}

    def by_store_entry(self) -> Dict[int, Detection]:
        """Latest detection per store entry (with a photo if any had one)."""
        out: Dict[int, Detection] = {}
        for d in sorted(self._rows.values(), key=lambda d: d.last_ts):
            if d.store_entry is None:
                continue
            prev = out.get(d.store_entry)
            if prev is None or d.jpeg_path or not prev.jpeg_path:
                out[d.store_entry] = d
        return out

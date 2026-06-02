"""LPR checkpoints — persisted in the reference map's metadata.

A checkpoint marks a Recognized location: an asserted pose plus the radius of
the locally-healed occupancy patch around it (see
docs/topological_localization_design.md §6 / Phase 2). The patch itself is NOT
stored — it is sliced from the map's occupancy at match time, so a checkpoint
always reflects the current (healed) map and stays tiny.

Stored as ``ReferenceMap.metadata["checkpoints"]`` (a JSON-friendly list of
dicts), so it round-trips through the existing ``meta_json`` npz key with no
format change. Written by the map editor's Recognize; read by the runtime
``CheckpointMatcher``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

Pose = Tuple[float, float, float]


@dataclass(frozen=True)
class Checkpoint:
    id: str
    x_m: float
    y_m: float
    theta_rad: float
    radius_m: float
    created_ts: float = 0.0

    @property
    def pose(self) -> Pose:
        return (self.x_m, self.y_m, self.theta_rad)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x_m": float(self.x_m),
            "y_m": float(self.y_m),
            "theta_rad": float(self.theta_rad),
            "radius_m": float(self.radius_m),
            "created_ts": float(self.created_ts),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        return cls(
            id=str(d["id"]),
            x_m=float(d["x_m"]),
            y_m=float(d["y_m"]),
            theta_rad=float(d["theta_rad"]),
            radius_m=float(d["radius_m"]),
            created_ts=float(d.get("created_ts", 0.0)),
        )


def checkpoints_from_metadata(metadata: Optional[Dict[str, Any]]) -> List[Checkpoint]:
    raw = (metadata or {}).get("checkpoints") or []
    out: List[Checkpoint] = []
    for d in raw:
        try:
            out.append(Checkpoint.from_dict(d))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def write_checkpoints_to_metadata(
    metadata: Dict[str, Any], checkpoints: Sequence[Checkpoint],
) -> None:
    """Replace ``metadata['checkpoints']`` with the given list (in place)."""
    metadata["checkpoints"] = [c.to_dict() for c in checkpoints]


def _next_id(checkpoints: Sequence[Checkpoint]) -> str:
    mx = -1
    for c in checkpoints:
        if c.id.startswith("cp_"):
            try:
                mx = max(mx, int(c.id[3:]))
            except ValueError:
                pass
    return f"cp_{mx + 1:03d}"


def upsert_checkpoint(
    checkpoints: Sequence[Checkpoint],
    pose: Pose,
    radius_m: float,
    *,
    created_ts: float = 0.0,
    merge_dist_m: float = 0.5,
) -> Tuple[List[Checkpoint], Checkpoint]:
    """Add a checkpoint at ``pose``, or update the existing one within
    ``merge_dist_m`` (re-Recognize corrects the same spot rather than piling
    up duplicates). Pure: returns (new_list, the_checkpoint)."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    out = list(checkpoints)
    for i, c in enumerate(out):
        if math.hypot(c.x_m - x, c.y_m - y) <= merge_dist_m:
            updated = Checkpoint(
                c.id, x, y, th, float(radius_m), created_ts or c.created_ts)
            out[i] = updated
            return out, updated
    new = Checkpoint(_next_id(out), x, y, th, float(radius_m), created_ts)
    out.append(new)
    return out, new

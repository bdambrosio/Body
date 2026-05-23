"""Convert legacy WorldGrid ``layers.npz`` snapshots to ReferenceMap."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np

from .reference_map import (
    ReferenceMap,
    build_reference_map_from_log_odds,
    load_reference_map,
    save_reference_map,
)


def convert_layers_npz(
    npz_path: str,
    *,
    out_path: Optional[str] = None,
    block_threshold: float = 1.0,
) -> ReferenceMap:
    """Build a ReferenceMap from a fuser ``layers.npz`` bundle.

    Uses ``block_votes > clear_votes`` (driveable layer when present) or
    ``block_votes >= block_threshold`` to infer occupied cells. Unknown
    cells get log-odds 0; occupied → +2; free → -2.
    """
    data = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    resolution_m = float(meta["resolution_m"])
    origin_x_m = float(meta["origin_x_m"])
    origin_y_m = float(meta["origin_y_m"])
    nx = int(meta["nx"])
    ny = int(meta["ny"])

    if "driveable" in data:
        drive = data["driveable"]
        occupied = drive == 0
        free = drive == 1
    else:
        block = data["block_votes"].astype(np.float32)
        clear = data.get("clear_votes")
        if clear is not None:
            clear = clear.astype(np.float32)
            occupied = block > clear
            free = clear > block
        else:
            occupied = block >= block_threshold
            free = np.zeros_like(occupied, dtype=bool)

    log_odds = np.zeros((nx, ny), dtype=np.float32)
    log_odds[occupied] = 2.0
    log_odds[free] = -2.0

    traj = None
    if "pose_trail" in data and data["pose_trail"].size >= 3:
        pt = data["pose_trail"].astype(np.float64).reshape(-1, 3)
        ts = np.arange(pt.shape[0], dtype=np.float64) * 0.1
        traj = np.column_stack([ts, pt[:, 0], pt[:, 1], pt[:, 2]])

    meta_out: Dict[str, Any] = {
        "converted_from": npz_path,
        "source_session_id": str(data.get("session_id") or ""),
        "conversion": "layers_npz_v1",
    }
    ref = build_reference_map_from_log_odds(
        log_odds,
        resolution_m=resolution_m,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        session_id=str(data.get("session_id") or "") or None,
        metadata=meta_out,
        trajectory=traj,
    )
    if out_path:
        save_reference_map(out_path, ref)
    return ref


def load_map_auto(path: str) -> ReferenceMap:
    """Load ``reference_map.npz`` or convert ``layers.npz`` on the fly."""
    if path.endswith("layers.npz"):
        return convert_layers_npz(path)
    try:
        return load_reference_map(path)
    except (KeyError, OSError, ValueError):
        return convert_layers_npz(path)

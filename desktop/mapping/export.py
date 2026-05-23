"""Export reference map bundles."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from desktop.reference_map.reference_map import save_reference_map

if TYPE_CHECKING:
    from desktop.localization.controller import LocalizationController
    from desktop.mapping.controller import MappingController


def _default_base_dir() -> str:
    return os.path.join(os.path.expanduser("~"), "Body", "maps")


def export_reference_map_bundle(
    controller: Any,
    *,
    base_dir: Optional[str] = None,
) -> str:
    """Save ReferenceMap.npz + summary for localization or mapping controller."""
    ref = controller.reference_map
    out_root = base_dir or _default_base_dir()
    sid = ref.session_id or "map"
    ts_dir = time.strftime("map_%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, sid, ts_dir)
    os.makedirs(out_dir, exist_ok=True)
    map_path = os.path.join(out_dir, "reference_map.npz")
    save_reference_map(map_path, ref)
    summary = {
        "session_id": ref.session_id,
        "resolution_m": ref.resolution_m,
        "nx": ref.nx,
        "ny": ref.ny,
        "origin_x_m": ref.origin_x_m,
        "origin_y_m": ref.origin_y_m,
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return out_dir


def export_mapping_session(
    controller: "MappingController",
    *,
    base_dir: Optional[str] = None,
) -> str:
    ref = controller.finalize_map()
    controller.reference_map = ref
    return export_reference_map_bundle(controller, base_dir=base_dir)

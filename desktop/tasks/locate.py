"""Where is <object>? VLM box → body bearing → range from depth → world point.

Frames: body x forward, y left, bearing positive to the left (CCW), matching
``desktop/nav/hierarchical_drive.py``. Image u grows to the right, so a box
right of centre has a negative bearing.

The RGB camera (CAM_A) and the depth map (aligned to stereo-left, CAM_B) sit
a few cm apart on the OAK-D and are not registered. At the 0.5–4 m ranges we
care about that parallax is a degree or two, so the depth lookup just maps
the box's bearing/elevation span into depth pixels with each camera's own
intrinsics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

Pose = Tuple[float, float, float]

# One object, server-enforced shape. bbox_2d on Qwen's native 0-1000 grid.
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "bbox_2d": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "minItems": 4, "maxItems": 4,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["present", "bbox_2d", "confidence"],
}

PROMPT = (
    "This image is from the forward camera of a small ground robot, about "
    "10 cm above the floor. Is there a {target} visible in the image? "
    "If yes, set present=true and give bbox_2d [x1, y1, x2, y2] of the most "
    "prominent {target} on a 0-1000 grid. If there is no {target}, set "
    "present=false and bbox_2d [0, 0, 0, 0]. Do not guess: a {target} that "
    "is only partly visible counts, a similar-looking different object does not."
)


def prompt_for(target: str) -> str:
    return PROMPT.format(target=target)


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics in pixels for an image of width x height."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled(self, width: int, height: int) -> "CameraModel":
        sx, sy = width / self.width, height / self.height
        return CameraModel(self.fx * sx, self.fy * sy, self.cx * sx,
                           self.cy * sy, width, height)

    @classmethod
    def from_config(cls, oakd_cfg: Mapping[str, Any]) -> "CameraModel":
        """RGB model from ``config.json`` ``oakd`` (defaults: the Pi's CAM_A
        factory calibration at 640x400, already in the 180°-rotated frame)."""
        w = int(oakd_cfg.get("rgb_preview_width", 640))
        h = int(oakd_cfg.get("rgb_preview_height", 400))
        return cls(float(oakd_cfg.get("rgb_fx", 558.3)),
                   float(oakd_cfg.get("rgb_fy", 558.3)),
                   float(oakd_cfg.get("rgb_cx", 327.6)),
                   float(oakd_cfg.get("rgb_cy", 194.3)), w, h)


def bbox_to_bearing(bbox_px: Sequence[float], cam: CameraModel) -> float:
    """Body bearing (rad, + left) of the box's horizontal centre."""
    u = 0.5 * (bbox_px[0] + bbox_px[2])
    return math.atan2(cam.cx - u, cam.fx)


def norm1000_to_px(bbox: Sequence[float], width: int,
                   height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (min(x1, x2) * width / 1000.0, min(y1, y2) * height / 1000.0,
            max(x1, x2) * width / 1000.0, max(y1, y2) * height / 1000.0)


def depth_range_m(bbox_px: Sequence[float], rgb: CameraModel,
                  depth_mm: np.ndarray, depth: CameraModel,
                  min_valid: int = 4, percentile: float = 30.0
                  ) -> Optional[float]:
    """Horizontal range (m) to the object in the box, or None.

    Maps the box's bearing and elevation span from RGB into depth pixels,
    then takes a low percentile of the valid (non-zero) depths so the
    object's near face wins over background showing around it. Depth is
    distance along the optical axis; dividing by cos(bearing) gives range.
    """
    if depth_mm is None or depth_mm.ndim != 2:
        return None
    dh, dw = depth_mm.shape
    if (dw, dh) != (depth.width, depth.height):
        depth = depth.scaled(dw, dh)
    x1, y1, x2, y2 = bbox_px
    # RGB pixel → ray tangent → depth pixel, per axis.
    def _col(u: float) -> float:
        return depth.cx + depth.fx * (u - rgb.cx) / rgb.fx
    def _row(v: float) -> float:
        return depth.cy + depth.fy * (v - rgb.cy) / rgb.fy
    c1, c2 = sorted((_col(x1), _col(x2)))
    r1, r2 = sorted((_row(y1), _row(y2)))
    # Shrink to the central part of the box: box edges catch background.
    cw, rh = c2 - c1, r2 - r1
    c1, c2 = c1 + 0.2 * cw, c2 - 0.2 * cw
    r1, r2 = r1 + 0.2 * rh, r2 - 0.2 * rh
    ci1, ci2 = max(0, int(math.floor(c1))), min(dw, int(math.ceil(c2)) + 1)
    ri1, ri2 = max(0, int(math.floor(r1))), min(dh, int(math.ceil(r2)) + 1)
    if ci1 >= ci2 or ri1 >= ri2:
        return None
    patch = depth_mm[ri1:ri2, ci1:ci2]
    valid = patch[patch > 0]
    if valid.size < min_valid:
        return None
    z_m = float(np.percentile(valid, percentile)) / 1000.0
    bearing = bbox_to_bearing(bbox_px, rgb)
    return z_m / max(0.2, math.cos(bearing))


def world_point(pose: Pose, bearing: float, range_m: float
                ) -> Tuple[float, float]:
    x, y, th = pose
    a = th + bearing
    return (x + range_m * math.cos(a), y + range_m * math.sin(a))


@dataclass(frozen=True)
class Sighting:
    label: str
    present: bool
    confidence: float
    bbox_px: Optional[Tuple[float, float, float, float]]
    bearing: Optional[float]                  # body frame, rad, + left
    range_m: Optional[float]
    pose: Optional[Pose]                      # robot pose at capture
    world_xy: Optional[Tuple[float, float]]
    ts: float                                 # Pi capture time


def sighting_from_answer(label: str, answer: Optional[Mapping[str, Any]],
                         cam: CameraModel, ts: float,
                         pose: Optional[Pose] = None,
                         depth_mm: Optional[np.ndarray] = None,
                         depth_cam: Optional[CameraModel] = None,
                         min_conf: float = 0.0) -> Optional[Sighting]:
    """Turn a ``SCHEMA`` answer into a Sighting; None if the answer is junk.

    A degenerate box with present=true is treated as not present."""
    if not isinstance(answer, Mapping):
        return None
    present = answer.get("present")
    bbox = answer.get("bbox_2d")
    try:
        conf = float(answer.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not isinstance(present, bool):
        return None
    if present and isinstance(bbox, Sequence) and len(bbox) == 4:
        px = norm1000_to_px(bbox, cam.width, cam.height)
        if px[2] - px[0] < 2 or px[3] - px[1] < 2 or conf < min_conf:
            present = False
    else:
        present = False
    if not present:
        return Sighting(label, False, conf, None, None, None, pose, None, ts)
    bearing = bbox_to_bearing(px, cam)
    rng = None
    if depth_mm is not None and depth_cam is not None:
        rng = depth_range_m(px, cam, depth_mm, depth_cam)
    wxy = world_point(pose, bearing, rng) if (pose and rng) else None
    return Sighting(label, True, conf, px, bearing, rng, pose, wxy, ts)

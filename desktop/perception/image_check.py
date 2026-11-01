"""Camera fallback for the post-fault pose check: does the view agree?

Compares the last RGB frame taken before a fault with a fresh one after it.
ORB features + RANSAC partial-affine fit give the image shift; the shift at
the image centre converts to a yaw change with the camera's focal length.
If the believed (posterior) heading changed by the same amount between the
two frames, the heading survived the fault; if the camera clearly says
otherwise, it did not. Translation is not checked (monocular, no depth), so
the check declines when the robot moved more than ``max_translation_m``.

Only consulted when the lidar check is inconclusive (pose_verify.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from desktop.localization.pose_verify import BAD, OK, UNKNOWN, Verdict


@dataclass(frozen=True)
class ImageCheckConfig:
    hfov_deg: float = 69.0          # OAK-D Lite RGB (config.json has depth 70°)
    n_features: int = 800
    min_inliers: int = 25
    ok_err_rad: float = math.radians(6.0)
    bad_err_rad: float = math.radians(15.0)
    max_translation_m: float = 0.5
    ransac_px: float = 3.0


def _decode(jpeg: bytes) -> Optional[np.ndarray]:
    import cv2
    buf = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img if img is not None and img.size > 0 else None


def image_yaw(img_a: np.ndarray, img_b: np.ndarray,
              cfg: ImageCheckConfig = ImageCheckConfig()) -> Tuple[Optional[float], int]:
    """Yaw of the camera from A to B (rad, CCW +) and the RANSAC inlier
    count; (None, n) when no fit."""
    import cv2
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    orb = cv2.ORB_create(nfeatures=cfg.n_features)
    ka, da = orb.detectAndCompute(img_a, None)
    kb, db = orb.detectAndCompute(img_b, None)
    if da is None or db is None or len(ka) < cfg.min_inliers or len(kb) < cfg.min_inliers:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < cfg.min_inliers:
        return None, len(matches)
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    A, mask = cv2.estimateAffinePartial2D(
        pa, pb, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_px)
    if A is None or mask is None:
        return None, 0
    inliers = int(mask.sum())
    h, w = img_a.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    dx = float(A[0, 0] * cx + A[0, 1] * cy + A[0, 2]) - cx
    f = (w / 2.0) / math.tan(math.radians(cfg.hfov_deg) / 2.0)
    # Turning left (CCW, +yaw) moves the scene right in the image (+x).
    return math.atan2(dx, f), inliers


def check_frames(
    jpeg_before: bytes, pose_before: Tuple[float, float, float],
    jpeg_after: bytes, pose_after: Tuple[float, float, float],
    cfg: ImageCheckConfig = ImageCheckConfig(),
) -> Verdict:
    moved = math.hypot(pose_after[0] - pose_before[0], pose_after[1] - pose_before[1])
    if moved > cfg.max_translation_m:
        return Verdict(UNKNOWN, f"moved {moved:.2f} m between frames", "image")
    a, b = _decode(jpeg_before), _decode(jpeg_after)
    if a is None or b is None:
        return Verdict(UNKNOWN, "frame decode failed", "image")
    yaw, inl = image_yaw(a, b, cfg)
    if yaw is None or inl < cfg.min_inliers:
        return Verdict(UNKNOWN, f"too few feature matches ({inl})", "image")
    believed = math.atan2(math.sin(pose_after[2] - pose_before[2]),
                          math.cos(pose_after[2] - pose_before[2]))
    err = abs(math.atan2(math.sin(yaw - believed), math.cos(yaw - believed)))
    detail = (f"camera turn {math.degrees(yaw):+.0f}° vs pose {math.degrees(believed):+.0f}° "
              f"({inl} inliers)")
    if err <= cfg.ok_err_rad:
        return Verdict(OK, detail, "image")
    if err >= cfg.bad_err_rad:
        return Verdict(BAD, detail, "image")
    return Verdict(UNKNOWN, detail, "image")

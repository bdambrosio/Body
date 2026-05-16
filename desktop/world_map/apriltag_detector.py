"""AprilTag detector — Phase 3 of the localization redesign.

Thin wrapper around ``pupil_apriltags.Detector`` so callers don't have
to think about JPEG decoding, grayscale conversion, or the
``estimate_tag_pose`` keyword soup. Stateless apart from the Detector
object held internally (which caches some C-side allocations).

Frames and conventions
----------------------
- Image is OAK-D RGB, JPEG-encoded by the Pi (oakd_driver._jpeg_b64_from_imgframe).
- Camera frame: +x right, +y down, +z forward (OpenCV convention,
  which pupil-apriltags follows).
- The detection's ``pose_R`` + ``pose_t`` form the homogeneous 4×4
  transform ``T_cam_tag`` (tag pose expressed in the camera frame).

Tag dictionary
--------------
Default tag36h11 — industry standard, robust to perspective and
partial occlusion. Encode 587 unique IDs (more than enough for a
household-scale map). Tags printed on letter-size paper with the
black-border edge at the size declared in ``tag_size_m``.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics: focal lengths + principal point in pixels."""
    fx: float
    fy: float
    cx: float
    cy: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)


@dataclass(frozen=True)
class TagDetection:
    """One tag observed in one frame.

    Fields:
    - tag_id: integer ID from the dictionary (tag36h11 → 0..586).
    - T_cam_tag: 4×4 homogeneous transform giving the tag's pose in the
      camera frame. Rotation is the upper-left 3×3, translation is the
      right column.
    - decision_margin: pupil-apriltags' detection confidence (>~30 is
      typical for a clean detection; <10 is a candidate for filtering).
    - pose_err: rough numerical pose-estimation residual.
    """
    tag_id: int
    T_cam_tag: np.ndarray   # (4, 4) float64
    decision_margin: float
    pose_err: float


class AprilTagDetector:
    """Stateful detector. Construct once per process — the underlying
    ``pupil_apriltags.Detector`` caches per-image-size scratch buffers
    and recreating it each call adds avoidable cost.
    """

    def __init__(
        self,
        *,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
    ):
        # Imported lazily so a CPU-only minimal install (or unit tests
        # that mock the detector) doesn't require the C extension.
        from pupil_apriltags import Detector
        self._detector = Detector(
            families=families,
            nthreads=int(nthreads),
            quad_decimate=float(quad_decimate),
            quad_sigma=float(quad_sigma),
            refine_edges=int(refine_edges),
            decode_sharpening=float(decode_sharpening),
        )

    def detect_jpeg(
        self,
        jpeg_bytes: bytes,
        *,
        intrinsics: CameraIntrinsics,
        tag_size_m: float,
        min_decision_margin: float = 20.0,
    ) -> List[TagDetection]:
        """Decode a JPEG and detect tags. Returns possibly-empty list.

        ``tag_size_m`` is the edge length of the *black border* of the
        printed tag, in meters. This is the canonical AprilTag size
        convention; the printed white margin around the tag is not
        counted.

        ``min_decision_margin`` is a soft filter on detection
        confidence — pupil-apriltags reports decision_margin for every
        detection, and values below ~10–15 are usually noise. 20 is a
        conservative default; lower it if you're missing valid
        detections in low-contrast conditions.
        """
        gray = _jpeg_to_grayscale(jpeg_bytes)
        return self.detect_array(
            gray,
            intrinsics=intrinsics,
            tag_size_m=tag_size_m,
            min_decision_margin=min_decision_margin,
        )

    def detect_array(
        self,
        gray: np.ndarray,
        *,
        intrinsics: CameraIntrinsics,
        tag_size_m: float,
        min_decision_margin: float = 20.0,
    ) -> List[TagDetection]:
        """Detect on a pre-decoded grayscale uint8 image."""
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError(
                f"detect_array expects a 2D uint8 image, got "
                f"shape={gray.shape} dtype={gray.dtype}"
            )
        raw = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=intrinsics.as_tuple(),
            tag_size=float(tag_size_m),
        )
        out: List[TagDetection] = []
        for r in raw:
            if r.decision_margin < min_decision_margin:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = np.asarray(r.pose_R, dtype=np.float64)
            T[:3, 3] = np.asarray(r.pose_t, dtype=np.float64).reshape(3)
            out.append(TagDetection(
                tag_id=int(r.tag_id),
                T_cam_tag=T,
                decision_margin=float(r.decision_margin),
                pose_err=float(getattr(r, "pose_err", 0.0)),
            ))
        return out


def _jpeg_to_grayscale(jpeg_bytes: bytes) -> np.ndarray:
    """Decode a JPEG (raw bytes — not base64) to a grayscale uint8
    image. Uses OpenCV; if cv2 isn't available, raises a clear error
    so the caller knows to install opencv-python-headless.
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "AprilTag detector needs opencv to decode JPEG. "
            "Install opencv-python-headless."
        ) from e
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("cv2.imdecode returned None — JPEG malformed?")
    return img


def decode_b64_jpeg(b64: str) -> bytes:
    """Helper: undo the base64 encoding the Pi applies to body/oakd/rgb
    payloads. Trims any surrounding whitespace and ``data:image/...;base64,``
    URL prefix the operator might paste in by accident.
    """
    s = b64.strip()
    if s.startswith("data:"):
        comma = s.find(",")
        if comma >= 0:
            s = s[comma + 1:]
    return base64.standard_b64decode(s)

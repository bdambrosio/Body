"""AprilTag observer — Phase 3 of the localization redesign.

Pairs with ``shadow_pf_driver`` (or any future PoseSource-implementing
filter): subscribes to ``body/oakd/rgb``, decodes the JPEG, runs the
AprilTag detector, and for every detection whose tag is known in the
calibration file, applies a Gaussian (x, y) + θ observation to the
particle filter.

RGB acquisition mode
--------------------
``request_hz=0`` (default): passive — only process captures someone
else (UI, scripts) drives. Useful for opportunistic tag observation
during normal teleop, where the UI captures RGB occasionally for
display purposes anyway.

``request_hz>0``: active — publish ``body/oakd/config`` capture_rgb
requests on a timer. The OAK-D's RGB stream is request-gated by
design, so this is how we "stream" at a chosen rate.

The plan's Phase 3 §"Open questions" flagged Pi-side detection vs.
desktop-side; we picked desktop-side per §5 risk #1 ("If the Pi can't
handle it, detect on desktop using streamed RGB"). Bandwidth at 1–2 Hz
of 640×400 JPEG is ~50–100 KB/s — fine over LAN.

Threading
---------
Zenoh callbacks fire on the session's callback threads. The detect +
SE(3) math + filter update all happen inside ``_on_rgb``; the per-call
work is dominated by JPEG decode + detector (~10–30 ms on a desktop
CPU at 640×400). Mutations to the filter take ``pf_lock`` for the
brief update window.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .apriltag_calibration import (
    AprilTagCalibration,
    AprilTagWorldPose,
    implied_body_world_pose,
)
from .apriltag_detector import (
    AprilTagDetector,
    TagDetection,
    decode_b64_jpeg,
)
from .particle_filter_pose import ParticleFilterPose

logger = logging.getLogger(__name__)


@dataclass
class AprilTagObserverConfig:
    # 0 = passive (only consume captures driven by others). >0 = the
    # observer publishes body/oakd/config capture_rgb at this rate.
    request_hz: float = 1.0

    # Detector confidence floor. pupil-apriltags decision_margin > 20
    # is typical for a clean detection; lower to catch low-contrast
    # tags at the cost of more false positives.
    min_decision_margin: float = 20.0

    # Treat the calibration's per-tag σ as the floor; multiply by this
    # if the detection looks marginal (high pose_err). 1.0 = use the
    # configured σ as-is.
    sigma_scale: float = 1.0

    # Apply yaw observation? Tags mounted on walls give a well-defined
    # yaw, but if your mount is on a flexible surface (e.g. a printed
    # sign that flaps), you may want xy-only.
    use_yaw_observation: bool = True


class AprilTagObserver:
    """Phase 3 observer. Optional attachment to a particle filter."""

    RGB_TOPIC = "body/oakd/rgb"
    OAKD_CONFIG_TOPIC = "body/oakd/config"

    def __init__(
        self,
        *,
        session: Any,
        pf: ParticleFilterPose,
        pf_lock: threading.RLock,
        calibration: AprilTagCalibration,
        detector: Optional[AprilTagDetector] = None,
        config: Optional[AprilTagObserverConfig] = None,
        on_detection: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._session = session
        self._pf = pf
        self._pf_lock = pf_lock
        self._calibration = calibration
        self._detector = detector if detector is not None else AprilTagDetector()
        self._config = config or AprilTagObserverConfig()
        self._on_detection = on_detection

        self._subs: List[Any] = []
        self._pub_config: Optional[Any] = None
        self._stop = threading.Event()
        self._request_thread: Optional[threading.Thread] = None

        self._counters: Dict[str, int] = {
            "rgb_received": 0,
            "rgb_malformed": 0,
            "rgb_error_payload": 0,
            "frames_processed": 0,
            "detections_total": 0,
            "detections_unknown_tag": 0,
            "observations_applied": 0,
            "capture_requests_sent": 0,
        }

    # ── Lifecycle ────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._subs:
            return
        self._subs.append(
            self._session.declare_subscriber(self.RGB_TOPIC, self._on_rgb),
        )
        if self._config.request_hz > 0.0:
            self._pub_config = self._session.declare_publisher(self.OAKD_CONFIG_TOPIC)
            self._stop.clear()
            self._request_thread = threading.Thread(
                target=self._request_loop, name="apriltag-rgb-requester",
                daemon=True,
            )
            self._request_thread.start()
        logger.info(
            "apriltag_observer: subscribed to %s, request_hz=%.2f, "
            "%d tag(s) calibrated",
            self.RGB_TOPIC, self._config.request_hz, len(self._calibration.tags),
        )

    def disconnect(self) -> None:
        self._stop.set()
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                logger.debug("apriltag_observer: sub undeclare failed", exc_info=True)
        self._subs.clear()
        if self._pub_config is not None:
            try:
                self._pub_config.undeclare()
            except Exception:
                logger.debug("apriltag_observer: pub undeclare failed", exc_info=True)
            self._pub_config = None
        if self._request_thread is not None:
            self._request_thread.join(timeout=1.0)
            self._request_thread = None
        logger.info("apriltag_observer: disconnected. counters=%s", self._counters)

    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    # ── Active RGB requesting ────────────────────────────────────────

    def _request_loop(self) -> None:
        period = 1.0 / max(0.01, self._config.request_hz)
        # Stagger first request slightly so we don't collide with
        # startup activity on the Pi.
        if self._stop.wait(0.5):
            return
        while not self._stop.is_set():
            try:
                if self._pub_config is not None:
                    payload = json.dumps({
                        "action": "capture_rgb",
                        "request_id": uuid.uuid4().hex,
                    }).encode("utf-8")
                    self._pub_config.put(payload)
                    self._counters["capture_requests_sent"] += 1
            except Exception:
                logger.exception("apriltag_observer: capture request failed")
            if self._stop.wait(period):
                return

    # ── RGB subscriber callback ──────────────────────────────────────

    def _payload_bytes(self, sample: Any) -> bytes:
        try:
            return bytes(sample.payload.to_bytes())
        except AttributeError:
            return bytes(sample.payload)

    def _on_rgb(self, sample: Any) -> None:
        self._counters["rgb_received"] += 1
        try:
            msg = json.loads(self._payload_bytes(sample).decode("utf-8"))
        except Exception:
            self._counters["rgb_malformed"] += 1
            return
        if not msg.get("ok"):
            self._counters["rgb_error_payload"] += 1
            return
        b64 = msg.get("data")
        if not isinstance(b64, str):
            self._counters["rgb_malformed"] += 1
            return
        try:
            jpeg_bytes = decode_b64_jpeg(b64)
        except Exception:
            self._counters["rgb_malformed"] += 1
            return

        try:
            detections = self._detector.detect_jpeg(
                jpeg_bytes,
                intrinsics=self._calibration.intrinsics,
                tag_size_m=self._calibration.default_tag_size_m,
                min_decision_margin=self._config.min_decision_margin,
            )
        except Exception:
            logger.exception("apriltag_observer: detector failed")
            return

        self._counters["frames_processed"] += 1
        self._counters["detections_total"] += len(detections)
        if not detections:
            return

        scan_ts = float(msg.get("ts") or time.time())
        for det in detections:
            self._apply_detection(det, scan_ts)

    # ── Observation application ──────────────────────────────────────

    def _apply_detection(self, det: TagDetection, ts: float) -> None:
        tag_def = self._calibration.tags.get(det.tag_id)
        if tag_def is None:
            self._counters["detections_unknown_tag"] += 1
            return

        # Re-detect with the tag's specific size if it overrides default.
        # (For multi-size deployments. Cheap relative to one full detect.)
        # NOTE: pupil-apriltags' pose_t scales linearly with tag_size,
        # so if the default size used at detection time differs from
        # the per-tag size, rescale t in-place rather than re-detecting:
        if not np.isclose(tag_def.tag_size_m, self._calibration.default_tag_size_m):
            scale = tag_def.tag_size_m / self._calibration.default_tag_size_m
            T_corrected = det.T_cam_tag.copy()
            T_corrected[:3, 3] *= scale
            T_cam_tag = T_corrected
        else:
            T_cam_tag = det.T_cam_tag

        x_w, y_w, theta_w = implied_body_world_pose(
            T_world_tag=tag_def.T_world_tag,
            T_cam_tag=T_cam_tag,
            T_body_cam=self._calibration.T_body_cam,
        )

        sigma_xy = tag_def.sigma_xy_m * self._config.sigma_scale
        sigma_theta = tag_def.sigma_theta_rad * self._config.sigma_scale

        with self._pf_lock:
            self._pf.observe_xy_world(x_w, y_w, sigma_xy)
            if self._config.use_yaw_observation:
                self._pf.observe_imu_yaw(theta_w, sigma_rad=sigma_theta)
        self._counters["observations_applied"] += 1

        if self._on_detection is not None:
            try:
                self._on_detection({
                    "type": "apriltag_obs",
                    "ts": ts,
                    "tag_id": det.tag_id,
                    "decision_margin": det.decision_margin,
                    "pose_err": det.pose_err,
                    "implied_world_pose": [x_w, y_w, theta_w],
                    "sigma_xy_m": sigma_xy,
                    "sigma_theta_rad": sigma_theta,
                })
            except Exception:
                logger.exception("apriltag_observer: on_detection callback raised")

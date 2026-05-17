"""Phase 6.3 — Shadow VPR driver.

Subscribes to ``body/oakd/rgb`` on the fuser's Zenoh session, runs
the DINOv2 extractor + bank query + mixture-conversion on every
frame, and writes a JSONL trace of *would-be* filter updates. The
production particle filter is **never mutated** — we read its state,
compute the per-particle log-likelihoods our observation would have
added, and log summary stats. Strictly observational, like
``ShadowParticleFilterDriver``.

Trace records (all JSON one per line):

- ``session_start`` — opening header with config + bank metadata.
- ``vpr_obs`` — one per processed RGB frame::

      {
        "type": "vpr_obs",
        "rgb_recv_ts": <float>,        # desktop wall-clock
        "rgb_ts": <float>,             # Pi sensor clock
        "top_k": [{idx, sim, pose_xytheta}, ...],
        "mixture": {                   # null if similarity_floor rejected
          "positions_xy": [[x, y], ...],
          "weights":      [w, ...],
          "sigma_m":       <float>,
        },
        "current_pose": [x, y, theta] | null,
        "would_be": {                  # null when mixture is null
          "mean_xy_before":  [x, y],
          "mean_xy_after":   [x, y],
          "n_eff_before":    <float>,
          "n_eff_after":     <float>,
          "log_lik_stats":   {"mean": .., "std": .., "min": .., "max": ..},
        },
      }

- ``no_match`` — emitted when similarity_floor empties the query;
  carries the (rejected) raw top-k for diagnostic purposes.
- ``session_end`` — closing record.

Why no mutation
---------------
6.3 is a measurement phase: does VPR actually improve localization
on this bank, on this drive? Mutating the live filter would
contaminate the very signal we're trying to measure. 6.4 promotes
to real observation behind the ``--vpr`` flag with σ/motion gating.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch

from .anchor import AnchorOffsetConfig, AnchorOffsetEstimator
from .bank import VPRBank, mixture_observation_from_query
from .extractor import DinoV2Extractor

logger = logging.getLogger(__name__)


RGB_TOPIC = "body/oakd/rgb"
OAKD_CONFIG_TOPIC = "body/oakd/config"


@dataclass
class ShadowVPRConfig:
    """Tunables for the observer. Defaults are tuned from the
    Phase 6.3 shadow trace (2026-05-17, 126 vpr_obs over a 125 s
    drive)."""

    # 0 = passive (consume captures driven elsewhere). >0 = publisher
    # drives body/oakd/config at this rate, same mechanism AprilTag uses.
    request_hz: float = 1.0

    # Top-K bank matches per query. K=5 gives the mixture room to
    # represent ambiguity without dominating the trace size.
    top_k: int = 5

    # Drop any match below this cosine. In the Phase 6.3 trace, clear
    # same-location matches sit at 0.85+; the 0.40 first-cut floor let
    # too much noise through (52% of matches were <0.80, mostly weak).
    # 0.80 is a defensible production floor; below it, an observation
    # is more likely to inject noise than to anchor.
    similarity_floor: float = 0.80

    # Softmax temperature on cosine similarity → mixture weights.
    # Smaller = sharper (top-1 dominates); larger = flatter.
    softmax_temperature: float = 0.05

    # Per-component Gaussian σ on (x, y) in meters. VPR is a room-scale
    # anchor, not a tag-style cm anchor.
    sigma_m: float = 0.5

    # Refuse to emit a would-be update if fewer than this many components
    # cleared the floor. 1 = single match is fine.
    min_components: int = 1

    # Buffered writes amortize syscalls; flushed on disconnect anyway.
    trace_flush_every: int = 20

    # ── Phase 6.4 — live mode + gating ────────────────────────────

    # When True, after computing the would-be record, also apply
    # the observation to the production filter via
    # ParticleFilterPose.observe_xy_mixture. False = pure shadow
    # (the original 6.3 behavior, useful for measurement).
    # Live mode additionally requires the anchor offset to be
    # calibrated and the gating checks to pass.
    live: bool = False

    # Bank↔session SE(2) calibration knobs. See anchor.py.
    anchor: AnchorOffsetConfig = field(default_factory=AnchorOffsetConfig)

    # σ gate: skip the observation when the cloud is already so
    # concentrated that no observation at sigma_m can discriminate
    # among particles. Condition: skip if
    #     sqrt(trace(cov[:2, :2])) < gate_sigma_floor_ratio * sigma_m
    # The intuition: VPR with σ=0.5 m can't usefully reweight a
    # cloud whose XY spread is 5 cm — every particle gets ≈ the
    # same log-likelihood. Setting the ratio to 0.5 means we fire
    # VPR only when the cloud spread is at least σ/2.
    gate_sigma_floor_ratio: float = 0.5

    # Motion gate: skip the observation if the bot hasn't moved at
    # least this far since the last applied observation. Avoids
    # double-anchoring at one physical location which would
    # over-concentrate the cloud at the bank pose.
    gate_min_distance_m: float = 0.3


class ShadowVPRDriver:
    """RGB → DINOv2 → bank → mixture-obs trace. Pure observer.

    Threading
    ---------
    Zenoh fires RGB callbacks on the session's threads. Bank query +
    extraction can take ~10–30 ms on GPU; rather than hold the
    production ``pf_lock`` across that whole window, we:
      1. lock briefly to snapshot ``pf.state`` and ``pf._log_w``;
      2. release; do extraction + query + would-be math off-lock;
      3. lock again briefly to read fresh posterior stats for the trace.
    The production filter is never written to.
    """

    def __init__(
        self,
        *,
        session: Any,
        pf: Any,                       # ParticleFilterPose
        pf_lock: threading.RLock,
        bank: VPRBank,
        extractor: DinoV2Extractor,
        trace_path: Path,
        pose_source: Optional[Any] = None,  # PoseSource — for current_pose annotation
        config: Optional[ShadowVPRConfig] = None,
        on_trace: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._session = session
        self._pf = pf
        self._pf_lock = pf_lock
        self._bank = bank
        self._extractor = extractor
        self._trace_path = Path(trace_path)
        self._pose_source = pose_source
        self._config = config or ShadowVPRConfig()
        self._on_trace = on_trace

        self._subs: List[Any] = []
        self._pub_config: Optional[Any] = None
        self._stop = threading.Event()
        self._request_thread: Optional[threading.Thread] = None
        self._trace_fp: Optional[TextIO] = None
        self._trace_lock = threading.Lock()
        self._trace_pending = 0

        self._counters: Dict[str, int] = {
            "rgb_received": 0,
            "rgb_malformed": 0,
            "rgb_error_payload": 0,
            "frames_processed": 0,
            "frames_no_match": 0,
            "frames_observed": 0,
            "would_be_updates_logged": 0,
            "capture_requests_sent": 0,
            # 6.4 additions
            "anchor_pairs_collected": 0,
            "anchor_calibrations": 0,
            "live_obs_applied": 0,
            "live_obs_gated_anchor": 0,
            "live_obs_gated_sigma": 0,
            "live_obs_gated_distance": 0,
        }

        # 6.4 — bank↔session SE(2) calibration. Always constructed;
        # only consulted in live mode. In shadow mode the trace still
        # carries the calibration state so post-hoc analysis can use it.
        self._anchor = AnchorOffsetEstimator(self._config.anchor)
        # 6.4 — gating bookkeeping.
        self._last_applied_xy: Optional[Tuple[float, float]] = None

    # ── Lifecycle ────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._subs:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_fp = self._trace_path.open("a", buffering=1)
        self._write_trace({
            "type": "session_start",
            "ts": time.time(),
            "bank": {
                "n_frames": self._bank.n_frames,
                "feature_dim": self._bank.feature_dim,
                "metadata": self._bank.metadata,
            },
            "config": {
                "request_hz": self._config.request_hz,
                "top_k": self._config.top_k,
                "similarity_floor": self._config.similarity_floor,
                "softmax_temperature": self._config.softmax_temperature,
                "sigma_m": self._config.sigma_m,
                "min_components": self._config.min_components,
            },
            "extractor": {
                "model_name": self._extractor.config.model_name,
                "input_size": self._extractor.config.input_size,
                "device": str(self._extractor.device),
                "feature_dim": self._extractor.feature_dim,
            },
        })

        self._subs.append(
            self._session.declare_subscriber(RGB_TOPIC, self._on_rgb),
        )
        if self._config.request_hz > 0.0:
            self._pub_config = self._session.declare_publisher(OAKD_CONFIG_TOPIC)
            self._stop.clear()
            self._request_thread = threading.Thread(
                target=self._request_loop,
                name="vpr-shadow-rgb-requester", daemon=True,
            )
            self._request_thread.start()
        logger.info(
            "shadow_vpr: subscribed to %s, trace=%s, request_hz=%.2f, "
            "bank=%d frames",
            RGB_TOPIC, self._trace_path, self._config.request_hz,
            self._bank.n_frames,
        )

    def disconnect(self) -> None:
        self._stop.set()
        if self._request_thread is not None:
            self._request_thread.join(timeout=1.0)
            self._request_thread = None
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                logger.debug("shadow_vpr: sub undeclare failed", exc_info=True)
        self._subs.clear()
        if self._pub_config is not None:
            try:
                self._pub_config.undeclare()
            except Exception:
                logger.debug("shadow_vpr: pub undeclare failed", exc_info=True)
            self._pub_config = None
        if self._trace_fp is not None:
            self._write_trace({
                "type": "session_end", "ts": time.time(),
                "counters": dict(self._counters),
            })
            try:
                self._trace_fp.flush()
                self._trace_fp.close()
            except Exception:
                logger.debug("shadow_vpr: trace close failed", exc_info=True)
            self._trace_fp = None
        logger.info(
            "shadow_vpr: disconnected. counters=%s", self._counters,
        )

    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    # ── Active RGB requesting ────────────────────────────────────────

    def _request_loop(self) -> None:
        period = 1.0 / max(0.01, self._config.request_hz)
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
                logger.exception("shadow_vpr: capture request failed")
            if self._stop.wait(period):
                return

    # ── RGB subscriber ───────────────────────────────────────────────

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
            jpeg_bytes = _decode_b64_jpeg(b64)
            rgb = _jpeg_bytes_to_rgb(jpeg_bytes)
        except Exception:
            self._counters["rgb_malformed"] += 1
            return

        rgb_recv_ts = time.time()
        rgb_ts = float(msg.get("ts") or rgb_recv_ts)
        self.process_frame(rgb, rgb_recv_ts=rgb_recv_ts, rgb_ts=rgb_ts)

    # ── Public processing entry (also used by tests / replay) ─────────

    def process_frame(
        self,
        rgb: np.ndarray,
        *,
        rgb_recv_ts: Optional[float] = None,
        rgb_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Extract → query → (anchor feed) → would-be math → gate →
        (live apply) → trace. Returns the written record (also written
        to disk + on_trace callback)."""
        rgb_recv_ts = rgb_recv_ts if rgb_recv_ts is not None else time.time()
        rgb_ts = rgb_ts if rgb_ts is not None else rgb_recv_ts

        feat = self._extractor.extract(rgb)
        result = self._bank.query(
            feat, top_k=self._config.top_k,
            similarity_floor=self._config.similarity_floor,
        )
        self._counters["frames_processed"] += 1

        top_k_records = [
            {
                "idx": int(result.indices[i].item()),
                "sim": float(result.similarities[i].item()),
                "pose_xytheta": [float(v) for v in result.poses[i].tolist()],
            }
            for i in range(result.indices.shape[0])
        ]
        # Snapshot the filter for would-be math, anchor pair, gating.
        snapshot = self._snapshot_pf()
        current_pose_pf = snapshot["mean"] if snapshot else None
        current_pose = (
            list(current_pose_pf) if current_pose_pf else
            self._snapshot_current_pose(rgb_ts)
        )

        # Feed the top-1 match to the anchor estimator (no-op once
        # calibrated; also no-op for low-similarity matches).
        if top_k_records and current_pose_pf:
            top = top_k_records[0]
            before = self._anchor.n_pairs_collected
            self._anchor.observe(
                bank_xy=(top["pose_xytheta"][0], top["pose_xytheta"][1]),
                current_xy=(current_pose_pf[0], current_pose_pf[1]),
                similarity=top["sim"],
            )
            if self._anchor.n_pairs_collected > before:
                self._counters["anchor_pairs_collected"] += 1
            new_cal = self._anchor.calibrate_if_ready()
            if new_cal is not None and self._counters["anchor_calibrations"] == 0:
                self._counters["anchor_calibrations"] += 1

        mixture = mixture_observation_from_query(
            result,
            temperature=self._config.softmax_temperature,
            sigma_m=self._config.sigma_m,
            min_components=self._config.min_components,
        )

        anchor_state, anchor_payload = self._anchor_payload()

        record: Dict[str, Any]
        if mixture is None:
            self._counters["frames_no_match"] += 1
            record = {
                "type": "no_match",
                "rgb_recv_ts": rgb_recv_ts,
                "rgb_ts": rgb_ts,
                "top_k": top_k_records,
                "current_pose": current_pose,
                "anchor": anchor_payload,
            }
        else:
            raw_positions_xy, weights, sigma_m = mixture
            # Apply the anchor offset if calibrated, so mixture
            # positions are in the live session's frame.
            calib = self._anchor.calibration
            if calib is not None:
                positions_xy = calib.apply_xy(raw_positions_xy)
            else:
                positions_xy = raw_positions_xy

            would_be = self._compute_would_be(
                positions_xy, weights, sigma_m, snapshot=snapshot,
            )
            self._counters["frames_observed"] += 1
            if would_be is not None:
                self._counters["would_be_updates_logged"] += 1

            gate = self._evaluate_gate(snapshot, sigma_m)
            applied = False
            if self._config.live:
                applied = self._maybe_apply_live(
                    positions_xy, weights, sigma_m, snapshot, gate,
                )

            record = {
                "type": "vpr_obs",
                "rgb_recv_ts": rgb_recv_ts,
                "rgb_ts": rgb_ts,
                "top_k": top_k_records,
                "mixture": {
                    "positions_xy": positions_xy.cpu().tolist(),
                    "positions_xy_bank_frame": raw_positions_xy.cpu().tolist(),
                    "weights": weights.cpu().tolist(),
                    "sigma_m": float(sigma_m),
                },
                "current_pose": current_pose,
                "would_be": would_be,
                "anchor": anchor_payload,
                "gating": gate,
                "applied": applied,
            }
        self._write_trace(record)
        if self._on_trace is not None:
            try:
                self._on_trace(record)
            except Exception:
                logger.exception("shadow_vpr: on_trace callback raised")
        return record

    # ── Snapshot, gating, would-be, apply ────────────────────────────

    def _snapshot_pf(self) -> Optional[Dict[str, Any]]:
        """Clone filter state under lock; do all heavy math off-lock.

        Returns ``{state, log_w, cov_xy, mean}`` or ``None`` if the
        filter isn't seeded yet. ``cov_xy`` is (2, 2), ``mean`` is the
        full SE(2) posterior mean tuple.
        """
        try:
            with self._pf_lock:
                if getattr(self._pf, "_state", None) is None:
                    return None
                state = self._pf.state.clone().detach()
                log_w = self._pf._log_w.clone().detach()
                cov = self._pf.posterior_cov().clone().detach()
                mean = self._pf.posterior_mean()
        except Exception:
            logger.exception("shadow_vpr: pf snapshot failed")
            return None
        return {
            "state": state,
            "log_w": log_w,
            "cov_xy": cov[:2, :2],
            "mean": mean,
        }

    def _compute_would_be(
        self,
        positions_xy: torch.Tensor, weights: torch.Tensor,
        sigma_m: float,
        *, snapshot: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Per-particle log-likelihood + posterior-shift stats. Does
        not mutate pf — uses the off-lock snapshot."""
        if snapshot is None:
            return None
        state = snapshot["state"]
        log_w = snapshot["log_w"]
        device = state.device
        pos = positions_xy.to(device, dtype=state.dtype)
        w = weights.to(device, dtype=state.dtype)
        w = w / w.sum().clamp_min(1e-30)
        diff = state[:, None, :2] - pos[None, :, :]
        sq = (diff * diff).sum(dim=-1)
        eps = torch.finfo(state.dtype).tiny
        log_terms = torch.log(w.clamp_min(eps))[None, :] - 0.5 * sq / (sigma_m * sigma_m)
        log_lik = torch.logsumexp(log_terms, dim=-1)

        w_before = torch.softmax(log_w, dim=0)
        mean_before = (state[:, :2] * w_before[:, None]).sum(dim=0)
        n_eff_before = float(1.0 / (w_before * w_before).sum().clamp_min(eps))
        new_log_w = log_w + log_lik.to(log_w.dtype)
        w_after = torch.softmax(new_log_w, dim=0)
        mean_after = (state[:, :2] * w_after[:, None]).sum(dim=0)
        n_eff_after = float(1.0 / (w_after * w_after).sum().clamp_min(eps))

        return {
            "mean_xy_before": [float(mean_before[0]), float(mean_before[1])],
            "mean_xy_after":  [float(mean_after[0]),  float(mean_after[1])],
            "n_eff_before":   n_eff_before,
            "n_eff_after":    n_eff_after,
            "log_lik_stats": {
                "mean": float(log_lik.mean()),
                "std":  float(log_lik.std(unbiased=False)),
                "min":  float(log_lik.min()),
                "max":  float(log_lik.max()),
            },
        }

    def _evaluate_gate(
        self, snapshot: Optional[Dict[str, Any]], sigma_m: float,
    ) -> Dict[str, Any]:
        """Decide whether a live application would fire. Returns a dict
        always written to the trace, so post-hoc analysis can see why
        an observation was suppressed."""
        if snapshot is None:
            return {"passed": False, "reason": "no_pf_snapshot",
                    "sigma_xy_m": None, "dist_since_last_m": None}
        cov_xy = snapshot["cov_xy"]
        sigma_xy = float(torch.sqrt(cov_xy.diagonal().sum()).item())
        mean = snapshot["mean"]
        dist = (
            math.hypot(mean[0] - self._last_applied_xy[0],
                       mean[1] - self._last_applied_xy[1])
            if self._last_applied_xy is not None else float("inf")
        )
        threshold_sigma = self._config.gate_sigma_floor_ratio * sigma_m
        gate_sigma_ok = sigma_xy >= threshold_sigma
        gate_dist_ok = dist >= self._config.gate_min_distance_m
        passed = gate_sigma_ok and gate_dist_ok
        if not gate_sigma_ok:
            reason = "cloud_too_tight"
        elif not gate_dist_ok:
            reason = "too_close_to_last"
        else:
            reason = "passed"
        return {
            "passed": passed,
            "reason": reason,
            "sigma_xy_m": sigma_xy,
            "dist_since_last_m": dist if math.isfinite(dist) else None,
        }

    def _maybe_apply_live(
        self,
        positions_xy: torch.Tensor, weights: torch.Tensor,
        sigma_m: float,
        snapshot: Optional[Dict[str, Any]],
        gate: Dict[str, Any],
    ) -> bool:
        """Apply the mixture observation to the live filter, IFF
        anchor is calibrated AND gates pass. Returns whether it fired."""
        if self._anchor.calibration is None:
            self._counters["live_obs_gated_anchor"] += 1
            return False
        if not gate["passed"]:
            if gate["reason"] == "cloud_too_tight":
                self._counters["live_obs_gated_sigma"] += 1
            elif gate["reason"] == "too_close_to_last":
                self._counters["live_obs_gated_distance"] += 1
            return False
        try:
            with self._pf_lock:
                if getattr(self._pf, "_state", None) is None:
                    return False
                self._pf.observe_xy_mixture(
                    positions_xy=positions_xy,
                    weights=weights,
                    sigma_xy_m=sigma_m,
                )
        except Exception:
            logger.exception("shadow_vpr: live observe_xy_mixture failed")
            return False
        if snapshot is not None:
            self._last_applied_xy = (snapshot["mean"][0], snapshot["mean"][1])
        self._counters["live_obs_applied"] += 1
        return True

    def _anchor_payload(self) -> Tuple[str, Dict[str, Any]]:
        """Trace-friendly snapshot of anchor state."""
        state = self._anchor.state
        payload: Dict[str, Any] = {
            "state": state,
            "n_pairs_collected": self._anchor.n_pairs_collected,
        }
        cal = self._anchor.calibration
        if cal is not None:
            payload["offset"] = {
                "dx": cal.dx, "dy": cal.dy, "dtheta_rad": cal.dtheta_rad,
                "n_pairs": cal.n_pairs, "residual_rms_m": cal.residual_rms_m,
            }
        return state, payload

    def _snapshot_current_pose(self, ts: float) -> Optional[List[float]]:
        if self._pose_source is None:
            return None
        try:
            pose = self._pose_source.pose_at(ts)
        except Exception:
            return None
        if pose is None:
            return None
        return [float(pose[0]), float(pose[1]), float(pose[2])]

    # ── Trace I/O ─────────────────────────────────────────────────────

    def log_event(self, record_type: str, payload: Dict[str, Any]) -> None:
        """Write a non-RGB-driven event to the trace (used by the
        Phase 6.4.2 calibration sweep). Adds ``type`` + a timestamp
        if not already present, then writes the line."""
        record = {"type": record_type, "ts": time.time(), **payload}
        self._write_trace(record)

    def anchor(self):
        """Expose the AnchorOffsetEstimator so external orchestrators
        (calibration sweep) can read its pairs and inject a calibration."""
        return self._anchor

    def _write_trace(self, record: Dict[str, Any]) -> None:
        if self._trace_fp is None:
            return
        line = json.dumps(record) + "\n"
        with self._trace_lock:
            self._trace_fp.write(line)
            self._trace_pending += 1
            if self._trace_pending >= self._config.trace_flush_every:
                try:
                    self._trace_fp.flush()
                except Exception:
                    logger.debug("shadow_vpr: trace flush failed", exc_info=True)
                self._trace_pending = 0


# ── Helpers ──────────────────────────────────────────────────────────


def _decode_b64_jpeg(b64: str) -> bytes:
    s = b64.strip()
    if s.startswith("data:"):
        comma = s.find(",")
        if comma >= 0:
            s = s[comma + 1:]
    return base64.standard_b64decode(s)


def _jpeg_bytes_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG → HxWx3 uint8 RGB. Pillow preferred (already a desktop
    dep); falls back to OpenCV (BGR→RGB swap) if Pillow isn't there."""
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(jpeg_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)
    except ImportError:
        pass
    try:
        import cv2
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError as e:
        raise RuntimeError(
            "shadow_vpr: need Pillow or OpenCV to decode JPEG; "
            "neither is installed."
        ) from e


__all__ = [
    "ShadowVPRDriver",
    "ShadowVPRConfig",
]

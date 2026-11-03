"""Change reporting (explore plan B2/B3).

``ChangeReporter`` appends every change event (and every later label) to
``~/body-logs/explore/<session>/events.jsonl`` and keeps a short in-memory
list for the UI. ``ChangeLabeler`` asks the VLM what a change is when it is
in front of the camera: grab a pose-tagged frame, crop it around the change's
bearing (from the frame's own pose) and ask the VLM to classify the crop
against a short closed class list. Labels are advisory — nothing here moves
the robot.

11-03: the old open-vocabulary ``detect()`` on the whole frame + "box nearest
the bearing" labelled a clearly visible box ``unknown`` and the owner's legs
``latch`` / ``jeans`` — never "person".
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from desktop.perception.change_detector import (
    KIND_MOVING, KIND_NEW, STATUS_APPEARED, ChangeEvent,
)
from desktop.perception.frames import PosedFrame
from desktop.perception.vlm_worker import VlmJob, VlmResult
from desktop.tasks.locate import CameraModel, bbox_to_bearing

logger = logging.getLogger(__name__)

Pose = Tuple[float, float, float]

# Closed class list for change labels (owner, 11-03: "small stuffed animal"
# because a toy is easily mistaken for a pet).
CHANGE_CLASSES = (
    "person", "pet", "small stuffed animal", "box or bag", "furniture",
    "door", "nothing there", "other",
)
_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(CHANGE_CLASSES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "what": {"type": "string", "maxLength": 60},
    },
    "required": ["label", "confidence", "what"],
}
_CLASSIFY_PROMPT = (
    "This image is cropped from a camera on a small floor-level robot in a house. "
    "Its lidar just saw something {kind} near floor level about {range_m:.1f} m away, "
    "roughly in the CENTRE of this image. What is that thing?\n"
    "Answer with one label: {classes}.\n"
    "- Legs, feet, shoes or trousers of someone standing or walking: person.\n"
    "- A live cat or dog: pet. A soft toy shaped like an animal: small stuffed animal.\n"
    "- An open or moving door, or its edge: door.\n"
    "- If the centre shows only floor, wall or skirting board: nothing there.\n"
    "Give your confidence (0-1) and a few words saying what you see."
)


def crop_box_px(cam: CameraModel, bearing: float, range_m: float, *,
                half_width_m: float = 0.5, min_half_deg: float = 12.0) -> Optional[Tuple[int, int]]:
    """(u0, u1) pixel columns of a crop centred on ``bearing`` (rad, + left)
    wide enough for ~``half_width_m`` either side at ``range_m``. None when
    the bearing is outside the image."""
    if abs(bearing) >= math.radians(89.0):
        return None
    u = cam.cx - cam.fx * math.tan(bearing)
    if not (0.0 <= u < cam.width):
        return None
    half_ang = max(math.radians(min_half_deg),
                   math.atan2(half_width_m, max(range_m, 0.3)))
    half_px = cam.fx * math.tan(half_ang)
    u0 = int(max(0, math.floor(u - half_px)))
    u1 = int(min(cam.width, math.ceil(u + half_px)))
    return (u0, u1) if u1 - u0 >= 16 else None


def crop_jpeg(jpeg: bytes, u0: int, u1: int) -> bytes:
    """Full-height vertical strip [u0, u1) of a JPEG (a standing person fits)."""
    import io
    from PIL import Image
    with Image.open(io.BytesIO(jpeg)) as im:
        out = io.BytesIO()
        im.crop((u0, 0, u1, im.height)).convert("RGB").save(out, format="JPEG", quality=90)
        return out.getvalue()


def classify_crop(jpeg: bytes, *, kind: str, range_m: float,
                  chat_json: Optional[Callable[..., Any]] = None) -> Optional[Dict[str, Any]]:
    """Ask the VLM which CHANGE_CLASSES the crop's centre is (schema-enforced)."""
    if chat_json is None:
        from desktop.vision_service import chat_json
    prompt = _CLASSIFY_PROMPT.format(
        kind="appear" if kind == KIND_NEW else "move", range_m=range_m,
        classes=", ".join(CHANGE_CLASSES))
    ans = chat_json([{"role": "user", "content": prompt}], _CLASSIFY_SCHEMA,
                    images=[jpeg], max_tokens=120, name="change_label")
    if not isinstance(ans, dict) or ans.get("label") not in CHANGE_CLASSES:
        return None
    return ans


def default_explore_dir() -> str:
    return os.path.join(os.path.expanduser("~/body-logs/explore"),
                        time.strftime("%Y%m%d_%H%M%S"))


class ChangeReporter:
    def __init__(self, out_dir: Optional[str], *, keep: int = 50,
                 clock: Callable[[], float] = time.time):
        self.out_dir = out_dir
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=keep)
        self.counts: Dict[str, int] = {}

    def _write(self, rec: Dict[str, Any]) -> None:
        if not self.out_dir:
            return
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, "events.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            logger.exception("change report: write failed")

    def report(self, ev: ChangeEvent, robot_pose: Optional[Pose] = None) -> Dict[str, Any]:
        rec = {"type": "change", "wall_ts": self._clock(), **ev.to_dict()}
        if robot_pose is not None:
            rec["robot"] = [round(float(v), 3) for v in robot_pose]
        key = f"{ev.kind}/{ev.status}"
        with self._lock:
            self._recent.append(rec)
            self.counts[key] = self.counts.get(key, 0) + 1
        extra = ""
        if ev.speed_mps is not None:
            extra += f" speed {ev.speed_mps:.2f} m/s"
        if ev.path_m is not None:
            extra += f" path {ev.path_m:.1f} m"
        if ev.duration_s is not None:
            extra += f" after {ev.duration_s:.0f} s"
        logger.info("change #%d %s %s at (%.2f, %.2f)%s", ev.change_id, ev.kind,
                    ev.status, ev.x, ev.y, extra)
        self._write(rec)
        return rec

    def label(self, change_id: int, label: str, **detail: Any) -> None:
        rec = {"type": "label", "wall_ts": self._clock(), "change_id": change_id,
               "label": label, **detail}
        with self._lock:
            for r in self._recent:
                if r.get("change_id") == change_id and r.get("type") == "change":
                    r["label"] = label
        logger.info("change #%d labelled %r %s", change_id, label,
                    {k: v for k, v in detail.items() if k != "jpeg_path"})
        self._write(rec)

    def save_jpeg(self, name: str, jpeg: bytes) -> Optional[str]:
        if not self.out_dir:
            return None
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, name)
            with open(path, "wb") as f:
                f.write(jpeg)
            return path
        except OSError:
            logger.exception("change report: jpeg write failed")
            return None

    def recent(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._recent)


def expected_bearing(pose: Pose, x: float, y: float) -> Tuple[float, float]:
    """(bearing rad + left, range m) of world (x, y) from ``pose``."""
    dx, dy = x - pose[0], y - pose[1]
    b = math.atan2(dy, dx) - pose[2]
    return (b + math.pi) % (2 * math.pi) - math.pi, math.hypot(dx, dy)


def pick_box(boxes: List[Any], cam: CameraModel, bearing: float,
             tol_rad: float) -> Optional[Tuple[Any, float]]:
    """The detected box whose centre bearing is closest to ``bearing``,
    within ``tol_rad``; (box, bearing error) or None."""
    best = None
    for b in boxes:
        err = abs(bbox_to_bearing(b.bbox, cam) - bearing)
        if err <= tol_rad and (best is None or err < best[1]):
            best = (b, err)
    return best


class ChangeLabeler:
    """At most one label request in flight; each change labelled once.

    Changes are labelled while they are *present*, not only when they first
    appear: ``track()`` every event, ``poll()`` with the robot pose often
    (every scan). A change reported while behind the robot gets its photo
    the first time it is in the camera's view within range and still there
    (11-02: a black cat in the hallway was reported from behind and was dead
    ahead of the camera for 10 s before leaving; the labeller had only tried
    at the report)."""

    def __init__(self, grab: Callable[..., Optional[str]], worker: Any,
                 camera: CameraModel, reporter: ChangeReporter, *,
                 detect_fn: Optional[Callable[[bytes], Any]] = None,
                 max_range_m: float = 3.0, fov_margin: float = 0.85,
                 bearing_tol_rad: float = math.radians(12.0),
                 min_interval_s: float = 1.0,
                 clock: Callable[[], float] = time.monotonic):
        """``grab(on_frame, on_fail, ...)``: ``FrameGrabber.grab``.
        ``detect_fn(jpeg, kind=, range_m=)`` classifies a crop (default
        ``classify_crop``)."""
        if detect_fn is None:
            detect_fn = classify_crop
        self._grab = grab
        self._worker = worker
        self._cam = camera
        self._rep = reporter
        self._detect = detect_fn
        self._max_range = max_range_m
        self._half_fov = fov_margin * math.atan2(camera.width / 2.0, camera.fx)
        self._tol = bearing_tol_rad
        self._min_interval = min_interval_s
        self._clock = clock
        self._lock = threading.Lock()
        self._busy = False
        self._last_req = -1e9
        self._done: set = set()
        self._pending: Dict[int, ChangeEvent] = {}   # present, not yet labelled

    def track(self, ev: ChangeEvent) -> None:
        """Feed every change event: appeared moving/new changes become
        pending; gone ones are dropped."""
        with self._lock:
            if ev.status == STATUS_APPEARED and ev.kind in (KIND_MOVING, KIND_NEW):
                if ev.change_id not in self._done:
                    self._pending[ev.change_id] = ev
            else:
                self._pending.pop(ev.change_id, None)

    def poll(self, pose: Optional[Pose]) -> bool:
        """Try to label the nearest pending change that is in view.
        Returns True when a frame request went out."""
        if pose is None:
            return False
        with self._lock:
            cands = [e for e in self._pending.values() if self.in_view(pose, e.x, e.y)]
        if not cands:
            return False
        cands.sort(key=lambda e: expected_bearing(pose, e.x, e.y)[1])
        return self.maybe_label(cands[0], pose)

    def pending(self) -> List[int]:
        with self._lock:
            return sorted(self._pending)

    def in_view(self, pose: Pose, x: float, y: float) -> bool:
        b, r = expected_bearing(pose, x, y)
        return abs(b) <= self._half_fov and r <= self._max_range

    def maybe_label(self, ev: ChangeEvent, pose: Optional[Pose]) -> bool:
        """Request a label for an appeared moving/new change in view.
        Returns True when a frame request went out."""
        if pose is None or ev.status != STATUS_APPEARED or ev.kind not in (KIND_MOVING, KIND_NEW):
            return False
        if not self.in_view(pose, ev.x, ev.y):
            return False
        now = self._clock()
        with self._lock:
            if self._busy or ev.change_id in self._done or now - self._last_req < self._min_interval:
                return False
            self._busy = True
            self._last_req = now
            self._done.add(ev.change_id)
            self._pending.pop(ev.change_id, None)

        def on_fail(_rid: str, err: str) -> None:
            logger.info("change #%d: frame grab failed (%s)", ev.change_id, err)
            self._release()

        rid = self._grab(lambda fr: self._on_frame(ev, fr), on_fail)
        if rid is None:
            self._release()
            return False
        return True

    def _release(self) -> None:
        with self._lock:
            self._busy = False

    def _on_frame(self, ev: ChangeEvent, frame: PosedFrame) -> None:
        if frame.pose is None:
            logger.info("change #%d: frame without pose — no label", ev.change_id)
            self._release()
            return
        cam = self._cam.scaled(frame.width or self._cam.width, frame.height or self._cam.height)
        bearing, rng = expected_bearing(frame.pose, ev.x, ev.y)
        box = crop_box_px(cam, bearing, rng)
        if box is None:
            logger.info("change #%d: out of the frame by the time it arrived", ev.change_id)
            self._release()
            return
        u0, u1 = box
        detect = self._detect
        kind = ev.kind

        def query(jpeg: bytes) -> Any:
            crop = crop_jpeg(jpeg, u0, u1)
            return {"answer": detect(crop, kind=kind, range_m=rng), "crop": crop,
                    "crop_px": [u0, u1], "range_m": rng}

        ok = self._worker.submit(VlmJob(
            frame=frame, query=query, kind="change_label",
            on_result=lambda res: self._on_result(ev, res)))
        if not ok:
            self._release()

    def _on_result(self, ev: ChangeEvent, res: VlmResult) -> None:
        try:
            v = res.verdict if isinstance(res.verdict, dict) else None
            if res.error or v is None:
                logger.info("change #%d: no label (%s)", ev.change_id, res.error or "no answer")
                return
            jpeg_path = self._rep.save_jpeg(f"change_{ev.change_id:05d}.jpg", v["crop"])
            ans = v.get("answer")
            if not ans:
                self._rep.label(ev.change_id, "unknown", crop_px=v["crop_px"],
                                range_m=round(v["range_m"], 2), jpeg_path=jpeg_path,
                                latency_s=round(res.latency_s, 2))
                return
            self._rep.label(ev.change_id, ans["label"],
                            confidence=round(float(ans.get("confidence", 0.0)), 2),
                            what=str(ans.get("what", ""))[:60], crop_px=v["crop_px"],
                            range_m=round(v["range_m"], 2), jpeg_path=jpeg_path,
                            latency_s=round(res.latency_s, 2))
        except Exception:
            logger.exception("change #%d: labelling failed", ev.change_id)
        finally:
            self._release()

"""Camera + vision dock group for the nav shell.

Two tabified QDockWidgets on the bottom area:
- CameraFeedsDock (new): RGB + depth side-by-side, plus a Request RGB
  button (RGB is on-demand, not streaming; depth streams).
- VisionDock (reused from chassis): chat + detect panel.

VisionDriver wires VisionDock's send_chat / run_detect signals to the
direct-VLM path via the existing _VisionWorker. Jill routing is not
wired here — operators who want Jill still have `python -m desktop.chassis`.

Exposed as CameraPanels, mirroring TeleopPanels' shape, so main_window
gets a single toggle for the group.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QDockWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from desktop.chassis.controller import StubController
from desktop.chassis.ui_qt import (
    VisionDock, _VisionWorker, _overlay_boxes, depth_to_pixmap,
)

logger = logging.getLogger(__name__)


class LidarFeedView(QWidget):
    """Top-down body-frame plot of the latest lidar scan.

    Robot at center, forward (+x) up. Concentric range rings at
    1/2/5 m. Returns rendered as dots at body-frame (x, y) computed
    from (angle, range). Self-scales to fit min(width, height) so
    the widget can share a row with the camera panels without
    forcing a fixed aspect.
    """

    PLOT_RANGE_M = 5.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scan: Optional[Dict[str, Any]] = None
        self._scan_ts: float = 0.0
        self.setMinimumSize(160, 80)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self.setStyleSheet("background-color:#111;")

    def update_scan(self, scan: Optional[Dict[str, Any]], ts: float) -> None:
        self._scan = scan
        self._scan_ts = ts
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D401
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.fillRect(self.rect(), QColor(17, 17, 17))
            w = self.width()
            h = self.height()
            side = min(w, h)
            cx = w / 2.0
            cy = h / 2.0
            scale = (side / 2.0 - 6.0) / self.PLOT_RANGE_M

            # Range rings (1 m, 2 m, 5 m) for spatial reference.
            p.setPen(QPen(QColor(60, 60, 60), 1))
            for r in (1.0, 2.0, 5.0):
                rr = r * scale
                if rr <= 0:
                    continue
                p.drawEllipse(QPointF(cx, cy), rr, rr)

            # Lidar returns. Body frame: +x forward, +y left. Map to
            # screen with forward = up: screen_x = cx - y_body*scale,
            # screen_y = cy - x_body*scale.
            if self._scan is not None:
                ranges = self._scan.get("ranges") or []
                angle_min = float(self._scan.get("angle_min", 0.0))
                angle_inc = self._scan.get("angle_increment")
                angle_inc = (
                    float(angle_inc)
                    if isinstance(angle_inc, (int, float)) and angle_inc > 0
                    else (2.0 * math.pi) / max(1, len(ranges))
                )
                range_max = self._scan.get("range_max")
                range_max_v = (
                    float(range_max)
                    if isinstance(range_max, (int, float)) and range_max > 0
                    else math.inf
                )
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(128, 200, 255))
                for i, r in enumerate(ranges):
                    try:
                        rv = float(r)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(rv) or rv <= 0.0 or rv > range_max_v:
                        continue
                    if rv > self.PLOT_RANGE_M:
                        rv = self.PLOT_RANGE_M
                    a = angle_min + i * angle_inc
                    x_b = rv * math.cos(a)
                    y_b = rv * math.sin(a)
                    sx = cx - y_b * scale
                    sy = cy - x_b * scale
                    p.drawEllipse(QPointF(sx, sy), 1.5, 1.5)

            # Robot triangle (forward = up).
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255))
            tri = QPolygonF([
                QPointF(cx, cy - 7),
                QPointF(cx - 5, cy + 5),
                QPointF(cx + 5, cy + 5),
            ])
            p.drawPolygon(tri)

            # Age + count text in top-left.
            p.setPen(QColor(140, 140, 140))
            if self._scan is None or self._scan_ts <= 0:
                p.drawText(QRectF(4, 2, w - 8, 16), 0, "no scan")
            else:
                age = max(0.0, time.time() - self._scan_ts)
                n = len(self._scan.get("ranges") or [])
                p.drawText(
                    QRectF(4, 2, w - 8, 16), 0,
                    f"lidar  n={n}  age={age:4.2f}s",
                )
        finally:
            p.end()


class CameraFeedsDock(QDockWidget):
    """RGB + depth feeds + Request RGB button."""

    request_rgb_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Cameras", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )

        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        feeds = QHBoxLayout()

        rgb_col = QVBoxLayout()
        rgb_col.addWidget(QLabel("OAK-D RGB (on request)"))
        self.rgb_label = QLabel("no image")
        self.rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rgb_label.setMinimumSize(160, 80)
        # Labels must expand with the splitter pane; render_rgb rescales
        # the pixmap to the current label size each tick, so growing the
        # label grows the displayed image. Same for depth.
        self.rgb_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self.rgb_label.setStyleSheet("background-color:#111;color:#aaa;")
        rgb_col.addWidget(self.rgb_label, stretch=1)
        self.rgb_meta = QLabel("—")
        self.rgb_meta.setStyleSheet("color:#888;")
        rgb_col.addWidget(self.rgb_meta)

        depth_col = QVBoxLayout()
        depth_col.addWidget(QLabel("OAK-D depth"))
        self.depth_label = QLabel("no depth")
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_label.setMinimumSize(160, 80)
        self.depth_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self.depth_label.setStyleSheet("background-color:#111;color:#aaa;")
        depth_col.addWidget(self.depth_label, stretch=1)
        self.depth_meta = QLabel("—")
        self.depth_meta.setStyleSheet("color:#888;")
        depth_col.addWidget(self.depth_meta)

        lidar_col = QVBoxLayout()
        lidar_col.addWidget(QLabel("Lidar (top-down)"))
        self.lidar_view = LidarFeedView()
        lidar_col.addWidget(self.lidar_view, stretch=1)
        self.lidar_meta = QLabel("—")
        self.lidar_meta.setStyleSheet("color:#888;")
        lidar_col.addWidget(self.lidar_meta)

        feeds.addLayout(rgb_col, stretch=1)
        feeds.addLayout(depth_col, stretch=1)
        feeds.addLayout(lidar_col, stretch=1)
        v.addLayout(feeds, stretch=1)

        btn_row = QHBoxLayout()
        self.request_btn = QPushButton("Request RGB")
        self.request_btn.setToolTip(
            "One-shot capture: publishes body/oakd/config capture_rgb. "
            "The Pi replies on body/oakd/rgb; the frame arrives on the "
            "next redraw tick."
        )
        btn_row.addWidget(self.request_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)

        self.setWidget(body)
        self.request_btn.clicked.connect(self.request_rgb_clicked)

    # ── Render (called from main-window tick) ───────────────────────

    def render_rgb(
        self, snap: dict, boxes: list, boxes_for_ts: float,
    ) -> None:
        pending = snap["pending_rgb"]
        err = snap["rgb_error"]
        jpeg = snap["rgb_jpeg"]
        if err:
            self.rgb_label.setText(f"error: {err}")
        elif jpeg:
            pm = QPixmap()
            if not pm.loadFromData(jpeg):
                self.rgb_label.setText("jpeg decode failed")
            else:
                scaled = pm.scaled(
                    max(1, self.rgb_label.width()),
                    max(1, self.rgb_label.height()),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                if boxes and snap["rgb_ts"] == boxes_for_ts:
                    scaled = _overlay_boxes(
                        scaled, boxes,
                        snap["rgb_width"], snap["rgb_height"],
                    )
                self.rgb_label.setPixmap(scaled)
        elif pending:
            self.rgb_label.setText("awaiting RGB reply…")
        else:
            self.rgb_label.setText("no image")
        if snap["rgb_ts"] > 0 and jpeg:
            age_s = time.time() - snap["rgb_ts"]
            streaming_on = bool(snap.get("streaming_on"))
            misses = int(snap.get("streaming_misses") or 0)
            stalled = streaming_on and misses >= 3
            if stalled:
                self.rgb_meta.setText(
                    f"⚠ stalled  miss={misses}  "
                    f"{snap['rgb_width']}×{snap['rgb_height']}  "
                    f"age={age_s:4.2f}s"
                )
                self.rgb_meta.setStyleSheet("color: #e8a;")
            else:
                self.rgb_meta.setText(
                    f"{snap['rgb_width']}×{snap['rgb_height']}  "
                    f"req={(snap['rgb_request_id'] or '')[:8]}…  "
                    f"age={age_s:4.2f}s"
                )
                self.rgb_meta.setStyleSheet("color: #888;")

    def render_lidar(self, snap: dict) -> None:
        scan = snap.get("lidar_scan")
        ts = float(snap.get("lidar_ts") or 0.0)
        self.lidar_view.update_scan(scan, ts)
        if scan is None or ts <= 0:
            self.lidar_meta.setText("—")
            return
        ranges = scan.get("ranges") or []
        age_s = time.time() - ts
        self.lidar_meta.setText(
            f"{len(ranges)} rays  age={age_s:4.2f}s"
        )

    def render_depth(self, snap: dict) -> None:
        img = snap["depth_image"]
        fmt = snap["depth_format"]
        if img is None:
            msg = f"no depth (format={fmt!r})" if fmt else "no depth"
            self.depth_label.setText(msg)
            self.depth_meta.setText("—")
            return
        try:
            pm = depth_to_pixmap(img)
        except Exception as e:
            logger.exception("depth render failed")
            self.depth_label.setText(f"render error: {e}")
            return
        # Fit current label size in both dimensions, preserve aspect —
        # letterbox rather than stretch. render_rgb does the same.
        scaled = pm.scaled(
            max(1, self.depth_label.width()),
            max(1, self.depth_label.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.depth_label.setPixmap(scaled)
        age_s = time.time() - snap["depth_ts"] if snap["depth_ts"] else 0.0
        valid_frac = float((img > 0).mean()) if img.size else 0.0
        self.depth_meta.setText(
            f"{snap['depth_width']}×{snap['depth_height']} "
            f"valid={valid_frac*100:4.1f}%  age={age_s:4.2f}s"
        )


class VisionDriver:
    """Wires VisionDock signals to direct-VLM calls via _VisionWorker.

    Jill mode intentionally unsupported in nav (the Jill client lives
    in chassis.jill_client and depends on a separate Zenoh topic
    routing story; not re-plumbing that here). Users selecting Jill
    in the dock get a polite error.
    """

    def __init__(
        self,
        chassis: StubController,
        vision_dock: VisionDock,
    ) -> None:
        self.chassis = chassis
        self.vision_dock = vision_dock
        self._transcript: list[dict] = []
        self._worker: Optional[_VisionWorker] = None
        self._pending_detect_ts: float = 0.0
        # Boxes tied to the rgb_ts they were computed for, so the RGB
        # render can auto-clear them when a newer frame arrives.
        self.boxes: list = []
        self.boxes_for_ts: float = 0.0

        vision_dock.send_chat.connect(self._on_send)
        vision_dock.run_detect.connect(self._on_detect)

    def _current_frame(self) -> tuple[Optional[bytes], float]:
        s = self.chassis.state
        with s.lock:
            return s.rgb_jpeg, s.rgb_ts

    def _on_send(self, text: str, attach_frame: bool) -> None:
        if self.vision_dock.mode() == "jill":
            self.vision_dock.append_turn(
                "error",
                "Jill chat is not wired in nav yet. Use "
                "`python -m desktop.chassis` for Jill chat.",
            )
            return
        if self._worker is not None:
            return
        jpeg, _ts = self._current_frame()
        if attach_frame and not jpeg:
            self.vision_dock.append_turn(
                "error",
                "No RGB frame to attach (click Request RGB first).",
            )
            return
        self._transcript.append({"role": "user", "content": text})
        self.vision_dock.append_turn(
            "user", text + (" [+frame]" if attach_frame else ""),
        )
        images = [jpeg] if (attach_frame and jpeg) else None
        self._start_worker(
            "chat",
            {"messages": list(self._transcript), "images": images},
        )

    def _on_detect(self) -> None:
        if self._worker is not None:
            return
        jpeg, ts = self._current_frame()
        if not jpeg:
            self.vision_dock.append_turn(
                "error", "No RGB frame — click Request RGB first.",
            )
            return
        self._pending_detect_ts = ts
        self.vision_dock.append_turn(
            "user", "[detect objects in current frame]",
        )
        self._start_worker("detect", {"jpeg_bytes": jpeg})

    def _start_worker(self, mode: str, kwargs: dict) -> None:
        worker = _VisionWorker(mode, kwargs, parent=self.vision_dock)
        worker.chat_result.connect(self._on_chat_result)
        worker.detect_result.connect(self._on_detect_result)
        worker.error.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self.vision_dock.set_busy(True, f"{mode}…")
        worker.start()

    def _on_chat_result(self, text: str) -> None:
        self._transcript.append({"role": "assistant", "content": text})
        self.vision_dock.append_turn("assistant", text)

    def _on_detect_result(self, result: Any) -> None:
        text = getattr(result, "text", "")
        self._transcript.append({"role": "assistant", "content": text})
        boxes = getattr(result, "boxes", None) or []
        if boxes:
            summary = "detected: " + ", ".join(
                f"{b.label}"
                + (f" ({b.confidence:.2f})"
                   if getattr(b, "confidence", None) is not None else "")
                for b in boxes
            )
            self.vision_dock.append_turn("assistant", summary)
        else:
            self.vision_dock.append_turn("assistant", text)
        self.boxes = boxes
        self.boxes_for_ts = self._pending_detect_ts

    def _on_error(self, msg: str) -> None:
        self.vision_dock.append_turn("error", msg)

    def _on_finished(self) -> None:
        worker = self._worker
        self._worker = None
        self.vision_dock.set_busy(False)
        if worker is not None:
            worker.deleteLater()


def _wrap_with_header(inner: QWidget, title: str) -> QWidget:
    """Prepend a small bold title above `inner` so the section is
    labeled once we strip the QDockWidget titlebar.
    """
    wrapper = QWidget()
    v = QVBoxLayout(wrapper)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(2)
    label = QLabel(title)
    label.setStyleSheet("color:#888; font-weight:bold;")
    v.addWidget(label)
    v.addWidget(inner, stretch=1)
    return wrapper


class CameraPanels:
    """Holds the camera-feed and vision widgets.

    Construction reuses the existing CameraFeedsDock / VisionDock
    classes (because that's where render_rgb / render_depth / chat /
    TTS live), but we pull out their .widget() bodies and expose them
    as plain widgets so main_window can place them independently —
    feeds in the vertical splitter with the maps, vision in its own
    narrow column alongside.
    """

    def __init__(self, chassis: StubController) -> None:
        self.chassis = chassis
        self._feeds_dock = CameraFeedsDock()
        self._vision_dock = VisionDock()
        self.vision_driver = VisionDriver(chassis, self._vision_dock)

        self.feeds_widget = _wrap_with_header(
            self._feeds_dock.widget(), "Cameras",
        )
        self.vision_widget = _wrap_with_header(
            self._vision_dock.widget(), "Vision",
        )

        self._feeds_dock.request_rgb_clicked.connect(self._on_request_rgb)

    def set_visible(self, visible: bool) -> None:
        # Only the feeds widget lives in the central splitter; vision
        # is a separate left-dock-area dock with its own toggle in
        # main_window. So this toggle now refers strictly to feeds.
        self.feeds_widget.setVisible(visible)

    def is_visible(self) -> bool:
        return self.feeds_widget.isVisible()

    def update_state(self, snap: dict) -> None:
        """Render feeds only while the feeds pane is visible."""
        if not self.feeds_widget.isVisible():
            return
        self._feeds_dock.render_rgb(
            snap, self.vision_driver.boxes, self.vision_driver.boxes_for_ts,
        )
        self._feeds_dock.render_depth(snap)
        self._feeds_dock.render_lidar(snap)

    def _on_request_rgb(self) -> None:
        req = self.chassis.request_rgb()
        if req is None:
            self._feeds_dock.rgb_meta.setText(
                "request failed (not connected?)"
            )
        else:
            self._feeds_dock.rgb_meta.setText(
                f"request_id {req[:8]}… pending"
            )


def build_camera_snapshot(chassis: StubController) -> dict:
    """Pull the fields CameraFeedsDock's render methods consume.

    `streaming_misses` is sampled outside the state lock — it lives on
    the controller as a plain attribute, not in the shared state object.
    Caller is expected to add `streaming_on` before render so the stall
    indicator only fires when streaming is actually engaged.
    """
    s = chassis.state
    with s.lock:
        snap = dict(
            rgb_jpeg=s.rgb_jpeg, rgb_width=s.rgb_width,
            rgb_height=s.rgb_height, rgb_ts=s.rgb_ts,
            rgb_error=s.rgb_error, rgb_request_id=s.rgb_request_id,
            pending_rgb=s.pending_rgb_request_id,
            depth_image=s.depth_image, depth_width=s.depth_width,
            depth_height=s.depth_height, depth_format=s.depth_format,
            depth_ts=s.depth_ts,
            lidar_scan=s.lidar_scan, lidar_ts=s.lidar_ts,
        )
    snap["streaming_misses"] = chassis.streaming_rgb_misses
    return snap

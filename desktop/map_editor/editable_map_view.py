"""`WorldDriveableView` subclass that adds a paint mode.

Keeps shared production `map_views.py` untouched. When paint mode is ON,
left-drag paints (emits world-frame points the window maps to cells);
middle/right-drag still pans. When paint mode is OFF, the widget behaves
exactly like `WorldDriveableView` (pan/zoom, and — Phase 2 — left-click
relocate via the shared locate callback).

It also renders an optional live-scan overlay (Phase 2): world-frame
points drawn as small dots on top of the map.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen

from desktop.world_map.map_views import WorldDriveableView


class EditableMapView(WorldDriveableView):
    # Emitted with world (x, y) on paint press and each paint drag step.
    paintAtWorld = pyqtSignal(float, float)
    # Emitted when a paint stroke ends (mouse release). For undo grouping.
    strokeStarted = pyqtSignal()
    strokeEnded = pyqtSignal()

    def __init__(self, parent=None, *, shared=None) -> None:
        # The editor's map is static — it must never dim with the
        # "stale — fuser idle?" overlay. A huge stale_s disables it.
        super().__init__(parent, stale_s=1e12, shared=shared)
        self._paint_mode: bool = False
        self._painting: bool = False
        self._scan_world_xy: Optional[np.ndarray] = None

    # ── External API ────────────────────────────────────────────────

    def set_paint_mode(self, on: bool) -> None:
        self._paint_mode = bool(on)
        if not on:
            self._painting = False
        self.setCursor(
            Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor
        )

    def set_scan_points(self, world_xy: Optional[np.ndarray]) -> None:
        """Install live-scan endpoints (N,2) in world frame, or None to
        clear. Phase 2."""
        self._scan_world_xy = (
            np.asarray(world_xy, dtype=np.float64)
            if world_xy is not None and len(world_xy) else None
        )
        self.update()

    # ── Mouse: paint when armed, else delegate to base ──────────────

    def _emit_paint_at(self, event) -> bool:
        if self._paint_geom is None:
            return False
        x_w, y_w = self._widget_to_world(
            float(event.position().x()), float(event.position().y())
        )
        self.paintAtWorld.emit(x_w, y_w)
        return True

    def mousePressEvent(self, event) -> None:
        if (self._paint_mode
                and event.button() == Qt.MouseButton.LeftButton
                and self._paint_geom is not None):
            self._painting = True
            self.strokeStarted.emit()
            self._emit_paint_at(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._painting:
            self._emit_paint_at(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._painting and event.button() == Qt.MouseButton.LeftButton:
            self._painting = False
            self.strokeEnded.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── Overlay rendering ───────────────────────────────────────────

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._scan_world_xy is None or self._paint_geom is None:
            return
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.setPen(QPen(QColor(80, 200, 255, 220), 0))
            p.setBrush(QColor(80, 200, 255, 220))
            for x_w, y_w in self._scan_world_xy:
                rx, ry = self._world_to_widget(float(x_w), float(y_w))
                p.drawRect(int(rx), int(ry), 2, 2)
        finally:
            p.end()

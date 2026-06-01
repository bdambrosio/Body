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
    # Emitted with a world-frame translation delta (dx, dy) while in
    # align mode (left-drag moves the scan overlay to seat it on walls).
    alignDragWorld = pyqtSignal(float, float)

    def __init__(self, parent=None, *, shared=None) -> None:
        # The editor's map is static — it must never dim with the
        # "stale — fuser idle?" overlay. A huge stale_s disables it.
        super().__init__(parent, stale_s=1e12, shared=shared)
        self._paint_mode: bool = False
        self._painting: bool = False
        self._align_mode: bool = False
        self._align_last: Optional[tuple] = None  # last drag world pt
        self._scan_world_xy: Optional[np.ndarray] = None
        # Keep-out overlay: world-frame cell centers (N,2) + cell size (m).
        self._nogo_centers: Optional[np.ndarray] = None
        self._nogo_res_m: float = 0.0
        # Accept key focus so arrow-key nudge works in align mode.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── External API ────────────────────────────────────────────────

    def set_paint_mode(self, on: bool) -> None:
        self._paint_mode = bool(on)
        if not on:
            self._painting = False
        self.setCursor(
            Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor
        )

    def set_align_mode(self, on: bool) -> None:
        self._align_mode = bool(on)
        self._align_last = None
        if on:
            self._paint_mode = False
            self._painting = False
        self.setCursor(
            Qt.CursorShape.SizeAllCursor if on else Qt.CursorShape.ArrowCursor
        )

    def set_scan_points(self, world_xy: Optional[np.ndarray]) -> None:
        """Install live-scan endpoints (N,2) in world frame, or None to
        clear. Phase 2."""
        self._scan_world_xy = (
            np.asarray(world_xy, dtype=np.float64)
            if world_xy is not None and len(world_xy) else None
        )
        self.update()

    def set_nogo_cells(
        self, centers_xy: Optional[np.ndarray], res_m: float,
    ) -> None:
        """Install keep-out cell centers (N,2) in world frame for the
        translucent orange overlay, or None to clear. `res_m` is the cell
        size so the squares stay glued to cells at any zoom."""
        self._nogo_centers = (
            np.asarray(centers_xy, dtype=np.float64)
            if centers_xy is not None and len(centers_xy) else None
        )
        self._nogo_res_m = float(res_m)
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
        if (self._align_mode
                and event.button() == Qt.MouseButton.LeftButton
                and self._paint_geom is not None):
            self._align_last = self._widget_to_world(
                float(event.position().x()), float(event.position().y()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._painting:
            self._emit_paint_at(event)
            event.accept()
            return
        if self._align_mode and self._align_last is not None:
            x_w, y_w = self._widget_to_world(
                float(event.position().x()), float(event.position().y()))
            self.alignDragWorld.emit(x_w - self._align_last[0],
                                     y_w - self._align_last[1])
            self._align_last = (x_w, y_w)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._painting and event.button() == Qt.MouseButton.LeftButton:
            self._painting = False
            self.strokeEnded.emit()
            event.accept()
            return
        if self._align_mode and event.button() == Qt.MouseButton.LeftButton:
            self._align_last = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        # Arrow keys nudge the aligned scan in screen-aligned directions
        # (Shift = coarse). The view isn't rotated, so screen↔world is
        # fixed: +world_x is up, +world_y is left (see _world_to_widget).
        if self._align_mode and self._paint_geom is not None:
            coarse = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            step = 0.10 if coarse else 0.02
            d = {
                Qt.Key.Key_Up: (step, 0.0),
                Qt.Key.Key_Down: (-step, 0.0),
                Qt.Key.Key_Left: (0.0, step),
                Qt.Key.Key_Right: (0.0, -step),
            }.get(event.key())
            if d is not None:
                self.alignDragWorld.emit(d[0], d[1])
                event.accept()
                return
        super().keyPressEvent(event)

    # ── Overlay rendering ───────────────────────────────────────────

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._paint_geom is None:
            return
        # Keep-out under the live scan so scan dots stay visible on top.
        if self._nogo_centers is not None:
            self._draw_nogo()
        if self._scan_world_xy is not None:
            self._draw_scan()

    def _draw_nogo(self) -> None:
        ppm = self._paint_geom[3]  # px per metre (cwx, cwy, side_px, ppm, …)
        side = max(1, int(round(self._nogo_res_m * ppm)))
        half = side / 2.0
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 140, 0, 110))  # translucent orange
            for x_w, y_w in self._nogo_centers:
                rx, ry = self._world_to_widget(float(x_w), float(y_w))
                p.drawRect(int(rx - half), int(ry - half), side, side)
        finally:
            p.end()

    def _draw_scan(self) -> None:
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

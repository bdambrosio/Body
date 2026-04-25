"""Map view widgets. Originally copied from dev/body_stub/ui_qt.py;
extended in place with:

  - auto-fit-to-data: render the populated cell rect + margin instead
    of the whole pre-allocated 40 m × 40 m grid;
  - in-widget zoom + pan: wheel zooms anchored at cursor, middle/right-
    drag pans, double-click resets to auto-fit;
  - pose trail overlay: polyline of recent poses, transformed under the
    same view as the grid image.

If a third consumer ever appears, promote the shared base class to
desktop/shared/widgets/.
"""
from __future__ import annotations

import math
import time
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget


def _turbo_rgb(x: np.ndarray) -> np.ndarray:
    """x: float32 in [0,1], shape (h,w). Returns uint8 (h,w,3) RGB.
    Anton Mikhailov's polynomial Turbo approximation, Apache 2.0.
    """
    r = 0.1357 + x*(4.5744 + x*(-42.3335 + x*(130.8988 + x*(-152.6574 + x*59.9032))))
    g = 0.0914 + x*(2.1915 + x*(  4.9271 + x*(-14.1846 + x*(  4.2755 + x* 2.8289))))
    b = 0.1067 + x*(12.5989 + x*(-60.1846 + x*(109.2364 + x*(-88.7840 + x*27.0060))))
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


# Display orientation: world +x → widget up, world +y → widget left.
# (Matches the original WorldHeightView convention.)


class _WorldViewBase(QWidget):
    """Shared transform / zoom / pan / overlay logic for world-frame
    map widgets. Subclasses provide grid colorization via _grid_to_rgb.

    Coordinate mapping. With the view rect (square in world frame)
    `(vx_min, vx_max, vy_min, vy_max)` rendered into the centered
    square `(ox, oy, side_px, side_px)` in the widget,
        widget_x = (ox + side_px/2) - (y_w - vcy) * px_per_m
        widget_y = (oy + side_px/2) - (x_w - vcx) * px_per_m
    where vcx, vcy = view-rect center, px_per_m = side_px / side_world.
    """

    AUTO_MARGIN_CELLS = 4
    MIN_VIEW_SIDE_M = 0.4   # ~5 cells at 0.08 m/cell
    MAX_VIEW_SIDE_M = 80.0  # 2× the world extent
    MARGIN_PX = 6

    def __init__(self, parent: Optional[QWidget] = None, *, stale_s: float = 2.0):
        super().__init__(parent)
        self.setMinimumSize(160, 160)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )

        self._meta: Optional[dict] = None
        self._ts: float = 0.0
        self._stale_s = stale_s
        self._pose: Optional[Tuple[float, float, float]] = None
        self._pose_history: list[Tuple[float, float, float]] = []
        self._bounds_ij: Optional[Tuple[int, int, int, int]] = None

        # View state. _view_rect is None ⇒ auto-fit.
        self._view_rect: Optional[Tuple[float, float, float, float]] = None
        self._dragging = False
        self._drag_anchor_widget: Optional[QPointF] = None
        self._drag_anchor_view_center: Optional[Tuple[float, float]] = None

        # Cached paint-time geometry so mouse handlers can convert
        # widget coords ↔ world coords using the same numbers the last
        # paint used. Tuple: (cwx, cwy, side_px, px_per_m, view_rect).
        self._paint_geom: Optional[Tuple[float, float, float, float,
                                          Tuple[float, float, float, float]]] = None

    # ── Public update API ────────────────────────────────────────────

    def update_inputs(
        self,
        *,
        meta: Optional[dict],
        ts: float,
        pose: Optional[Tuple[float, float, float]],
        pose_history: Optional[Sequence[Tuple[float, float, float]]],
        bounds_ij: Optional[Tuple[int, int, int, int]],
    ) -> None:
        self._meta = meta
        self._ts = ts
        self._pose = pose
        self._pose_history = list(pose_history) if pose_history else []
        self._bounds_ij = bounds_ij
        self.update()

    # ── Auto-fit / view-rect helpers ─────────────────────────────────

    def _auto_view_rect(
        self, meta: dict,
    ) -> Tuple[float, float, float, float]:
        res = float(meta["resolution_m"])
        ox = float(meta["origin_x_m"])
        oy = float(meta["origin_y_m"])
        nx = int(meta["nx"])
        ny = int(meta["ny"])
        if self._bounds_ij is not None:
            i0, i1, j0, j1 = self._bounds_ij
            x_min = ox + i0 * res
            x_max = ox + (i1 + 1) * res
            y_min = oy + j0 * res
            y_max = oy + (j1 + 1) * res
            m = self.AUTO_MARGIN_CELLS * res
            x_min -= m
            x_max += m
            y_min -= m
            y_max += m
        else:
            x_min, x_max = ox, ox + nx * res
            y_min, y_max = oy, oy + ny * res

        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        side = max(x_max - x_min, y_max - y_min, self.MIN_VIEW_SIDE_M)
        side = min(side, self.MAX_VIEW_SIDE_M)
        return (cx - side / 2, cx + side / 2,
                cy - side / 2, cy + side / 2)

    def _effective_view_rect(self, meta: dict) -> Tuple[float, float, float, float]:
        if self._view_rect is not None:
            return self._view_rect
        return self._auto_view_rect(meta)

    def _set_view_rect(self, rect: Tuple[float, float, float, float]) -> None:
        # Clamp side to sane limits.
        cx = 0.5 * (rect[0] + rect[1])
        cy = 0.5 * (rect[2] + rect[3])
        side = max(rect[1] - rect[0], self.MIN_VIEW_SIDE_M)
        side = min(side, self.MAX_VIEW_SIDE_M)
        self._view_rect = (cx - side / 2, cx + side / 2,
                           cy - side / 2, cy + side / 2)

    def reset_view(self) -> None:
        """Drop manual zoom/pan; resume auto-fit."""
        self._view_rect = None
        self.update()

    def is_auto_fit(self) -> bool:
        return self._view_rect is None

    # ── Transform helpers (paint-time geometry must be set) ──────────

    def _world_to_widget(self, x_w: float, y_w: float) -> Tuple[float, float]:
        g = self._paint_geom
        if g is None:
            return 0.0, 0.0
        cwx, cwy, _side_px, ppm, (vx0, vx1, vy0, vy1) = g
        vcx = 0.5 * (vx0 + vx1)
        vcy = 0.5 * (vy0 + vy1)
        return (cwx - (y_w - vcy) * ppm,
                cwy - (x_w - vcx) * ppm)

    def _widget_to_world(self, rx: float, ry: float) -> Tuple[float, float]:
        g = self._paint_geom
        if g is None:
            return 0.0, 0.0
        cwx, cwy, _side_px, ppm, (vx0, vx1, vy0, vy1) = g
        vcx = 0.5 * (vx0 + vx1)
        vcy = 0.5 * (vy0 + vy1)
        return (vcx - (ry - cwy) / ppm,
                vcy - (rx - cwx) / ppm)

    # ── Subclass hook ────────────────────────────────────────────────

    def _grid_to_rgb(self) -> Optional[np.ndarray]:
        """Return uint8 (nx, ny, 3) RGB for the *unflipped* grid.
        The base class handles the display flip. Return None if the
        subclass has no grid yet."""
        raise NotImplementedError

    def _legend_text(self) -> str:
        return ""

    # ── Painting ─────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            w, h = self.width(), self.height()
            p.fillRect(0, 0, w, h, QColor(10, 10, 10))

            if self._meta is None:
                p.setPen(QColor(160, 160, 160))
                p.drawText(10, 16, "no world map (drive to populate)")
                return

            meta = self._meta
            res = float(meta.get("resolution_m", 0.0))
            if res <= 0.0:
                p.setPen(QColor(255, 100, 100))
                p.drawText(10, 16, f"bad resolution_m={res}")
                return

            # Centered square draw rect.
            avail = max(0, min(w, h) - 2 * self.MARGIN_PX)
            if avail <= 0:
                return
            ox = (w - avail) // 2
            oy = (h - avail) // 2
            side_px = avail

            view_rect = self._effective_view_rect(meta)
            vx0, vx1, vy0, vy1 = view_rect
            side_world = max(vx1 - vx0, 1e-6)
            ppm = side_px / side_world
            cwx = ox + side_px / 2
            cwy = oy + side_px / 2
            self._paint_geom = (cwx, cwy, side_px, ppm, view_rect)

            # Grid image.
            rgb = self._grid_to_rgb()
            if rgb is not None:
                nx, ny, _ = rgb.shape
                origin_x = float(meta["origin_x_m"])
                origin_y = float(meta["origin_y_m"])
                # Flip both axes so display row 0 = max-x, col 0 = max-y.
                flipped = np.ascontiguousarray(rgb[::-1, ::-1])
                qimg = QImage(
                    flipped.data, ny, nx, 3 * ny,
                    QImage.Format.Format_RGB888,
                ).copy()
                # Destination rect in widget pixels. The flipped image
                # spans world x ∈ [origin_x, origin_x+nx*res] and world
                # y ∈ [origin_y, origin_y+ny*res].
                left, top = self._world_to_widget(
                    origin_x + nx * res, origin_y + ny * res,
                )
                right, bottom = self._world_to_widget(origin_x, origin_y)
                dest = QRectF(left, top, right - left, bottom - top)
                p.save()
                p.setClipRect(QRectF(ox, oy, side_px, side_px))
                p.drawImage(dest, qimg, QRectF(qimg.rect()))
                p.restore()

            # Outer frame.
            p.setPen(QPen(QColor(70, 70, 70), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(ox, oy, side_px, side_px)

            # Pose trail (under markers).
            self._draw_pose_trail(p, ox, oy, side_px)

            # World origin (anchor) marker.
            self._draw_world_marker(
                p, 0.0, 0.0, 0.0, ox, oy, side_px,
                color=QColor(140, 140, 200), label="anchor",
            )

            # Robot pose marker.
            if self._pose is not None:
                self._draw_world_marker(
                    p, self._pose[0], self._pose[1], self._pose[2],
                    ox, oy, side_px,
                    color=QColor(255, 255, 255), label=None,
                )

            # Stale dimming overlay.
            age = time.time() - self._ts if self._ts > 0 else 0.0
            if age > self._stale_s:
                p.fillRect(
                    QRectF(ox, oy, side_px, side_px),
                    QColor(0, 0, 0, 140),
                )
                p.setPen(QColor(255, 200, 80))
                p.drawText(
                    ox + 6, oy + 16,
                    f"stale ({age:.1f}s) — fuser idle?",
                )

            # Header text.
            p.setPen(QColor(180, 180, 180))
            p.drawText(self.MARGIN_PX, self.MARGIN_PX + 10, self._legend_text())

            # View HUD: scale + zoom indicator (only when zoomed).
            self._draw_view_hud(p, ox, oy, side_px, side_world)
        finally:
            self._paint_geom = None
            p.end()

    def _draw_pose_trail(
        self, p: QPainter, ox: int, oy: int, side_px: int,
    ) -> None:
        if len(self._pose_history) < 2:
            return
        path = QPainterPath()
        first = True
        for x_w, y_w, _ in self._pose_history:
            rx, ry = self._world_to_widget(x_w, y_w)
            if first:
                path.moveTo(rx, ry)
                first = False
            else:
                path.lineTo(rx, ry)
        p.save()
        p.setClipRect(QRectF(ox, oy, side_px, side_px))
        p.setPen(QPen(QColor(255, 220, 120, 200), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.restore()

    def _draw_world_marker(
        self, p: QPainter,
        x_w: float, y_w: float, theta_w: float,
        ox: int, oy: int, side_px: int,
        *, color: QColor, label: Optional[str],
    ) -> None:
        rx, ry = self._world_to_widget(x_w, y_w)
        in_rect = (ox <= rx <= ox + side_px) and (oy <= ry <= oy + side_px)
        if not in_rect:
            # Pin the marker to the nearest edge and tint it to flag
            # it as off-screen (operator still wants to see "robot is
            # somewhere off to the right").
            rx = max(ox, min(ox + side_px - 1, rx))
            ry = max(oy, min(oy + side_px - 1, ry))
            color = QColor(255, 200, 80)

        p.setPen(QPen(color, 1))
        p.setBrush(color)

        size = 9.0
        ct, st = math.cos(-theta_w), math.sin(-theta_w)

        def rot(px, py):
            return (ct * px - st * py, st * px + ct * py)

        p0 = rot(0.0, -size)
        p1 = rot(-size * 0.7, size * 0.7)
        p2 = rot(size * 0.7, size * 0.7)
        tri = QPolygonF([
            QPointF(rx + p0[0], ry + p0[1]),
            QPointF(rx + p1[0], ry + p1[1]),
            QPointF(rx + p2[0], ry + p2[1]),
        ])
        p.drawPolygon(tri)
        if label:
            p.setPen(color)
            p.drawText(int(rx) + 12, int(ry) + 4, label)

    def _draw_view_hud(
        self, p: QPainter, ox: int, oy: int, side_px: int, side_world: float,
    ) -> None:
        # 1 m scale bar in the bottom-left of the draw rect.
        ppm = side_px / max(side_world, 1e-6)
        bar_m = self._pick_scale_bar_m(side_world)
        bar_px = bar_m * ppm
        bx0 = ox + 8
        bx1 = bx0 + bar_px
        by = oy + side_px - 12
        p.setPen(QPen(QColor(220, 220, 220), 2))
        p.drawLine(QPointF(bx0, by), QPointF(bx1, by))
        p.drawLine(QPointF(bx0, by - 4), QPointF(bx0, by + 4))
        p.drawLine(QPointF(bx1, by - 4), QPointF(bx1, by + 4))
        p.setPen(QColor(220, 220, 220))
        label = (
            f"{bar_m:.2g} m"
            if bar_m < 1.0 else f"{bar_m:.0f} m"
        )
        p.drawText(int(bx1) + 6, int(by) + 4, label)

        # Zoom badge in the top-right when not in auto-fit.
        if self._view_rect is not None:
            badge = f"{side_world:.1f} m view  (dbl-click to fit)"
            p.setPen(QColor(220, 220, 120))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(badge)
            p.drawText(ox + side_px - tw - 8, oy + 16, badge)

    @staticmethod
    def _pick_scale_bar_m(side_world: float) -> float:
        # ~1/5 of the visible side, snapped to a 1/2/5·10ⁿ value.
        target = max(side_world / 5.0, 0.05)
        e = math.floor(math.log10(target))
        base = 10.0 ** e
        for k in (1.0, 2.0, 5.0, 10.0):
            if k * base >= target:
                return k * base
        return 10.0 * base

    # ── Mouse / wheel ────────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        if self._meta is None or self._paint_geom is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 ** (delta / 120.0)  # >1 = zoom in
        # Cursor position in widget pixels. Use position() for HiDPI.
        pos = event.position()
        mx, my = float(pos.x()), float(pos.y())
        x_w, y_w = self._widget_to_world(mx, my)

        # Current view side; new side after zoom.
        meta = self._meta
        cur = self._effective_view_rect(meta)
        cur_side = cur[1] - cur[0]
        new_side = max(self.MIN_VIEW_SIDE_M,
                       min(self.MAX_VIEW_SIDE_M, cur_side / factor))

        # Recenter so (x_w, y_w) maps back to (mx, my) under new ppm.
        cwx, cwy, side_px, _ppm, _vr = self._paint_geom
        new_ppm = side_px / new_side
        vcy_new = y_w + (mx - cwx) / new_ppm
        vcx_new = x_w + (my - cwy) / new_ppm
        self._set_view_rect((
            vcx_new - new_side / 2, vcx_new + new_side / 2,
            vcy_new - new_side / 2, vcy_new + new_side / 2,
        ))
        event.accept()
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.MouseButton.MiddleButton,
                              Qt.MouseButton.RightButton,
                              Qt.MouseButton.LeftButton):
            if self._meta is None or self._paint_geom is None:
                return
            self._dragging = True
            self._drag_anchor_widget = event.position()
            vr = self._effective_view_rect(self._meta)
            self._drag_anchor_view_center = (
                0.5 * (vr[0] + vr[1]), 0.5 * (vr[2] + vr[3]),
            )
            # First drag beat: pin the current rect as the manual rect.
            if self._view_rect is None:
                self._set_view_rect(vr)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if not self._dragging or self._paint_geom is None:
            return
        if self._drag_anchor_widget is None or self._drag_anchor_view_center is None:
            return
        cwx, cwy, side_px, ppm, vr = self._paint_geom
        dx = float(event.position().x() - self._drag_anchor_widget.x())
        dy = float(event.position().y() - self._drag_anchor_widget.y())
        # widget_x = cwx - (y - vcy)*ppm  ⇒  Δy_world = -Δwidget_x / ppm
        # widget_y = cwy - (x - vcx)*ppm  ⇒  Δx_world = -Δwidget_y / ppm
        vcx0, vcy0 = self._drag_anchor_view_center
        vcx_new = vcx0 - dy / ppm
        vcy_new = vcy0 - dx / ppm
        side = vr[1] - vr[0]
        self._set_view_rect((
            vcx_new - side / 2, vcx_new + side / 2,
            vcy_new - side / 2, vcy_new + side / 2,
        ))
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging:
            self._dragging = False
            self._drag_anchor_widget = None
            self._drag_anchor_view_center = None
            self.unsetCursor()
            event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_view()
        event.accept()


class WorldHeightView(_WorldViewBase):
    """Top-down render of a height grid in world frame."""

    DEFAULT_MAX_HEIGHT_M = 2.2

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        stale_s: float = 2.0,
        max_height_m: float = DEFAULT_MAX_HEIGHT_M,
    ):
        super().__init__(parent, stale_s=stale_s)
        self._max_height_m = max_height_m
        self._grid: Optional[np.ndarray] = None

    def update_map(
        self,
        grid: Optional[np.ndarray],
        meta: Optional[dict],
        ts: float,
        pose: Optional[Tuple[float, float, float]] = None,
        pose_history: Optional[Sequence[Tuple[float, float, float]]] = None,
        bounds_ij: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self._grid = grid
        self.update_inputs(
            meta=meta, ts=ts, pose=pose,
            pose_history=pose_history, bounds_ij=bounds_ij,
        )

    def _grid_to_rgb(self) -> Optional[np.ndarray]:
        if self._grid is None:
            return None
        valid = ~np.isnan(self._grid)
        norm = np.zeros_like(self._grid, dtype=np.float32)
        np.divide(self._grid, self._max_height_m, out=norm, where=valid)
        np.clip(norm, 0.0, 1.0, out=norm)
        rgb = _turbo_rgb(norm)
        rgb[~valid] = (16, 16, 16)
        return rgb

    def _legend_text(self) -> str:
        return f"0–{self._max_height_m:.1f} m (turbo)"


class WorldDriveableView(_WorldViewBase):
    """Top-down render of the driveable layer in world frame."""

    COLOR_CLEAR = (60, 170, 90)
    COLOR_BLOCKED = (180, 60, 60)
    COLOR_UNKNOWN = (60, 60, 60)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        stale_s: float = 2.0,
    ):
        super().__init__(parent, stale_s=stale_s)
        self._drive: Optional[np.ndarray] = None

    def update_map(
        self,
        drive: Optional[np.ndarray],
        meta: Optional[dict],
        ts: float,
        pose: Optional[Tuple[float, float, float]] = None,
        pose_history: Optional[Sequence[Tuple[float, float, float]]] = None,
        bounds_ij: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self._drive = drive
        self.update_inputs(
            meta=meta, ts=ts, pose=pose,
            pose_history=pose_history, bounds_ij=bounds_ij,
        )

    def _grid_to_rgb(self) -> Optional[np.ndarray]:
        if self._drive is None:
            return None
        nx, ny = self._drive.shape
        rgb = np.empty((nx, ny, 3), dtype=np.uint8)
        rgb[...] = self.COLOR_UNKNOWN
        rgb[self._drive == 1] = self.COLOR_CLEAR
        rgb[self._drive == 0] = self.COLOR_BLOCKED
        return rgb

    def _legend_text(self) -> str:
        return "clear / blocked / unknown"

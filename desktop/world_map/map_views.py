"""Map view widgets. Originally copied from dev/body_stub/ui_qt.py;
extended in place with:

  - auto-fit-to-data: render the populated cell rect + margin instead
    of the whole pre-allocated 40 m × 40 m grid;
  - in-widget zoom + pan: wheel zooms anchored at cursor, middle/right-
    drag pans, double-click resets to auto-fit;
  - pose trail overlay: polyline of recent poses, transformed under the
    same view as the grid image;
  - shared view state (SharedMapView): pan/zoom on any panel moves all
    panels that subscribe to the same shared instance, so height,
    driveable, and (later) costmap stay aligned;
  - 1 m grid overlay + range rings (1/2/5 m) around the robot pose.

If a third consumer ever appears, promote the shared base class to
desktop/shared/widgets/.
"""
from __future__ import annotations

import math
import time
from typing import Iterable, List, Optional, Sequence, Tuple

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


class SharedMapView:
    """View state shared across one or more map panels.

    Holding the view rect (pan + zoom) plus the grid/ring toggles
    centrally means that wheel-zooming or drag-panning any panel
    moves all attached panels in lockstep. Each subscribed view
    delegates its rect-mutating operations here and reads the
    current rect on each paint.

    Drag-tracking state stays local to the view that received the
    mouse-press: only one view is dragged at a time, but the
    *result* of the drag (the new rect) is broadcast.
    """

    AUTO_MARGIN_CELLS = 4
    MIN_VIEW_SIDE_M = 0.4   # ~5 cells at 0.08 m/cell
    MAX_VIEW_SIDE_M = 80.0  # 2× the world extent

    def __init__(self) -> None:
        self._view_rect: Optional[Tuple[float, float, float, float]] = None
        self._show_grid: bool = True
        self._show_range_rings: bool = True
        self._views: List["_WorldViewBase"] = []

    # ── Subscriptions ───────────────────────────────────────────────

    def attach(self, view: "_WorldViewBase") -> None:
        if view not in self._views:
            self._views.append(view)

    def detach(self, view: "_WorldViewBase") -> None:
        if view in self._views:
            self._views.remove(view)

    def _notify(self) -> None:
        for v in self._views:
            v.update()

    # ── View rect ───────────────────────────────────────────────────

    def view_rect(self) -> Optional[Tuple[float, float, float, float]]:
        """Current manually-set rect, or None for auto-fit."""
        return self._view_rect

    def is_auto_fit(self) -> bool:
        return self._view_rect is None

    def reset_view(self) -> None:
        self._view_rect = None
        self._notify()

    def set_view_rect(self, rect: Tuple[float, float, float, float]) -> None:
        cx = 0.5 * (rect[0] + rect[1])
        cy = 0.5 * (rect[2] + rect[3])
        side = max(rect[1] - rect[0], self.MIN_VIEW_SIDE_M)
        side = min(side, self.MAX_VIEW_SIDE_M)
        self._view_rect = (cx - side / 2, cx + side / 2,
                           cy - side / 2, cy + side / 2)
        self._notify()

    # ── Overlay toggles ─────────────────────────────────────────────

    def show_grid(self) -> bool:
        return self._show_grid

    def set_show_grid(self, on: bool) -> None:
        self._show_grid = bool(on)
        self._notify()

    def show_range_rings(self) -> bool:
        return self._show_range_rings

    def set_show_range_rings(self, on: bool) -> None:
        self._show_range_rings = bool(on)
        self._notify()


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

    AUTO_MARGIN_CELLS = SharedMapView.AUTO_MARGIN_CELLS
    MIN_VIEW_SIDE_M = SharedMapView.MIN_VIEW_SIDE_M
    MAX_VIEW_SIDE_M = SharedMapView.MAX_VIEW_SIDE_M
    MARGIN_PX = 6

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        stale_s: float = 2.0,
        shared: Optional[SharedMapView] = None,
    ):
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

        # Shared view state (pan/zoom + overlay toggles). Default to a
        # private instance so single-view callers continue to work.
        self._shared: SharedMapView = shared if shared is not None else SharedMapView()
        self._shared.attach(self)

        # Drag tracking is local — only the view that captured the
        # mouse-press updates the shared rect on subsequent moves.
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
        manual = self._shared.view_rect()
        if manual is not None:
            return manual
        return self._auto_view_rect(meta)

    def _set_view_rect(self, rect: Tuple[float, float, float, float]) -> None:
        # Delegated to the shared view so all attached panels move
        # together. Clamping is enforced by the shared.
        self._shared.set_view_rect(rect)

    def reset_view(self) -> None:
        """Drop manual zoom/pan; resume auto-fit. Affects all panels
        sharing the same view state."""
        self._shared.reset_view()

    def is_auto_fit(self) -> bool:
        return self._shared.is_auto_fit()

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

            # Grid + range rings (subtle, sit between map image and
            # markers so they're visible against both the map and the
            # bare background outside it).
            if self._shared.show_grid():
                self._draw_grid(p, ox, oy, side_px, side_world)
            if self._shared.show_range_rings() and self._pose is not None:
                self._draw_range_rings(p, ox, oy, side_px, side_world)

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

    def _draw_grid(
        self, p: QPainter, ox: int, oy: int, side_px: int,
        side_world: float,
    ) -> None:
        """Draw a 1 m world-aligned grid (origin at world (0, 0)).
        Spacing scales with view side so the line density stays sane.
        """
        step = self._pick_grid_step_m(side_world)
        if step <= 0.0:
            return
        # World extents currently visible. Snap start/end outward to
        # the nearest grid line so we don't miss a stripe at the edge.
        g = self._paint_geom
        if g is None:
            return
        _cwx, _cwy, _side, _ppm, (vx0, vx1, vy0, vy1) = g

        x_first = math.floor(vx0 / step) * step
        x_last = math.ceil(vx1 / step) * step
        y_first = math.floor(vy0 / step) * step
        y_last = math.ceil(vy1 / step) * step

        p.save()
        p.setClipRect(QRectF(ox, oy, side_px, side_px))
        # World-axis lines (x=0, y=0) get a slightly stronger stroke
        # to anchor the operator's mental model. The anchor marker
        # sits at their intersection.
        thin_pen = QPen(QColor(255, 255, 255, 50), 1)
        thick_pen = QPen(QColor(255, 255, 255, 90), 1)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # Vertical-in-display lines = constant world-y. World +y goes
        # widget-left, so each y-line stretches top-to-bottom of the
        # draw rect at x_widget = world_to_widget(_, y_w).x.
        n_steps_x = int(round((x_last - x_first) / step))
        n_steps_y = int(round((y_last - y_first) / step))
        for k in range(n_steps_y + 1):
            y_w = y_first + k * step
            rx_top, _ = self._world_to_widget(vx1, y_w)
            rx_bot, _ = self._world_to_widget(vx0, y_w)
            p.setPen(thick_pen if abs(y_w) < step * 0.01 else thin_pen)
            p.drawLine(QPointF(rx_top, oy), QPointF(rx_bot, oy + side_px))
        # Horizontal-in-display lines = constant world-x. World +x
        # goes widget-up.
        for k in range(n_steps_x + 1):
            x_w = x_first + k * step
            _, ry_left = self._world_to_widget(x_w, vy1)
            _, ry_right = self._world_to_widget(x_w, vy0)
            p.setPen(thick_pen if abs(x_w) < step * 0.01 else thin_pen)
            p.drawLine(QPointF(ox, ry_left), QPointF(ox + side_px, ry_right))
        p.restore()

    @staticmethod
    def _pick_grid_step_m(side_world: float) -> float:
        """Choose 1/2/5 × 10ⁿ such that the visible square shows
        roughly 5–10 grid stripes per side."""
        if side_world <= 0:
            return 0.0
        target = side_world / 8.0
        e = math.floor(math.log10(max(target, 1e-3)))
        base = 10.0 ** e
        for k in (1.0, 2.0, 5.0, 10.0):
            if k * base >= target:
                return k * base
        return 10.0 * base

    def _draw_range_rings(
        self, p: QPainter, ox: int, oy: int, side_px: int,
        side_world: float,
    ) -> None:
        """Draw 1, 2, 5 m range rings centered on the current robot
        pose, with a small label on each. Only the rings that fall
        within the visible square are useful — but cheap to draw all
        three since QPainter clips, and they're informative even when
        partially out of view."""
        if self._pose is None:
            return
        x_w, y_w, _theta = self._pose
        cx, cy = self._world_to_widget(x_w, y_w)
        ppm = side_px / max(side_world, 1e-6)
        p.save()
        p.setClipRect(QRectF(ox, oy, side_px, side_px))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for r_m in (1.0, 2.0, 5.0):
            r_px = r_m * ppm
            if r_px < 6.0:
                continue  # too small to read, skip
            color = QColor(255, 220, 120, 110)
            p.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
            p.drawEllipse(QPointF(cx, cy), r_px, r_px)
            # Label NE of the ring intersection (world +x = widget up,
            # world +y = widget left, so NE in widget = +x & −y).
            label = f"{r_m:.0f} m"
            label_widget_x, label_widget_y = self._world_to_widget(
                x_w + r_m * 0.7, y_w - r_m * 0.7,
            )
            p.setPen(QColor(255, 220, 120, 180))
            p.drawText(int(label_widget_x) + 2,
                       int(label_widget_y) - 2, label)
        p.restore()

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
        if not self._shared.is_auto_fit():
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
            if self._shared.is_auto_fit():
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
        shared: Optional[SharedMapView] = None,
    ):
        super().__init__(parent, stale_s=stale_s, shared=shared)
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


class WorldCostmapView(_WorldViewBase):
    """Top-down render of the planner's costmap in world frame.

    Inputs are a `Costmap` instance (see `desktop.world_map.costmap`).
    The view itself owns no costmap construction logic — it only
    renders. Building the costmap on each redraw tick is the
    `MainWindow`'s job.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        stale_s: float = 2.0,
        shared: Optional[SharedMapView] = None,
    ):
        super().__init__(parent, stale_s=stale_s, shared=shared)
        self._costmap = None  # Costmap | None

    def update_map(
        self,
        costmap,
        meta: Optional[dict],
        ts: float,
        pose: Optional[Tuple[float, float, float]] = None,
        pose_history: Optional[Sequence[Tuple[float, float, float]]] = None,
        bounds_ij: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self._costmap = costmap
        self.update_inputs(
            meta=meta, ts=ts, pose=pose,
            pose_history=pose_history, bounds_ij=bounds_ij,
        )

    def _grid_to_rgb(self) -> Optional[np.ndarray]:
        if self._costmap is None:
            return None
        # Lazy-import to avoid a hard module dep if costmap.py is
        # ever swapped out at runtime.
        from .costmap import costmap_to_rgb
        return costmap_to_rgb(self._costmap)

    def _legend_text(self) -> str:
        return "lethal / halo / unknown / clear"


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
        shared: Optional[SharedMapView] = None,
    ):
        super().__init__(parent, stale_s=stale_s, shared=shared)
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

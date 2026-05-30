"""Body-frame local_map view for the Tier-3 debug console.

Top-down render with the robot fixed at the origin: +x_body (forward) is
up, +y_body (left) is left. Renders the driveable layer (clear/blocked/
unknown), the robot, the active goal, and accepts a click that maps the
pixel back to a body-frame (x, y) point. Self-contained (not the world-
frame WorldDriveableView, which is coupled to shared zoom/pan + a global
pose).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QWidget

COLOR_CLEAR = QColor(60, 170, 90)
COLOR_BLOCKED = QColor(180, 60, 60)
COLOR_BG = QColor(40, 40, 40)        # unknown / background
COLOR_ROBOT = QColor(255, 255, 255)
COLOR_GOAL = QColor(80, 160, 255)
COLOR_TARGET = QColor(255, 220, 80)      # Tier-2 manual target
COLOR_SUBGOAL = QColor(255, 170, 60)     # Tier-2 chosen sub-goal
COLOR_RAY = QColor(120, 200, 255, 160)   # bearing ray
MARGIN_PX = 8
# Drawn footprint radius (m); match config.json:local_drive.footprint_radius_m.
FOOTPRINT_M = 0.14


class BodyLocalMapView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self._drive: Optional[np.ndarray] = None
        self._meta: Optional[dict] = None
        self._goal_body: Optional[Tuple[float, float]] = None
        self._state_text: str = ""
        self._on_click: Optional[Callable[[float, float], None]] = None
        self._geom: Optional[Tuple[float, float, float, float, float]] = None
        # geom = (wc, hc, ppm, xc, yc): widget center px, m/px, body-center m
        # Tier-2 debug overlay (target / bearing ray / sub-goal).
        self._target_body: Optional[Tuple[float, float]] = None
        self._subgoal_body: Optional[Tuple[float, float]] = None
        self._bearing_rad: Optional[float] = None
        self._free_dist_m: float = 0.0

    def set_click_callback(self, cb: Callable[[float, float], None]) -> None:
        self._on_click = cb

    def set_overlay(
        self,
        target_body: Optional[Tuple[float, float]],
        subgoal_body: Optional[Tuple[float, float]],
        bearing_rad: Optional[float],
        free_dist_m: float = 0.0,
    ) -> None:
        """Tier-2 debug markers: manual target, bearing ray, chosen sub-goal."""
        self._target_body = target_body
        self._subgoal_body = subgoal_body
        self._bearing_rad = bearing_rad
        self._free_dist_m = free_dist_m
        self.update()

    def update_data(
        self,
        drive: Optional[np.ndarray],
        meta: Optional[dict],
        goal_body: Optional[Tuple[float, float]],
        state_text: str = "",
    ) -> None:
        self._drive = drive
        self._meta = meta
        self._goal_body = goal_body
        self._state_text = state_text
        self.update()

    # ── Transform helpers ────────────────────────────────────────────

    def _compute_geom(self, meta: dict) -> Optional[Tuple[float, float, float, float, float]]:
        res = float(meta.get("resolution_m", 0.0))
        if res <= 0:
            return None
        ox = float(meta["origin_x_m"])
        oy = float(meta["origin_y_m"])
        nx = int(meta["nx"])
        ny = int(meta["ny"])
        x_span = nx * res   # forward extent (screen vertical)
        y_span = ny * res   # lateral extent (screen horizontal)
        w = self.width() - 2 * MARGIN_PX
        h = self.height() - 2 * MARGIN_PX
        if w <= 0 or h <= 0 or x_span <= 0 or y_span <= 0:
            return None
        ppm = min(w / y_span, h / x_span)
        xc = ox + x_span / 2.0
        yc = oy + y_span / 2.0
        return (self.width() / 2.0, self.height() / 2.0, ppm, xc, yc)

    def _body_to_px(self, bx: float, by: float) -> Tuple[float, float]:
        wc, hc, ppm, xc, yc = self._geom  # type: ignore[misc]
        sx = wc - (by - yc) * ppm    # +y_body (left) → screen left
        sy = hc - (bx - xc) * ppm    # +x_body (forward) → screen up
        return sx, sy

    def _px_to_body(self, sx: float, sy: float) -> Tuple[float, float]:
        wc, hc, ppm, xc, yc = self._geom  # type: ignore[misc]
        bx = xc + (hc - sy) / ppm
        by = yc + (wc - sx) / ppm
        return bx, by

    # ── Paint ────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(0, 0, self.width(), self.height(), QColor(12, 12, 12))

        if self._drive is None or self._meta is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(10, 18, "no local map (is body.local_map running?)")
            return
        geom = self._compute_geom(self._meta)
        if geom is None:
            return
        self._geom = geom
        _, _, ppm, _, _ = geom
        res = float(self._meta["resolution_m"])
        ox = float(self._meta["origin_x_m"])
        oy = float(self._meta["origin_y_m"])
        cell_px = ppm * res + 1.0  # +1 to avoid hairline gaps

        drive = self._drive
        nx, ny = drive.shape
        # Background = unknown; draw only clear/blocked cells.
        for i in range(nx):
            bx = ox + (i + 0.5) * res
            for j in range(ny):
                v = int(drive[i, j])
                if v == -1:
                    continue
                by = oy + (j + 0.5) * res
                sx, sy = self._body_to_px(bx, by)
                p.fillRect(
                    QRectF(sx - cell_px / 2, sy - cell_px / 2, cell_px, cell_px),
                    COLOR_CLEAR if v == 1 else COLOR_BLOCKED,
                )

        # Goal marker + line from robot.
        if self._goal_body is not None:
            gx, gy = self._body_to_px(self._goal_body[0], self._goal_body[1])
            rx, ry = self._body_to_px(0.0, 0.0)
            p.setPen(QPen(COLOR_GOAL, 2))
            p.drawLine(int(rx), int(ry), int(gx), int(gy))
            p.setBrush(COLOR_GOAL)
            p.drawEllipse(QPointF(gx, gy), 6, 6)

        self._draw_tier2_overlay(p)

        # Robot at origin, a triangle pointing forward (+x → up).
        self._draw_robot(p, ppm)

        if self._state_text:
            p.setPen(QColor(220, 220, 220))
            p.drawText(10, 18, self._state_text)

    def _draw_tier2_overlay(self, p: QPainter) -> None:
        rx, ry = self._body_to_px(0.0, 0.0)
        # Bearing ray from the robot toward the target.
        if self._bearing_rad is not None and self._target_body is not None:
            d = max(0.3, math.hypot(*self._target_body))
            ex, ey = self._body_to_px(d * math.cos(self._bearing_rad),
                                      d * math.sin(self._bearing_rad))
            p.setPen(QPen(COLOR_RAY, 1, Qt.PenStyle.DashLine))
            p.drawLine(int(rx), int(ry), int(ex), int(ey))
        # Manual target: hollow yellow circle + cross.
        if self._target_body is not None:
            tx, ty = self._body_to_px(*self._target_body)
            p.setPen(QPen(COLOR_TARGET, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(tx, ty), 7, 7)
            p.drawLine(int(tx - 9), int(ty), int(tx + 9), int(ty))
            p.drawLine(int(tx), int(ty - 9), int(tx), int(ty + 9))
        # Tier-2 sub-goal: filled orange dot + free-dist annotation.
        if self._subgoal_body is not None:
            sx, sy = self._body_to_px(*self._subgoal_body)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(COLOR_SUBGOAL)
            p.drawEllipse(QPointF(sx, sy), 4, 4)
            p.setPen(COLOR_SUBGOAL)
            p.drawText(QRectF(sx + 6, sy - 8, 90, 14), 0, f"free={self._free_dist_m:.2f}")

    def _draw_robot(self, p: QPainter, ppm: float) -> None:
        # Footprint circle = the Tier-3 swept-check radius. Keep in sync with
        # config.json:local_drive.footprint_radius_m (the actual block radius
        # is this + half a cell, so a pixel just outside the ring can still
        # block until the half-cell / directional refinements land).
        fp = FOOTPRINT_M
        r = fp * ppm
        cx, cy = self._body_to_px(0.0, 0.0)
        nose_x, nose_y = self._body_to_px(fp, 0.0)
        left_x, left_y = self._body_to_px(-0.4 * fp, 0.55 * fp)
        right_x, right_y = self._body_to_px(-0.4 * fp, -0.55 * fp)
        tri = QPolygonF([
            QPointF(nose_x, nose_y), QPointF(left_x, left_y), QPointF(right_x, right_y),
        ])
        p.setPen(QPen(COLOR_ROBOT, 1))
        p.setBrush(QColor(255, 255, 255, 60))
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.setBrush(COLOR_ROBOT)
        p.drawPolygon(tri)

    # ── Click → body point ──────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if self._geom is None or self._on_click is None:
            return
        pos = event.position()
        bx, by = self._px_to_body(pos.x(), pos.y())
        self._on_click(bx, by)

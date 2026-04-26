"""Main window for the nav operator UI.

Hosts both controllers in one Qt process. Central widget = the two
world-map views (height + driveable). A persistent SafetyToolbar owns
Connect/ALL-STOP/Live/pills (steps 3 and 4 will add teleop + camera
docks). A small Map toolbar carries world_map-specific actions (reset
today; goto/explore/etc. later).
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QSplitter, QToolBar, QVBoxLayout, QWidget,
)
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtCore import QUrl

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.world_map.config import FuserConfig
from desktop.world_map.controller import FuserController
from desktop.world_map.costmap import CostmapConfig, build_costmap
from desktop.world_map.map_views import (
    SharedMapView, WorldCostmapView, WorldDriveableView, WorldHeightView,
)
from .follower import (
    Follower, FollowerConfig, FollowerOutput,
    STATUS_ARRIVED, STATUS_FOLLOWING, STATUS_NO_PATH, STATUS_ROTATING,
)
from .mission import Mission, MissionConfig, MissionState
from .planner import AStarConfig, PlanResult, plan_path
from .recovery import (
    PRIM_ABORTED, PRIM_DONE, PRIM_RUNNING,
    REASON_NO_POSE, RecoveryPolicy, RecoveryPrimitive,
    classify_replan_failure,
)
from .safety import SafetyConfig, forward_arc_blocked

from .camera_panels import CameraPanels, build_camera_snapshot
from .safety_toolbar import SafetyToolbar
from .teleop_panels import TeleopPanels, build_chassis_snapshot

logger = logging.getLogger(__name__)


class NavMainWindow(QMainWindow):
    def __init__(
        self,
        fuser: FuserController,
        fuser_config: FuserConfig,
        chassis: StubController,
        chassis_config: StubConfig,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Body Nav")
        self.fuser = fuser
        self.fuser_config = fuser_config
        self.chassis = chassis
        self.chassis_config = chassis_config

        self._build_toolbars()
        self._build_ui()
        self._build_docks()
        self._build_menu()
        self._build_timer()

    # ── Layout ───────────────────────────────────────────────────────

    def _build_toolbars(self) -> None:
        self._safety_toolbar = SafetyToolbar(self.chassis, self.fuser, parent=self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._safety_toolbar)

        # Map toolbar: world_map-scoped actions. Starts with Reset-world;
        # grows as nav primitives (goto/explore/...) come online.
        self._map_toolbar = QToolBar("Map", self)
        self._map_toolbar.setMovable(False)
        self._map_toolbar.setFloatable(False)
        self._map_toolbar.setContextMenuPolicy(
            Qt.ContextMenuPolicy.PreventContextMenu
        )
        reset_act = QAction("Reset world", self)
        reset_act.setToolTip(
            "Clear the accumulated world map and re-anchor to the "
            "robot's current pose."
        )
        reset_act.triggered.connect(
            lambda: self.fuser.request_reset(reason="ui_reset")
        )
        self._map_toolbar.addAction(reset_act)

        save_act = QAction("Save snapshot", self)
        save_act.setToolTip(
            "Write a self-contained snapshot bundle (layers.npz, "
            "PNGs, summary.json) for offline inspection. Path is "
            "shown on completion; default ~/Body/sessions/<sid>/."
        )
        save_act.triggered.connect(self._on_save_snapshot)
        self._map_toolbar.addAction(save_act)

        load_act = QAction("Load snapshot", self)
        load_act.setToolTip(
            "Restore a saved layers.npz into the world grid as a "
            "prior. Vote decay continues normally — re-observed "
            "cells stay confident, unobserved cells gradually fade "
            "toward the floor."
        )
        load_act.triggered.connect(self._on_load_snapshot)
        self._map_toolbar.addAction(load_act)

        fit_act = QAction("Fit maps", self)
        fit_act.setToolTip(
            "Reset map zoom/pan to auto-fit the populated region. "
            "(Shortcut: double-click a map.)"
        )
        fit_act.triggered.connect(self._on_fit_maps)
        self._map_toolbar.addAction(fit_act)

        clear_goal_act = QAction("Clear goal", self)
        clear_goal_act.setToolTip(
            "Remove the current goal pin and any planned path. "
            "Right-click a map to set a new goal."
        )
        clear_goal_act.triggered.connect(self._on_clear_goal)
        self._map_toolbar.addAction(clear_goal_act)

        self._go_act = QAction("Go", self)
        self._go_act.setToolTip(
            "Begin autonomous follow of the planned path. "
            "Requires Live cmd ON and a successful plan."
        )
        self._go_act.triggered.connect(self._on_go)
        self._map_toolbar.addAction(self._go_act)

        self._cancel_act = QAction("Stop", self)
        self._cancel_act.setToolTip(
            "Halt the autonomous follow and zero cmd_vel. "
            "Live cmd remains on so manual driving stays available."
        )
        self._cancel_act.triggered.connect(self._on_cancel)
        self._map_toolbar.addAction(self._cancel_act)

        self._stream_rgb_act = QAction("Stream RGB", self)
        self._stream_rgb_act.setCheckable(True)
        self._stream_rgb_act.setChecked(False)
        self._stream_rgb_act.setToolTip(
            "Toggle low-rate (2 Hz) streaming of OAK-D RGB into the "
            "feed pane — useful when the robot is out of sight. "
            "Default off; on-demand Request RGB still works."
        )
        self._stream_rgb_act.toggled.connect(self._on_toggle_stream_rgb)
        self._map_toolbar.addAction(self._stream_rgb_act)

        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._map_toolbar)

    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Horizontal split: [ left column (maps + camera feeds stacked)
        # | vision column ]. The left column is itself a vertical
        # splitter so maps and feeds can rebalance without stealing
        # height from the chat. The vision column is narrow by default
        # but resizable — grabbing a wide transcript is a click-drag
        # away. Each image widget aspect-preserves its own render.
        self._h_splitter = QSplitter(Qt.Orientation.Horizontal, central)
        self._h_splitter.setChildrenCollapsible(True)

        self._left_splitter = QSplitter(Qt.Orientation.Vertical)
        self._left_splitter.setChildrenCollapsible(True)

        maps_widget = QWidget()
        maps = QHBoxLayout(maps_widget)
        maps.setContentsMargins(0, 0, 0, 0)
        # Shared view state so all map panels pan/zoom together and
        # share the grid + range-ring toggles.
        self._shared_view = SharedMapView()
        self._height_view = WorldHeightView(
            stale_s=self.fuser_config.map_stale_s,
            shared=self._shared_view,
        )
        self._drive_view = WorldDriveableView(
            stale_s=self.fuser_config.map_stale_s,
            shared=self._shared_view,
        )
        self._costmap_view = WorldCostmapView(
            stale_s=self.fuser_config.map_stale_s,
            shared=self._shared_view,
        )
        maps.addWidget(self._height_view, stretch=1)
        maps.addWidget(self._drive_view, stretch=1)
        maps.addWidget(self._costmap_view, stretch=1)
        self._costmap_config = CostmapConfig(
            footprint_radius_m=self.fuser_config.footprint_radius_m,
        )
        self._astar_config = AStarConfig()
        self._last_plan: Optional[PlanResult] = None
        self._last_costmap = None  # cached for replanning when goal changes
        self._shared_view.set_goal_callback(self._on_goal_requested)
        # Stage 4: pure-pursuit follower computes the cmd_vel that
        # *would* be published. Stage 5: when self._mission is in
        # FOLLOWING, the redraw tick pushes the follower output to
        # chassis.set_cmd_vel() and lets chassis publish it through
        # its existing 5 Hz publisher.
        self._follower = Follower(FollowerConfig())
        self._last_follower: Optional[FollowerOutput] = None
        self._mission = Mission()
        self._mission_config = MissionConfig()
        # Recovery policy + currently-running primitive (None unless
        # mission is RECOVERING). Phase 1c ships the stub policy
        # (WaitAndResume for every reason); Phase 2c upgrades.
        self._recovery_policy = RecoveryPolicy()
        self._active_recovery: Optional[RecoveryPrimitive] = None
        # Stage 5b: forward-arc lethal-cell check overrides cmd_vel
        # to zero when an obstacle appears between replans. Mission
        # stays FOLLOWING so we resume when the arc clears.
        self._safety_config = SafetyConfig()
        self._safety_blocked: bool = False
        self._left_splitter.addWidget(maps_widget)

        self._cameras = CameraPanels(self.chassis)
        self._left_splitter.addWidget(self._cameras.feeds_widget)

        # Maps and feeds share the left column 50/50 by default.
        self._left_splitter.setStretchFactor(0, 1)
        self._left_splitter.setStretchFactor(1, 1)

        self._h_splitter.addWidget(self._left_splitter)
        self._h_splitter.addWidget(self._cameras.vision_widget)

        # Left column dominates horizontally; vision column is narrow
        # but user-resizable.
        self._h_splitter.setStretchFactor(0, 4)
        self._h_splitter.setStretchFactor(1, 1)
        self._splitter_balanced = False

        outer.addWidget(self._h_splitter, stretch=1)

        # Bottom status block: 3 narrow rows of fixed-width labels +
        # a notes row. Fixed widths and Ignored size policies keep the
        # central widget's width-hint stable across redraws — without
        # this, per-tick text variation (grace countdown, drift, plan
        # ms, hb seq) pumps the window width and produces visible
        # jitter under XWayland/Mutter. Any label whose CONTENT may
        # vary in width MUST live in this block with a fixed width.
        small = self.font()
        small.setPointSize(max(7, small.pointSize() - 1))
        self._pose_lbl = self._mk_status_label("pose: —", 220, small)
        self._rates_lbl = self._mk_status_label("rates: —", 250, small)
        self._cells_lbl = self._mk_status_label("cells: —", 180, small)
        self._session_lbl = self._mk_status_label("session: —", 210, small)
        self._slam_lbl = self._mk_status_label("slam: —", 270, small)
        self._plan_lbl = self._mk_status_label("plan: —", 220, small)
        self._follow_lbl = self._mk_status_label("follow: —", 360, small)
        self._chassis_lbl = self._mk_status_label("chassis: —", 200, small)
        self._notes_lbl = QLabel("")
        self._notes_lbl.setStyleSheet("color: #e8a; font-weight: bold;")
        self._notes_lbl.setFont(small)

        status = QVBoxLayout()
        status.setContentsMargins(0, 0, 0, 0)
        status.setSpacing(2)
        # Row 1: data freshness — pose / rates / cells / session.
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(8)
        for w in (self._pose_lbl, self._rates_lbl,
                  self._cells_lbl, self._session_lbl):
            row1.addWidget(w)
        row1.addStretch(1)
        status.addLayout(row1)
        # Row 2: control state — slam / plan / follow / chassis.
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(8)
        for w in (self._slam_lbl, self._plan_lbl,
                  self._follow_lbl, self._chassis_lbl):
            row2.addWidget(w)
        row2.addStretch(1)
        status.addLayout(row2)
        # Row 3: notes — stretches to the available width.
        row3 = QHBoxLayout()
        row3.setContentsMargins(0, 0, 0, 0)
        row3.addWidget(self._notes_lbl, stretch=1)
        status.addLayout(row3)
        outer.addLayout(status)

        self.setCentralWidget(central)
        self.resize(1400, 900)

    def _mk_status_label(
        self, initial_text: str, width_px: int, font,
    ) -> QLabel:
        """Build a status-strip label with a fixed pixel width and an
        Ignored horizontal size policy, so per-tick text changes don't
        pump the layout's width-hint. width_px is the maximum the label
        will need at any point in its lifetime — see _refresh_fuser_panel
        for the width-stable formatters that keep content within this.
        """
        from PyQt6.QtWidgets import QSizePolicy
        lbl = QLabel(initial_text)
        lbl.setStyleSheet("color: #ccc;")
        lbl.setFont(font)
        lbl.setFixedWidth(width_px)
        lbl.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred,
        )
        return lbl

    def _build_docks(self) -> None:
        # Cameras + vision live in the central splitter (built in
        # _build_ui); only teleop remains a dock-area panel.
        self._teleop = TeleopPanels(self.chassis, self.chassis_config)
        self._teleop.attach_to(self)

    def _build_menu(self) -> None:
        view_menu = self.menuBar().addMenu("&View")
        self._teleop_action = QAction("Teleop panels", self)
        self._teleop_action.setCheckable(True)
        self._teleop_action.setChecked(self._teleop.is_visible())
        self._teleop_action.setShortcut("Ctrl+T")
        self._teleop_action.triggered.connect(self._teleop.set_visible)
        view_menu.addAction(self._teleop_action)

        self._camera_action = QAction("Camera panels", self)
        self._camera_action.setCheckable(True)
        self._camera_action.setChecked(self._cameras.is_visible())
        self._camera_action.setShortcut("Ctrl+Shift+C")
        self._camera_action.triggered.connect(self._cameras.set_visible)
        view_menu.addAction(self._camera_action)

        view_menu.addSeparator()

        self._grid_action = QAction("Map grid (1 m)", self)
        self._grid_action.setCheckable(True)
        self._grid_action.setChecked(self._shared_view.show_grid())
        self._grid_action.setShortcut("Ctrl+G")
        self._grid_action.toggled.connect(self._shared_view.set_show_grid)
        view_menu.addAction(self._grid_action)

        self._rings_action = QAction("Range rings (1/2/5 m)", self)
        self._rings_action.setCheckable(True)
        self._rings_action.setChecked(self._shared_view.show_range_rings())
        self._rings_action.setShortcut("Ctrl+R")
        self._rings_action.toggled.connect(
            self._shared_view.set_show_range_rings
        )
        view_menu.addAction(self._rings_action)

    def _build_timer(self) -> None:
        period_ms = int(1000.0 / max(1.0, self.fuser_config.ui_redraw_hz))
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(period_ms)
        self._redraw_timer.timeout.connect(self._on_redraw_tick)
        self._redraw_timer.start()

        # Streaming-RGB timer (off by default). The toggle action
        # starts/stops it. Rate is fixed at 2 Hz for v1; cheap to
        # adjust later if testing reveals a different sweet spot.
        self._stream_rgb_hz = 2.0
        self._stream_rgb_timer = QTimer(self)
        self._stream_rgb_timer.setInterval(
            int(1000.0 / self._stream_rgb_hz)
        )
        self._stream_rgb_timer.timeout.connect(self._on_stream_rgb_tick)
        # Not started — _on_toggle_stream_rgb starts it on demand.

    # ── Tick ─────────────────────────────────────────────────────────

    def _on_redraw_tick(self) -> None:
        self._safety_toolbar.refresh()
        self._refresh_fuser_panel()
        self._refresh_chassis_panel()
        # Dock groups only consume ticks when visible. Each group owns
        # its own snapshot builder so nav doesn't have to know which
        # state fields either group reads.
        if self._teleop.is_visible():
            self._teleop.update_state(
                build_chassis_snapshot(self.chassis), time.time(),
            )
        if self._cameras.is_visible():
            self._cameras.update_state(build_camera_snapshot(self.chassis))
        # Keep View menu checkmarks in sync if the user closed a dock
        # via its titlebar X rather than via the menu action.
        if self._teleop_action.isChecked() != self._teleop.is_visible():
            self._teleop_action.setChecked(self._teleop.is_visible())
        if self._camera_action.isChecked() != self._cameras.is_visible():
            self._camera_action.setChecked(self._cameras.is_visible())

    def _refresh_fuser_panel(self) -> None:
        snap = self.fuser.snapshot_for_ui()
        latest = self.fuser.pose_source.latest_pose()
        pose = latest[0] if latest is not None else None
        trail = self.fuser.pose_trail()
        # Pull status_summary up here so pose_age is available for the
        # mission tick. ages["odom"] is local-arrival-time age — what
        # we actually want for freshness (Pi clock skew doesn't matter).
        st = self.fuser.status_summary()
        ages = st["ages"] or {}
        pose_age: Optional[float] = ages.get("odom")
        ts = time.time()
        cm = None
        if snap is not None:
            self._height_view.update_map(
                snap["grid"], snap["meta"], ts, pose=pose,
                pose_history=trail, bounds_ij=snap.get("bounds_ij"),
            )
            self._drive_view.update_map(
                snap["driveable"], snap["meta"], ts, pose=pose,
                pose_history=trail, bounds_ij=snap.get("bounds_ij"),
            )
            try:
                cm = build_costmap(snap, self._costmap_config)
            except Exception:
                logger.exception("costmap build failed; skipping panel update")
                cm = None
            self._costmap_view.update_map(
                cm, snap["meta"], ts, pose=pose,
                pose_history=trail, bounds_ij=snap.get("bounds_ij"),
            )
            # Cache for replanning when goal changes; if a goal is
            # already set, replan against the freshly-built costmap
            # so the path keeps up as the map fills in. Skip during
            # RECOVERING — the active primitive owns cmd_vel and we
            # don't want a stale path replaced under it.
            self._last_costmap = cm
            if (
                cm is not None
                and self._shared_view.goal() is not None
                and not self._mission.is_recovering()
            ):
                self._replan(cm, pose)
        else:
            self._height_view.update_map(
                None, None, 0.0, pose=pose,
                pose_history=trail, bounds_ij=None,
            )
            self._drive_view.update_map(
                None, None, 0.0, pose=pose,
                pose_history=trail, bounds_ij=None,
            )
            self._costmap_view.update_map(
                None, None, 0.0, pose=pose,
                pose_history=trail, bounds_ij=None,
            )

        # Run the follower whenever a path exists and we have a
        # live pose. The output renders on the map either way; in
        # FOLLOWING state it also drives the chassis.
        path = self._shared_view.planned_path()
        out = self._follower.update(path, pose)
        self._last_follower = out
        self._shared_view.set_lookahead(out.lookahead_world)

        # cmd_vel decision: hard gates → pose freshness → state dispatch.
        # Single helper so the per-state logic is readable.
        if self._mission.is_active():
            self._drive_mission_tick(out, cm, pose, pose_age)
        else:
            self._safety_blocked = False
            self._active_recovery = None

        # All status labels below use width-stable formats: every
        # numeric field has a fixed min-width via `:>N.Mf` / `:>Nd` so
        # 3.20 and 12.34 render the same width, and "—" placeholders
        # are padded to the same. This is what stops the bottom block
        # from pumping the window's preferred width every redraw.
        if pose is not None:
            self._pose_lbl.setText(
                f"pose: x={pose[0]:>+6.2f} y={pose[1]:>+6.2f} "
                f"θ={math.degrees(pose[2]):>+6.1f}°"
            )
        else:
            self._pose_lbl.setText("pose: (no odom)")

        rates = st["rates"]
        ages = st["ages"]

        def _hz(v: Optional[float]) -> str:
            return f"{v:>4.1f}" if v is not None else "  — "

        def _age(v: Optional[float]) -> str:
            return f"{v:>4.2f}" if v is not None else "  — "

        self._rates_lbl.setText(
            f"rates: lm {_hz(rates.get('local_map'))}Hz/"
            f"{_age(ages.get('local_map'))}s  "
            f"od {_hz(rates.get('odom'))}Hz/{_age(ages.get('odom'))}s"
        )

        self._cells_lbl.setText(
            f"cells: obs={st['cells_observed']:>5d} "
            f"trav={st['cells_traversed']:>5d}"
        )

        # SLAM health: pose-unavailable streak (≥10 = sticky note set)
        # plus cumulative scan-match correction since session reset.
        unavail = int(st.get("pose_unavail_streak") or 0)
        corr = st.get("correction_summary") or {}
        corr_m = float(corr.get("total_m") or 0.0)
        corr_deg = math.degrees(float(corr.get("total_rad") or 0.0))
        n_corr = int(corr.get("n_applied") or 0)
        self._slam_lbl.setText(
            f"slam: lost={unavail:>2d}  "
            f"drift={corr_m:>5.2f}m/{corr_deg:>+5.0f}°  "
            f"n={n_corr:>3d}"
        )
        if unavail >= 10:
            self._slam_lbl.setStyleSheet("color: #e8a;")
        elif unavail > 0:
            self._slam_lbl.setStyleSheet("color: #ec8;")
        else:
            self._slam_lbl.setStyleSheet("color: #ccc;")

        # Plan status: "—" with no goal; compact one-shot otherwise.
        goal = self._shared_view.goal()
        if goal is None:
            self._plan_lbl.setText("plan: —")
            self._plan_lbl.setStyleSheet("color: #ccc;")
        else:
            plan = self._last_plan
            if plan is None:
                self._plan_lbl.setText("plan: pending")
                self._plan_lbl.setStyleSheet("color: #cc8;")
            elif plan.ok:
                self._plan_lbl.setText(
                    f"plan: {plan.distance_m:>5.2f}m / "
                    f"{plan.elapsed_ms:>4.0f}ms"
                )
                self._plan_lbl.setStyleSheet("color: #8cf;")
            else:
                # Truncate failure msg so the label width never bursts.
                self._plan_lbl.setText(f"plan: {plan.msg[:14]}")
                self._plan_lbl.setStyleSheet("color: #e8a;")

        # Follow / mission status — kept compact so width never bursts.
        f = self._last_follower
        ms = self._mission.state
        active = self._mission.is_active()
        max_att = self._mission_config.max_recovery_attempts

        def _short_reason(r: str, n: int = 18) -> str:
            r2 = r[len("no_path:"):] if r.startswith("no_path:") else r
            return r2[:n]

        if ms == MissionState.ARRIVED:
            self._follow_lbl.setText(
                f"follow: ARRIVED  goal={f.distance_to_goal_m:>5.2f}m"
                if f is not None else "follow: ARRIVED"
            )
            self._follow_lbl.setStyleSheet("color: #8f8;")
        elif ms == MissionState.CANCELED:
            self._follow_lbl.setText("follow: canceled")
            self._follow_lbl.setStyleSheet("color: #cc8;")
        elif ms == MissionState.FAILED:
            self._follow_lbl.setText(
                f"follow: FAILED  {self._mission.failure_reason[:24]}"
            )
            self._follow_lbl.setStyleSheet("color: #e8a;")
        elif ms == MissionState.PAUSED:
            grace = self._mission_config.pause_grace_s
            elapsed = max(0.0, time.time() - self._mission.pause_started_at)
            grace_left = max(0.0, grace - elapsed)
            self._follow_lbl.setText(
                f"follow: PAUSED  "
                f"{_short_reason(self._mission.pause_reason)}  "
                f"{grace_left:>3.1f}s  "
                f"{self._mission.recovery_attempts}/{max_att}"
            )
            self._follow_lbl.setStyleSheet("color: #ec8;")
        elif ms == MissionState.RECOVERING:
            self._follow_lbl.setText(
                f"follow: REC  {self._mission.recovery_action[:20]}  "
                f"{self._mission.recovery_attempts}/{max_att}"
            )
            self._follow_lbl.setStyleSheet("color: #ec8;")
        elif f is None or f.status == STATUS_NO_PATH:
            self._follow_lbl.setText("follow: —")
            self._follow_lbl.setStyleSheet("color: #ccc;")
        elif active and self._safety_blocked:
            self._follow_lbl.setText(
                f"follow: GO BLOCKED  goal={f.distance_to_goal_m:>5.2f}m"
            )
            self._follow_lbl.setStyleSheet("color: #e8a;")
        elif f.status == STATUS_ROTATING:
            tag = "GO " if active else "dry"
            self._follow_lbl.setText(
                f"follow: {tag} ROT  "
                f"α={math.degrees(f.heading_error_rad):>+4.0f}°  "
                f"goal={f.distance_to_goal_m:>5.2f}m"
            )
            self._follow_lbl.setStyleSheet("color: #ec8;")
        else:  # FOLLOWING (follower's view)
            tag = "GO " if active else "dry"
            self._follow_lbl.setText(
                f"follow: {tag}  "
                f"v={f.v_mps:>4.2f} ω={f.omega_radps:>+5.2f}  "
                f"goal={f.distance_to_goal_m:>5.2f}m"
            )
            self._follow_lbl.setStyleSheet(
                "color: #8f8;" if active else "color: #8cf;"
            )

        # Go/Stop button enable mirrors mission state.
        self._go_act.setEnabled(self._mission.can_start())
        self._cancel_act.setEnabled(self._mission.can_cancel())

        # Session id is fixed-8; pose-source label can be "odom" or
        # "imu+scan_match" — clamp so the label width is bounded.
        src = st.get("pose_source") or "—"
        if len(src) > 14:
            src = src[:13] + "…"
        self._session_lbl.setText(
            f"session: {st['session_id'][:8]} ({src})"
        )
        self._notes_lbl.setText(st.get("notes") or "")

    # ── Mission tick ────────────────────────────────────────────────

    def _drive_mission_tick(
        self,
        out: FollowerOutput,
        cm,
        pose,
        pose_age: Optional[float],
    ) -> None:
        """Decide cmd_vel for this tick. Called only when the mission
        is in an active state (FOLLOWING / PAUSED / RECOVERING).

        Order of concerns:
            1. Hard gates (chassis disconnect, Live cmd dropped) — both
               are terminal; recovery doesn't help.
            2. Pose freshness — pause if stale, resume if fresh after
               a no_pose pause. Stale pose is treated as universal:
               applies in any active state and overrides the others.
            3. Per-state behavior.
        """
        with self.chassis.state.lock:
            connected = self.chassis.state.connected
            live = self.chassis.state.live_command
        if not connected:
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._mission.fail("chassis disconnect")
            self._cancel_recovery()
            self._safety_blocked = False
            return
        if not live:
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._mission.fail("Live cmd dropped")
            self._cancel_recovery()
            self._safety_blocked = False
            return

        # Pose freshness applies regardless of current state. None pose
        # is treated as max-stale.
        threshold = self._mission_config.pose_age_threshold_s
        stale = pose is None or (pose_age is not None and pose_age > threshold)
        if stale:
            self._cancel_recovery()
            # pause() is idempotent for the same reason — won't reset
            # the pause clock if we're already in no_pose pause.
            self._mission.pause(REASON_NO_POSE)
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._safety_blocked = False
            return
        if (
            self._mission.is_paused()
            and self._mission.pause_reason == REASON_NO_POSE
        ):
            self._mission.resume()

        # Dispatch on state.
        if self._mission.is_following():
            self._tick_following(out, pose)
        elif self._mission.is_paused():
            self._tick_paused(cm, pose)
        elif self._mission.is_recovering():
            self._tick_recovering(pose, cm)

    def _tick_following(self, out: FollowerOutput, pose) -> None:
        if out.status == STATUS_ARRIVED:
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._mission.arrive()
            self._safety_blocked = False
            return
        if out.status == STATUS_NO_PATH:
            # Replan failed (or path degenerate). Classify the failure
            # so the policy can pick an appropriate recovery action.
            reason = classify_replan_failure(
                self._last_costmap, pose, self._shared_view.goal(),
            )
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._mission.pause(reason)
            self._safety_blocked = False
            return
        # FOLLOWING / ROTATING — forward-arc safety check.
        self._safety_blocked = (
            self._last_costmap is not None
            and forward_arc_blocked(
                self._last_costmap, pose, self._safety_config,
            )
        )
        if self._safety_blocked:
            self.chassis.set_cmd_vel(0.0, 0.0)
        else:
            self.chassis.set_cmd_vel(out.v_mps, out.omega_radps)

    def _tick_paused(self, cm, pose) -> None:
        """While paused: hold cmd_vel zero, watch for the pause
        condition to clear, and on grace-expiry escalate to recovery
        (or fail if the policy is exhausted).
        """
        self.chassis.set_cmd_vel(0.0, 0.0)
        self._safety_blocked = False

        # Auto-resume if the planner now has a path. Last tick's
        # follower output already saw the freshly-replanned path; we
        # consult its status to know whether the no_path condition has
        # cleared.
        if self._last_follower is not None:
            follower_status = self._last_follower.status
            if (
                self._mission.pause_reason.startswith("no_path:")
                and follower_status not in (STATUS_NO_PATH,)
            ):
                self._mission.resume()
                return

        # Grace window — give the world a moment before swinging at it.
        elapsed = max(0.0, time.time() - self._mission.pause_started_at)
        if elapsed < self._mission_config.pause_grace_s:
            return

        # Escalate. Reasons that recovery can't address (currently just
        # NO_POSE — handled above) shouldn't reach here. Anything else
        # goes through the policy.
        action = self._recovery_policy.select(
            reason=self._mission.pause_reason,
            attempts=self._mission.recovery_attempts,
            max_attempts=self._mission_config.max_recovery_attempts,
        )
        if action is None:
            self._mission.fail(
                f"recovery exhausted ({self._mission.recovery_attempts} "
                f"attempts; last reason: {self._mission.pause_reason})"
            )
            return
        self._active_recovery = action
        self._mission.begin_recovery(action.name())

    def _tick_recovering(self, pose, cm) -> None:
        action = self._active_recovery
        if action is None:
            # State drift — recover by forcing back to PAUSED so the
            # next tick selects fresh.
            self._mission.end_recovery(success=False)
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._safety_blocked = False
            return
        out = action.update(pose, cm)
        self._safety_blocked = False
        if out.status == PRIM_RUNNING:
            self.chassis.set_cmd_vel(out.v_mps, out.omega_radps)
            return
        # Primitive finished one way or another. Drop cmd_vel, clear the
        # active handle, and notify the mission.
        self.chassis.set_cmd_vel(0.0, 0.0)
        self._active_recovery = None
        self._mission.end_recovery(success=(out.status == PRIM_DONE))

    def _cancel_recovery(self) -> None:
        if self._active_recovery is not None:
            try:
                self._active_recovery.cancel()
            except Exception:
                logger.exception("recovery primitive cancel raised")
            self._active_recovery = None

    def _refresh_chassis_panel(self) -> None:
        """Text summary with values the pills can't convey (status age,
        heartbeat seq). Gate colors live on the safety toolbar.

        Width-stable formatters: status age is `{:>4.1f}s` so 0.1 and
        12.3 share width; hb seq is rendered as the last 4 digits so it
        doesn't grow without bound across a long session.
        """
        s = self.chassis.state
        with s.lock:
            connected = s.connected
            status_ts = s.status_ts
            hb_seq = s.heartbeat_seq
        if not connected:
            self._chassis_lbl.setText("chassis: disconnected")
            self._chassis_lbl.setStyleSheet("color: #a88;")
            return
        if status_ts > 0:
            age_s = f"{time.time() - status_ts:>4.1f}"
        else:
            age_s = "  — "
        self._chassis_lbl.setText(
            f"chassis: {age_s}s  #{hb_seq % 10000:04d}"
        )
        self._chassis_lbl.setStyleSheet("color: #ccc;")

    # ── Planning ─────────────────────────────────────────────────────

    def _on_goal_requested(self, x_w: float, y_w: float) -> None:
        """Right-click in any map view → set goal here, plan now."""
        self._shared_view.set_goal((x_w, y_w))
        cm = self._last_costmap
        latest = self.fuser.pose_source.latest_pose()
        pose = latest[0] if latest is not None else None
        if cm is None or pose is None:
            self._shared_view.set_planned_path([])
            self._notes_lbl.setText(
                "goal set; waiting for map + pose before planning"
            )
            return
        self._replan(cm, pose)

    def _replan(self, costmap, pose) -> None:
        goal = self._shared_view.goal()
        if goal is None or costmap is None or pose is None:
            return
        result = plan_path(
            costmap,
            start_world=(pose[0], pose[1]),
            goal_world=goal,
            config=self._astar_config,
        )
        self._last_plan = result
        if result.ok:
            self._shared_view.set_planned_path(result.waypoints_world)
        else:
            self._shared_view.set_planned_path([])

    def _on_clear_goal(self) -> None:
        # Cancel any active mission first so we don't keep driving
        # toward a goal the operator just cleared.
        if self._mission.is_active():
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._mission.cancel()
        self._cancel_recovery()
        self._mission.reset()
        self._shared_view.set_goal(None)
        self._last_plan = None

    def _on_go(self) -> None:
        """Validate preconditions and transition the mission to
        FOLLOWING. Each subsequent redraw tick pushes the follower's
        cmd_vel to chassis until ARRIVED, CANCELED, or FAILED."""
        if not self._mission.can_start():
            return  # already FOLLOWING — nothing to do
        plan = self._last_plan
        if plan is None or not plan.ok:
            self._mission.fail("no plan — drop a goal pin first")
            return
        latest = self.fuser.pose_source.latest_pose()
        if latest is None:
            self._mission.fail("no pose — wait for odom before starting")
            return
        with self.chassis.state.lock:
            connected = self.chassis.state.connected
            live = self.chassis.state.live_command
        if not connected:
            self._mission.fail("chassis disconnected")
            return
        if not live:
            self._mission.fail(
                "Live cmd is OFF — enable it on the safety toolbar first"
            )
            return
        self._mission.start()

    def _on_cancel(self) -> None:
        """Operator-initiated stop. Zero cmd_vel and transition to
        CANCELED. Live cmd is left ON so the operator can drive
        manually without a second click."""
        if self._mission.is_active():
            self.chassis.set_cmd_vel(0.0, 0.0)
        self._cancel_recovery()
        self._mission.cancel()

    # ── Toolbar handlers ─────────────────────────────────────────────

    def _on_load_snapshot(self) -> None:
        """Pick a layers.npz via file dialog and restore it into the
        live world grid. Cancelling any active mission first so we
        don't drive against a freshly-replaced map."""
        # Default to the conventional sessions directory; fall back
        # to home if it doesn't exist yet.
        default_dir = os.path.expanduser("~/Body/sessions")
        if not os.path.isdir(default_dir):
            default_dir = os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load snapshot — pick a layers.npz",
            default_dir,
            "World snapshots (layers.npz)",
        )
        if not path:
            return
        # Cancel a running mission so we don't drive on stale
        # follower state with a freshly-replaced grid underneath.
        if self._mission.is_active():
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._mission.cancel()
        self._cancel_recovery()
        try:
            summary = self.fuser.load_snapshot(path)
        except Exception as e:
            logger.exception("load_snapshot failed")
            QMessageBox.warning(
                self, "Load snapshot failed",
                f"Could not load snapshot:\n{type(e).__name__}: {e}",
            )
            return
        QMessageBox.information(
            self, "Snapshot loaded",
            f"Loaded {summary['cells_observed']} cells from\n"
            f"{path}\n\n"
            f"loaded session: {summary['loaded_session_id'][:8]}\n"
            f"live session:   {summary['current_session_id'][:8]}"
        )

    def _on_save_snapshot(self) -> None:
        try:
            out_dir = self.fuser.save_snapshot_bundle()
        except Exception as e:
            logger.exception("snapshot bundle write failed")
            QMessageBox.warning(
                self, "Snapshot failed",
                f"Could not write snapshot bundle:\n{type(e).__name__}: {e}",
            )
            return
        # Non-modal toast; let the operator click through to the
        # directory if they want to inspect it.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Snapshot saved")
        box.setText(f"Snapshot bundle written:\n{out_dir}")
        open_btn = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))

    def _on_fit_maps(self) -> None:
        # Single call: shared view propagates to all attached panels.
        self._shared_view.reset_view()

    def _on_toggle_stream_rgb(self, checked: bool) -> None:
        if checked:
            self._stream_rgb_timer.start()
        else:
            self._stream_rgb_timer.stop()

    def _on_stream_rgb_tick(self) -> None:
        # Streaming-mode capture: in-flight gating in the controller
        # keeps a slow Pi from accruing a backlog. If the chassis is
        # disconnected, request_rgb_streaming() returns None and the
        # tick is a no-op.
        self.chassis.request_rgb_streaming()

    # ── Lifecycle ────────────────────────────────────────────────────

    def _balance_splitter_once(self) -> None:
        # Applied via QTimer.singleShot(0, …) from showEvent: by the
        # time this fires, the splitters have real sizes from the
        # first layout pass. Idempotent so spurious re-fires don't
        # stomp on a user's drag.
        if self._splitter_balanced:
            return
        h_total = self._h_splitter.width()
        v_total = self._left_splitter.height()
        if h_total <= 0 or v_total <= 0:
            return
        # Vision column defaults to ~320 px; left column takes the
        # rest. Narrow enough to keep the images wide, wide enough
        # for a readable chat.
        vision_w = min(360, max(260, h_total // 5))
        self._h_splitter.setSizes([h_total - vision_w, vision_w])
        # Maps ≈ feeds in the left column.
        self._left_splitter.setSizes([v_total // 2, v_total // 2])
        self._splitter_balanced = True

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._splitter_balanced:
            QTimer.singleShot(0, self._balance_splitter_once)

    def closeEvent(self, event) -> None:
        try:
            self._redraw_timer.stop()
        except Exception:
            pass
        super().closeEvent(event)


def run_app(
    fuser: FuserController,
    fuser_config: FuserConfig,
    chassis: StubController,
    chassis_config: StubConfig,
) -> int:
    app = QApplication.instance() or QApplication([])
    win = NavMainWindow(fuser, fuser_config, chassis, chassis_config)
    win.show()
    return app.exec()

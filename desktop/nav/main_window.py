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
import shutil
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QSplitter, QToolBar, QVBoxLayout, QWidget,
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
from .health import LivenessWatcher
from .mission import Mission, MissionConfig, MissionState
from . import patrol as patrol_mod
from .patrol import Patrol, PatrolRunner
from .patrol_panel import PatrolDock
from .planner import AStarConfig, PlanResult, plan_path
from .primitives import RotateToHeading
from .recovery import (
    PRIM_ABORTED, PRIM_DONE, PRIM_RUNNING,
    REASON_NO_LIVE_CMD, REASON_NO_POSE, RecoveryPolicy, RecoveryPrimitive,
    classify_replan_failure,
)
from .safety import SafetyConfig, forward_arc_blocked_local, rear_arc_blocked_local
from .tracing import (
    CAT_PLAN, CAT_SAFETY, LEVEL_WARN, Tracer, git_sha,
)

from .camera_panels import CameraPanels, build_camera_snapshot
from .safety_toolbar import SafetyToolbar
from .teleop_panels import TeleopPanels, build_chassis_snapshot

logger = logging.getLogger(__name__)


# Forward-arc block must persist this long before an auto-snapshot
# fires. Transient cross-overs (a person walking past) shouldn't burn
# disk; a real "stuck looking at an obstacle" should.
_SUSTAINED_BLOCK_S = 3.0

# Patrol waypoint adaptation: when a saved waypoint sits in a lethal
# cell or deep in inflation halo, snap to the nearest accessible cell
# within this radius. Generous enough to escape inflation halo,
# tight enough that a wall-buried wp isn't relocated halfway across
# the room.
_WP_SNAP_RADIUS_M = 1.0

# Below this displacement, the snap is considered "no relocation" and
# no trace event is emitted (avoids per-mission spam when the wp is
# already in clear space and just rounds to a different sub-cell).
_WP_SNAP_TRIVIAL_M = 0.05


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

        relocate_act = QAction("Re-localize", self)
        relocate_act.setToolTip(
            "Snap the robot pose to a wide global scan-match against "
            "the current world map. Use after the steady-state matcher "
            "has diverged (huge SLAM drift); the map is kept."
        )
        relocate_act.triggered.connect(self._on_relocate)
        self._map_toolbar.addAction(relocate_act)

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

        self._patrol_edit_act = QAction("Patrol edit", self)
        self._patrol_edit_act.setCheckable(True)
        self._patrol_edit_act.setChecked(False)
        self._patrol_edit_act.setToolTip(
            "Toggle right-click semantics: while on, right-clicking a "
            "map appends a waypoint to the active patrol (creating one "
            "if none is loaded). While off, right-click sets the single "
            "goal as before."
        )
        self._patrol_edit_act.toggled.connect(self._on_patrol_edit_action)
        self._map_toolbar.addAction(self._patrol_edit_act)

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

        # Vertical stack: [ maps | feeds | vision ]. Pulling vision out
        # of a right-hand column lets the patrol dock + maps fit on a
        # 1440-wide laptop screen without horizontal overflow. Each row
        # is independently collapsible via splitter handles.
        self._v_splitter = QSplitter(Qt.Orientation.Vertical, central)
        self._v_splitter.setChildrenCollapsible(True)

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
        # Tracing: one JSONL file per mission, edge-triggered emits.
        # Pose sampler + auto-snapshot callback are attached here so
        # the Mission and LivenessWatcher can use them as soon as the
        # first event fires. See `tracing.py` and `health.py`.
        self._tracer = Tracer()
        self._tracer.attach_pose_sampler(self._sample_pose_for_trace)
        self._tracer.attach_snapshot_cb(self._auto_snapshot_for_trace)
        self._mission.tracer = self._tracer
        self._liveness = LivenessWatcher(
            self._tracer, fuser=self.fuser, chassis=self.chassis,
        )
        # Edge-trigger state. None = uninitialized (first observation
        # establishes baseline without emitting).
        self._last_plan_ok: Optional[bool] = None
        self._last_safety_blocked: Optional[bool] = None
        self._safety_block_started_at: Optional[float] = None
        self._safety_block_snapped: bool = False
        self._mission_was_active: bool = False
        # Patrol execution state. Populated by `_on_go` when a patrol
        # with waypoints is loaded; None for single-goal missions.
        # `_active_rotation` is the currently-running RotateToHeading
        # primitive (only set while ROTATING_TO_NEXT). `_pending_advance`
        # holds (new_wp_index, new_lap_index, lap_completed) — the
        # values committed to the mission once the primitive reports
        # DONE.
        self._patrol_runner: Optional[PatrolRunner] = None
        self._active_rotation: Optional[RotateToHeading] = None
        self._pending_advance: Optional[Tuple[int, int, bool]] = None
        # Per-mission cache of effective (snapped) waypoint coords,
        # keyed by wp_index. Each entry is computed once per mission
        # when the wp first becomes the active target, emits a
        # `patrol.waypoint_snapped` event if relocation was needed,
        # then is reused for the leg's goal + the rotate-to-face
        # heading. Cleared on mission terminal.
        self._snapped_wp_xys: Dict[int, Tuple[float, float]] = {}
        # Right-click append target for patrol-edit mode. Wired to
        # SharedMapView in `_build_ui` (next block).
        self._shared_view.set_patrol_append_callback(
            self._on_patrol_append_requested
        )
        self._v_splitter.addWidget(maps_widget)

        self._cameras = CameraPanels(self.chassis)
        self._v_splitter.addWidget(self._cameras.feeds_widget)
        # Vision is now a left-dock-area dock (stacks vertically with
        # Patrol) — see _build_docks. Removed from the central splitter
        # so chat doesn't compete with maps + feeds for vertical room.

        # Maps dominates; feeds is a thin strip below it.
        self._v_splitter.setStretchFactor(0, 3)
        self._v_splitter.setStretchFactor(1, 1)
        self._splitter_balanced = False

        outer.addWidget(self._v_splitter, stretch=1)

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
        self.resize(960, 880)

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
        # Camera feeds live in the central splitter (built in _build_ui).
        # Teleop, Patrol, and Vision are dock-area panels.
        self._teleop = TeleopPanels(self.chassis, self.chassis_config)
        self._teleop.attach_to(self)
        # Patrol dock: lives on the left dock area to avoid stealing
        # space from the teleop column on the right.
        self._patrol_dock = PatrolDock(
            self._shared_view,
            get_live_session_id=lambda: self.fuser.grid.session_id,
            parent=self,
        )
        self._patrol_dock.attach_to(self)
        self._patrol_dock.edit_mode_toggled.connect(self._on_patrol_edit_toggled)
        # Vision dock: same left dock area as Patrol so they share the
        # vertical column. Qt stacks dock widgets added to the same
        # area; user can drag handles to rebalance. Visible by default
        # (parity with the previous central-splitter placement).
        self._vision_dock = QDockWidget("Vision", self)
        self._vision_dock.setWidget(self._cameras.vision_widget)
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self._vision_dock,
        )

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

        self._patrol_action = QAction("Patrol panel", self)
        self._patrol_action.setCheckable(True)
        self._patrol_action.setChecked(self._patrol_dock.is_visible())
        self._patrol_action.setShortcut("Ctrl+P")
        self._patrol_action.triggered.connect(self._patrol_dock.set_visible)
        view_menu.addAction(self._patrol_action)

        self._vision_action = QAction("Vision panel", self)
        self._vision_action.setCheckable(True)
        self._vision_action.setChecked(self._vision_dock.isVisible())
        self._vision_action.setShortcut("Ctrl+Shift+V")
        self._vision_action.triggered.connect(self._vision_dock.setVisible)
        view_menu.addAction(self._vision_action)

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
        # Liveness watcher self-throttles to its own 1 Hz cadence;
        # cheap to call on every redraw.
        self._liveness.tick()
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
            cam_snap = build_camera_snapshot(self.chassis)
            cam_snap["streaming_on"] = self._stream_rgb_act.isChecked()
            self._cameras.update_state(cam_snap)
        # Keep View menu checkmarks in sync if the user closed a dock
        # via its titlebar X rather than via the menu action.
        if self._teleop_action.isChecked() != self._teleop.is_visible():
            self._teleop_action.setChecked(self._teleop.is_visible())
        if self._camera_action.isChecked() != self._cameras.is_visible():
            self._camera_action.setChecked(self._cameras.is_visible())
        if self._vision_action.isChecked() != self._vision_dock.isVisible():
            self._vision_action.setChecked(self._vision_dock.isVisible())
        if self._patrol_action.isChecked() != self._patrol_dock.is_visible():
            self._patrol_action.setChecked(self._patrol_dock.is_visible())
        # Sync the patrol dock's widgets with the current shared-view
        # state (session-match hint, waypoint count). Cheap.
        if self._patrol_dock.is_visible():
            self._patrol_dock.refresh()

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
                cm = build_costmap(snap, self._costmap_config, pose=pose)
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
            self._update_safety_block_trace(False)
            self._active_recovery = None
        # Trace lifecycle: close the per-mission file on the
        # active→terminal edge so each mission yields one self-
        # contained JSONL artifact.
        if self._mission_was_active and not self._mission.is_active():
            self._tracer.close()
            self._mission_was_active = False
            # Reset edge state so a fresh mission's first events
            # (re)establish baselines silently.
            self._last_plan_ok = None
            self._last_safety_blocked = None
            self._safety_block_started_at = None
            self._safety_block_snapped = False
            # Patrol bookkeeping: drop runner / pending advance so a
            # subsequent Go on the same patrol starts at wp[0] again,
            # clear the snap cache (next mission rebuilds against the
            # current costmap), and unlock the patrol dock for edits.
            self._patrol_runner = None
            self._active_rotation = None
            self._pending_advance = None
            self._snapped_wp_xys.clear()
            self._patrol_dock.set_mission_active(False)
        elif self._mission.is_active():
            self._mission_was_active = True

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
            1. Chassis disconnect — terminal; recovery doesn't help.
            2. Live cmd flag — pause if dropped, resume if restored,
               fail on short timeout. Operator ALL-STOP / Live toggle /
               chassis reconnect all drop the flag; the timeout still
               ends the mission deliberately while cushioning transient
               flag races. cmd_loop stops publishing while live=False,
               so the Pi watchdog halts motors during the pause window
               regardless.
            3. Pose freshness — pause if stale, resume if fresh after
               a no_pose pause. Stale pose is treated as universal:
               applies in any active state and overrides the others.
            4. Per-state behavior.
        """
        with self.chassis.state.lock:
            connected = self.chassis.state.connected
            live = self.chassis.state.live_command
        if not connected:
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._mission.fail("chassis disconnect")
            self._cancel_recovery()
            self._safety_blocked = False
            self._update_safety_block_trace(False)
            return
        if not live:
            self._cancel_recovery()
            # pause() is idempotent for the same reason — won't reset
            # the pause clock if we're already in no_live_cmd pause.
            self._mission.pause(REASON_NO_LIVE_CMD)
            if (
                self._mission.is_paused()
                and self._mission.pause_reason == REASON_NO_LIVE_CMD
            ):
                elapsed = time.time() - self._mission.pause_started_at
                if elapsed > self._mission_config.no_live_cmd_timeout_s:
                    self._mission.fail(
                        f"live cmd lost for {elapsed:.0f}s "
                        f"(threshold {self._mission_config.no_live_cmd_timeout_s:.0f}s)"
                    )
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._safety_blocked = False
            self._update_safety_block_trace(False)
            return
        if (
            self._mission.is_paused()
            and self._mission.pause_reason == REASON_NO_LIVE_CMD
        ):
            self._mission.resume()

        # Pose freshness applies regardless of current state. None pose
        # is treated as max-stale.
        threshold = self._mission_config.pose_age_threshold_s
        stale = pose is None or (pose_age is not None and pose_age > threshold)
        if stale:
            self._cancel_recovery()
            # pause() is idempotent for the same reason — won't reset
            # the pause clock if we're already in no_pose pause.
            self._mission.pause(REASON_NO_POSE)
            # Bail out of the wait if pose has been gone too long. This
            # is the hard escape from PAUSED("no_pose") — the recovery
            # policy doesn't fire for no_pose (no primitive helps a
            # missing pose), so without this the mission would idle in
            # PAUSED forever on a dead Pi-side local_map publisher.
            if self._mission.is_paused() and self._mission.pause_reason == REASON_NO_POSE:
                elapsed = time.time() - self._mission.pause_started_at
                if elapsed > self._mission_config.no_pose_timeout_s:
                    self._mission.fail(
                        f"pose lost for {elapsed:.0f}s "
                        f"(threshold {self._mission_config.no_pose_timeout_s:.0f}s)"
                    )
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._safety_blocked = False
            self._update_safety_block_trace(False)
            return
        if (
            self._mission.is_paused()
            and self._mission.pause_reason == REASON_NO_POSE
        ):
            self._mission.resume()

        # Dispatch on state.
        if self._mission.is_following():
            self._tick_following(out, pose)
        elif self._mission.is_rotating_to_next():
            self._tick_rotating_to_next(pose, cm)
        elif self._mission.is_paused():
            self._tick_paused(cm, pose)
        elif self._mission.is_recovering():
            self._tick_recovering(pose, cm)

    def _tick_following(self, out: FollowerOutput, pose) -> None:
        if out.status == STATUS_ARRIVED:
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._update_safety_block_trace(False)
            # Patrol arrival branch: advance to the next waypoint
            # (rotating to face it first) instead of terminating, when
            # there's another leg to drive.
            if self._patrol_runner is not None:
                self._handle_patrol_arrival(pose)
                return
            self._mission.arrive()
            return
        if out.status == STATUS_NO_PATH:
            # Replan failed (or path degenerate). Classify the failure
            # so the policy can pick an appropriate recovery action.
            reason = classify_replan_failure(
                self._last_costmap, pose, self._shared_view.goal(),
            )
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._mission.pause(reason)
            self._update_safety_block_trace(False)
            return
        # FOLLOWING / ROTATING — forward / rear arc safety check.
        # Reads the body-frame local_map.driveable directly (the freshest
        # fused lidar+depth observation from the Pi), not the world-frame
        # costmap. Drift-immune: a pose error doesn't shift our view of
        # what's physically in front of the robot.
        #
        # Staleness rule: if local_map is missing or older than 2× its
        # median publish period (fallback 1.0 s), treat as BLOCKED. We
        # would rather refuse to drive than drive blind on stale data.
        with self.chassis.state.lock:
            lm_drive = self.chassis.state.local_map_driveable
            lm_meta = self.chassis.state.local_map_meta
            lm_ts = self.chassis.state.local_map_ts
        lm_period = self.chassis.state.local_map_period_s() or 0.5
        lm_stale_threshold_s = max(1.0, 2.0 * lm_period)
        lm_age_s = time.time() - lm_ts if lm_ts > 0 else float("inf")
        if lm_drive is None or lm_meta is None or lm_age_s > lm_stale_threshold_s:
            fwd_blocked = True
            rear_blocked = True
        else:
            fwd_blocked = forward_arc_blocked_local(
                lm_drive, lm_meta, self._safety_config,
            )
            rear_blocked = rear_arc_blocked_local(
                lm_drive, lm_meta, self._safety_config,
            )
        # Clip the *commanded translation* by the direction-appropriate
        # arc, but always pass ω through. Rotation in place doesn't
        # advance the body, so an obstacle in the forward arc must not
        # prevent the bot from rotating to face a clear direction —
        # rotation is precisely how it escapes. Zeroing ω here was a
        # deadlock: facing a bookshelf, forward arc fires every tick,
        # bot stuck unable to turn away. Direction-appropriate clip
        # keeps the safety semantics (don't translate into obstacles)
        # without making rotation impossible.
        v_cmd = out.v_mps
        omega_cmd = out.omega_radps
        if v_cmd > 0.0 and fwd_blocked:
            v_cmd = 0.0
        elif v_cmd < 0.0 and rear_blocked:
            v_cmd = 0.0
        # `_safety_blocked` drives the GO BLOCKED status label and the
        # safety.* trace event. Edge on "the commanded forward motion
        # was clipped" — purely-rotating ticks are not "blocked" in the
        # user-facing sense, they're driving around the problem.
        blocked = (v_cmd != out.v_mps)
        self._safety_blocked = blocked
        self.chassis.set_cmd_vel(v_cmd, omega_cmd)
        self._update_safety_block_trace(blocked)

    def _update_safety_block_trace(self, blocked: bool) -> None:
        """Edge-triggered safety.* emits + sustained-block auto-snap.

        The forward arc flicks on/off as the robot's heading sweeps past
        nearby lethal cells — we want one event when an episode begins
        and one when it ends, not per-tick spam. If the same episode
        persists past `_SUSTAINED_BLOCK_S`, fire `safety.sustained_block`
        (in AUTO_SNAP_EVENTS) so the trace gets a costmap snapshot at
        the moment of trouble. Snap is rate-limited per episode — one
        bundle per stuck-stretch, not one per tick after the threshold.
        """
        prev = self._last_safety_blocked
        now = time.time()
        if prev is None:
            self._last_safety_blocked = blocked
            if blocked:
                self._safety_block_started_at = now
                self._tracer.emit(
                    CAT_SAFETY, "forward_arc_blocked", {},
                    level=LEVEL_WARN,
                )
            return
        if blocked != prev:
            self._last_safety_blocked = blocked
            if blocked:
                self._safety_block_started_at = now
                self._safety_block_snapped = False
                self._tracer.emit(
                    CAT_SAFETY, "forward_arc_blocked", {},
                    level=LEVEL_WARN,
                )
            else:
                self._safety_block_started_at = None
                self._safety_block_snapped = False
                self._tracer.emit(CAT_SAFETY, "cleared", {})
            return
        # No edge — but check sustained-block threshold while blocked.
        if (
            blocked
            and not self._safety_block_snapped
            and self._safety_block_started_at is not None
            and (now - self._safety_block_started_at) > _SUSTAINED_BLOCK_S
        ):
            self._safety_block_snapped = True
            self._tracer.emit(
                CAT_SAFETY, "sustained_block",
                {"duration_s": now - self._safety_block_started_at},
                level=LEVEL_WARN,
            )

    def _tick_paused(self, cm, pose) -> None:
        """While paused: hold cmd_vel zero, watch for the pause
        condition to clear, and on grace-expiry escalate to recovery
        (or fail if the policy is exhausted).
        """
        self.chassis.set_cmd_vel(0.0, 0.0)
        self._safety_blocked = False
        self._update_safety_block_trace(False)

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
            self._update_safety_block_trace(False)
            return
        out = action.update(pose, cm)
        self._safety_blocked = False
        self._update_safety_block_trace(False)
        if out.status == PRIM_RUNNING:
            self.chassis.set_cmd_vel(out.v_mps, out.omega_radps)
            return
        # Primitive finished one way or another. Drop cmd_vel, clear the
        # active handle, and notify the mission.
        self.chassis.set_cmd_vel(0.0, 0.0)
        self._active_recovery = None
        self._mission.end_recovery(success=(out.status == PRIM_DONE))

    # ── Patrol arrival / rotation ───────────────────────────────────

    def _get_effective_wp_xy(
        self, wp, wp_index: int,
    ) -> Tuple[float, float]:
        """Return the (x, y) the planner/follower should aim at for
        `wp`. Snaps the saved waypoint to the nearest accessible cell
        (cost < halo_max/2, non-lethal) within `_WP_SNAP_RADIUS_M` if
        the saved cell isn't itself accessible. Cached per wp_index
        for the duration of the mission so each leg gets a consistent
        target and the trace event fires at most once per wp per run.

        Emits:
          - `patrol.waypoint_snapped` when a relocation > trivial
            distance was applied.
          - `patrol.waypoint_snap_failed` when no acceptable cell
            exists within the radius (fall back to raw xy; planner
            relaxation will still try its own 8-cell search).
        """
        cached = self._snapped_wp_xys.get(wp_index)
        if cached is not None:
            return cached
        raw_xy = (float(wp.x_m), float(wp.y_m))
        cm = self._last_costmap
        if cm is None:
            # No costmap yet — can't snap. Use raw and don't cache
            # (so the next call retries once a costmap arrives).
            return raw_xy
        try:
            result = patrol_mod.snap_to_accessible(
                cm, raw_xy, radius_m=_WP_SNAP_RADIUS_M,
            )
        except Exception:
            logger.exception("snap_to_accessible raised")
            self._snapped_wp_xys[wp_index] = raw_xy
            return raw_xy
        if result is None:
            # No accessible cell within radius — fall back to raw.
            # Trace the failure so a reviewer can see *why* the
            # planner relaxation took over.
            self._snapped_wp_xys[wp_index] = raw_xy
            try:
                self._tracer.emit(
                    "patrol", "waypoint_snap_failed",
                    {
                        "wp_index": wp_index,
                        "original_xy": [raw_xy[0], raw_xy[1]],
                        "radius_m": _WP_SNAP_RADIUS_M,
                    },
                    level=LEVEL_WARN,
                )
            except Exception:
                logger.exception("waypoint_snap_failed emit raised")
            return raw_xy
        eff_xy = result.snapped_xy
        self._snapped_wp_xys[wp_index] = eff_xy
        if result.snapped and result.distance_m >= _WP_SNAP_TRIVIAL_M:
            try:
                self._tracer.emit(
                    "patrol", "waypoint_snapped",
                    {
                        "wp_index": wp_index,
                        "original_xy": [
                            result.original_xy[0], result.original_xy[1],
                        ],
                        "snapped_xy": [eff_xy[0], eff_xy[1]],
                        "distance_m": result.distance_m,
                        "cost_at_original": result.cost_at_original,
                        "cost_at_snapped": result.cost_at_snapped,
                    },
                )
            except Exception:
                logger.exception("waypoint_snapped emit raised")
        return eff_xy

    def _handle_patrol_arrival(self, pose) -> None:
        """Called from `_tick_following` when the follower reports
        ARRIVED *and* a patrol is active. Compute the next leg, decide
        terminal vs. advance, and (if advancing) kick off a
        RotateToHeading primitive aimed at the next waypoint's bearing.
        """
        runner = self._patrol_runner
        if runner is None:
            self._mission.arrive()
            return
        # `face_next` is a per-waypoint flag — if the just-reached
        # waypoint says false, skip the rotation step.
        face_next = bool(
            runner.patrol.waypoints[runner.wp_index].face_next
        )
        new_idx, lap_completed = runner.on_arrived()
        if new_idx is None:
            # Patrol terminal. If this terminal arrival also closed a
            # lap (loop=True, laps=K and K-th arrival at wp[0]), emit
            # patrol.lap_complete before mission.arrive() so the trace
            # records the closure — complete_rotation_to_next is the
            # usual emit site but we skip it on terminate-via-lap.
            if lap_completed:
                # Sync the mission's lap counter to the runner's
                # before emitting + arriving, so the trace event and
                # the mission state agree on lap_index.
                self._mission.lap_index = runner.lap_index
                try:
                    self._tracer.emit(
                        "patrol", "lap_complete",
                        {"lap_index": runner.lap_index},
                    )
                except Exception:
                    logger.exception("lap_complete emit raised")
            self._patrol_runner = None
            self._active_rotation = None
            self._pending_advance = None
            self._mission.arrive()
            return
        # Cache the values to commit when rotation completes. We don't
        # update the goal yet — the follower keeps reporting ARRIVED
        # against wp[i] until we advance, which is fine since cmd_vel
        # is owned by the rotation primitive while ROTATING_TO_NEXT.
        self._pending_advance = (new_idx, runner.lap_index, lap_completed)
        # Snap the NEW active wp to an accessible cell ONCE per
        # mission. Used both as the rotation target's heading and (in
        # _commit_pending_advance) as the goal for the next leg.
        next_wp = runner.patrol.waypoints[new_idx]
        eff_next_xy = self._get_effective_wp_xy(next_wp, new_idx)
        if not face_next or pose is None:
            # Skip rotation — commit advance immediately and resume
            # FOLLOWING in the next tick. Use 0-radian dummy target;
            # the begin/complete pair still fires patrol.advance.
            self._mission.begin_rotation_to_next(0.0, to_wp_index=new_idx)
            self._commit_pending_advance()
            return
        target_theta = math.atan2(
            eff_next_xy[1] - pose[1], eff_next_xy[0] - pose[0],
        )
        self._mission.begin_rotation_to_next(
            target_theta, to_wp_index=new_idx,
        )
        self._active_rotation = RotateToHeading(target_theta)

    def _tick_rotating_to_next(self, pose, cm) -> None:
        """RotateToHeading-driven cmd_vel until the primitive reports
        DONE (or ABORTED), then commit the pending advance and resume
        FOLLOWING."""
        action = self._active_rotation
        if action is None:
            # State drift: shouldn't happen, but recover by aborting
            # the rotation and resuming FOLLOWING with no advance.
            self._mission.abort_rotation_to_next()
            self.chassis.set_cmd_vel(0.0, 0.0)
            return
        out = action.update(pose, cm)
        self._update_safety_block_trace(False)
        if out.status == PRIM_RUNNING:
            self.chassis.set_cmd_vel(out.v_mps, out.omega_radps)
            return
        # DONE or ABORTED — drop cmd_vel and commit.
        self.chassis.set_cmd_vel(0.0, 0.0)
        self._active_rotation = None
        if out.status == PRIM_DONE:
            self._commit_pending_advance()
        else:
            # ABORTED — fall back to FOLLOWING with no advance; the
            # next ARRIVED tick will re-enter this path. cmd_vel
            # already zeroed.
            self._mission.abort_rotation_to_next()
            self._pending_advance = None

    def _commit_pending_advance(self) -> None:
        """Apply the (wp_index, lap_index, lap_completed) cached by
        `_handle_patrol_arrival` and shift the goal pin to the new
        waypoint so the planner / follower target it on the next tick.
        """
        adv = self._pending_advance
        runner = self._patrol_runner
        if adv is None or runner is None:
            self._mission.abort_rotation_to_next()
            return
        new_idx, new_lap, lap_completed = adv
        self._pending_advance = None
        self._mission.complete_rotation_to_next(
            new_wp_index=new_idx,
            new_lap_index=new_lap,
            lap_completed=lap_completed,
        )
        # Shift the goal pin to the new active waypoint's effective
        # (snapped) position. The planner will replan against this on
        # the next redraw tick. The effective xy was computed in
        # _handle_patrol_arrival and cached, so the rotation target
        # and the new goal agree exactly.
        wp = runner.patrol.waypoints[new_idx]
        eff_xy = self._get_effective_wp_xy(wp, new_idx)
        self._shared_view.set_goal(eff_xy)
        self._shared_view.set_patrol_active_wp_index(new_idx)

    def _cancel_recovery(self) -> None:
        if self._active_recovery is not None:
            try:
                self._active_recovery.cancel()
            except Exception:
                logger.exception("recovery primitive cancel raised")
            self._active_recovery = None
        # An active rotate-to-next primitive is the patrol equivalent
        # of a recovery primitive — same teardown story (cancel +
        # drop the handle; mission state transitioned by the caller).
        if self._active_rotation is not None:
            try:
                self._active_rotation.cancel()
            except Exception:
                logger.exception("rotation primitive cancel raised")
            self._active_rotation = None
            self._pending_advance = None

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
        # Edge-triggered plan tracing: emit on fail→ok and ok→fail
        # transitions only. Every-tick plan.ok would flood the file.
        # The first observed result establishes the baseline silently.
        prev_ok = self._last_plan_ok
        cur_ok = bool(result.ok)
        if prev_ok is None:
            self._last_plan_ok = cur_ok
        elif cur_ok != prev_ok:
            self._last_plan_ok = cur_ok
            if cur_ok:
                self._tracer.emit(
                    CAT_PLAN, "ok",
                    {
                        "distance_m": result.distance_m,
                        "expansions": result.n_expansions,
                        "elapsed_ms": result.elapsed_ms,
                        "goal": [goal[0], goal[1]],
                    },
                )
            else:
                self._tracer.emit(
                    CAT_PLAN, "fail",
                    {
                        "msg": result.msg,
                        "expansions": result.n_expansions,
                        "elapsed_ms": result.elapsed_ms,
                        "goal": [goal[0], goal[1]],
                    },
                    level=LEVEL_WARN,
                )

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

    def _on_relocate(self) -> None:
        # Wide global scan-match snap. Zero cmd_vel first — relocate
        # rewrites the world offset, and the follower's last cmd_vel
        # was computed against the pre-relocate pose.
        if self._mission.is_active():
            self.chassis.set_cmd_vel(0.0, 0.0)
            self._mission.cancel()
        result = self.fuser.request_relocate(reason="ui_relocate")
        if result.get("success"):
            # Phase B: apply the same SE(2) transform to the patrol's
            # waypoints + the single goal pin so they stay glued to the
            # physical environment, not the drifting world frame. The
            # world map cells are NOT transformed here (deferred); the
            # sum-bounded vote model will reconcile old votes with new
            # observations as the bot drives.
            shift = self._apply_relocate_to_patrol(result)
            self._snapped_wp_xys.clear()
            QMessageBox.information(
                self, "Re-localize",
                f"Snapped pose by "
                f"dx={result['dx']:+.2f} m, dy={result['dy']:+.2f} m, "
                f"dθ={math.degrees(result['dtheta']):+.1f}° "
                f"(improvement {result['improvement']:.0f} over "
                f"{result['evidence_cells']} evidence cells)."
                + (
                    f"\n\nShifted {shift} waypoint(s) / goal pin to "
                    f"match the new frame." if shift > 0 else ""
                ),
            )
        else:
            QMessageBox.warning(
                self, "Re-localize failed",
                f"reason: {result.get('reason', 'unknown')}\n"
                + "\n".join(
                    f"{k}: {v}" for k, v in result.items()
                    if k not in ("success", "reason")
                ),
            )

    def _apply_relocate_to_patrol(self, result: Dict[str, Any]) -> int:
        """Transform the active patrol's waypoints and the single goal
        pin by the SE(2) given by `result["prior_pose"]` +
        (dx, dy, dtheta). Saved Patrol on disk is NOT mutated — the
        operator persists via Save if they want the new coords.

        Returns the count of points transformed (waypoints + 1 for the
        goal if present).
        """
        prior = result.get("prior_pose")
        if not prior or len(prior) < 3:
            return 0
        dx = float(result.get("dx", 0.0))
        dy = float(result.get("dy", 0.0))
        dtheta = float(result.get("dtheta", 0.0))
        bot_old = (float(prior[0]), float(prior[1]))
        bot_new = (bot_old[0] + dx, bot_old[1] + dy)
        cos_d = math.cos(dtheta)
        sin_d = math.sin(dtheta)

        def transform(xy):
            rx = xy[0] - bot_old[0]
            ry = xy[1] - bot_old[1]
            return (
                bot_new[0] + cos_d * rx - sin_d * ry,
                bot_new[1] + sin_d * rx + cos_d * ry,
            )

        n_transformed = 0
        wp_changes: list = []

        patrol = self._shared_view.patrol()
        if patrol is not None:
            for i, wp in enumerate(patrol.waypoints):
                old_xy = (wp.x_m, wp.y_m)
                new_xy = transform(old_xy)
                wp.x_m = new_xy[0]
                wp.y_m = new_xy[1]
                wp_changes.append({
                    "wp_index": i,
                    "old_xy": [old_xy[0], old_xy[1]],
                    "new_xy": [new_xy[0], new_xy[1]],
                })
                n_transformed += 1
            # Re-set on shared view to fire notify so map re-renders.
            self._shared_view.set_patrol(patrol)

        goal = self._shared_view.goal()
        goal_change = None
        if goal is not None:
            new_goal = transform(goal)
            self._shared_view.set_goal(new_goal)
            goal_change = {
                "old_xy": [goal[0], goal[1]],
                "new_xy": [new_goal[0], new_goal[1]],
            }
            n_transformed += 1

        if n_transformed > 0:
            try:
                self._tracer.emit(
                    "patrol", "world_relocated",
                    {
                        "prior_pose": [
                            float(prior[0]), float(prior[1]), float(prior[2]),
                        ],
                        "dx": dx,
                        "dy": dy,
                        "dtheta": dtheta,
                        "n_waypoints": (
                            len(patrol.waypoints) if patrol is not None else 0
                        ),
                        "waypoints": wp_changes,
                        "goal": goal_change,
                    },
                )
            except Exception:
                logger.exception("world_relocated emit raised")

        return n_transformed

    def _on_go(self) -> None:
        """Validate preconditions and transition the mission to
        FOLLOWING. Each subsequent redraw tick pushes the follower's
        cmd_vel to chassis until ARRIVED, CANCELED, or FAILED.

        Patrol precedence: if a patrol with at least one waypoint is
        loaded, Go drives the patrol (sets goal to wp[0]; advances on
        each ARRIVED via the PatrolRunner). Otherwise Go drives to the
        single goal pin as before. The toolbar Plan label / Stop /
        Cancel paths are common between both modes.
        """
        if not self._mission.can_start():
            return  # already FOLLOWING — nothing to do
        # If a patrol is loaded with waypoints, override the goal pin
        # to wp[0]'s effective (snapped) position before validating the
        # plan — the existing planner only knows about
        # `_shared_view.goal()`, so we hand it the current target.
        # The snap cache starts empty at each Go so a re-run sees the
        # current costmap, not the prior mission's snap result.
        patrol = self._shared_view.patrol()
        patrol_runner: Optional[PatrolRunner] = None
        self._snapped_wp_xys.clear()
        if patrol is not None and len(patrol.waypoints) > 0:
            wp0 = patrol.waypoints[0]
            eff_xy = self._get_effective_wp_xy(wp0, 0)
            self._shared_view.set_goal(eff_xy)
            self._shared_view.set_patrol_active_wp_index(0)
            patrol_runner = PatrolRunner(patrol)
            # Force a synchronous replan so the plan precondition below
            # sees a fresh result against the freshly-set wp[0] goal.
            cm = self._last_costmap
            latest = self.fuser.pose_source.latest_pose()
            pose0 = latest[0] if latest is not None else None
            if cm is not None and pose0 is not None:
                self._replan(cm, pose0)
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
        # Open the per-mission trace file before start() so the very
        # first emit (mission.start) lands in the file rather than only
        # the ring buffer. A separate Go that fails a precondition above
        # leaves no file behind — those fail() emits ring-only, which
        # matches the operator-visible UI (nothing to report).
        try:
            self._tracer.open(
                session_id=self.fuser.grid.session_id,
                configs=self._trace_configs_snapshot(),
                patrol=patrol.to_dict() if patrol_runner is not None else None,
                git_sha=git_sha(),
                snapshot_at_start=None,
            )
        except Exception:
            logger.exception("tracer open failed; continuing without trace")
        self._patrol_runner = patrol_runner
        self._active_rotation = None
        self._pending_advance = None
        self._mission_was_active = True
        self._patrol_dock.set_mission_active(True)
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
        # If the bundle contains a patrols.json, load it into the
        # shared view so the operator gets the patrol back alongside
        # the world layers. Best-effort — a malformed sidecar
        # shouldn't block the layer reload.
        patrol_info = ""
        patrol_path = os.path.join(os.path.dirname(path), "patrols.json")
        if os.path.isfile(patrol_path):
            try:
                p = patrol_mod.load_from_file(patrol_path)
                self._shared_view.set_patrol(p)
                self._shared_view.set_patrol_active_wp_index(0)
                patrol_info = (
                    f"\n\npatrol loaded: {p.name} "
                    f"({len(p.waypoints)} waypoints)"
                )
            except Exception:
                logger.exception("patrols.json load from bundle failed")
                patrol_info = "\n\npatrols.json found but failed to load"
        QMessageBox.information(
            self, "Snapshot loaded",
            f"Loaded {summary['cells_observed']} cells from\n"
            f"{path}\n\n"
            f"loaded session: {summary['loaded_session_id'][:8]}\n"
            f"live session:   {summary['current_session_id'][:8]}"
            + patrol_info
        )

    def _on_save_snapshot(self) -> None:
        try:
            out_dir = self._save_bundle_with_trace()
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

    # ── Trace integration helpers ────────────────────────────────────

    def _save_bundle_with_trace(self) -> str:
        """Write a snapshot bundle and, if a trace is currently open,
        copy the active trace.jsonl into the bundle directory. Also
        embeds the active patrol (when one is loaded) as
        `patrols.json`, so a reload of the bundle restores the patrol
        alongside the world layers. Returns the bundle path. Raises
        on bundle-write errors; trace/patrol side files are best-
        effort.
        """
        out_dir = self.fuser.save_snapshot_bundle()
        trace_path = self._tracer.current_path()
        if trace_path:
            try:
                shutil.copy(
                    trace_path, os.path.join(out_dir, "trace.jsonl"),
                )
            except Exception:
                logger.exception(
                    "trace copy into snapshot bundle failed"
                )
        patrol = self._shared_view.patrol()
        if patrol is not None and len(patrol.waypoints) > 0:
            try:
                patrol_mod.write_to_file(
                    patrol, os.path.join(out_dir, "patrols.json"),
                )
            except Exception:
                logger.exception(
                    "patrol embed into snapshot bundle failed"
                )
        return out_dir

    def _sample_pose_for_trace(self):
        """Pose sampler registered with the Tracer. Returns the latest
        world-frame pose or None when none is available."""
        try:
            latest = self.fuser.pose_source.latest_pose()
        except Exception:
            return None
        if latest is None:
            return None
        return latest[0]

    def _auto_snapshot_for_trace(self, event_code: str) -> Optional[str]:
        """Tracer-invoked auto-snapshot. Writes a full bundle (including
        the active trace prefix) and returns the bundle path so it can
        be stamped into the triggering event's data.
        """
        try:
            return self._save_bundle_with_trace()
        except Exception:
            logger.exception(
                f"auto-snapshot for {event_code} raised; skipping"
            )
            return None

    def _trace_configs_snapshot(self) -> Dict[str, Any]:
        """Snapshot the frozen run-time config (mission, follower,
        safety, A*, costmap, fuser, chassis) into the trace header so
        a reviewer doesn't need access to the source tree to interpret
        thresholds and tunables.
        """
        cfg: Dict[str, Any] = {}
        try:
            cfg["mission"] = asdict(self._mission_config)
            cfg["follower"] = asdict(self._follower.config)
            cfg["safety"] = asdict(self._safety_config)
            cfg["astar"] = asdict(self._astar_config)
            cfg["costmap"] = asdict(self._costmap_config)
        except Exception:
            logger.exception("config snapshot raised")
        # Fuser + chassis configs include router; useful for "which Pi
        # was this run against." Captured loosely so unrelated dataclass
        # shape changes can't break trace opening.
        try:
            cfg["fuser"] = asdict(self.fuser_config)
        except Exception:
            cfg["fuser"] = {"router": getattr(self.fuser_config, "router", None)}
        try:
            cfg["chassis"] = asdict(self.chassis_config)
        except Exception:
            cfg["chassis"] = {"router": getattr(self.chassis_config, "router", None)}
        cfg["sustained_block_s"] = _SUSTAINED_BLOCK_S
        return cfg

    def _on_fit_maps(self) -> None:
        # Single call: shared view propagates to all attached panels.
        self._shared_view.reset_view()

    def _on_toggle_stream_rgb(self, checked: bool) -> None:
        if checked:
            self._stream_rgb_timer.start()
        else:
            self._stream_rgb_timer.stop()

    # ── Patrol UI handlers ───────────────────────────────────────────

    def _on_patrol_edit_action(self, checked: bool) -> None:
        """Toolbar toggle → propagate to shared view + dock checkbox.
        The dock has its own checkbox; we keep both in sync via the
        shared-view boolean so either control reflects reality."""
        self._shared_view.set_patrol_edit_mode(bool(checked))
        # Keep the dock checkbox visually in sync without firing its
        # toggle signal (which would re-enter this handler).
        if self._patrol_dock._edit_box.isChecked() != bool(checked):
            blk = self._patrol_dock._edit_box.blockSignals(True)
            self._patrol_dock._edit_box.setChecked(bool(checked))
            self._patrol_dock._edit_box.blockSignals(blk)

    def _on_patrol_edit_toggled(self, on: bool) -> None:
        """Dock checkbox → propagate to toolbar action."""
        if self._patrol_edit_act.isChecked() != bool(on):
            blk = self._patrol_edit_act.blockSignals(True)
            self._patrol_edit_act.setChecked(bool(on))
            self._patrol_edit_act.blockSignals(blk)

    def _on_patrol_append_requested(self, x_w: float, y_w: float) -> None:
        """Right-click append handler — wired into the SharedMapView.
        Creates an empty patrol if none is loaded (so the operator can
        start placing pins without an explicit New click), then appends
        the world point.
        """
        if self._mission.is_active():
            # Edit-mid-mission is locked by design.
            return
        p = self._shared_view.patrol()
        if p is None:
            live_sid = self.fuser.grid.session_id
            p = patrol_mod.new_empty(session_id=live_sid)
        p.append(x_w, y_w)
        self._shared_view.set_patrol(p)

    def _on_stream_rgb_tick(self) -> None:
        # Streaming-mode capture: in-flight gating in the controller
        # keeps a slow Pi from accruing a backlog. If the chassis is
        # disconnected, request_rgb_streaming() returns None and the
        # tick is a no-op.
        self.chassis.request_rgb_streaming()

    # ── Lifecycle ────────────────────────────────────────────────────

    def _balance_splitter_once(self) -> None:
        # Applied via QTimer.singleShot(0, …) from showEvent: by the
        # time this fires, the splitter has a real size from the first
        # layout pass. Idempotent so spurious re-fires don't stomp on
        # a user's drag.
        if self._splitter_balanced:
            return
        v_total = self._v_splitter.height()
        if v_total <= 0:
            return
        # Maps gets ~75% of the column, feeds the remaining ~25%.
        # Vision lives in the left dock area now (alongside Patrol).
        maps_h = int(v_total * 0.75)
        feeds_h = max(0, v_total - maps_h)
        self._v_splitter.setSizes([maps_h, feeds_h])
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
        # Flush + close any open trace file so the last events aren't
        # left in an unclosed handle (line-buffering covers writes, but
        # close releases the fd cleanly for any tail consumer).
        try:
            self._tracer.close()
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

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
import time
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QSplitter, QToolBar, QVBoxLayout, QWidget,
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
from .planner import AStarConfig, PlanResult, plan_path

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

        # Bottom status strip: fuser (pose + rates + cells + session) on
        # the left, chassis text summary on the right. The safety pills
        # at the top handle *gate* state (conn/hb/estop); this strip is
        # for values the pills can't convey (ages, counts, session id).
        bot = QHBoxLayout()
        self._pose_lbl = QLabel("pose: —")
        self._rates_lbl = QLabel("rates: —")
        self._cells_lbl = QLabel("cells: —")
        self._slam_lbl = QLabel("slam: —")
        self._plan_lbl = QLabel("plan: —")
        self._session_lbl = QLabel("session: —")
        self._chassis_lbl = QLabel("chassis: —")
        self._notes_lbl = QLabel("")
        self._notes_lbl.setStyleSheet("color: #e8a; font-weight: bold;")
        # Slightly-smaller font across the strip — the line is wide
        # enough to crowd at default size once slam_health is added.
        small = self.font()
        small.setPointSize(max(7, small.pointSize() - 1))
        for w in (self._pose_lbl, self._rates_lbl,
                  self._cells_lbl, self._slam_lbl, self._plan_lbl,
                  self._session_lbl, self._chassis_lbl):
            w.setStyleSheet("color: #ccc;")
            w.setFont(small)
            bot.addWidget(w)
        self._notes_lbl.setFont(small)
        bot.addWidget(self._notes_lbl, stretch=1)
        outer.addLayout(bot)

        self.setCentralWidget(central)
        self.resize(1400, 900)

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
        ts = time.time()
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
            # so the path keeps up as the map fills in.
            self._last_costmap = cm
            if cm is not None and self._shared_view.goal() is not None:
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

        st = self.fuser.status_summary()
        if pose is not None:
            self._pose_lbl.setText(
                f"pose: x={pose[0]:+.2f} y={pose[1]:+.2f} "
                f"θ={math.degrees(pose[2]):+.1f}°"
            )
        else:
            self._pose_lbl.setText("pose: (no odom)")

        rates = st["rates"]
        ages = st["ages"]
        parts = []
        for name, key in (("lm", "local_map"), ("od", "odom")):
            hz = rates.get(key)
            age = ages.get(key)
            hz_s = f"{hz:.1f}Hz" if hz is not None else "—"
            age_s = f"{age:.2f}s" if age is not None else "—"
            parts.append(f"{name} {hz_s}/{age_s}")
        self._rates_lbl.setText("rates: " + "  ".join(parts))

        self._cells_lbl.setText(
            f"cells: obs={st['cells_observed']} trav={st['cells_traversed']}"
        )

        # SLAM health: pose-unavailable streak (≥10 = sticky note set)
        # plus cumulative scan-match correction since session reset.
        # Both rise when SLAM is struggling; both reset on Reset world.
        unavail = int(st.get("pose_unavail_streak") or 0)
        corr = st.get("correction_summary") or {}
        corr_m = float(corr.get("total_m") or 0.0)
        corr_deg = math.degrees(float(corr.get("total_rad") or 0.0))
        n_corr = int(corr.get("n_applied") or 0)
        slam_text = (
            f"slam: lost={unavail}  drift={corr_m:.2f} m / "
            f"{corr_deg:.1f}°  n={n_corr}"
        )
        self._slam_lbl.setText(slam_text)
        # Color cue: red when pose has been lost recently, amber when
        # it's been lost for ≥10 frames (already sticky-noted).
        if unavail >= 10:
            self._slam_lbl.setStyleSheet("color: #e8a;")
        elif unavail > 0:
            self._slam_lbl.setStyleSheet("color: #ec8;")
        else:
            self._slam_lbl.setStyleSheet("color: #ccc;")

        # Plan status: "—" with no goal; details when planned.
        goal = self._shared_view.goal()
        if goal is None:
            self._plan_lbl.setText("plan: —")
            self._plan_lbl.setStyleSheet("color: #ccc;")
        else:
            plan = self._last_plan
            if plan is None:
                self._plan_lbl.setText(
                    f"plan: pending ({goal[0]:+.2f}, {goal[1]:+.2f})"
                )
                self._plan_lbl.setStyleSheet("color: #cc8;")
            elif plan.ok:
                self._plan_lbl.setText(
                    f"plan: {plan.distance_m:.2f} m  "
                    f"({plan.elapsed_ms:.0f} ms)"
                )
                self._plan_lbl.setStyleSheet("color: #8cf;")
            else:
                self._plan_lbl.setText(f"plan: {plan.msg}")
                self._plan_lbl.setStyleSheet("color: #e8a;")

        self._session_lbl.setText(
            f"session: {st['session_id'][:8]}  ({st['pose_source']})"
        )
        self._notes_lbl.setText(st.get("notes") or "")

    def _refresh_chassis_panel(self) -> None:
        """Text summary with values the pills can't convey (status age,
        heartbeat seq). Gate colors live on the safety toolbar.
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
        age_s = (
            f"{time.time() - status_ts:.1f}s" if status_ts > 0 else "—"
        )
        self._chassis_lbl.setText(
            f"chassis: status/{age_s}  hb#{hb_seq}"
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
        self._shared_view.set_goal(None)
        self._last_plan = None

    # ── Toolbar handlers ─────────────────────────────────────────────

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

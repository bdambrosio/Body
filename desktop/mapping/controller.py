"""Mapping session controller — EKF + pose-graph SLAM."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import os

from desktop.fusion.ekf_pose_tracker import EkfPoseTracker
from desktop.fusion.load_slam_config import load_slam_config
from desktop.fusion.rotation_contest import RotationContestJudge
from desktop.mapping.bag import BagRecorder
from desktop.mapping.export import export_mapping_session
from desktop.mapping.pose_graph_mapper import PoseGraphMapper
from desktop.nav.slam.scan_matcher import lidar_scan_to_xy
from desktop.nav.slam.types import ImuReading
from desktop.reference_map.legacy_convert import load_map_auto
from desktop.reference_map.reference_map import ReferenceMap
from desktop.chassis.transport import open_session
from desktop.utils.stream_epoch import epoch_fields

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


def _decode_json(payload: bytes) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None


def _odom_xyt(msg: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    try:
        ts = float(msg.get("ts") or _now())
        x = float(msg.get("x") or 0.0)
        y = float(msg.get("y") or 0.0)
        th = float(msg.get("theta", msg.get("theta_rad", msg.get("yaw", 0.0))))
        return ts, x, y, th
    except (TypeError, ValueError):
        return None


class MappingConfig:
    def __init__(
        self,
        *,
        router: str = "tcp/127.0.0.1:7447",
        extent_m: float = 33.0,
        resolution_m: float = 0.05,
        scan_match_hz: float = 2.0,
        map_stale_s: float = 2.0,
        ui_redraw_hz: float = 5.0,
        record: bool = True,
        bag_dir: Optional[str] = None,
        prior_map: Optional[str] = None,
        seed_pose: Optional[Tuple[float, float, float]] = None,
        keyframe_spacing_m: float = 0.5,
    ):
        self.router = router
        self.extent_m = extent_m
        self.resolution_m = resolution_m
        self.scan_match_hz = scan_match_hz
        self.map_stale_s = map_stale_s
        self.ui_redraw_hz = ui_redraw_hz
        # Raw recording (desktop/mapping/bag.py): on by default; the bag
        # lands under ~/Body/maps/<session>/bag_<ts>/ unless bag_dir is set.
        self.record = record
        self.bag_dir = bag_dir
        # An older reference_map.npz whose hand layers (checkpoints in
        # metadata, no-go mask) are carried into the new map. Only
        # meaningful with a seed_pose that puts the run in the old frame.
        self.prior_map = prior_map
        # World pose the robot's start is declared to be (x, y, theta rad):
        # the new map comes out roughly in an older map's frame.
        self.seed_pose = seed_pose
        # RGB keyframe every this much travel (bag only).
        self.keyframe_spacing_m = keyframe_spacing_m


class MappingController:
    """Builds a pose-graph map during a teleop mapping drive."""

    def __init__(self, config: MappingConfig):
        self.config = config
        slam_cfg = load_slam_config()
        self._slam_cfg = slam_cfg
        self.ekf = EkfPoseTracker(noise=slam_cfg.fusion)
        self.mapper = PoseGraphMapper(
            slam_config=slam_cfg.slam,
            fusion_config=slam_cfg.fusion,
            ekf=self.ekf,
        )
        self._session_id = uuid.uuid4().hex[:12]
        self.reference_map: Optional[ReferenceMap] = None
        self._lock = threading.RLock()
        self._session: Optional[Any] = None
        self._subs: List[Any] = []
        self._on_update: Optional[Callable[[], None]] = None
        # Wheel-vs-IMU heading disputes get a lidar verdict here too (the
        # 11-01 nav fix); before, every dispute expired as "keep wheels" and
        # a slip's heading error went straight into the map.
        self._judge = RotationContestJudge(self.ekf, tag="mapping")
        self._seed_pending = config.seed_pose is not None
        self.prior: Optional[ReferenceMap] = None
        if config.prior_map:
            try:
                self.prior = load_map_auto(config.prior_map)
                logger.info("mapping: prior map %s (%d checkpoints, nogo %s)",
                            config.prior_map,
                            len(self.prior.metadata.get("checkpoints") or []),
                            self.prior.nogo_mask is not None)
            except Exception:
                logger.exception("mapping: prior map %s failed to load", config.prior_map)
        self.bag = self._open_bag()
        self._last_frame_pose: Optional[Tuple[float, float, float]] = None

    def _open_bag(self) -> BagRecorder:
        if not self.config.record:
            return BagRecorder(None)
        d = self.config.bag_dir or os.path.join(
            os.path.expanduser("~"), "Body", "maps", self._session_id,
            time.strftime("bag_%Y%m%d_%H%M%S"))
        info = {"session_id": self._session_id,
                "slam": dict(self._slam_cfg.slam.__dict__),
                "seed_pose": list(self.config.seed_pose) if self.config.seed_pose else None,
                "prior_map": self.config.prior_map}
        logger.info("mapping: recording bag to %s", d)
        return BagRecorder(d, info=info)

    @property
    def grid(self) -> MappingController:
        return self

    @property
    def session_id(self) -> str:
        return self._session_id

    def set_on_grid_update(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_update = cb

    def connect(self) -> Tuple[bool, Optional[str]]:
        if self._session is not None:
            return True, None
        try:
            self._session = open_session(self.config.router)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        self._subs.append(
            self._session.declare_subscriber("body/odom", self._on_odom),
        )
        self._subs.append(
            self._session.declare_subscriber("body/imu", self._on_imu),
        )
        self._subs.append(
            self._session.declare_subscriber("body/lidar/scan", self._on_scan),
        )
        return True, None

    def shutdown(self) -> None:
        try:
            self.bag.close()
        except Exception:
            logger.exception("bag close failed")
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                pass
        self._subs.clear()
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    disconnect = shutdown

    @property
    def connected(self) -> bool:
        return self._session is not None

    @property
    def session(self) -> Optional[Any]:
        return self._session

    def _payload_bytes(self, sample: Any) -> bytes:
        try:
            return bytes(sample.payload.to_bytes())
        except AttributeError:
            return bytes(sample.payload)

    def _on_odom(self, sample: Any) -> None:
        msg = _decode_json(self._payload_bytes(sample))
        if msg is not None:
            self.ingest_odom(msg)

    def _on_imu(self, sample: Any) -> None:
        msg = _decode_json(self._payload_bytes(sample))
        if msg is not None:
            self.ingest_imu(msg)

    def _on_scan(self, sample: Any) -> None:
        msg = _decode_json(self._payload_bytes(sample))
        if msg is not None:
            self.ingest_scan(msg)

    # ---- message ingestion (Zenoh callbacks and bag replay) ---------------

    def ingest_odom(self, msg: Dict[str, Any]) -> None:
        triple = _odom_xyt(msg)
        if triple is None:
            return
        self.bag.add_odom(msg)
        ts, x, y, theta = triple
        session_id, seq = epoch_fields(msg)
        self.ekf.update_odom(ts, x, y, theta, session_id=session_id, seq=seq)

    def ingest_imu(self, msg: Dict[str, Any]) -> None:
        reading = ImuReading.from_payload(msg)
        if reading is not None:
            self.bag.add_imu(msg)
            self.ekf.update_imu(reading)

    def ingest_scan(self, msg: Dict[str, Any]) -> bool:
        """Returns True when the scan became a graph node."""
        ts = float(msg.get("ts") or _now())
        ranges = msg.get("ranges")
        angle_min = float(msg.get("angle_min", 0.0))
        angle_inc = float(msg.get("angle_increment", 0.0))
        if not isinstance(ranges, list) or angle_inc <= 0:
            return False
        self.bag.add_scan(msg)
        angles = np.arange(len(ranges), dtype=np.float64) * angle_inc + angle_min
        ranges_arr = np.asarray(
            [r if isinstance(r, (int, float)) else np.nan for r in ranges],
            dtype=np.float64,
        )
        if self._seed_pending and self.ekf.is_ready():
            self._apply_seed()
        if msg.get("ts"):
            try:
                self._judge.observe(ts, lidar_scan_to_xy(ranges_arr, angles))
            except Exception:
                logger.exception("rotation contest judge failed")
        added = self.mapper.add_scan(ts, ranges_arr, angles)
        if added:
            cb = self._on_update
            if cb is not None:
                try:
                    cb()
                except Exception:
                    logger.exception("on_grid_update failed")
        return added

    def _apply_seed(self) -> None:
        seed = self.config.seed_pose
        self._seed_pending = False
        if seed is None:
            return
        out = self.ekf.rebind_world_to(seed)
        self.mapper.reset()
        self._judge.clear()
        logger.info("mapping: world frame seeded at (%.2f, %.2f, %.1f°) -> %s",
                    seed[0], seed[1], math.degrees(seed[2]), out)

    # ---- RGB keyframes for the bag -----------------------------------------

    def wants_keyframe(self) -> bool:
        """UI poll: has the robot moved keyframe_spacing_m since the last
        recorded frame (bag enabled, EKF ready)?"""
        if not self.bag.enabled or not self.ekf.is_ready():
            return False
        pose = self.mapper.display_pose()
        if pose is None:
            return False
        last = self._last_frame_pose
        if last is None:
            return True
        return (math.hypot(pose[0] - last[0], pose[1] - last[1])
                >= self.config.keyframe_spacing_m)

    def record_frame(self, jpeg: bytes, *, pi_ts: float = 0.0) -> None:
        pose = self.mapper.display_pose()
        self._last_frame_pose = pose
        self.bag.add_frame(_now(), pose, jpeg, pi_ts=pi_ts)

    def snapshot_for_ui(self) -> Optional[Dict[str, Any]]:
        snap = self.mapper.snapshot_for_ui()
        if snap is None:
            return None
        snap["session_id"] = self._session_id
        drive = snap["driveable"]
        snap["grid"] = np.full(drive.shape, np.nan, dtype=np.float32)
        return snap

    def pose_trail(self) -> List[Tuple[float, float, float]]:
        return self.mapper.trajectory()

    def status_summary(self) -> Dict[str, Any]:
        pose = self.mapper.display_pose()
        ekf_diag = self.ekf.diagnostics()
        slam_diag = self.mapper.diagnostics()
        # Pi-clock ts of the newest EKF state — never the desktop clock (a
        # cross-host comparison that returns None under any skew).
        latest_ts = self.ekf.latest_state_ts()
        cov = self.ekf.cov_at(latest_ts) if latest_ts is not None else None
        cov_xy = None
        if cov is not None:
            cov_xy = float(cov[0, 0] + cov[1, 1])
        return {
            "session_id": self._session_id,
            "bag_dir": self.bag.out_dir,
            "bag_scans": self.bag.n_scans,
            "bag_frames": self.bag.n_frames,
            "keyframes_skipped": slam_diag.get("keyframes_skipped"),
            "edge_rejects": slam_diag.get("edge_rejects"),
            "contests_judged": self._judge.counts["judged"],
            "contests_imu_wins": self._judge.counts["imu_wins"],
            "pose": pose,
            "pose_source": "pose_graph",
            "imu_settled": ekf_diag.get("imu_settled"),
            "heading_source": ekf_diag.get("heading_source"),
            "yaw_at_misses": ekf_diag.get("yaw_at_misses"),
            "imu_distrusted": ekf_diag.get("imu_distrusted"),
            "odom_resets": ekf_diag.get("odom_resets"),
            "ekf_cov_trace_xy": cov_xy,
            "graph_nodes": slam_diag.get("node_count"),
            "graph_edges": slam_diag.get("edge_count"),
            "last_match_improvement": slam_diag.get("last_match_improvement"),
            "loop_closures": slam_diag.get("loop_closure_count"),
        }

    def finalize_map(self) -> ReferenceMap:
        metadata: Dict[str, Any] = {"mapping_version": 2, "slam": "pose_graph"}
        nogo = None
        prior = self.prior
        if prior is not None:
            cps = prior.metadata.get("checkpoints")
            if cps:
                metadata["checkpoints"] = cps
                metadata["checkpoints_from"] = self.config.prior_map
            snap = self.mapper.snapshot_for_ui()
            meta = (snap or {}).get("meta") or {}
            same_grid = (
                prior.nogo_mask is not None
                and int(meta.get("nx", -1)) == prior.nx and int(meta.get("ny", -1)) == prior.ny
                and abs(float(meta.get("resolution_m", 0)) - prior.resolution_m) < 1e-9
                and abs(float(meta.get("origin_x_m", 1e9)) - prior.origin_x_m) < 1e-6
                and abs(float(meta.get("origin_y_m", 1e9)) - prior.origin_y_m) < 1e-6)
            if same_grid:
                nogo = prior.nogo_mask
            elif prior.nogo_mask is not None:
                logger.warning("mapping: prior no-go mask dropped (grid geometry differs)")
        ref = self.mapper.to_reference_map(
            session_id=self._session_id, metadata=metadata, nogo_mask=nogo,
        )
        self.reference_map = ref
        return ref

    def save_snapshot_bundle(self, base_dir: Optional[str] = None) -> str:
        # export_mapping_session finalizes once (optimise + rebuild are not
        # free at large N — don't do them twice).
        self.bag.flush_scans()
        return export_mapping_session(self, base_dir=base_dir)

    def request_reset(self, *, reason: str = "ui_reset") -> None:
        if self.config.seed_pose is not None:
            self.ekf.rebind_world_to(self.config.seed_pose)
            self._seed_pending = False
        else:
            self.ekf.rebind_world_to_current()
        self.mapper.reset()
        self._judge.clear()
        self._last_frame_pose = None
        self._session_id = uuid.uuid4().hex[:12]
        self.reference_map = None
        self.bag.add_json("events.jsonl", {"ts": _now(), "event": "reset", "reason": reason,
                                           "session_id": self._session_id})

    def request_relocate(self, *, reason: str = "ui_relocate") -> dict:
        return {"success": False, "reason": "not_supported_in_mapping"}

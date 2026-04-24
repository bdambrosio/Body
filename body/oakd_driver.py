"""OAK-D Lite driver: DepthAI depth/RGB pipeline and diagnostic IMU aggregation.

The IMU publish path was retired when the external BNO085 took over ``body/imu`` (see
``body/imu_driver.py`` and docs/imu_driver_spec.md). The OAK-D IMU loop here is kept only
to exercise the DepthAI IMU pipeline during bring-up (log lines over
``imu_aggregate_interval_s``); nothing is published on Zenoh.

Supports **DepthAI v2** (``dai.node.XLinkOut``) and **v3** (output queues on node outputs;
no XLink).
"""

from __future__ import annotations

import base64
import queue
import signal
import sys
import time
from collections.abc import Callable
from typing import Any

import cv2
import depthai as dai
import numpy as np

from body.lib import schemas, zenoh_helpers

# Populated once per device open by ``_log_oakd_calibration`` (stereo-L intrinsics at the configured
# depth output resolution, adjusted for ``rgb_rotate_deg``). Consumed by ``_depth_frame_to_stream_msg``
# so downstream readers (e.g. local_map) can unproject with device-true fx/fy/cx/cy.
_DEPTH_INTRINSICS: dict[str, float] | None = None


def _depthai_is_v3() -> bool:
    return not hasattr(dai.node, "XLinkOut")


def _is_depthai_imu_missing_error(exc: BaseException) -> bool:
    msg = str(exc)
    return "IMU not detected" in msg or "IMU invalid settings" in msg


def _log_imu_missing_fallback(exc: RuntimeError) -> None:
    err = str(exc)
    print(
        "[oakd] DepthAI IMU error:\n"
        f"  {err}\n"
        "  Falling back to synthetic IMU on Zenoh. To skip opening the OAK for IMU entirely, "
        "set oakd.imu_hardware_present to false. If you have a BNO IMU that needs flashing, set "
        "oakd.imu_enable_firmware_update to true (DepthAI v3 uses device.startIMUFirmwareUpdate), "
        "keep USB stable, wait for 100%.",
        file=sys.stderr,
        flush=True,
    )


def _log_oakd_calibration(device: Any, oakd_cfg: dict[str, Any]) -> None:
    """Print factory calibration per camera and cache stereo-L intrinsics for the depth stream.

    The OAK-D depth stream is aligned to stereo-L (CAM_B). We store rotation-corrected intrinsics at
    the published ``(out_w, out_h)`` resolution in ``_DEPTH_INTRINSICS`` so depth messages can carry
    device-true ``fx``/``fy``/``cx``/``cy`` to consumers (avoids HFOV-based approximations).
    """
    global _DEPTH_INTRINSICS
    try:
        calib = device.readCalibration()
    except Exception as e:
        print(f"[oakd] readCalibration failed: {e}", flush=True)
        return
    rw = int(oakd_cfg.get("rgb_preview_width", 640))
    rh = int(oakd_cfg.get("rgb_preview_height", 400))
    dw = int(oakd_cfg.get("depth_out_width", 120))
    dh = int(oakd_cfg.get("depth_out_height", 90))
    for name, sock, w, h in (
        ("RGB(CAM_A)", dai.CameraBoardSocket.CAM_A, rw, rh),
        ("stereo_L(CAM_B)", dai.CameraBoardSocket.CAM_B, dw, dh),
        ("stereo_R(CAM_C)", dai.CameraBoardSocket.CAM_C, dw, dh),
    ):
        try:
            hfov = float(calib.getFov(sock))
            K = calib.getCameraIntrinsics(sock, w, h)
            fx, fy = float(K[0][0]), float(K[1][1])
            cx, cy = float(K[0][2]), float(K[1][2])
            print(
                f"[oakd] calib {name} @ {w}x{h}: hfov={hfov:.2f}° "
                f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}",
                flush=True,
            )
        except Exception:
            continue

    try:
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, dw, dh)
        fx, fy = float(K[0][0]), float(K[1][1])
        cx, cy = float(K[0][2]), float(K[1][2])
    except Exception as e:
        print(f"[oakd] depth intrinsics unavailable (falling back to hfov-derived): {e}", flush=True)
        _DEPTH_INTRINSICS = None
        return
    deg = int(oakd_cfg.get("rgb_rotate_deg", 0)) % 360
    # ``_apply_oakd_image_rotate`` runs before the cv2.resize that finalizes (dw, dh). For 0° and
    # 180° the output aspect matches getCameraIntrinsics'; for 90°/270° aspect flips so cx/cy swap
    # and fx/fy swap (reported values are approximate in that case due to the subsequent resize).
    if deg == 90:
        fx, fy = fy, fx
        cx, cy = cy, (dw - 1) - cx
    elif deg == 180:
        cx = (dw - 1) - cx
        cy = (dh - 1) - cy
    elif deg == 270:
        fx, fy = fy, fx
        cx, cy = (dh - 1) - cy, cx
    _DEPTH_INTRINSICS = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    print(
        f"[oakd] depth msg intrinsics (CAM_B @ {dw}x{dh}, rotate={deg}°): "
        f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}",
        flush=True,
    )


def _cleanup_v3_after_failed_pipeline_start(device: Any, pipeline: Any) -> None:
    try:
        if getattr(pipeline, "isRunning", lambda: False)():
            pipeline.stop()
    except Exception:
        pass
    try:
        close = getattr(device, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def _configure_imu_node(imu: Any, oakd_cfg: dict[str, Any], *, depthai_v3: bool) -> None:
    # DepthAI v2: optional IMU node firmware flag (deprecated on v3 — use Device.startIMUFirmwareUpdate).
    if (
        not depthai_v3
        and oakd_cfg.get("imu_enable_firmware_update", False)
        and hasattr(imu, "enableFirmwareUpdate")
    ):
        imu.enableFirmwareUpdate(True)

    accel_hz = int(oakd_cfg.get("imu_accel_hz", 500))
    gyro_hz = int(oakd_cfg.get("imu_gyro_hz", 400))
    rot_hz = int(oakd_cfg.get("imu_rotation_vector_hz", 400))
    use_rot = bool(oakd_cfg.get("rotation_vector_enabled", True))

    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, accel_hz)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, gyro_hz)
    if use_rot:
        imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, rot_hz)

    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)


def _pick_device(device_id: str | None) -> dai.DeviceInfo | None:
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        return None
    if not device_id:
        return devices[0]
    for d in devices:
        if device_id in d.deviceId or device_id == d.name:
            return d
    return None


def _rgb_preview_dims(oakd_cfg: dict[str, Any]) -> tuple[int, int]:
    w = int(oakd_cfg.get("rgb_preview_width", 640))
    h = int(oakd_cfg.get("rgb_preview_height", 400))
    return max(16, w), max(16, h)


def _depth_output_dims(oakd_cfg: dict[str, Any]) -> tuple[int, int]:
    w = int(oakd_cfg.get("depth_out_width", 120))
    h = int(oakd_cfg.get("depth_out_height", 90))
    return max(2, w), max(2, h)


def _depth_drain_latest(depth_queue: Any) -> Any | None:
    last: Any | None = None
    while True:
        f = depth_queue.tryGet()
        if f is None:
            break
        last = f
    return last


def _apply_oakd_image_rotate(arr: np.ndarray, oakd_cfg: dict[str, Any]) -> np.ndarray:
    """Shared mount correction for RGB and depth (``oakd.rgb_rotate_deg``: 0, 90, 180, 270)."""
    deg = int(oakd_cfg.get("rgb_rotate_deg", 0)) % 360
    if deg == 90:
        return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(arr, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return arr


def _depth_frame_to_stream_msg(
    frame: Any,
    out_w: int,
    out_h: int,
    oakd_cfg: dict[str, Any],
    smooth_prev_holder: list[np.ndarray | None],
) -> dict[str, Any]:
    """Resize (and optionally rotate) depth to ``out_w``×``out_h``, apply temporal IIR smoothing.

    Smoothing: where the new sample ``D`` is valid (``D>0``), ``S = (1/3)*S_prev + (2/3)*D`` (mm).
    Where ``D`` is invalid (0) but ``S_prev > 0``, **hold** ``S = S_prev``. Where both are invalid,
    ``S = 0``. ``smooth_prev_holder[0]`` holds float64 ``S`` (same shape as output).
    """
    h0 = int(frame.getHeight())
    w0 = int(frame.getWidth())
    arr = frame.getFrame()
    if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
        buf = frame.getData()
        arr = np.frombuffer(bytes(buf), dtype=np.uint16).reshape((h0, w0))
    else:
        arr = np.ascontiguousarray(arr, dtype=np.uint16)
    arr = _apply_oakd_image_rotate(arr, oakd_cfg)
    small = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    curr = small.astype(np.float64)
    prev = smooth_prev_holder[0]
    if prev is None or prev.shape != curr.shape:
        prev = np.zeros_like(curr)
        smooth_prev_holder[0] = prev
    valid_c = curr > 0.0
    valid_p = prev > 0.0
    out = np.zeros_like(curr)
    both = valid_c & valid_p
    w_old, w_new = 1.0 / 3.0, 2.0 / 3.0
    out[both] = w_old * prev[both] + w_new * curr[both]
    out[valid_c & ~valid_p] = curr[valid_c & ~valid_p]
    hold = ~valid_c & valid_p
    out[hold] = prev[hold]
    np.copyto(prev, out)
    small_out = np.clip(np.round(out), 0, 65535).astype(np.uint16)
    b64 = base64.standard_b64encode(small_out.tobytes()).decode("ascii")
    return schemas.oakd_depth_stream_frame(out_w, out_h, b64, intrinsics=_DEPTH_INTRINSICS)


def _create_v3_stereo_depth_queue(pipeline: Any, oakd_cfg: dict[str, Any]) -> Any:
    mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    mono_left.requestFullResolutionOutput().link(stereo.left)
    mono_right.requestFullResolutionOutput().link(stereo.right)
    stereo.setRectification(bool(oakd_cfg.get("stereo_rectification", True)))
    stereo.setExtendedDisparity(bool(oakd_cfg.get("stereo_extended_disparity", True)))
    stereo.setLeftRightCheck(bool(oakd_cfg.get("stereo_left_right_check", True)))
    return stereo.depth.createOutputQueue(maxSize=4, blocking=False)


def _create_v3_rgb_queue(pipeline: Any, oakd_cfg: dict[str, Any]) -> Any | None:
    if not oakd_cfg.get("rgb_enabled", False):
        return None
    rw, rh = _rgb_preview_dims(oakd_cfg)
    rgb_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    return rgb_cam.requestOutput((rw, rh)).createOutputQueue(maxSize=4, blocking=False)


def _rgb_drain_latest(rgb_queue: Any) -> Any | None:
    last: Any | None = None
    while True:
        f = rgb_queue.tryGet()
        if f is None:
            break
        last = f
    return last


def _wait_rgb_frame(rgb_queue: Any, timeout_s: float = 3.0) -> Any | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        f = _rgb_drain_latest(rgb_queue)
        if f is not None:
            return f
        time.sleep(0.005)
    return _rgb_drain_latest(rgb_queue)


def _jpeg_b64_from_imgframe(frame: Any, oakd_cfg: dict[str, Any]) -> tuple[str, int, int]:
    bgr = _apply_oakd_image_rotate(frame.getCvFrame(), oakd_cfg)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok or buf is None:
        raise RuntimeError("cv2.imencode failed for RGB frame")
    h, w = bgr.shape[:2]
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return b64, w, h


def _process_oakd_config_queue(
    session: Any,
    config_pending: queue.Queue[dict[str, Any]],
    rgb_queue: Any | None,
    oakd_cfg: dict[str, Any],
) -> None:
    while True:
        try:
            req = config_pending.get_nowait()
        except queue.Empty:
            break
        action = req.get("action")
        rid = str(req.get("request_id", ""))
        if action != "capture_rgb":
            continue
        if rgb_queue is None:
            zenoh_helpers.publish_json(
                session,
                "body/oakd/rgb",
                schemas.oakd_rgb_capture_error(
                    rid,
                    "no_rgb_pipeline_set_oakd_rgb_enabled_true_or_fix_imu_rgb_fallback",
                ),
            )
            continue
        frame = _wait_rgb_frame(rgb_queue)
        if frame is None:
            zenoh_helpers.publish_json(
                session,
                "body/oakd/rgb",
                schemas.oakd_rgb_capture_error(rid, "no_rgb_frame_timeout"),
            )
            continue
        try:
            jpeg_b64, fw, fh = _jpeg_b64_from_imgframe(frame, oakd_cfg)
            zenoh_helpers.publish_json(
                session,
                "body/oakd/rgb",
                schemas.oakd_rgb_capture_ok(rid, jpeg_b64, fw, fh),
            )
            print(f"[oakd] capture_rgb ok request_id={rid} {fw}x{fh}", flush=True)
        except Exception as e:
            zenoh_helpers.publish_json(
                session,
                "body/oakd/rgb",
                schemas.oakd_rgb_capture_error(rid, f"encode_failed:{e}"),
            )


def _run_imu_loop(
    imu_queue: dai.MessageQueue,
    session: Any,
    oakd_cfg: dict[str, Any],
    interval_s: float,
    depth_period: float,
    continue_fn: Callable[[], bool],
    config_pending: queue.Queue[dict[str, Any]],
    rgb_queue: Any | None,
    depth_queue: Any | None = None,
) -> None:
    window_ax: list[float] = []
    window_ay: list[float] = []
    window_az: list[float] = []
    window_gx: list[float] = []
    window_gy: list[float] = []
    window_gz: list[float] = []
    last_quat: tuple[float, float, float, float] | None = None
    depth_smooth_prev: list[np.ndarray | None] = [None]

    window_start = time.monotonic()
    next_depth = time.monotonic()

    while continue_fn():
        _process_oakd_config_queue(session, config_pending, rgb_queue, oakd_cfg)
        imu_data = imu_queue.tryGet()
        if imu_data is not None:
            for pkt in imu_data.packets:
                a = pkt.acceleroMeter
                g = pkt.gyroscope
                window_ax.append(float(a.x))
                window_ay.append(float(a.y))
                window_az.append(float(a.z))
                window_gx.append(float(g.x))
                window_gy.append(float(g.y))
                window_gz.append(float(g.z))
                rv = pkt.rotationVector
                if rv is not None:
                    last_quat = (float(rv.real), float(rv.i), float(rv.j), float(rv.k))

        now = time.monotonic()

        if now >= window_start + interval_s and window_ax:
            n = len(window_ax)
            ax = sum(window_ax) / n
            ay = sum(window_ay) / n
            az = sum(window_az) / n
            gx = sum(window_gx) / n
            gy = sum(window_gy) / n
            gz = sum(window_gz) / n
            ts = time.time()
            quat = last_quat if oakd_cfg.get("rotation_vector_enabled", True) else None
            # body/imu is published by body/imu_driver.py (BNO085). OAK-D IMU is retained
            # here only as a redundant diagnostic log; no Zenoh publish.
            _ = (ts, ax, ay, az, gx, gy, gz, quat)
            q_str = (
                f" quat=({quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f})" if quat else ""
            )
            print(
                f"[oakd] imu mean over {interval_s:.2f}s: n={n} "
                f"accel=({ax:.3f},{ay:.3f},{az:.3f}) m/s² "
                f"gyro=({gx:.4f},{gy:.4f},{gz:.4f}) rad/s{q_str}",
                flush=True,
            )
            window_ax.clear()
            window_ay.clear()
            window_az.clear()
            window_gx.clear()
            window_gy.clear()
            window_gz.clear()
            window_start = now

        if now >= next_depth:
            if depth_queue is not None:
                df = _depth_drain_latest(depth_queue)
                if df is not None:
                    dw, dh = _depth_output_dims(oakd_cfg)
                    try:
                        zenoh_helpers.publish_json(
                            session,
                            "body/oakd/depth",
                            _depth_frame_to_stream_msg(
                                df, dw, dh, oakd_cfg, depth_smooth_prev
                            ),
                        )
                    except Exception as e:
                        print(f"[oakd] depth stream encode error: {e}", file=sys.stderr, flush=True)
            else:
                zenoh_helpers.publish_json(
                    session, "body/oakd/depth", schemas.oakd_depth_placeholder()
                )
            next_depth += depth_period

        if imu_data is None:
            time.sleep(0.001)


def _run_synthetic_imu_depth_config_loop(
    session: Any,
    oakd_cfg: dict[str, Any],
    interval_s: float,
    depth_period: float,
    stop_ref: list[bool],
    config_pending: queue.Queue[dict[str, Any]],
    rgb_queue: Any | None,
    depth_queue: Any | None = None,
) -> None:
    """Synthetic ``body/oakd/imu``; depth placeholder or StereoDepth stream; optional ``rgb_queue``."""
    next_imu = time.monotonic()
    next_depth = time.monotonic()
    depth_smooth_prev: list[np.ndarray | None] = [None]
    while not stop_ref[0]:
        _process_oakd_config_queue(session, config_pending, rgb_queue, oakd_cfg)
        now = time.monotonic()
        if now >= next_imu:
            # body/imu comes from body/imu_driver.py now; no synthetic publish here.
            next_imu += interval_s
        if now >= next_depth:
            if depth_queue is not None:
                df = _depth_drain_latest(depth_queue)
                if df is not None:
                    dw, dh = _depth_output_dims(oakd_cfg)
                    try:
                        zenoh_helpers.publish_json(
                            session,
                            "body/oakd/depth",
                            _depth_frame_to_stream_msg(
                                df, dw, dh, oakd_cfg, depth_smooth_prev
                            ),
                        )
                    except Exception as e:
                        print(f"[oakd] depth stream encode error: {e}", file=sys.stderr, flush=True)
            else:
                zenoh_helpers.publish_json(
                    session, "body/oakd/depth", schemas.oakd_depth_placeholder()
                )
            next_depth += depth_period
        time.sleep(0.02)


def _run_stub_imu_publisher(
    session: Any,
    oakd_cfg: dict[str, Any],
    interval_s: float,
    depth_period: float,
    stop_ref: list[bool],
    config_pending: queue.Queue[dict[str, Any]],
) -> None:
    """No DepthAI IMU (e.g. OAK-D-Lite Kickstarter boards without IMU): synthetic Zenoh traffic."""
    _run_synthetic_imu_depth_config_loop(
        session,
        oakd_cfg,
        interval_s,
        depth_period,
        stop_ref,
        config_pending,
        None,
    )


def _run_depthai_v2_rgb_only(
    oakd_cfg: dict[str, Any],
    dev_info: dai.DeviceInfo,
    session: Any,
    interval_s: float,
    depth_period: float,
    stop_ref: list[bool],
    config_pending: queue.Queue[dict[str, Any]],
) -> None:
    """IMU pipeline unusable: ColorCamera-only pipeline + synthetic IMU/depth on Zenoh (DepthAI v2)."""
    rw, rh = _rgb_preview_dims(oakd_cfg)
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(rw, rh)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xlink_rgb = pipeline.create(dai.node.XLinkOut)
    xlink_rgb.setStreamName("rgb")
    cam.preview.link(xlink_rgb.input)
    try:
        with dai.Device(pipeline, dev_info) as device:
            _log_oakd_calibration(device, oakd_cfg)
            rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
            print("[oakd] RGB-only pipeline running (synthetic IMU/depth on Zenoh)", flush=True)
            _run_synthetic_imu_depth_config_loop(
                session,
                oakd_cfg,
                interval_s,
                depth_period,
                stop_ref,
                config_pending,
                rgb_queue,
            )
    except Exception as e:
        print(f"[oakd] RGB-only pipeline failed: {e}", file=sys.stderr, flush=True)
        _run_stub_imu_publisher(
            session,
            oakd_cfg,
            interval_s,
            depth_period,
            stop_ref,
            config_pending,
        )


def _run_depthai_v3_rgb_only(
    oakd_cfg: dict[str, Any],
    dev_info: dai.DeviceInfo,
    session: Any,
    interval_s: float,
    depth_period: float,
    stop_ref: list[bool],
    config_pending: queue.Queue[dict[str, Any]],
) -> None:
    """IMU unusable: StereoDepth + RGB + synthetic IMU; depth stream + optional capture_rgb (DepthAI v3)."""
    device = dai.Device(dev_info)
    _log_oakd_calibration(device, oakd_cfg)
    pipeline = dai.Pipeline(device)
    depth_queue = _create_v3_stereo_depth_queue(pipeline, oakd_cfg)
    rgb_queue = _create_v3_rgb_queue(pipeline, oakd_cfg)

    try:
        pipeline.start()
    except Exception as e:
        print(f"[oakd] stereo+RGB pipeline start failed: {e}", file=sys.stderr, flush=True)
        _cleanup_v3_after_failed_pipeline_start(device, pipeline)
        _run_stub_imu_publisher(
            session,
            oakd_cfg,
            interval_s,
            depth_period,
            stop_ref,
            config_pending,
        )
        return

    print(
        "[oakd] stereo + RGB pipeline running (synthetic IMU; depth uint16 stream on body/oakd/depth)",
        flush=True,
    )
    with pipeline:
        _run_synthetic_imu_depth_config_loop(
            session,
            oakd_cfg,
            interval_s,
            depth_period,
            stop_ref,
            config_pending,
            rgb_queue,
            depth_queue,
        )


def _v3_imu_firmware_update(device: dai.Device, oakd_cfg: dict[str, Any], timeout_s: float = 120.0) -> None:
    """DepthAI v3: optional BNO IMU flash via Device API. No-op if disabled in config."""
    if not oakd_cfg.get("imu_enable_firmware_update", False):
        return
    print("[oakd] IMU firmware update: starting (keep USB connected; may take ~1–2 min)…", flush=True)
    device.startIMUFirmwareUpdate(False)
    deadline = time.monotonic() + timeout_s
    last_pct = -1.0
    while time.monotonic() < deadline:
        done, progress = device.getIMUFirmwareUpdateStatus()
        if int(progress) != int(last_pct):
            print(f"[oakd] IMU firmware update: {progress:.0f}%", flush=True)
            last_pct = progress
        if not done:
            time.sleep(0.15)
            continue
        if progress >= 99.99:
            print("[oakd] IMU firmware update: complete.", flush=True)
            return
        print(
            f"[oakd] IMU firmware update failed (device reported done with progress={progress}). "
            "If this OAK-D-Lite has no IMU chip, set oakd.imu_hardware_present to false.",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError("IMU firmware update failed")
    raise TimeoutError("IMU firmware update timed out")


def _run_depthai_v2(
    oakd_cfg: dict[str, Any],
    dev_info: dai.DeviceInfo,
    session: Any,
    interval_s: float,
    depth_period: float,
    stop_ref: list[bool],
    config_pending: queue.Queue[dict[str, Any]],
) -> None:
    def _continue() -> bool:
        return not stop_ref[0]

    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)
    xlink = pipeline.create(dai.node.XLinkOut)
    xlink.setStreamName("imu")
    _configure_imu_node(imu, oakd_cfg, depthai_v3=False)
    imu.out.link(xlink.input)

    rgb_wants = bool(oakd_cfg.get("rgb_enabled", False))
    if rgb_wants:
        rw, rh = _rgb_preview_dims(oakd_cfg)
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(rw, rh)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        xlink_rgb = pipeline.create(dai.node.XLinkOut)
        xlink_rgb.setStreamName("rgb")
        cam.preview.link(xlink_rgb.input)

    try:
        with dai.Device(pipeline, dev_info) as device:
            _log_oakd_calibration(device, oakd_cfg)
            imu_queue = device.getOutputQueue("imu", maxSize=50, blocking=False)
            rgb_queue = None
            if rgb_wants:
                rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
            _run_imu_loop(
                imu_queue,
                session,
                oakd_cfg,
                interval_s,
                depth_period,
                _continue,
                config_pending,
                rgb_queue,
            )
    except RuntimeError as e:
        if not _is_depthai_imu_missing_error(e):
            raise
        _log_imu_missing_fallback(e)
        if oakd_cfg.get("rgb_enabled", False):
            print("[oakd] IMU failed: trying RGB-only pipeline for capture_rgb", flush=True)
            _run_depthai_v2_rgb_only(
                oakd_cfg, dev_info, session, interval_s, depth_period, stop_ref, config_pending
            )
        else:
            _run_stub_imu_publisher(
                session,
                oakd_cfg,
                interval_s,
                depth_period,
                stop_ref,
                config_pending,
            )


def _run_depthai_v3(
    oakd_cfg: dict[str, Any],
    dev_info: dai.DeviceInfo,
    session: Any,
    interval_s: float,
    depth_period: float,
    stop_ref: list[bool],
    config_pending: queue.Queue[dict[str, Any]],
) -> None:
    # v3: bind an opened Device to the Pipeline, attach queues to node outputs, then start().
    device = dai.Device(dev_info)
    _log_oakd_calibration(device, oakd_cfg)
    _v3_imu_firmware_update(device, oakd_cfg)
    pipeline = dai.Pipeline(device)
    imu = pipeline.create(dai.node.IMU)
    _configure_imu_node(imu, oakd_cfg, depthai_v3=True)
    imu_queue = imu.out.createOutputQueue(maxSize=50, blocking=False)

    depth_queue_v3 = _create_v3_stereo_depth_queue(pipeline, oakd_cfg)
    rgb_queue_v3 = _create_v3_rgb_queue(pipeline, oakd_cfg)

    def _continue() -> bool:
        return pipeline.isRunning() and not stop_ref[0]

    try:
        pipeline.start()
    except RuntimeError as e:
        if not _is_depthai_imu_missing_error(e):
            raise
        _log_imu_missing_fallback(e)
        _cleanup_v3_after_failed_pipeline_start(device, pipeline)
        print(
            "[oakd] IMU failed: trying StereoDepth + optional RGB (synthetic IMU, depth stream)",
            flush=True,
        )
        _run_depthai_v3_rgb_only(
            oakd_cfg, dev_info, session, interval_s, depth_period, stop_ref, config_pending
        )
        return
    with pipeline:
        _run_imu_loop(
            imu_queue,
            session,
            oakd_cfg,
            interval_s,
            depth_period,
            _continue,
            config_pending,
            rgb_queue_v3,
            depth_queue_v3,
        )


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    oakd_cfg = body_cfg.get("oakd", {})
    if not oakd_cfg.get("imu_enabled", True):
        print("oakd_driver: imu_enabled is false; nothing to run", file=sys.stderr)
        sys.exit(1)

    interval_s = float(oakd_cfg.get("imu_aggregate_interval_s", 1.0))
    depth_fps = float(oakd_cfg.get("depth_fps", 15))
    depth_period = 1.0 / max(1.0, depth_fps)
    device_id = oakd_cfg.get("device_id")

    session = zenoh_helpers.open_session(body_cfg)
    config_pending: queue.Queue[dict[str, Any]] = queue.Queue()

    def on_oakd_config(_key: str, msg: dict[str, Any]) -> None:
        config_pending.put(msg)

    zenoh_helpers.declare_subscriber_json(session, "body/oakd/config", on_oakd_config)
    stop_ref: list[bool] = [False]

    def handle_sigterm(_sig: int, _frame: Any) -> None:
        stop_ref[0] = True

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    try:
        if not oakd_cfg.get("imu_hardware_present", True):
            print(
                "[oakd] imu_hardware_present=false — no DepthAI IMU node; "
                "Luxonis notes some OAK-D-Lite (e.g. Kickstarter) have no IMU chip.",
                flush=True,
            )
            dev_info = _pick_device(device_id if isinstance(device_id, str) else None)
            if dev_info is None:
                print(
                    "[oakd] no DepthAI device found; running synthetic IMU/depth stub.",
                    flush=True,
                )
                _run_stub_imu_publisher(
                    session, oakd_cfg, interval_s, depth_period, stop_ref, config_pending
                )
            else:
                api = "v3 (no XLinkOut)" if _depthai_is_v3() else "v2 (XLinkOut)"
                print(
                    f"[oakd] depthai API: {api}; library {getattr(dai, '__version__', '?')} "
                    "(RGB + depth pipeline, synthetic IMU)",
                    flush=True,
                )
                if _depthai_is_v3():
                    _run_depthai_v3_rgb_only(
                        oakd_cfg, dev_info, session, interval_s, depth_period, stop_ref, config_pending
                    )
                else:
                    _run_depthai_v2_rgb_only(
                        oakd_cfg, dev_info, session, interval_s, depth_period, stop_ref, config_pending
                    )
        else:
            dev_info = _pick_device(device_id if isinstance(device_id, str) else None)
            if dev_info is None:
                print(
                    "oakd_driver: no DepthAI device found (connect OAK-D-Lite, check udev / USB)",
                    file=sys.stderr,
                )
                sys.exit(1)

            api = "v3 (no XLinkOut)" if _depthai_is_v3() else "v2 (XLinkOut)"
            print(f"[oakd] depthai API: {api}; library {getattr(dai, '__version__', '?')}", flush=True)
            if _depthai_is_v3():
                _run_depthai_v3(
                    oakd_cfg, dev_info, session, interval_s, depth_period, stop_ref, config_pending
                )
            else:
                _run_depthai_v2(
                    oakd_cfg, dev_info, session, interval_s, depth_period, stop_ref, config_pending
                )
    finally:
        session.close()

    sys.exit(0)


if __name__ == "__main__":
    main()

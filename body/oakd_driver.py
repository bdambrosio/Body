"""OAK-D Lite driver: DepthAI IMU pipeline, Zenoh publish, optional depth placeholder.

Aggregates high-rate accel/gyro into **arithmetic means** over ``imu_aggregate_interval_s`` and
publishes ``body/oakd/imu`` at that rate. **Mean gyro ≠ integrated angle** and mean accel ≠
velocity change over the window; for strapdown-equivalent dead reckoning you would integrate
at full sensor rate on the Pi (or use fused orientation only as a slow state estimate).
See module docstring in body_project_spec.md §5.6.

Supports **DepthAI v2** (``dai.node.XLinkOut``) and **v3** (output queues on node outputs;
no XLink).
"""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable
from typing import Any

import depthai as dai

from body.lib import schemas, zenoh_helpers


def _depthai_is_v3() -> bool:
    return not hasattr(dai.node, "XLinkOut")


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


def _run_imu_loop(
    imu_queue: dai.MessageQueue,
    session: Any,
    oakd_cfg: dict[str, Any],
    interval_s: float,
    depth_period: float,
    continue_fn: Callable[[], bool],
) -> None:
    window_ax: list[float] = []
    window_ay: list[float] = []
    window_az: list[float] = []
    window_gx: list[float] = []
    window_gy: list[float] = []
    window_gz: list[float] = []
    last_quat: tuple[float, float, float, float] | None = None

    window_start = time.monotonic()
    next_depth = time.monotonic()

    while continue_fn():
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
            msg = schemas.oakd_imu_report(ts, (ax, ay, az), (gx, gy, gz), quat_wxyz=quat)
            zenoh_helpers.publish_json(session, "body/oakd/imu", msg)
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
            zenoh_helpers.publish_json(session, "body/oakd/depth", schemas.oakd_depth_placeholder())
            next_depth += depth_period

        if imu_data is None:
            time.sleep(0.001)


def _run_stub_imu_publisher(
    session: Any,
    oakd_cfg: dict[str, Any],
    interval_s: float,
    depth_period: float,
    stop_ref: list[bool],
) -> None:
    """No DepthAI IMU (e.g. OAK-D-Lite Kickstarter boards without IMU): synthetic Zenoh traffic."""
    next_imu = time.monotonic()
    next_depth = time.monotonic()
    while not stop_ref[0]:
        now = time.monotonic()
        if now >= next_imu:
            zenoh_helpers.publish_json(session, "body/oakd/imu", schemas.oakd_imu())
            print("[oakd] synthetic IMU (imu_hardware_present=false in config)", flush=True)
            next_imu += interval_s
        if now >= next_depth:
            zenoh_helpers.publish_json(
                session, "body/oakd/depth", schemas.oakd_depth_placeholder()
            )
            next_depth += depth_period
        time.sleep(0.02)


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
) -> None:
    def _continue() -> bool:
        return not stop_ref[0]

    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)
    xlink = pipeline.create(dai.node.XLinkOut)
    xlink.setStreamName("imu")
    _configure_imu_node(imu, oakd_cfg, depthai_v3=False)
    imu.out.link(xlink.input)

    with dai.Device(pipeline, dev_info) as device:
        imu_queue = device.getOutputQueue("imu", maxSize=50, blocking=False)
        _run_imu_loop(imu_queue, session, oakd_cfg, interval_s, depth_period, _continue)


def _run_depthai_v3(
    oakd_cfg: dict[str, Any],
    dev_info: dai.DeviceInfo,
    session: Any,
    interval_s: float,
    depth_period: float,
    stop_ref: list[bool],
) -> None:
    # v3: bind an opened Device to the Pipeline, attach queues to node outputs, then start().
    device = dai.Device(dev_info)
    _v3_imu_firmware_update(device, oakd_cfg)
    pipeline = dai.Pipeline(device)
    imu = pipeline.create(dai.node.IMU)
    _configure_imu_node(imu, oakd_cfg, depthai_v3=True)
    imu_queue = imu.out.createOutputQueue(maxSize=50, blocking=False)

    def _continue() -> bool:
        return pipeline.isRunning() and not stop_ref[0]

    try:
        pipeline.start()
    except RuntimeError as e:
        err = str(e)
        if "IMU not detected" in err or "IMU invalid settings" in err:
            print(
                "[oakd] DepthAI IMU error:\n"
                f"  {err}\n"
                "  If you have a BNO IMU: set oakd.imu_enable_firmware_update to true (DepthAI v3 "
                "uses device.startIMUFirmwareUpdate), USB stable, wait for 100%. "
                "If your OAK-D-Lite has no IMU (e.g. some Kickstarter units), set "
                "oakd.imu_hardware_present to false.",
                file=sys.stderr,
                flush=True,
            )
        raise
    with pipeline:
        _run_imu_loop(imu_queue, session, oakd_cfg, interval_s, depth_period, _continue)


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
            _run_stub_imu_publisher(session, oakd_cfg, interval_s, depth_period, stop_ref)
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
                _run_depthai_v3(oakd_cfg, dev_info, session, interval_s, depth_period, stop_ref)
            else:
                _run_depthai_v2(oakd_cfg, dev_info, session, interval_s, depth_period, stop_ref)
    finally:
        session.close()

    sys.exit(0)


if __name__ == "__main__":
    main()

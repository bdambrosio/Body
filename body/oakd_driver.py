"""OAK-D Lite driver: DepthAI IMU pipeline, Zenoh publish, optional depth placeholder.

Aggregates high-rate accel/gyro into **arithmetic means** over ``imu_aggregate_interval_s`` and
publishes ``body/oakd/imu`` at that rate. **Mean gyro ≠ integrated angle** and mean accel ≠
velocity change over the window; for strapdown-equivalent dead reckoning you would integrate
at full sensor rate on the Pi (or use fused orientation only as a slow state estimate).
See module docstring in body_project_spec.md §5.6.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Any

import depthai as dai

from body.lib import schemas, zenoh_helpers


def _build_imu_pipeline(oakd_cfg: dict[str, Any]) -> dai.Pipeline:
    accel_hz = int(oakd_cfg.get("imu_accel_hz", 500))
    gyro_hz = int(oakd_cfg.get("imu_gyro_hz", 400))
    rot_hz = int(oakd_cfg.get("imu_rotation_vector_hz", 400))
    use_rot = bool(oakd_cfg.get("rotation_vector_enabled", True))

    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)
    xlink = pipeline.create(dai.node.XLinkOut)
    xlink.setStreamName("imu")

    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, accel_hz)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, gyro_hz)
    if use_rot:
        imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, rot_hz)

    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    imu.out.link(xlink.input)
    return pipeline


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
    stop = False

    def handle_sigterm(_sig: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    pipeline = _build_imu_pipeline(oakd_cfg)
    dev_info = _pick_device(device_id if isinstance(device_id, str) else None)
    if dev_info is None:
        print(
            "oakd_driver: no DepthAI device found (connect OAK-D-Lite, check udev / USB)",
            file=sys.stderr,
        )
        session.close()
        sys.exit(1)

    window_ax: list[float] = []
    window_ay: list[float] = []
    window_az: list[float] = []
    window_gx: list[float] = []
    window_gy: list[float] = []
    window_gz: list[float] = []
    last_quat: tuple[float, float, float, float] | None = None

    window_start = time.monotonic()
    next_depth = time.monotonic()

    with dai.Device(pipeline, dev_info) as device:
        imu_queue = device.getOutputQueue("imu", maxSize=50, blocking=False)

        while not stop:
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
                q_str = f" quat=({quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f})" if quat else ""
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
                zenoh_helpers.publish_json(
                    session, "body/oakd/depth", schemas.oakd_depth_placeholder()
                )
                next_depth += depth_period

            if imu_data is None:
                time.sleep(0.001)

    session.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

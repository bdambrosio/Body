"""BNO085 IMU driver: i2c → SH-2 fusion → Zenoh body/imu.

Sole owner of the BNO085 i2c transaction. Reads accel, gyro, and the
configured rotation quaternion from the SH-2 FIFO, publishes one fused
report per cycle on ``body/imu`` at ``imu.publish_hz`` (default 100 Hz).

See docs/imu_driver_spec.md (producer spec) and docs/imu_integration_spec.md
(consumer contract). Hardware pin allocation in the former §4.
"""

from __future__ import annotations

import queue
import signal
import sys
import time
from typing import Any

from body.lib import schemas, zenoh_helpers

_VALID_CAL_ACTIONS = ("start", "save", "status")


def _pulse_reset(gpio: int, chip_idx: int) -> None:
    """Drive RST low ~20 ms then release. Waits 200 ms for SH-2 to boot."""
    try:
        import lgpio
    except ModuleNotFoundError:
        print(
            "imu_driver: lgpio unavailable — skipping RST pulse. Install python3-lgpio "
            "(sudo apt install python3-lgpio) or set imu.reset_gpio to a negative value to disable.",
            file=sys.stderr,
            flush=True,
        )
        return
    h = lgpio.gpiochip_open(chip_idx)
    try:
        ret = lgpio.gpio_claim_output(h, gpio, 1)
        if ret < 0:
            print(
                f"imu_driver: gpio_claim_output RST={gpio} failed: {ret}; skipping reset pulse.",
                file=sys.stderr,
                flush=True,
            )
            return
        lgpio.gpio_write(h, gpio, 0)
        time.sleep(0.02)
        lgpio.gpio_write(h, gpio, 1)
        time.sleep(0.20)
    finally:
        try:
            lgpio.gpio_free(h, gpio)
        except Exception:
            pass
        lgpio.gpiochip_close(h)


_RV_ACCURACY_Q = 2 ** (-12)


def _import_bno() -> tuple[Any, Any, Any, dict[str, int]]:
    """Return (board, busio, Bno085WithAccuracy, feature_ids) or exit with a diagnostic.

    The subclass captures the 2-bit calibration status from each sensor report and, for
    ``BNO_REPORT_ROTATION_VECTOR``, the 14-bit estimated-accuracy field (Q12 radians) that
    the stock Adafruit driver does not parse. See SH-2 Reference Manual §6.5.18.
    """
    try:
        import board
        import busio
        from adafruit_bno08x.i2c import BNO08X_I2C
        from adafruit_bno08x import (
            BNO_REPORT_ACCELEROMETER,
            BNO_REPORT_GYROSCOPE,
            BNO_REPORT_LINEAR_ACCELERATION,
            BNO_REPORT_ROTATION_VECTOR,
            BNO_REPORT_GAME_ROTATION_VECTOR,
        )
    except ModuleNotFoundError as e:
        print(
            f"imu_driver: missing dependency {e.name}. Install on the Pi venv with:\n"
            "  pip install adafruit-circuitpython-bno08x adafruit-blinka\n"
            "See docs/imu_driver_spec.md §9.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)
    features = {
        "accel": BNO_REPORT_ACCELEROMETER,
        "gyro": BNO_REPORT_GYROSCOPE,
        "linear_accel": BNO_REPORT_LINEAR_ACCELERATION,
        "rotation_vector": BNO_REPORT_ROTATION_VECTOR,
        "game_rotation_vector": BNO_REPORT_GAME_ROTATION_VECTOR,
    }
    from struct import unpack_from
    from adafruit_bno08x import _AVAIL_SENSOR_REPORTS, _separate_batch

    class Bno085WithAccuracy(BNO08X_I2C):
        """BNO08X subclass that captures rotation-vector accuracy and survives
        SH-2 reports the upstream Adafruit driver doesn't know about.

        SH-2 emits high-rate auxiliary reports (e.g. 0xDE Gyro Integrated RV)
        after reset that Adafruit's ``_separate_batch`` cannot parse, turning
        into ``KeyError`` that kill the whole packet. We catch that on our side
        and drop only the offending packet so fusion keeps ticking.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._status_by_report: dict[int, int] = {}
            self._rv_accuracy_rad: float | None = None
            super().__init__(*args, **kwargs)

        def _handle_packet(self, packet: Any) -> None:
            try:
                _separate_batch(packet, self._packet_slices)
            except (KeyError, RuntimeError):
                self._packet_slices.clear()
                return
            while self._packet_slices:
                self._process_report(*self._packet_slices.pop())

        def _process_report(self, report_id: int, report_bytes: bytearray) -> None:
            if report_id < 0xF0 and report_id not in _AVAIL_SENSOR_REPORTS:
                return
            if report_id < 0xF0 and len(report_bytes) >= 3:
                self._status_by_report[report_id] = report_bytes[2] & 0b11
                if report_id == BNO_REPORT_ROTATION_VECTOR and len(report_bytes) >= 14:
                    raw = unpack_from("<h", bytes(report_bytes), 12)[0]
                    self._rv_accuracy_rad = float(raw) * _RV_ACCURACY_Q
            super()._process_report(report_id, report_bytes)

    return board, busio, Bno085WithAccuracy, features


def _read_quat(bno: Any, mode: str) -> tuple[float, float, float, float] | None:
    """Return (i, j, k, real) for the active mode, or None if the lib has no sample yet."""
    if mode == "rotation_vector":
        q = bno.quaternion
    else:
        q = bno.game_quaternion
    if q is None:
        return None
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _read_accuracy_rad(bno: Any, mode: str, fallback_rad: float) -> float:
    """Return latest per-report accuracy in radians.

    For ``rotation_vector`` we use the Q12 estimated-accuracy field captured by
    ``Bno085WithAccuracy``. For ``game_rotation_vector`` the sensor does not emit an
    accuracy field (no absolute reference) so we return ``fallback_rad``.
    """
    if mode == "rotation_vector":
        val = getattr(bno, "_rv_accuracy_rad", None)
        if val is None:
            return fallback_rad
        try:
            return float(val)
        except (TypeError, ValueError):
            return fallback_rad
    return fallback_rad


def _read_report_status(bno: Any, report_id: int) -> int | None:
    status_map = getattr(bno, "_status_by_report", None)
    if not isinstance(status_map, dict):
        return None
    val = status_map.get(report_id)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _run_cal_action(bno: Any, action: str) -> None:
    """Execute a single BNO085 calibration command on the main thread.

    ``start`` begins mag+accel+gyro self-calibration; ``save`` writes the current
    DCD (dynamic calibration data) to the sensor's flash so it survives reboot;
    ``status`` polls the magnetometer accuracy (0=unreliable, 3=high) via a
    synchronous ME_GET_CAL command. Each action blocks the publish loop for one
    or more i2c round-trips — typically a few ms.
    """
    try:
        if action == "start":
            bno.begin_calibration()
            print(
                "imu_driver: calibration started (move the robot in a figure-8 for "
                "~10 s, then publish {\"action\":\"save\"} on body/imu/calibrate).",
                flush=True,
            )
        elif action == "save":
            bno.save_calibration_data()
            print(
                "imu_driver: calibration DCD saved to BNO085 flash. Survives power cycle.",
                flush=True,
            )
        elif action == "status":
            status = bno.calibration_status
            labels = {0: "unreliable", 1: "low", 2: "medium", 3: "high"}
            print(
                f"imu_driver: mag calibration_status={status} ({labels.get(int(status), '?')}).",
                flush=True,
            )
    except Exception as e:
        print(
            f"imu_driver: calibrate action {action!r} failed: {type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )


def _enable(bno: Any, feature_id: int, label: str) -> None:
    try:
        bno.enable_feature(feature_id)
    except Exception as e:
        print(
            f"imu_driver: enable_feature({label}) failed: {type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )
        raise


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    imu_cfg = body_cfg.get("imu", {})

    if not bool(imu_cfg.get("enabled", True)):
        print("imu_driver: imu.enabled is false; exiting.", file=sys.stderr, flush=True)
        sys.exit(1)

    address = int(imu_cfg.get("i2c_address", 0x4B))
    publish_hz = float(imu_cfg.get("publish_hz", 100))
    publish_period = 1.0 / max(1.0, publish_hz)
    fusion_mode = str(imu_cfg.get("fusion_mode", "rotation_vector"))
    fusion_fallback = str(imu_cfg.get("fusion_fallback", "game_rotation_vector"))
    mag_accuracy_fallback_rad = float(imu_cfg.get("mag_accuracy_fallback_rad", 0.087))
    mag_accuracy_fallback_count = int(imu_cfg.get("mag_accuracy_fallback_count", 20))
    calibration_threshold_rad = float(imu_cfg.get("calibration_stable_threshold_rad", 0.087))
    grv_accuracy_constant_rad = float(imu_cfg.get("game_rotation_vector_accuracy_rad", 0.175))
    settle_time_s = float(imu_cfg.get("settle_time_s", 2.0))
    linear_accel_enabled = bool(imu_cfg.get("linear_accel_enabled", False))
    reset_gpio = int(imu_cfg.get("reset_gpio", 25))
    gpio_chip = int(imu_cfg.get("gpio_chip", 0))

    if fusion_mode not in ("rotation_vector", "game_rotation_vector"):
        print(
            f"imu_driver: invalid imu.fusion_mode={fusion_mode!r}; expected "
            "'rotation_vector' or 'game_rotation_vector'.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    if reset_gpio >= 0:
        print(f"imu_driver: pulsing RST on BCM {reset_gpio}...", flush=True)
        _pulse_reset(reset_gpio, gpio_chip)

    board_mod, busio_mod, BNO08X_I2C, features = _import_bno()

    try:
        i2c = busio_mod.I2C(board_mod.SCL, board_mod.SDA)
    except Exception as e:
        print(
            f"imu_driver: failed to open i2c bus: {type(e).__name__}: {e}. "
            "Check dtparam=i2c_arm=on in /boot/firmware/config.txt and `ls /dev/i2c-*`.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    try:
        bno = BNO08X_I2C(i2c, address=address)
    except Exception as e:
        print(
            f"imu_driver: BNO085 init failed at 0x{address:02x}: {type(e).__name__}: {e}. "
            "Run `sudo i2cdetect -y 1` — expect 0x4b (ADDR open) or 0x4a (ADDR shorted). "
            "See docs/imu_driver_spec.md §3.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    active_mode = fusion_mode
    _enable(bno, features["accel"], "accelerometer")
    _enable(bno, features["gyro"], "gyroscope")
    _enable(bno, features[active_mode], active_mode)
    if linear_accel_enabled:
        _enable(bno, features["linear_accel"], "linear_acceleration")

    print(
        f"imu_driver: BNO085 ready at 0x{address:02x}, mode={active_mode}, "
        f"publish_hz={publish_hz:.0f}, linear_accel={'on' if linear_accel_enabled else 'off'}.",
        flush=True,
    )

    session = zenoh_helpers.open_session(body_cfg)
    cal_actions: queue.Queue[str] = queue.Queue()

    def on_calibrate(_key: str, msg: dict[str, Any]) -> None:
        action = str(msg.get("action", "")).strip().lower()
        if action not in _VALID_CAL_ACTIONS:
            print(
                f"imu_driver: body/imu/calibrate ignored, unknown action {action!r}. "
                f"Expected one of {_VALID_CAL_ACTIONS}.",
                file=sys.stderr,
                flush=True,
            )
            return
        cal_actions.put(action)

    zenoh_helpers.declare_subscriber_json(session, "body/imu/calibrate", on_calibrate)

    stop = False

    def handle_sigterm(_sig: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    start_mono = time.monotonic()
    settled = False
    last_settle_log = 0.0
    fallback_count = 0
    fallback_done = False
    next_tick = time.monotonic()

    try:
        while not stop:
            try:
                while True:
                    action = cal_actions.get_nowait()
                    _run_cal_action(bno, action)
            except queue.Empty:
                pass

            try:
                accel = bno.acceleration
                gyro = bno.gyro
                quat = _read_quat(bno, active_mode)
                accuracy_rad = _read_accuracy_rad(
                    bno, active_mode, grv_accuracy_constant_rad
                )
                lin_accel = bno.linear_acceleration if linear_accel_enabled else None
                calib = _read_report_status(bno, features[active_mode])
            except Exception as e:
                print(
                    f"imu_driver: i2c read error: {type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(0.05)
                next_tick = time.monotonic()
                continue

            now_mono = time.monotonic()
            now_wall = time.time()

            if accel is None or gyro is None or quat is None:
                if now_mono - last_settle_log > 1.0:
                    print(
                        "imu_driver: waiting for first full report set…",
                        flush=True,
                    )
                    last_settle_log = now_mono
                time.sleep(0.01)
                next_tick = time.monotonic()
                continue

            if not settled:
                elapsed = now_mono - start_mono
                if elapsed >= settle_time_s and accuracy_rad <= calibration_threshold_rad:
                    settled = True
                    print(
                        f"imu_driver: settled after {elapsed:.2f}s, "
                        f"accuracy_rad={accuracy_rad:.4f}. Publishing body/imu.",
                        flush=True,
                    )
                else:
                    if now_mono - last_settle_log >= 1.0:
                        print(
                            f"imu_driver: calibrating, elapsed={elapsed:.1f}s, "
                            f"accuracy_rad={accuracy_rad:.4f}, "
                            f"threshold={calibration_threshold_rad:.4f} "
                            "(hold the robot still).",
                            flush=True,
                        )
                        last_settle_log = now_mono

            if (
                settled
                and not fallback_done
                and active_mode == "rotation_vector"
                and fusion_fallback == "game_rotation_vector"
                and accuracy_rad > mag_accuracy_fallback_rad
            ):
                fallback_count += 1
                if fallback_count >= mag_accuracy_fallback_count:
                    print(
                        f"imu_driver: rotation_vector accuracy exceeded "
                        f"{mag_accuracy_fallback_rad:.4f} rad for "
                        f"{fallback_count} consecutive samples; switching to "
                        "game_rotation_vector for the remainder of this session.",
                        flush=True,
                    )
                    _enable(bno, features["game_rotation_vector"], "game_rotation_vector")
                    active_mode = "game_rotation_vector"
                    fallback_done = True
                    fallback_count = 0
            else:
                fallback_count = 0

            if settled:
                ax, ay, az = (float(accel[0]), float(accel[1]), float(accel[2]))
                gx, gy, gz = (float(gyro[0]), float(gyro[1]), float(gyro[2]))
                i, j, k, real = quat
                quat_wxyz = (real, i, j, k)
                lin_tuple: tuple[float, float, float] | None = None
                if lin_accel is not None:
                    lin_tuple = (
                        float(lin_accel[0]),
                        float(lin_accel[1]),
                        float(lin_accel[2]),
                    )
                msg = schemas.imu_report(
                    ts=now_wall,
                    accel_xyz=(ax, ay, az),
                    gyro_xyz=(gx, gy, gz),
                    quat_wxyz=quat_wxyz,
                    fusion_mode=active_mode,
                    fusion_accuracy_rad=accuracy_rad,
                    linear_accel_xyz=lin_tuple,
                    calibration_status=calib,
                )
                zenoh_helpers.publish_json(session, "body/imu", msg)

            next_tick += publish_period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    finally:
        try:
            session.close()
        except Exception:
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()

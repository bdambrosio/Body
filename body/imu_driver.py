"""BNO085 IMU driver: i2c → SH-2 fusion → Zenoh body/imu.

Sole owner of the BNO085 i2c transaction. Enables accel, gyro and the
configured rotation-vector report at ``imu.report_interval_ms`` (default
10 ms), drains the SH-2 FIFO every ``imu.poll_interval_ms`` and publishes on
``body/imu`` exactly once per *new* rotation report, stamped with that
report's arrival time and carrying ``session_id`` + ``seq``. If no new report
arrives for ``imu.stale_reinit_ms`` the driver stops publishing, re-inits the
sensor under a new ``session_id`` and, after ``imu.max_reinit_attempts``,
exits so the launcher restarts it.

See docs/imu_driver_spec.md (producer spec) and docs/imu_integration_spec.md
(consumer contract). Hardware pin allocation in the former §4.
"""

from __future__ import annotations

import queue
import secrets
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
            # Freshness bookkeeping (see _process_report): per report id, the
            # wall-clock time the report was parsed, a monotonically increasing
            # count of reports received, and the last SH-2 sequence byte.
            self._report_arrival: dict[int, float] = {}
            self._report_count: dict[int, int] = {}
            self._report_sh2_seq: dict[int, int] = {}
            self._sh2_resets = 0
            super().__init__(*args, **kwargs)

        def _handle_packet(self, packet: Any) -> None:
            # Channel 0 (SHTP advertisement) and 1 (executable: reset
            # complete) carry no sensor reports. An unsolicited "reset
            # complete" means the hub dropped every enabled report.
            channel = getattr(getattr(packet, "header", None), "channel_number", None)
            if channel is None:
                channel = getattr(packet, "channel_number", None)
            if channel in (0, 1):
                try:
                    data = packet.data
                    if channel == 1 and len(data) >= 1 and int(data[0]) == 1:
                        self._sh2_resets += 1
                except Exception:
                    pass
                return
            try:
                _separate_batch(packet, self._packet_slices)
            except (KeyError, RuntimeError, IndexError):
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
            if report_id < 0xF0:
                self._report_arrival[report_id] = time.time()
                self._report_count[report_id] = self._report_count.get(report_id, 0) + 1
                if len(report_bytes) >= 2:
                    self._report_sh2_seq[report_id] = int(report_bytes[1])

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


_BNO_CHANNEL_CONTROL = 2


def _enable_with_interval_legacy(bno: Any, feature_id: int, interval_us: int, label: str) -> None:
    """Older adafruit_bno08x ``enable_feature`` has no ``report_interval`` kwarg (fixed
    50 ms default). Build and send the Set Feature command ourselves when the private
    helpers exist; otherwise fall back to the library default and say so."""
    build = getattr(bno, "_get_feature_enable_report", None)
    send = getattr(bno, "_send_packet", None)
    process = getattr(bno, "_process_available_packets", None)
    if build is None or send is None or process is None:
        print(
            f"imu_driver: adafruit_bno08x cannot set report_interval for {label}; using the "
            "library default (~20 Hz). Upgrade adafruit-circuitpython-bno08x.",
            file=sys.stderr,
            flush=True,
        )
        bno.enable_feature(feature_id)
        return
    try:
        report = build(feature_id, report_interval=interval_us)
    except TypeError:
        report = build(feature_id, interval_us)
    send(_BNO_CHANNEL_CONTROL, report)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            process(max_packets=10)
        except TypeError:
            process()
        if feature_id in getattr(bno, "_readings", {}):
            return
    raise RuntimeError(f"feature {label} did not start reporting")


def _enable(bno: Any, feature_id: int, label: str, interval_ms: float | None = None) -> None:
    """Enable one SH-2 report, at ``interval_ms`` when given (library default otherwise)."""
    try:
        if interval_ms is None or interval_ms <= 0:
            bno.enable_feature(feature_id)
            return
        interval_us = max(1000, int(round(float(interval_ms) * 1000.0)))
        try:
            bno.enable_feature(feature_id, report_interval=interval_us)
        except TypeError:
            _enable_with_interval_legacy(bno, feature_id, interval_us, label)
    except Exception as e:
        print(
            f"imu_driver: enable_feature({label}) failed: {type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )
        raise


def _new_session_id() -> str:
    return secrets.token_hex(4)


class ReportFreshness:
    """Track whether the orientation report stream is delivering *new* reports.

    ``poll(count, arrival_wall, now_wall)`` takes the per-report-id counter and
    arrival time recorded by ``Bno085WithAccuracy._process_report`` and returns
    ``(is_new, is_stale)``. ``is_new`` is True exactly once per new report;
    ``is_stale`` is True once no new report has arrived for ``stale_s`` (or, before
    the first report after (re)init, for ``first_report_timeout_s``).
    """

    def __init__(self, stale_s: float, first_report_timeout_s: float, now_wall: float) -> None:
        self.stale_s = max(0.01, float(stale_s))
        self.first_report_timeout_s = max(self.stale_s, float(first_report_timeout_s))
        self.reset(now_wall)

    def reset(self, now_wall: float) -> None:
        self._armed_at = now_wall
        self._last_count: int | None = None
        self.last_arrival: float | None = None

    def poll(self, count: int | None, arrival_wall: float | None, now_wall: float) -> tuple[bool, bool]:
        new = False
        if count is not None and arrival_wall is not None:
            # Any change counts (a lower count means a new library object).
            new = self._last_count is None or count != self._last_count
            if new:
                self._last_count = count
                self.last_arrival = arrival_wall
        if self.last_arrival is None:
            stale = (now_wall - self._armed_at) > self.first_report_timeout_s
        else:
            stale = (now_wall - self.last_arrival) > self.stale_s
        return new, stale


class SettleGate:
    """Boot-time settle: wait ``settle_time_s``, accuracy within threshold, and
    ``still_reports`` consecutive fresh reports with |gyro| below ``still_gyro_radps``
    (so a restart while moving does not lock in a biased yaw). The stillness
    requirement is waived after ``max_wait_s`` (``forced`` is then True).

    ``reset(now, quick=True)`` is for a mid-run re-init: the robot is usually
    moving and consumers re-baseline yaw on the new session, so waiting for
    stillness only blanks heading for ``max_wait_s``. Quick mode settles after
    ``quick_time_s`` with no stillness requirement."""

    def __init__(
        self,
        settle_time_s: float,
        accuracy_threshold_rad: float,
        still_gyro_radps: float,
        still_reports: int,
        max_wait_s: float,
        now_mono: float,
        quick_time_s: float = 0.3,
    ) -> None:
        self.settle_time_s = float(settle_time_s)
        self.accuracy_threshold_rad = float(accuracy_threshold_rad)
        self.still_gyro_radps = float(still_gyro_radps)
        self.still_reports = max(0, int(still_reports))
        self.max_wait_s = max(self.settle_time_s, float(max_wait_s))
        self.quick_time_s = max(0.0, float(quick_time_s))
        self.reset(now_mono)

    def reset(self, now_mono: float, quick: bool = False) -> None:
        self.start_mono = now_mono
        self.still_count = 0
        self.settled = False
        self.forced = False
        self.quick = bool(quick)

    def update(self, now_mono: float, accuracy_rad: float, gyro_norm_radps: float) -> bool:
        if self.settled:
            return True
        if self.still_gyro_radps <= 0.0 or gyro_norm_radps <= self.still_gyro_radps:
            self.still_count += 1
        else:
            self.still_count = 0
        elapsed = now_mono - self.start_mono
        min_time_s = self.quick_time_s if self.quick else self.settle_time_s
        if elapsed < min_time_s or accuracy_rad > self.accuracy_threshold_rad:
            return False
        if self.quick or self.still_count >= self.still_reports:
            self.settled = True
        elif elapsed >= self.max_wait_s:
            self.settled = True
            self.forced = True
        return self.settled


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    imu_cfg = body_cfg.get("imu", {})

    if not bool(imu_cfg.get("enabled", True)):
        print("imu_driver: imu.enabled is false; exiting.", file=sys.stderr, flush=True)
        sys.exit(1)

    address = int(imu_cfg.get("i2c_address", 0x4B))
    fusion_mode = str(imu_cfg.get("fusion_mode", "rotation_vector"))
    fusion_fallback = str(imu_cfg.get("fusion_fallback", "game_rotation_vector"))
    mag_accuracy_fallback_rad = float(imu_cfg.get("mag_accuracy_fallback_rad", 0.087))
    mag_accuracy_fallback_count = int(imu_cfg.get("mag_accuracy_fallback_count", 20))
    calibration_threshold_rad = float(imu_cfg.get("calibration_stable_threshold_rad", 0.087))
    grv_accuracy_constant_rad = float(imu_cfg.get("game_rotation_vector_accuracy_rad", 0.175))
    settle_time_s = float(imu_cfg.get("settle_time_s", 2.0))
    settle_still_gyro = float(imu_cfg.get("settle_still_gyro_rad_s", 0.05))
    settle_still_reports = int(imu_cfg.get("settle_still_reports", 50))
    settle_max_wait_s = float(imu_cfg.get("settle_max_wait_s", 10.0))
    reinit_settle_time_s = float(imu_cfg.get("reinit_settle_time_s", 0.3))
    linear_accel_enabled = bool(imu_cfg.get("linear_accel_enabled", False))
    mag_when_idle_enabled = bool(imu_cfg.get("mag_when_idle_enabled", False))
    mag_idle_pwm_threshold = float(imu_cfg.get("mag_idle_pwm_threshold", 0.02))
    mag_idle_settle_ms = float(imu_cfg.get("mag_idle_settle_ms", 300.0))
    mag_idle_max_accuracy_rad = float(
        imu_cfg.get("mag_idle_max_accuracy_rad", mag_accuracy_fallback_rad)
    )
    reset_gpio = int(imu_cfg.get("reset_gpio", 25))
    gpio_chip = int(imu_cfg.get("gpio_chip", 0))
    # SH-2 report intervals (ms). The library default is 50 ms (~20 Hz).
    rot_interval_ms = float(imu_cfg.get("report_interval_ms", 10.0))
    gyro_interval_ms = float(imu_cfg.get("gyro_report_interval_ms", 10.0))
    accel_interval_ms = float(imu_cfg.get("accel_report_interval_ms", 10.0))
    mag_rv_interval_ms = float(imu_cfg.get("mag_report_interval_ms", 50.0))
    poll_s = max(0.0005, float(imu_cfg.get("poll_interval_ms", 2.0)) / 1000.0)
    stale_s = float(imu_cfg.get("stale_reinit_ms", 200.0)) / 1000.0
    first_report_timeout_s = float(imu_cfg.get("first_report_timeout_s", 2.0))
    max_reinit_attempts = max(1, int(imu_cfg.get("max_reinit_attempts", 3)))

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

    active_mode = fusion_mode

    def _init_sensor() -> Any:
        """Construct the SH-2 object and enable reports at the configured rates."""
        dev = BNO08X_I2C(i2c, address=address)
        _enable(dev, features["accel"], "accelerometer", accel_interval_ms)
        _enable(dev, features["gyro"], "gyroscope", gyro_interval_ms)
        _enable(dev, features[active_mode], active_mode, rot_interval_ms)
        if mag_when_idle_enabled and active_mode == "game_rotation_vector":
            _enable(
                dev, features["rotation_vector"], "rotation_vector (mag-when-idle)",
                mag_rv_interval_ms,
            )
        if linear_accel_enabled:
            _enable(dev, features["linear_accel"], "linear_acceleration", accel_interval_ms)
        return dev

    try:
        bno = _init_sensor()
    except Exception as e:
        print(
            f"imu_driver: BNO085 init failed at 0x{address:02x}: {type(e).__name__}: {e}. "
            "Run `sudo i2cdetect -y 1` — expect 0x4b (ADDR open) or 0x4a (ADDR shorted). "
            "See docs/imu_driver_spec.md §3.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    session_id = _new_session_id()
    print(
        f"imu_driver: BNO085 ready at 0x{address:02x}, mode={active_mode}, "
        f"report_interval={rot_interval_ms:.1f}ms (gyro {gyro_interval_ms:.1f}, "
        f"accel {accel_interval_ms:.1f}), stale_reinit={stale_s * 1000:.0f}ms, "
        f"linear_accel={'on' if linear_accel_enabled else 'off'}, "
        f"mag_when_idle={'on' if mag_when_idle_enabled else 'off'}, session={session_id}.",
        flush=True,
    )

    session = zenoh_helpers.open_session(body_cfg)
    cal_actions: queue.Queue[str] = queue.Queue()
    motor_ref: dict[str, float] = {
        "left_pwm": 0.0,
        "right_pwm": 0.0,
        "last_motion_mono": time.monotonic(),
        "seen": 0.0,
    }

    def on_motor_state(_key: str, msg: dict[str, Any]) -> None:
        left_pwm = abs(float(msg.get("left_pwm", 0.0)))
        right_pwm = abs(float(msg.get("right_pwm", 0.0)))
        motor_ref["left_pwm"] = left_pwm
        motor_ref["right_pwm"] = right_pwm
        motor_ref["seen"] = 1.0
        if left_pwm > mag_idle_pwm_threshold or right_pwm > mag_idle_pwm_threshold:
            motor_ref["last_motion_mono"] = time.monotonic()

    if mag_when_idle_enabled:
        zenoh_helpers.declare_subscriber_json(session, "body/motor_state", on_motor_state)

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

    publish_unsettled = False
    last_settle_log = 0.0
    last_err_log = 0.0
    fallback_count = 0
    fallback_done = False
    mag_idle_settle_s = max(0.0, mag_idle_settle_ms / 1000.0)
    seq = 0
    reinit_failures = 0
    resets_baseline = int(getattr(bno, "_sh2_resets", 0))
    fresh = ReportFreshness(stale_s, first_report_timeout_s, time.time())
    settle = SettleGate(
        settle_time_s,
        calibration_threshold_rad,
        settle_still_gyro,
        settle_still_reports,
        settle_max_wait_s,
        time.monotonic(),
        quick_time_s=reinit_settle_time_s,
    )

    def _motors_idle() -> bool:
        if motor_ref["seen"] <= 0.0:
            return False  # never heard from motor_controller: do not assume idle
        if (
            motor_ref["left_pwm"] > mag_idle_pwm_threshold
            or motor_ref["right_pwm"] > mag_idle_pwm_threshold
        ):
            return False
        return (time.monotonic() - motor_ref["last_motion_mono"]) >= mag_idle_settle_s

    try:
        while not stop:
            try:
                while True:
                    action = cal_actions.get_nowait()
                    _run_cal_action(bno, action)
                    if action == "start" and not publish_unsettled:
                        publish_unsettled = True
                        print(
                            "imu_driver: publishing unsettled body/imu during calibration.",
                            flush=True,
                        )
            except queue.Empty:
                pass

            count: int | None = None
            arrival: float | None = None
            sh2_resets = resets_baseline
            mag_sample: tuple[tuple[float, float, float, float], float] | None = None
            accel = gyro = quat = lin_accel = None
            calib: int | None = None
            accuracy_rad = grv_accuracy_constant_rad
            try:
                # Every property access drains the SH-2 FIFO; the subclass stamps
                # each report's arrival, so ``count``/``arrival`` identify new ones.
                quat = _read_quat(bno, active_mode)
                accel = bno.acceleration
                gyro = bno.gyro
                accuracy_rad = _read_accuracy_rad(bno, active_mode, grv_accuracy_constant_rad)
                lin_accel = bno.linear_acceleration if linear_accel_enabled else None
                rid = features[active_mode]
                calib = _read_report_status(bno, rid)
                count = getattr(bno, "_report_count", {}).get(rid)
                arrival = getattr(bno, "_report_arrival", {}).get(rid)
                sh2_resets = int(getattr(bno, "_sh2_resets", resets_baseline))
                if mag_when_idle_enabled and active_mode == "game_rotation_vector":
                    rv_id = features["rotation_vector"]
                    rv_arrival = getattr(bno, "_report_arrival", {}).get(rv_id)
                    if rv_arrival is not None and time.time() - rv_arrival < 0.25:
                        rv_quat = _read_quat(bno, "rotation_vector")
                        if rv_quat is not None:
                            mag_sample = (
                                rv_quat,
                                _read_accuracy_rad(
                                    bno, "rotation_vector", mag_accuracy_fallback_rad
                                ),
                            )
            except Exception as e:
                now_m = time.monotonic()
                if now_m - last_err_log > 1.0:
                    print(
                        f"imu_driver: i2c read error: {type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_err_log = now_m
                count = None

            now_wall = time.time()
            now_mono = time.monotonic()
            is_new, stale = fresh.poll(count, arrival, now_wall)
            if sh2_resets > resets_baseline:
                print("imu_driver: unsolicited SH-2 reset detected.", file=sys.stderr, flush=True)
                stale = True

            if stale:
                # No new orientation report: publishing has already stopped (we
                # only publish new reports). Re-initialise; give up after N tries
                # so the launcher restarts the process.
                age_ms = (
                    (now_wall - fresh.last_arrival) * 1000.0
                    if fresh.last_arrival is not None
                    else float("nan")
                )
                reinit_failures += 1
                if reinit_failures > max_reinit_attempts:
                    print(
                        f"imu_driver: {active_mode} still stale after {max_reinit_attempts} "
                        "re-init attempts; exiting for launcher restart.",
                        file=sys.stderr,
                        flush=True,
                    )
                    sys.exit(1)
                print(
                    f"imu_driver: {active_mode} stale (last report {age_ms:.0f} ms ago); "
                    f"re-initialising BNO085 (attempt {reinit_failures}/{max_reinit_attempts}).",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    if reset_gpio >= 0:
                        _pulse_reset(reset_gpio, gpio_chip)
                    bno = _init_sensor()
                except Exception as e:
                    print(
                        f"imu_driver: re-init failed: {type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(0.2)
                # New yaw origin → new session; consumers re-baseline on the change.
                session_id = _new_session_id()
                seq = 0
                resets_baseline = int(getattr(bno, "_sh2_resets", 0))
                fresh.reset(time.time())
                settle.reset(time.monotonic(), quick=True)
                fallback_count = 0
                print(f"imu_driver: new session {session_id}.", flush=True)
                continue

            if not is_new:
                time.sleep(poll_s)
                continue
            reinit_failures = 0

            if accel is None or gyro is None or quat is None:
                if now_mono - last_settle_log > 1.0:
                    print(
                        "imu_driver: waiting for first full report set…",
                        flush=True,
                    )
                    last_settle_log = now_mono
                time.sleep(poll_s)
                continue

            gx, gy, gz = (float(gyro[0]), float(gyro[1]), float(gyro[2]))
            if not settle.settled:
                gyro_norm = (gx * gx + gy * gy + gz * gz) ** 0.5
                if settle.update(now_mono, accuracy_rad, gyro_norm):
                    elapsed = now_mono - settle.start_mono
                    print(
                        f"imu_driver: settled after {elapsed:.2f}s"
                        f"{' (re-init)' if settle.quick else ''}, "
                        f"accuracy_rad={accuracy_rad:.4f}"
                        f"{' (stillness not reached; forced after max wait)' if settle.forced else ''}"
                        ". Publishing body/imu.",
                        flush=True,
                    )
                elif now_mono - last_settle_log >= 1.0:
                    elapsed = now_mono - settle.start_mono
                    print(
                        f"imu_driver: calibrating, elapsed={elapsed:.1f}s, "
                        f"accuracy_rad={accuracy_rad:.4f}, "
                        f"threshold={calibration_threshold_rad:.4f}, "
                        f"still_reports={settle.still_count}/{settle.still_reports} "
                        "(hold the robot still).",
                        flush=True,
                    )
                    last_settle_log = now_mono
            settled = settle.settled

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
                    try:
                        _enable(
                            bno, features["game_rotation_vector"], "game_rotation_vector",
                            rot_interval_ms,
                        )
                    except Exception:
                        pass  # stale detection re-initialises if GRV never arrives
                    active_mode = "game_rotation_vector"
                    fallback_done = True
                    fallback_count = 0
                    # Different quaternion origin → new session for consumers.
                    session_id = _new_session_id()
                    seq = 0
                    fresh.reset(time.time())
            else:
                fallback_count = 0

            if (settled or publish_unsettled) and arrival is not None:
                ax, ay, az = (float(accel[0]), float(accel[1]), float(accel[2]))
                i, j, k, real = quat
                quat_wxyz = (real, i, j, k)
                lin_tuple: tuple[float, float, float] | None = None
                if lin_accel is not None:
                    lin_tuple = (
                        float(lin_accel[0]),
                        float(lin_accel[1]),
                        float(lin_accel[2]),
                    )
                mag_quat_wxyz: tuple[float, float, float, float] | None = None
                mag_accuracy: float | None = None
                mag_valid: bool | None = None
                if mag_when_idle_enabled and active_mode == "game_rotation_vector":
                    mag_valid = False
                    if settled and _motors_idle() and mag_sample is not None:
                        (i_rv, j_rv, k_rv, real_rv), mag_accuracy = mag_sample
                        mag_quat_wxyz = (real_rv, i_rv, j_rv, k_rv)
                        mag_valid = mag_accuracy <= mag_idle_max_accuracy_rad
                msg = schemas.imu_report(
                    ts=arrival,
                    accel_xyz=(ax, ay, az),
                    gyro_xyz=(gx, gy, gz),
                    quat_wxyz=quat_wxyz,
                    fusion_mode=active_mode,
                    fusion_accuracy_rad=accuracy_rad,
                    linear_accel_xyz=lin_tuple,
                    calibration_status=calib,
                    mag_quat_wxyz=mag_quat_wxyz,
                    mag_accuracy_rad=mag_accuracy,
                    mag_valid=mag_valid,
                    session_id=session_id,
                    seq=seq,
                )
                zenoh_helpers.publish_json(session, "body/imu", msg)
                seq += 1

            time.sleep(poll_s)
    finally:
        try:
            session.close()
        except Exception:
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()

"""Start Body processes, prefix logs, restart with backoff, graceful shutdown.

Also (config ``launcher``):
  * liveness supervision — a process whose primary topic was flowing and then
    goes stale (hung, not exited) is SIGTERM'd, SIGKILL'd after a grace, and
    respawned with the usual backoff;
  * motor failsafe — whenever motor_controller exits (or is found hung) the
    launcher zeroes/disables the motor PWM channels via sysfs, because kernel
    PWM state outlives the process.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from body.lib.zenoh_helpers import load_body_config, repo_root

PROCESSES: list[dict[str, Any]] = [
    {"name": "watchdog", "cmd": [sys.executable, "-m", "body.watchdog"]},
    {"name": "motor_controller", "cmd": [sys.executable, "-m", "body.motor_controller"]},
    {"name": "lidar_driver", "cmd": [sys.executable, "-m", "body.lidar_driver"]},
    {"name": "imu_driver", "cmd": [sys.executable, "-m", "body.imu_driver"]},
    {"name": "oakd_driver", "cmd": [sys.executable, "-m", "body.oakd_driver"]},
    {"name": "local_map", "cmd": [sys.executable, "-m", "body.local_map"]},
    {"name": "local_drive", "cmd": [sys.executable, "-m", "body.local_drive"]},
]

MOTOR_PROCESS = "motor_controller"

# Defaults for ``launcher.liveness`` (process -> {topic, timeout_ms}).
DEFAULT_LIVENESS: dict[str, dict[str, Any]] = {
    "watchdog": {"topic": "body/status", "timeout_ms": 5000},
    "motor_controller": {"topic": "body/odom", "timeout_ms": 1000},
    "lidar_driver": {"topic": "body/lidar/scan", "timeout_ms": 3000},
    "imu_driver": {"topic": "body/imu", "timeout_ms": 5000},
    "oakd_driver": {"topic": "body/oakd/depth", "timeout_ms": 5000},
    "local_map": {"topic": "body/map/local_2p5d", "timeout_ms": 5000},
    "local_drive": {"topic": "body/drive/status", "timeout_ms": 3000},
}


@dataclass
class Managed:
    name: str
    cmd: list[str]
    proc: subprocess.Popen[bytes] | None = None
    failures: int = 0
    stable_since: float | None = None
    next_start_monotonic: float = field(default_factory=time.monotonic)
    kill_deadline: float | None = None


class LivenessMonitor:
    """Tracks per-process topic freshness (pure; times are monotonic seconds).

    A process is only *armed* once its topic has arrived since its latest
    start, so a driver whose device is absent (never publishes) is left alone
    rather than restart-looped; only "was flowing, then stopped" counts.
    Staleness is also suppressed unless some other monitored topic arrived
    within ``transport_window_s`` — if nothing at all is arriving the router
    or link is down, and restarting every process would not help.
    """

    def __init__(self, specs: dict[str, tuple[str, float]], transport_window_s: float = 2.0) -> None:
        self.specs = dict(specs)  # name -> (topic, timeout_s)
        self.transport_window_s = transport_window_s
        self.last_seen: dict[str, float] = {}
        self.armed: dict[str, bool] = {name: False for name in self.specs}

    def topics(self) -> set[str]:
        return {t for t, _ in self.specs.values()}

    def on_sample(self, topic: str, now: float) -> None:
        self.last_seen[topic] = now
        for name, (t, _) in self.specs.items():
            if t == topic:
                self.armed[name] = True

    def on_start(self, name: str) -> None:
        if name in self.armed:
            self.armed[name] = False

    def stale(self, now: float, running: set[str]) -> list[str]:
        out: list[str] = []
        for name, (topic, timeout_s) in self.specs.items():
            if name not in running or not self.armed.get(name):
                continue
            seen = self.last_seen.get(topic, 0.0)
            if now - seen <= timeout_s:
                continue
            others_fresh = any(
                now - ts <= self.transport_window_s
                for t, ts in self.last_seen.items()
                if t != topic
            )
            if others_fresh:
                out.append(name)
        return out


def liveness_specs(launcher_cfg: dict[str, Any]) -> dict[str, tuple[str, float]]:
    raw = launcher_cfg.get("liveness", DEFAULT_LIVENESS)
    specs: dict[str, tuple[str, float]] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or not spec.get("topic"):
            continue
        specs[str(name)] = (str(spec["topic"]), float(spec.get("timeout_ms", 3000)) / 1000.0)
    return specs


def _log(text: str) -> None:
    try:
        sys.stdout.write(f"[launcher] {text}\n")
        sys.stdout.flush()
    except Exception:
        pass


def _pump_stdout(managed: Managed, proc: subprocess.Popen[bytes]) -> None:
    """Forward a child's output with a name prefix.

    Reads bytes and decodes with ``errors="replace"`` (a non-UTF-8 byte must not
    kill the pump), and keeps draining until EOF even if our own stdout fails —
    a pump that stops reading lets the pipe fill and blocks the child in print().
    """
    if proc.stdout is None:
        return
    prefix = managed.name
    for raw in iter(proc.stdout.readline, b""):
        try:
            line = raw.decode("utf-8", errors="replace")
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
        except Exception:
            pass


def _backoff_seconds(failures: int) -> float:
    if failures <= 0:
        return 0.0
    return float(min(30, 2 ** (failures - 1)))


def _zero_motor_pwm(motor_cfg: dict[str, Any], why: str) -> None:
    if not bool(motor_cfg.get("gpio_enabled", False)):
        return
    try:
        from body.lib import motor_gpio

        dirs = motor_gpio.failsafe_zero_pwm(motor_cfg)
        _log(f"motor failsafe ({why}): zeroed PWM on {dirs or 'no channels found'}")
    except Exception as e:
        _log(f"motor failsafe ({why}) failed: {e!r}")


class _LivenessFeed:
    """Zenoh side of liveness: raw subscribers stamping arrival time. Opened lazily."""

    def __init__(self, body_cfg: dict[str, Any], monitor: LivenessMonitor) -> None:
        self.body_cfg = body_cfg
        self.monitor = monitor
        self.lock = threading.Lock()
        self.session: Any = None
        self.subs: list[Any] = []
        self.next_try = 0.0

    def ensure_open(self, now: float) -> None:
        if self.session is not None or now < self.next_try:
            return
        self.next_try = now + 5.0
        try:
            from body.lib import zenoh_helpers

            session = zenoh_helpers.open_session(self.body_cfg)
            for topic in self.monitor.topics():
                def _cb(_sample: Any, _topic: str = topic) -> None:
                    with self.lock:
                        self.monitor.on_sample(_topic, time.monotonic())

                self.subs.append(session.declare_subscriber(topic, _cb))
            self.session = session
            _log(f"liveness supervision on {sorted(self.monitor.topics())}")
        except Exception as e:
            _log(f"liveness: zenoh session unavailable ({e!r}); retrying")

    def close(self) -> None:
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None


def main() -> None:
    root = repo_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        body_cfg = load_body_config()
    except (OSError, json.JSONDecodeError) as e:
        _log(f"config.json unreadable ({e!r}); liveness + motor failsafe use defaults")
        body_cfg = {}
    launcher_cfg = body_cfg.get("launcher", {})
    motor_cfg = body_cfg.get("motor", {})
    zero_pwm_on_motor_exit = bool(launcher_cfg.get("zero_pwm_on_motor_exit", True))
    liveness_enabled = bool(launcher_cfg.get("liveness_enabled", True))
    kill_grace_s = float(launcher_cfg.get("liveness_kill_grace_s", 3.0))
    monitor = LivenessMonitor(
        liveness_specs(launcher_cfg),
        transport_window_s=float(launcher_cfg.get("liveness_transport_window_s", 2.0)),
    )
    feed = _LivenessFeed(body_cfg, monitor) if liveness_enabled else None

    managed_list = [
        Managed(name=str(spec["name"]), cmd=list(spec["cmd"]), next_start_monotonic=time.monotonic() + i * 0.15)
        for i, spec in enumerate(PROCESSES)
    ]

    shutdown = threading.Event()

    def handle_signal(_sig: int, _frame: Any) -> None:
        shutdown.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGHUP, handle_signal)

    while not shutdown.is_set() or any(m.proc is not None for m in managed_list):
        now = time.monotonic()
        if feed is not None and not shutdown.is_set():
            feed.ensure_open(now)
            with feed.lock:
                running = {m.name for m in managed_list if m.proc is not None and m.kill_deadline is None}
                stale = monitor.stale(now, running)
            for m in managed_list:
                if m.name in stale and m.proc is not None and m.kill_deadline is None:
                    _log(f"{m.name} is alive but its topic went stale — restarting")
                    if m.name == MOTOR_PROCESS and zero_pwm_on_motor_exit:
                        _zero_motor_pwm(motor_cfg, "motor_controller hung")
                    try:
                        m.proc.send_signal(signal.SIGTERM)
                    except OSError:
                        pass
                    m.kill_deadline = now + kill_grace_s

        for m in managed_list:
            if m.proc is not None and m.kill_deadline is not None and m.proc.poll() is None and now >= m.kill_deadline:
                _log(f"{m.name} ignored SIGTERM — SIGKILL")
                m.proc.kill()
                m.kill_deadline = now + kill_grace_s

            if m.proc is not None and m.proc.poll() is not None:
                code = m.proc.returncode
                _log(f"{m.name} exited rc={code}")
                m.proc = None
                m.kill_deadline = None
                if m.name == MOTOR_PROCESS and zero_pwm_on_motor_exit:
                    _zero_motor_pwm(motor_cfg, f"motor_controller exited rc={code}")
                if shutdown.is_set():
                    continue
                if m.stable_since is not None and (now - m.stable_since) >= 5.0:
                    m.failures = 0
                m.failures += 1
                delay = _backoff_seconds(m.failures)
                m.next_start_monotonic = now + delay
                m.stable_since = None
                _log(f"{m.name} will restart in {delay}s")

            if shutdown.is_set():
                continue

            if m.proc is None and now >= m.next_start_monotonic:
                try:
                    m.proc = subprocess.Popen(
                        m.cmd,
                        cwd=root,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                except OSError as e:
                    m.failures += 1
                    m.next_start_monotonic = now + _backoff_seconds(m.failures)
                    _log(f"failed to start {m.name}: {e!r}")
                    continue
                m.stable_since = time.monotonic()
                if feed is not None:
                    with feed.lock:
                        monitor.on_start(m.name)
                _log(f"started {m.name} pid={m.proc.pid}")
                threading.Thread(target=_pump_stdout, args=(m, m.proc), daemon=True).start()

        if shutdown.is_set():
            for m in managed_list:
                if m.proc is not None and m.proc.poll() is None:
                    m.proc.send_signal(signal.SIGTERM)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and any(m.proc is not None and m.proc.poll() is None for m in managed_list):
                time.sleep(0.05)
            for m in managed_list:
                if m.proc is not None and m.proc.poll() is None:
                    m.proc.kill()
            if zero_pwm_on_motor_exit:
                _zero_motor_pwm(motor_cfg, "launcher shutdown")
            break

        time.sleep(0.05)

    if feed is not None:
        feed.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Start Body processes, prefix logs, restart with backoff, graceful shutdown."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from body.lib.zenoh_helpers import repo_root

PROCESSES: list[dict[str, Any]] = [
    {"name": "watchdog", "cmd": [sys.executable, "-m", "body.watchdog"]},
    {"name": "motor_controller", "cmd": [sys.executable, "-m", "body.motor_controller"]},
    {"name": "lidar_driver", "cmd": [sys.executable, "-m", "body.lidar_driver"]},
    {"name": "oakd_driver", "cmd": [sys.executable, "-m", "body.oakd_driver"]},
]


@dataclass
class Managed:
    name: str
    cmd: list[str]
    proc: subprocess.Popen[str] | None = None
    failures: int = 0
    stable_since: float | None = None
    next_start_monotonic: float = field(default_factory=time.monotonic)


def _pump_stdout(managed: Managed, shutdown: threading.Event) -> None:
    proc = managed.proc
    if proc is None or proc.stdout is None:
        return
    prefix = f"{managed.name}"
    try:
        for line in proc.stdout:
            if shutdown.is_set():
                break
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
    except Exception:
        return


def _backoff_seconds(failures: int) -> float:
    if failures <= 0:
        return 0.0
    return float(min(30, 2 ** (failures - 1)))


def main() -> None:
    root = repo_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    managed_list = [
        Managed(name=str(spec["name"]), cmd=list(spec["cmd"]), next_start_monotonic=time.monotonic() + i * 0.15)
        for i, spec in enumerate(PROCESSES)
    ]

    shutdown = threading.Event()
    pump_threads: list[threading.Thread] = []

    def handle_signal(_sig: int, _frame: Any) -> None:
        shutdown.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while not shutdown.is_set() or any(m.proc is not None for m in managed_list):
        now = time.monotonic()
        for m in managed_list:
            if m.proc is not None and m.proc.poll() is not None:
                code = m.proc.returncode
                sys.stdout.write(f"[launcher] {m.name} exited rc={code}\n")
                sys.stdout.flush()
                m.proc = None
                if shutdown.is_set():
                    continue
                if m.stable_since is not None and (now - m.stable_since) >= 5.0:
                    m.failures = 0
                m.failures += 1
                delay = _backoff_seconds(m.failures)
                m.next_start_monotonic = now + delay
                m.stable_since = None
                sys.stdout.write(f"[launcher] {m.name} will restart in {delay}s\n")
                sys.stdout.flush()

            if shutdown.is_set():
                continue

            if m.proc is None and now >= m.next_start_monotonic:
                m.proc = subprocess.Popen(
                    m.cmd,
                    cwd=root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                m.stable_since = time.monotonic()
                sys.stdout.write(f"[launcher] started {m.name} pid={m.proc.pid}\n")
                sys.stdout.flush()
                t = threading.Thread(target=_pump_stdout, args=(m, shutdown), daemon=True)
                t.start()
                pump_threads.append(t)

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
            break

        time.sleep(0.05)

    sys.exit(0)


if __name__ == "__main__":
    main()

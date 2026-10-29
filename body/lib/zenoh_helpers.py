"""Zenoh session setup and JSON pub/sub helpers."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import zenoh

# Small, latency-critical control topics, all at INTERACTIVE_HIGH priority so
# they are not queued behind bulk sensor data (depth, local map, scans).
# Commands flowing *into* the robot are also published with
# CongestionControl.BLOCK (never silently dropped under congestion).
BLOCK_TOPICS: frozenset[str] = frozenset(
    {
        "body/cmd_vel",
        "body/cmd_direct",
        "body/drive/goto",
        "body/estop_rearm",
        "body/heartbeat",
    }
)
# Pi -> desktop reports keep DROP: zenohd forwards them over WiFi, and a BLOCK
# put on a stalled link would stall the publishing Pi process (local_drive's
# cmd_vel, the watchdog's emergency_stop) for seconds. A desktop-originated
# emergency_stop is the exception (see ``put_kwargs(sender="desktop")``).
PRIORITY_TOPICS: frozenset[str] = frozenset(
    {
        "body/drive/status",
        "body/emergency_stop",
    }
)
CONTROL_TOPICS: frozenset[str] = BLOCK_TOPICS | PRIORITY_TOPICS

_control_qos_enabled = True


def repo_root() -> Path:
    """Repository root (directory containing config.json)."""
    return Path(__file__).resolve().parents[2]


def load_body_config(path: Path | None = None) -> dict[str, Any]:
    """Load JSON config; merge Zenoh endpoints from env when set."""
    cfg_path = path or (repo_root() / "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    override = os.environ.get("ZENOH_CONNECT", "").strip()
    if override:
        zenoh_cfg = data.setdefault("zenoh", {})
        zenoh_cfg["connect_endpoints"] = [override]
    # Pi-only transport settings (set in deploy/body-launcher.service) so the shared
    # config.json stays valid for desktop tools that read it.
    mode = os.environ.get("ZENOH_MODE", "").strip()
    if mode:
        data.setdefault("zenoh", {})["mode"] = mode
    scouting = os.environ.get("ZENOH_MULTICAST_SCOUTING", "").strip().lower()
    if scouting:
        data.setdefault("zenoh", {})["multicast_scouting"] = scouting in ("1", "true", "yes", "on")
    return data


def zenoh_config_dict(body_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the zenoh session config dict from the ``zenoh`` block of config.json.

    Keys: ``connect_endpoints`` (list), optional ``mode`` ("peer" | "client"; omitted →
    zenoh default), optional ``multicast_scouting`` (bool; omitted → zenoh default).
    On the Pi, ``mode: "client"`` + ``multicast_scouting: false`` pins every process to
    the local router (see deploy/NETWORK.md).
    """
    zcfg = body_cfg.get("zenoh", {})
    endpoints = zcfg.get("connect_endpoints", ["tcp/127.0.0.1:7447"])
    merged: dict[str, Any] = {"connect": {"endpoints": endpoints}}
    mode = zcfg.get("mode")
    if mode:
        merged["mode"] = str(mode)
    if "multicast_scouting" in zcfg:
        merged["scouting"] = {"multicast": {"enabled": bool(zcfg["multicast_scouting"])}}
    return merged


def zenoh_config_from_body(body_cfg: dict[str, Any]) -> zenoh.Config:
    return zenoh.Config.from_json5(json.dumps(zenoh_config_dict(body_cfg)))


def open_session(body_cfg: dict[str, Any]) -> zenoh.Session:
    global _control_qos_enabled
    _control_qos_enabled = bool(body_cfg.get("zenoh", {}).get("control_qos_enabled", True))
    return zenoh.open(zenoh_config_from_body(body_cfg))


def put_kwargs(key_expr: str, *, sender: str = "pi") -> dict[str, Any]:
    """QoS kwargs for ``session.put`` on ``key_expr`` (empty for non-control topics).

    ``sender`` is ``"pi"`` or ``"desktop"``; the desktop also BLOCKs its own
    ``body/emergency_stop`` since that one travels toward the robot.
    """
    if not _control_qos_enabled or key_expr not in CONTROL_TOPICS:
        return {}
    block = key_expr in BLOCK_TOPICS or (sender == "desktop" and key_expr == "body/emergency_stop")
    return {
        "congestion_control": zenoh.CongestionControl.BLOCK if block else zenoh.CongestionControl.DROP,
        "priority": zenoh.Priority.INTERACTIVE_HIGH,
        "express": True,
    }


def publish_json(session: zenoh.Session, key_expr: str, payload: dict[str, Any]) -> None:
    session.put(key_expr, json.dumps(payload), **put_kwargs(key_expr))


def declare_subscriber_json(
    session: zenoh.Session,
    key_expr: str,
    handler: Callable[[str, dict[str, Any]], None],
) -> zenoh.Subscriber:
    last_err_log = [0.0]

    def _cb(sample: zenoh.Sample) -> None:
        key = str(sample.key_expr)
        try:
            text = sample.payload.to_string()
            obj = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return
        if not isinstance(obj, dict):
            return
        try:
            handler(key, obj)
        except Exception:
            # A malformed message must never kill the zenoh callback thread or
            # escape silently: log (rate-limited) and drop it.
            now = time.monotonic()
            if now - last_err_log[0] >= 1.0:
                last_err_log[0] = now
                print(
                    f"zenoh_helpers: handler for {key_expr} raised on {key}:\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )

    return session.declare_subscriber(key_expr, _cb)


def publish_json_when_matched(
    session: zenoh.Session,
    key_expr: str,
    payload: dict[str, Any],
    wait_s: float = 2.0,
) -> bool:
    """One-shot publish for short-lived tools (CLI): wait until a matching
    subscriber is known (up to ``wait_s``), put, then give the transport a moment
    to flush before the caller closes the session. Returns whether a subscriber
    was matched (the message is published either way)."""
    pub = session.declare_publisher(key_expr, **put_kwargs(key_expr))
    try:
        deadline = time.monotonic() + max(0.0, wait_s)
        matched = bool(pub.matching_status)
        while not matched and time.monotonic() < deadline:
            time.sleep(0.02)
            matched = bool(pub.matching_status)
        pub.put(json.dumps(payload))
        time.sleep(0.2)
        return matched
    finally:
        try:
            pub.undeclare()
        except Exception:
            pass

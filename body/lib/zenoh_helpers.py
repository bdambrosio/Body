"""Zenoh session setup and JSON pub/sub helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import zenoh


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
    return data


def zenoh_config_from_body(body_cfg: dict[str, Any]) -> zenoh.Config:
    endpoints = body_cfg.get("zenoh", {}).get("connect_endpoints", ["tcp/127.0.0.1:7447"])
    merged = {"connect": {"endpoints": endpoints}}
    return zenoh.Config.from_json5(json.dumps(merged))


def open_session(body_cfg: dict[str, Any]) -> zenoh.Session:
    return zenoh.open(zenoh_config_from_body(body_cfg))


def publish_json(session: zenoh.Session, key_expr: str, payload: dict[str, Any]) -> None:
    session.put(key_expr, json.dumps(payload))


def declare_subscriber_json(
    session: zenoh.Session,
    key_expr: str,
    handler: Callable[[str, dict[str, Any]], None],
) -> zenoh.Subscriber:
    def _cb(sample: zenoh.Sample) -> None:
        key = str(sample.key_expr)
        try:
            text = sample.payload.to_string()
            obj = json.loads(text)
            if isinstance(obj, dict):
                handler(key, obj)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

    return session.declare_subscriber(key_expr, _cb)

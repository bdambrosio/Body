"""Zenoh session helpers for the body stub.

Mirrors the pattern in src/utils/zenoh_utils.make_localhost_config, but
connects to an explicit remote router endpoint rather than scouting
localhost. Multicast scouting is disabled so the stub binds only to the
configured router (predictable behavior on a LAN with other Zenoh peers).
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def normalize_endpoint(endpoint: str) -> str:
    """Zenoh wants `<protocol>/<address>`. If the caller passed a bare
    `host:port` (no `/`), assume tcp. Whitespace is stripped.
    """
    s = endpoint.strip()
    if "/" in s:
        return s
    return f"tcp/{s}"


def make_remote_config(endpoint: str) -> Any:
    """Build a zenoh.Config that connects to one explicit router."""
    import zenoh
    ep = normalize_endpoint(endpoint)
    config = zenoh.Config()
    config.insert_json5("connect/endpoints", json.dumps([ep]))
    config.insert_json5("scouting/multicast/enabled", "false")
    config.insert_json5("scouting/gossip/enabled", "true")
    return config


def open_session(endpoint: str) -> Any:
    """Open a Zenoh session to the given router. Raises on failure."""
    import zenoh
    ep = normalize_endpoint(endpoint)
    logger.info(f"opening zenoh session → {ep}")
    return zenoh.open(make_remote_config(ep))


def control_qos_kwargs(key_expr: str) -> dict:
    """QoS kwargs for a desktop put/publisher on ``key_expr``.

    Control topics (cmd_vel, cmd_direct, drive/goto, heartbeat, estop_rearm,
    emergency_stop) go out with CongestionControl.BLOCK at INTERACTIVE_HIGH priority so a
    goto/cancel/stop is never silently dropped or queued behind bulk sensor
    traffic (X5). The topic list and kwargs come from the shared
    ``body.lib.zenoh_helpers.put_kwargs``; empty for non-control topics or if
    the helper is unavailable (older tree) — i.e. today's default QoS.
    """
    try:
        from body.lib.zenoh_helpers import put_kwargs
        return dict(put_kwargs(key_expr, sender="desktop"))
    except Exception:
        return {}


def declare_control_publisher(session: Any, key_expr: str) -> Any:
    """``session.declare_publisher`` with control-topic QoS when supported."""
    kw = control_qos_kwargs(key_expr)
    if kw:
        try:
            return session.declare_publisher(key_expr, **kw)
        except TypeError:
            logger.warning("zenoh declare_publisher rejected QoS kwargs for %s", key_expr)
    return session.declare_publisher(key_expr)


def control_put(session: Any, key_expr: str, payload: Any) -> None:
    """``session.put`` with control-topic QoS when supported."""
    kw = control_qos_kwargs(key_expr)
    if kw:
        try:
            session.put(key_expr, payload, **kw)
            return
        except TypeError:
            logger.warning("zenoh put rejected QoS kwargs for %s", key_expr)
    session.put(key_expr, payload)

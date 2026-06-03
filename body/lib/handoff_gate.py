"""Tier-handoff gate: arm/continue breakpoints + record publishing.

Shared by the desktop hierarchical drive (Tier-1/Tier-2 handoffs) and the Pi
``local_drive`` (Tier-3 handoff). At each handoff a producer calls
``record(tier, payload)`` to publish what it is about to hand down, then
``should_hold(tier)`` to decide whether to pause. Arm / disarm / continue
arrive on one control topic from the standalone Handoff Inspector; records go
out on a per-tier topic.

Single-step semantics: while armed, ``should_hold`` stays True until a
``continue`` token arrives (one-shot). The producer clears it via
``consume_continue`` once it proceeds, so the next handoff re-holds.

Only depends on the zenoh JSON helpers, so it is importable on both the
desktop and the Pi. All state is mutated under a lock because the ctrl
subscriber fires on a zenoh thread while producers call from their own loop.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, Optional

from body.lib import zenoh_helpers

RECORD_PREFIX = "drive/handoff"
CTRL_KEY = "drive/handoff/ctrl"


class HandoffGate:
    """Per-tier breakpoint gate driven by an inspector over zenoh.

    ``publish`` / ``subscribe`` are injectable for tests (default to the real
    ``zenoh_helpers`` functions). Pass ``subscribe=None`` to skip wiring the
    control subscriber (e.g. a record-only producer)."""

    def __init__(
        self,
        session: Any,
        *,
        tiers: Iterable[int] = (1, 2, 3),
        record_prefix: str = RECORD_PREFIX,
        ctrl_key: str = CTRL_KEY,
        publish=zenoh_helpers.publish_json,
        subscribe: Optional[Any] = zenoh_helpers.declare_subscriber_json,
    ) -> None:
        self._session = session
        self._publish = publish
        self._tiers = tuple(tiers)
        self._record_key = {t: f"{record_prefix}/t{t}" for t in self._tiers}
        self._lock = threading.Lock()
        self._armed: Dict[int, bool] = {t: False for t in self._tiers}
        self._continue: Dict[int, bool] = {t: False for t in self._tiers}
        self._seq: Dict[int, int] = {t: 0 for t in self._tiers}
        self._ctrl_sub = (
            subscribe(session, ctrl_key, self._on_ctrl)
            if subscribe is not None else None
        )

    # ── control (inspector → gate) ───────────────────────────────────
    def _on_ctrl(self, _key: str, msg: Dict[str, Any]) -> None:
        try:
            tier = int(msg.get("tier", 0))
        except (TypeError, ValueError):
            return
        action = msg.get("action")
        with self._lock:
            if tier not in self._armed:
                return
            if action == "arm":
                self._armed[tier] = True
            elif action == "disarm":
                self._armed[tier] = False
                self._continue[tier] = False
            elif action == "continue":
                self._continue[tier] = True

    # ── records (producer → inspector) ───────────────────────────────
    def record(self, tier: int, payload: Dict[str, Any]) -> None:
        with self._lock:
            if tier not in self._record_key:
                return
            self._seq[tier] += 1
            seq = self._seq[tier]
            key = self._record_key[tier]
        out = dict(payload)
        out["tier"] = tier
        out["seq"] = seq
        self._publish(self._session, key, out)

    # ── gate queries (producer) ──────────────────────────────────────
    def is_armed(self, tier: int) -> bool:
        with self._lock:
            return self._armed.get(tier, False)

    def should_hold(self, tier: int) -> bool:
        """True when this tier is armed and no continue token is pending."""
        with self._lock:
            return self._armed.get(tier, False) and not self._continue.get(tier, False)

    def consume_continue(self, tier: int) -> bool:
        """Clear and return the one-shot continue token (call when proceeding)."""
        with self._lock:
            if self._continue.get(tier, False):
                self._continue[tier] = False
                return True
            return False

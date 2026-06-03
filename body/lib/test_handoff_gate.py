"""Unit tests for the tier-handoff gate (arm → hold → single-step)."""
import unittest

from body.lib.handoff_gate import HandoffGate


class _FakeBus:
    """Captures publishes and the ctrl handler so tests can inject control."""
    def __init__(self):
        self.published = []          # list of (key, payload)
        self.ctrl_handler = None

    def publish(self, _session, key, payload):
        self.published.append((key, payload))

    def subscribe(self, _session, _key, handler):
        self.ctrl_handler = handler
        return object()

    def send_ctrl(self, tier, action):
        self.ctrl_handler("drive/handoff/ctrl", {"tier": tier, "action": action})


def _gate(bus):
    return HandoffGate(object(), publish=bus.publish, subscribe=bus.subscribe)


class TestHandoffGate(unittest.TestCase):
    def test_record_stamps_tier_seq_and_routes_topic(self):
        bus = _FakeBus()
        g = _gate(bus)
        g.record(2, {"src": "clear"})
        g.record(2, {"src": "blind"})
        keys = [k for k, _ in bus.published]
        self.assertEqual(keys, ["drive/handoff/t2", "drive/handoff/t2"])
        self.assertEqual(bus.published[0][1]["tier"], 2)
        self.assertEqual(bus.published[0][1]["seq"], 1)
        self.assertEqual(bus.published[1][1]["seq"], 2)        # per-tier counter
        self.assertEqual(bus.published[0][1]["src"], "clear")

    def test_unarmed_never_holds(self):
        bus = _FakeBus()
        g = _gate(bus)
        self.assertFalse(g.should_hold(2))
        self.assertFalse(g.is_armed(2))

    def test_arm_holds_until_continue_then_single_steps(self):
        bus = _FakeBus()
        g = _gate(bus)
        bus.send_ctrl(2, "arm")
        self.assertTrue(g.is_armed(2))
        self.assertTrue(g.should_hold(2))          # armed, no token → hold
        bus.send_ctrl(2, "continue")
        self.assertFalse(g.should_hold(2))         # token present → may pass
        self.assertTrue(g.consume_continue(2))     # one-shot consumed
        self.assertFalse(g.consume_continue(2))    # …only once
        self.assertTrue(g.should_hold(2))          # still armed → re-holds next leg

    def test_disarm_clears_pending_continue(self):
        bus = _FakeBus()
        g = _gate(bus)
        bus.send_ctrl(3, "arm")
        bus.send_ctrl(3, "continue")
        bus.send_ctrl(3, "disarm")
        self.assertFalse(g.is_armed(3))
        self.assertFalse(g.should_hold(3))
        self.assertFalse(g.consume_continue(3))    # continue was cleared by disarm

    def test_tiers_are_independent(self):
        bus = _FakeBus()
        g = _gate(bus)
        bus.send_ctrl(1, "arm")
        self.assertTrue(g.should_hold(1))
        self.assertFalse(g.should_hold(2))
        self.assertFalse(g.should_hold(3))

    def test_unknown_tier_ctrl_ignored(self):
        bus = _FakeBus()
        g = _gate(bus)
        bus.send_ctrl(9, "arm")                     # out of range → ignored
        self.assertFalse(g.is_armed(9))


if __name__ == "__main__":
    unittest.main()

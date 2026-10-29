"""StubController.rearm_estop publishes body/estop_rearm (E1)."""
from __future__ import annotations

import json
import unittest

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController


class _Pub:
    def __init__(self):
        self.sent = []

    def put(self, payload):
        self.sent.append(json.loads(payload))


class TestRearm(unittest.TestCase):
    def test_not_connected(self):
        self.assertFalse(StubController(StubConfig()).rearm_estop())

    def test_publishes_rearm(self):
        c = StubController(StubConfig())
        c._session = object()
        c._estop_rearm_pub = _Pub()
        self.assertTrue(c.rearm_estop("test"))
        msg = c._estop_rearm_pub.sent[-1]
        self.assertEqual(msg["source"], "test")
        self.assertIsInstance(msg["ts"], float)
        self.assertEqual(c.config.topics.estop_rearm, "body/estop_rearm")


if __name__ == "__main__":
    unittest.main()

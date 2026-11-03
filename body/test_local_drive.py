"""Smoke test for the local_drive process loop with a fake zenoh bus.

Drives ``body.local_drive.main`` tick by tick (the loop's sleep is replaced by
a script step) and checks the lifecycle contract: ack by status cmd_id,
REJECTED + watermark, FAULT e_stop at boot and its clean release, odom reset
→ FAULT odom_reset, body-frame goals anchored at ``anchor_ts``, and an
exception inside a tick → zero cmd + FAULT internal_error, loop alive.
"""
import math
import time
import unittest
from unittest import mock

from body import local_drive
from body.lib import zenoh_helpers
from body.lib.tier3_controller import Tier3Controller


class _Stop(BaseException):
    pass


class _FakeGate:
    def __init__(self, *_a, **_k):
        pass

    def record(self, *_a, **_k):
        pass

    def is_armed(self, _t):
        return False

    def should_hold(self, _t):
        return False

    def consume_continue(self, _t):
        return False


class _Bus:
    def __init__(self):
        self.handlers = {}
        self.pub = []

    def subscribe(self, _session, key, handler):
        self.handlers[key] = handler
        return object()

    def publish(self, _session, key, payload):
        self.pub.append((key, payload))

    def send(self, key, msg):
        self.handlers[key](key, msg)

    def statuses(self):
        return [p for k, p in self.pub if k == "body/drive/status"]

    def cmds(self):
        return [p for k, p in self.pub if k == "body/cmd_vel"]


def _scan(ts):
    # Open space: every beam a no-return.
    n = 360
    return {"ts": ts, "angle_min": 0.0, "angle_increment": 2 * math.pi / n,
            "ranges": [None] * n}


def _odom(ts, x=0.0, sid="sess0001", seq=0):
    return {"ts": ts, "x": x, "y": 0.0, "theta": 0.0, "vx": 0.0, "vtheta": 0.0,
            "source": "wheel_encoders", "session_id": sid, "seq": seq}


class TestLocalDriveLoop(unittest.TestCase):
    def test_lifecycle(self):
        bus = _Bus()
        cfg = zenoh_helpers.load_body_config()
        cfg = dict(cfg)
        cfg["local_drive"] = dict(cfg["local_drive"], enabled=True)
        cfg["local_drive"]["depth_veto"] = dict(cfg["local_drive"]["depth_veto"], enabled=False)
        seq = [0]

        def feed(x=0.0, sid="sess0001"):
            now = time.time()
            seq[0] += 1
            bus.send("body/odom", _odom(now - 0.02, x=x, sid=sid, seq=seq[0]))
            bus.send("body/odom", _odom(now, x=x, sid=sid, seq=seq[0] + 1))
            seq[0] += 1
            bus.send("body/lidar/scan", _scan(now - 0.01))

        log = {}
        boom = {"armed": False}
        orig_step = Tier3Controller.step

        def step(ctrl, **kw):
            if boom["armed"]:
                boom["armed"] = False
                raise RuntimeError("synthetic tick failure")
            return orig_step(ctrl, **kw)

        script = []

        def at(fn):
            script.append(fn)

        # 0: boot — watchdog says e-stop latched.
        at(lambda: (feed(), bus.send("body/status", {"e_stop_active": True})))
        # 1: check FAULT e_stop; goto refused while latched
        def s1():
            log["boot"] = bus.statuses()[-1]
            bus.send("body/drive/goto", {"kind": "goto", "cmd_id": 1, "frame": "odom",
                                         "x_m": 1.0, "y_m": 0.0})
        at(s1)
        # 2: re-armed
        def s2():
            log["refused"] = bus.statuses()[-1]
            bus.send("body/status", {"e_stop_active": False})
            bus.send("body/motor_state", {"e_stop_active": False, "stall_detected": False})
            feed()
        at(s2)
        def s3():
            log["after_rearm"] = bus.statuses()[-1]
            bus.send("body/drive/goto", {"kind": "goto", "cmd_id": 5, "frame": "odom",
                                         "x_m": 1.5, "y_m": 0.0})
            feed()
        at(s3)
        def s4():
            log["driving"] = bus.statuses()[-1]
            log["n_cmd_after_goto"] = len(bus.cmds())
            bus.send("body/drive/goto", {"kind": "goto", "cmd_id": 3, "frame": "odom",
                                         "x_m": 0.0, "y_m": 0.0})
            bus.send("body/drive/goto", {"kind": "goto", "cmd_id": 5, "frame": "odom",
                                         "x_m": 1.5, "y_m": 0.0})   # resend: idempotent
            feed()
        at(s4)
        def s5():
            log["after_reject"] = list(bus.statuses()[-3:])
            feed(sid="sess0002")                      # motor_controller restarted
        at(s5)
        def s6():
            log["odom_reset"] = bus.statuses()[-1]
            now = time.time()
            # robot is at x=0.3 now; goal 1 m ahead in the body frame of the
            # pose 0.5 s ago, when it was at x=0.0 (odom history in the ring)
            for ts, x in ((now, 0.0), (now + 0.001, 0.3)):
                seq[0] += 1
                bus.send("body/odom", _odom(ts, x=x, sid="sess0002", seq=seq[0]))
            bus.send("body/lidar/scan", _scan(now - 0.01))
            bus.send("body/drive/goto", {"kind": "goto", "cmd_id": 6, "frame": "body",
                                         "x_m": 1.0, "y_m": 0.0, "anchor_ts": now})
        at(s6)
        def s7():
            log["body_goal"] = bus.statuses()[-1]
            boom["armed"] = True
            feed(x=0.3, sid="sess0002")
        at(s7)
        def s8():
            log["exception"] = bus.statuses()[-1]
            log["exception_cmd"] = bus.cmds()[-1]
            feed(x=0.3, sid="sess0002")
        at(s8)
        def s9():
            log["after_exception"] = bus.statuses()[-1]
            raise _Stop()
        at(s9)

        def fake_sleep(next_tick, _period):
            fn = script.pop(0)
            fn()
            return next_tick

        with mock.patch.object(zenoh_helpers, "load_body_config", return_value=cfg), \
             mock.patch.object(zenoh_helpers, "open_session", return_value=object()), \
             mock.patch.object(zenoh_helpers, "declare_subscriber_json", bus.subscribe), \
             mock.patch.object(zenoh_helpers, "publish_json", bus.publish), \
             mock.patch.object(local_drive, "HandoffGate", _FakeGate), \
             mock.patch.object(local_drive, "git_sha", return_value="test"), \
             mock.patch.object(local_drive, "_sleep_to", fake_sleep), \
             mock.patch.object(local_drive.signal, "signal"), \
             mock.patch.object(Tier3Controller, "step", step):
            with self.assertRaises(_Stop):
                local_drive.main()

        self.assertEqual((log["boot"]["state"], log["boot"]["blocked_reason"]), ("FAULT", "e_stop"))
        self.assertEqual(log["refused"]["state"], "FAULT")
        self.assertEqual(log["refused"]["cmd_id"], 1)
        # The refused goto stays reported (FAULT, its cmd_id) until the
        # desktop sends something new — never decays to IDLE ("success").
        self.assertEqual(log["after_rearm"]["state"], "FAULT")
        self.assertEqual(log["after_rearm"]["cmd_id"], 1)
        self.assertEqual(log["driving"]["cmd_id"], 5)              # ack within a tick
        self.assertEqual(log["driving"]["state"], "DRIVING")
        self.assertGreater(log["n_cmd_after_goto"], 0)
        rej = [s for s in log["after_reject"] if s["state"] == "REJECTED"]
        self.assertEqual(len(rej), 1)                               # resend not rejected
        self.assertEqual((rej[0]["cmd_id"], rej[0]["watermark"]), (3, 5))
        self.assertEqual(log["after_reject"][-1]["cmd_id"], 5)
        self.assertEqual((log["odom_reset"]["state"], log["odom_reset"]["blocked_reason"],
                          log["odom_reset"]["cmd_id"]), ("FAULT", "odom_reset", 5))
        bg = log["body_goal"]
        self.assertEqual(bg["cmd_id"], 6)
        # anchored at x=0.0 → odom goal (1.0, 0); robot now at 0.3 → 0.7 ahead
        self.assertAlmostEqual(bg["goal_body_xy"][0], 0.7, delta=0.02)
        self.assertEqual((log["exception"]["state"], log["exception"]["blocked_reason"]),
                         ("FAULT", "internal_error"))
        self.assertEqual((log["exception_cmd"]["linear"], log["exception_cmd"]["angular"]),
                         (0.0, 0.0))
        self.assertEqual(log["after_exception"]["state"], "FAULT")  # latched, loop alive


class TestPathGoalLoop(unittest.TestCase):
    """goto kind "path" through the real loop: accept + caps, refresh in place
    (no restart), stale refresh ignored, anchoring, starvation → path_stale."""

    def test_path_lifecycle(self):
        bus = _Bus()
        cfg = dict(zenoh_helpers.load_body_config())
        cfg["local_drive"] = dict(cfg["local_drive"], enabled=True)
        cfg["local_drive"]["depth_veto"] = dict(cfg["local_drive"]["depth_veto"], enabled=False)
        seq = [0]
        x_now = [0.0]

        def feed():
            now = time.time()
            for ts in (now - 0.02, now):
                seq[0] += 1
                bus.send("body/odom", _odom(ts, x=x_now[0], seq=seq[0]))
            bus.send("body/lidar/scan", _scan(now - 0.01))
            return now

        def path_msg(path_seq, route_id=1, x0=0.0, ttl=0.2, anchor=None):
            return {"kind": "path", "cmd_id": 10, "frame": "body", "path_seq": path_seq,
                    "route_id": route_id, "ttl_s": ttl, "end_is_goal": False,
                    "anchor_ts": anchor, "path_xy": [[x0 + 0.1 * k, 0.0] for k in range(21)]}

        log = {}
        script = []
        at = script.append
        at(lambda: (feed(), bus.send("body/status", {"e_stop_active": False}),
                    bus.send("body/motor_state", {"e_stop_active": False, "stall_detected": False})))

        def s1():
            now = feed()
            bus.send("body/drive/goto", path_msg(1, anchor=now))
        at(s1)

        def s2():
            log["first"] = bus.statuses()[-1]
            x_now[0] = 0.5                     # robot moved 0.5 m
            now = feed()
            # refresh: path again starts 1 m ahead of the robot in ITS body
            # frame, anchored now → odom 1.5 .. 3.5
            bus.send("body/drive/goto", path_msg(2, x0=1.0, anchor=now))
        at(s2)

        def s3():
            log["refreshed"] = bus.statuses()[-1]
            now = feed()
            bus.send("body/drive/goto", path_msg(1, x0=-5.0, anchor=now))   # older seq: ignored
        at(s3)

        def s4():
            log["after_old"] = bus.statuses()[-1]
            time.sleep(0.45)                   # > 2 × ttl without a refresh
            feed()
        at(s4)

        def s5():
            log["stale"] = bus.statuses()[-1]
            raise _Stop()
        at(s5)

        def fake_sleep(next_tick, _period):
            script.pop(0)()
            return next_tick

        with mock.patch.object(zenoh_helpers, "load_body_config", return_value=cfg), \
             mock.patch.object(zenoh_helpers, "open_session", return_value=object()), \
             mock.patch.object(zenoh_helpers, "declare_subscriber_json", bus.subscribe), \
             mock.patch.object(zenoh_helpers, "publish_json", bus.publish), \
             mock.patch.object(local_drive, "HandoffGate", _FakeGate), \
             mock.patch.object(local_drive, "git_sha", return_value="test"), \
             mock.patch.object(local_drive, "_sleep_to", fake_sleep), \
             mock.patch.object(local_drive.signal, "signal"):
            with self.assertRaises(_Stop):
                local_drive.main()

        f = log["first"]
        self.assertEqual((f["cmd_id"], f["state"], f["path_seq"]), (10, "DRIVING", 1))
        self.assertIn("path", f["caps"])
        r = log["refreshed"]
        self.assertEqual((r["cmd_id"], r["path_seq"]), (10, 2))
        # end of the refreshed path: odom 3.5, robot at 0.5 → 3.0 ahead
        self.assertAlmostEqual(r["goal_body_xy"][0], 3.0, delta=0.05)
        self.assertEqual(log["after_old"]["path_seq"], 2)
        st = log["stale"]
        self.assertEqual((st["state"], st["blocked_reason"]), ("BLOCKED", "path_stale"))


if __name__ == "__main__":
    unittest.main()

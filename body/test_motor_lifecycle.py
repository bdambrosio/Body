"""Tests for process lifecycle / transport helpers: watchdog e-stop, launcher liveness,
motor PWM failsafe, zenoh QoS + config, odom/estop schemas."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from body import launcher, watchdog
from body.lib import motor_gpio, schemas, zenoh_helpers


class TestWatchdogEStop(unittest.TestCase):
    def test_latches_on_heartbeat_loss_and_announces_once(self):
        e = watchdog.EStopState()
        self.assertFalse(e.update_heartbeat(True))
        self.assertTrue(e.update_heartbeat(False))
        self.assertFalse(e.update_heartbeat(False))
        self.assertTrue(e.e_stop_active)

    def test_heartbeat_recovery_alone_does_not_clear(self):
        e = watchdog.EStopState()
        e.update_heartbeat(False)
        e.update_heartbeat(True)
        self.assertTrue(e.e_stop_active)

    def test_rearm_requires_heartbeat(self):
        e = watchdog.EStopState()
        e.update_heartbeat(False)
        self.assertFalse(e.rearm(hb_ok=False))
        self.assertTrue(e.e_stop_active)
        e.update_heartbeat(True)
        self.assertTrue(e.rearm(hb_ok=True))
        self.assertFalse(e.e_stop_active)
        # A later loss latches and announces again.
        self.assertTrue(e.update_heartbeat(False))


class TestLiveness(unittest.TestCase):
    def setUp(self):
        self.mon = launcher.LivenessMonitor(
            {"motor_controller": ("body/odom", 1.0), "lidar_driver": ("body/lidar/scan", 3.0)},
            transport_window_s=2.0,
        )
        self.running = {"motor_controller", "lidar_driver"}

    def test_never_published_is_not_restarted(self):
        self.mon.on_sample("body/lidar/scan", 10.0)
        self.assertEqual(self.mon.stale(10.5, self.running), [])
        self.assertEqual(self.mon.stale(100.0, self.running), [])

    def test_hung_after_flowing_is_stale(self):
        self.mon.on_sample("body/odom", 10.0)
        self.mon.on_sample("body/lidar/scan", 11.5)
        self.assertEqual(self.mon.stale(10.9, self.running), [])
        self.assertEqual(self.mon.stale(12.0, self.running), ["motor_controller"])

    def test_transport_down_suppresses(self):
        self.mon.on_sample("body/odom", 10.0)
        self.mon.on_sample("body/lidar/scan", 10.0)
        self.assertEqual(self.mon.stale(20.0, self.running), [])

    def test_restart_disarms_until_new_sample(self):
        self.mon.on_sample("body/odom", 10.0)
        self.mon.on_start("motor_controller")
        self.mon.on_sample("body/lidar/scan", 12.0)
        self.assertEqual(self.mon.stale(12.5, self.running), [])
        self.mon.on_sample("body/odom", 13.0)
        self.mon.on_sample("body/lidar/scan", 14.5)
        self.assertEqual(self.mon.stale(14.5, self.running), ["motor_controller"])

    def test_not_running_is_ignored(self):
        self.mon.on_sample("body/odom", 10.0)
        self.mon.on_sample("body/lidar/scan", 12.0)
        self.assertEqual(self.mon.stale(12.0, {"lidar_driver"}), [])

    def test_specs_from_config(self):
        specs = launcher.liveness_specs(
            {"liveness": {"a": {"topic": "t/a", "timeout_ms": 500}, "b": {"timeout_ms": 1}}}
        )
        self.assertEqual(specs, {"a": ("t/a", 0.5)})
        self.assertIn("local_drive", launcher.liveness_specs({}))

    def test_pump_survives_non_utf8(self):
        class FakeProc:
            def __init__(self):
                import io
                self.stdout = io.BytesIO(b"ok\n\xff\xfe bad\nafter\n")

        out = []
        with mock.patch("sys.stdout") as so:
            so.write.side_effect = lambda s: out.append(s)
            launcher._pump_stdout(launcher.Managed(name="x", cmd=[]), FakeProc())
        self.assertEqual(len(out), 3)
        self.assertIn("after", out[2])


class TestPwmFailsafe(unittest.TestCase):
    def test_zero_pwm_on_discovered_chip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            chip = root / "pwmchip2"
            for ch in (0, 1):
                d = chip / f"pwm{ch}"
                d.mkdir(parents=True)
                (d / "duty_cycle").write_text("500000\n")
                (d / "enable").write_text("1\n")
            with mock.patch.object(motor_gpio, "PWM_SYSFS_ROOT", root):
                touched = motor_gpio.failsafe_zero_pwm({})
            self.assertEqual(len(touched), 2)
            for ch in (0, 1):
                self.assertEqual((chip / f"pwm{ch}" / "duty_cycle").read_text().strip(), "0")
                self.assertEqual((chip / f"pwm{ch}" / "enable").read_text().strip(), "0")

    def test_no_chip_never_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(motor_gpio, "PWM_SYSFS_ROOT", Path(td)):
                self.assertEqual(motor_gpio.failsafe_zero_pwm({}), [])

    def test_duty_reenables_channel_after_zero_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            pwm = Path(td)
            (pwm / "duty_cycle").write_text("0\n")
            (pwm / "enable").write_text("1\n")
            motor_gpio.zero_outputs({"left_pwm_dir": pwm, "right_pwm_dir": None})
            self.assertEqual((pwm / "enable").read_text().strip(), "0")
            motor_gpio._set_pwm_duty(pwm, 1_000_000, 0.0)
            self.assertEqual((pwm / "enable").read_text().strip(), "0")
            motor_gpio._set_pwm_duty(pwm, 1_000_000, 0.3)
            self.assertEqual((pwm / "duty_cycle").read_text().strip(), "300000")
            self.assertEqual((pwm / "enable").read_text().strip(), "1")

    def test_zero_outputs_tolerates_missing_dirs(self):
        motor_gpio.zero_outputs({"left_pwm_dir": Path("/nonexistent/pwm0"), "right_pwm_dir": None})


class TestZenohHelpers(unittest.TestCase):
    def test_control_topics_get_block_and_priority(self):
        kw = zenoh_helpers.put_kwargs("body/cmd_vel")
        self.assertEqual(kw["congestion_control"], zenoh_helpers.zenoh.CongestionControl.BLOCK)
        self.assertEqual(kw["priority"], zenoh_helpers.zenoh.Priority.INTERACTIVE_HIGH)
        self.assertEqual(zenoh_helpers.put_kwargs("body/oakd/depth"), {})
        for t in ("body/estop_rearm", "body/heartbeat", "body/drive/goto", "body/emergency_stop"):
            self.assertIn(t, zenoh_helpers.CONTROL_TOPICS)

    def test_pi_reports_drop_but_desktop_estop_blocks(self):
        cc = zenoh_helpers.zenoh.CongestionControl
        for t in ("body/drive/status", "body/emergency_stop"):
            kw = zenoh_helpers.put_kwargs(t)
            self.assertEqual(kw["congestion_control"], cc.DROP)
            self.assertEqual(kw["priority"], zenoh_helpers.zenoh.Priority.INTERACTIVE_HIGH)
        kw = zenoh_helpers.put_kwargs("body/emergency_stop", sender="desktop")
        self.assertEqual(kw["congestion_control"], cc.BLOCK)
        self.assertEqual(zenoh_helpers.put_kwargs("body/drive/status", sender="desktop")["congestion_control"], cc.DROP)

    def test_config_dict_mode_and_scouting(self):
        d = zenoh_helpers.zenoh_config_dict({"zenoh": {"connect_endpoints": ["tcp/x:1"]}})
        self.assertEqual(d, {"connect": {"endpoints": ["tcp/x:1"]}})
        d = zenoh_helpers.zenoh_config_dict(
            {"zenoh": {"connect_endpoints": ["tcp/x:1"], "mode": "client", "multicast_scouting": False}}
        )
        self.assertEqual(d["mode"], "client")
        self.assertFalse(d["scouting"]["multicast"]["enabled"])
        zenoh_helpers.zenoh_config_from_body({"zenoh": {"mode": "client", "multicast_scouting": False}})

    def test_env_overrides(self):
        with mock.patch.dict(os.environ, {"ZENOH_MODE": "client", "ZENOH_MULTICAST_SCOUTING": "0"}):
            cfg = zenoh_helpers.load_body_config()
        self.assertEqual(cfg["zenoh"]["mode"], "client")
        self.assertFalse(cfg["zenoh"]["multicast_scouting"])

    def test_subscriber_handler_exception_is_contained(self):
        captured = {}

        class FakeSession:
            def declare_subscriber(self, key, cb):
                captured["cb"] = cb

        class FakeSample:
            key_expr = "body/cmd_vel"

            class payload:
                @staticmethod
                def to_string():
                    return '{"linear": 1}'

        def handler(_k, msg):
            raise KeyError("angular")

        zenoh_helpers.declare_subscriber_json(FakeSession(), "body/cmd_vel", handler)
        with mock.patch("builtins.print"):
            captured["cb"](FakeSample())  # must not raise


class TestSchemas(unittest.TestCase):
    def test_odom_session_and_seq_optional(self):
        m = schemas.odom(ts=1.0, dt_ms=19.7, source="wheel_encoders", session_id="deadbeef", seq=3)
        self.assertEqual(m["session_id"], "deadbeef")
        self.assertEqual(m["seq"], 3)
        self.assertEqual(m["dt_ms"], 19.7)
        self.assertNotIn("seq", schemas.odom(ts=1.0))

    def test_estop_rearm(self):
        m = schemas.estop_rearm("desktop", ts=5.0)
        self.assertEqual(m, {"ts": 5.0, "source": "desktop"})


if __name__ == "__main__":
    unittest.main()


class TestMotorControllerLoop(unittest.TestCase):
    """Drive motor_controller.main() with a fake zenoh layer (no GPIO)."""

    def _run(self, script, n_ticks=40):
        from body import motor_controller

        cfg = {
            "motor": {"gpio_enabled": False, "max_wheel_vel_ms": 0.9, "loop_hz": 500,
                      "wheel_base_m": 0.181, "cmd_timeout_max_ms": 1000},
            "watchdog": {"heartbeat_timeout_ms": 2000},
        }
        handlers = {}
        published = []

        class FakeSession:
            def close(self):
                pass

        def fake_sub(_session, key, handler):
            handlers[key] = handler

        def fake_pub(_session, key, payload):
            published.append((key, payload))
            if key == "body/odom":
                n = payload["seq"]
                if n in script:
                    for topic, msg in script[n]:
                        handlers[topic](topic, msg)
                if n >= n_ticks:
                    raise KeyboardInterrupt

        with mock.patch.object(zenoh_helpers, "load_body_config", return_value=cfg), \
             mock.patch.object(zenoh_helpers, "open_session", return_value=FakeSession()), \
             mock.patch.object(zenoh_helpers, "declare_subscriber_json", side_effect=fake_sub), \
             mock.patch.object(zenoh_helpers, "publish_json", side_effect=fake_pub), \
             mock.patch("signal.signal"), mock.patch("builtins.print"):
            with self.assertRaises(KeyboardInterrupt):
                motor_controller.main()
        odoms = [p for k, p in published if k == "body/odom"]
        states = [p for k, p in published if k == "body/motor_state"]
        return odoms, states

    def test_malformed_commands_ignored_and_odom_fields(self):
        script = {
            2: [("body/cmd_vel", {"linear": "x", "angular": 0}),
                ("body/cmd_direct", {"right": 0.3}),
                ("body/cmd_vel", {"linear": float("nan"), "angular": 0})],
            5: [("body/cmd_vel", {"ts": 1e12, "linear": 0.45, "angular": 0.0})],
        }
        odoms, states = self._run(script, n_ticks=10)
        self.assertEqual([o["seq"] for o in odoms], list(range(len(odoms))))
        self.assertEqual(len({o["session_id"] for o in odoms}), 1)
        self.assertEqual(len(odoms[0]["session_id"]), 8)
        self.assertEqual(odoms[0]["source"], "none")
        self.assertTrue(all(s["left_pwm"] == 0.0 for s in states[:6]))
        self.assertAlmostEqual(states[7]["left_pwm"], 0.5)
        self.assertGreater(odoms[7]["vx"], 0.0)

    def test_estop_needs_rearm_with_heartbeat(self):
        go = ("body/cmd_vel", {"linear": 0.45, "angular": 0.0})
        script = {
            1: [go],
            3: [("body/emergency_stop", {"reason": "test"}), go],
            5: [go],
            7: [("body/estop_rearm", {"source": "t"}), go],  # no heartbeat yet → refused
            9: [("body/heartbeat", {"seq": 1}), ("body/estop_rearm", {"source": "t"}), go],
        }
        _odoms, states = self._run(script, n_ticks=12)
        self.assertGreater(states[2]["left_pwm"], 0.0)
        for i in (4, 6, 8):
            self.assertEqual(states[i]["left_pwm"], 0.0, i)
            self.assertTrue(states[i]["e_stop_active"])
        self.assertFalse(states[10]["e_stop_active"])
        self.assertGreater(states[10]["left_pwm"], 0.0)

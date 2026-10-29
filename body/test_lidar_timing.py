"""Tests for lidar sweep timing, rear seam, per-point self-mask and unknown_bins."""
import math
import struct
import unittest

from body.lib.ldrobot_ldpacket import LdLidarPacket, LdPacketDecoder, crc8_ld
from body.lidar_driver import (
    LidarClock,
    SweepAssembler,
    _bearing_bin,
    _bin_revolution,
    packet_points_timed,
    point_spacing_s,
)

SPEED = 3600          # deg/s → 10 Hz
STEP = 0.8            # deg between samples
PKT_SPAN = STEP * 11  # deg covered by one 12-point packet


def _packet(start_deg, ts_ms, dist_mm=2000, inten=200, dists=None):
    start = int(round((start_deg % 360.0) * 100))
    end = int(round(((start_deg + PKT_SPAN) % 360.0) * 100))
    d = tuple(dists) if dists is not None else (dist_mm,) * 12
    return LdLidarPacket(
        speed_deg_s=SPEED,
        start_angle_0p01=start,
        end_angle_0p01=end,
        timestamp_ms=ts_ms % 30000,
        distances_mm=d,
        intensities=(inten,) * 12,
    )


def _frame_bytes(pkt):
    b = bytearray([0x54, 0x2C])
    b += struct.pack("<HH", pkt.speed_deg_s, pkt.start_angle_0p01)
    for d, i in zip(pkt.distances_mm, pkt.intensities):
        b += struct.pack("<HB", d, i)
    b += struct.pack("<HH", pkt.end_angle_0p01, pkt.timestamp_ms)
    b.append(crc8_ld(bytes(b)))
    return bytes(b)


def _stream(asm, start_deg, n_packets, t0=100.0, dist_fn=None, skip=()):
    """Feed consecutive packets (device CW angles) with exact timing; return sweeps."""
    sweeps = []
    dt_pkt = 12 * STEP / SPEED
    for k in range(n_packets):
        a0 = start_deg + k * 12 * STEP
        if k in skip:
            continue
        dists = None
        if dist_fn is not None:
            dists = [dist_fn((a0 + i * STEP) % 360.0) for i in range(12)]
        pkt = _packet(a0, int(k * dt_pkt * 1000), dists=dists)
        sweeps += asm.add_points(packet_points_timed(pkt, t0 + k * dt_pkt))
    return sweeps


class TestBinning(unittest.TestCase):
    def test_nearest_bin_no_half_bin_bias(self):
        self.assertEqual(_bearing_bin(0.4, 360), 0)
        self.assertEqual(_bearing_bin(0.6, 360), 1)
        self.assertEqual(_bearing_bin(359.6, 360), 0)

    def test_bin_revolution_device_cw_to_published_ccw(self):
        # Device 90° CW = published 270° (right side).
        ranges, _ = _bin_revolution([(90.0, 1.0, 10)], 360, 0.05, 12.0, True)
        self.assertEqual(ranges[270], 1.0)


class TestPacketTiming(unittest.TestCase):
    def test_point_spacing_from_speed(self):
        pkt = _packet(10.0, 0)
        self.assertAlmostEqual(point_spacing_s(pkt), STEP / SPEED, places=6)
        pts = packet_points_timed(pkt, 5.0)
        self.assertAlmostEqual(pts[0][3], 5.0)
        self.assertAlmostEqual(pts[11][3] - pts[0][3], 11 * STEP / SPEED, places=6)

    def test_decoder_roundtrip_keeps_timestamp(self):
        pkt = _packet(123.0, 29990)
        out = LdPacketDecoder().feed(b"\x00\x01" + _frame_bytes(pkt))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].timestamp_ms, 29990)


class TestLidarClock(unittest.TestCase):
    def test_min_latency_offset_and_wrap(self):
        clk = LidarClock(wrap_ms=30000, latency_s=0.0)
        base = 1000.0
        stamps = []
        # Device ms counts from 29900 and wraps at 30000; arrival jitter 0–20 ms.
        for k in range(200):
            dev = (29900 + 3 * k) % 30000
            true_t = base + 0.003 * k
            jitter = 0.020 if k % 3 else 0.001
            stamps.append((clk.stamp(dev, true_t + jitter), true_t))
        for est, true_t in stamps[50:]:
            self.assertIsNotNone(est)
            self.assertAlmostEqual(est, true_t + 0.001, delta=1e-6)
        self.assertEqual(clk.resyncs, 0)

    def test_auto_upgrade_to_16bit_wrap(self):
        clk = LidarClock(wrap_ms=30000)
        clk.stamp(40000, 10.0)
        self.assertEqual(clk.wrap_ms, 65536)

    def test_stalled_device_clock_returns_none(self):
        clk = LidarClock(stall_packets=3)
        out = [clk.stamp(0, 10.0 + 0.003 * k) for k in range(6)]
        self.assertIsNotNone(out[0])
        self.assertIsNone(out[-1])

    def test_gap_resyncs(self):
        clk = LidarClock(latency_s=0.0, max_gap_ms=1000)
        clk.stamp(100, 10.0)
        clk.stamp(103, 10.003)
        t = clk.stamp(5000, 20.0)      # 4.9 s jump in device clock
        self.assertEqual(clk.resyncs, 1)
        self.assertAlmostEqual(t, 20.0)


class TestSweepAssembler(unittest.TestCase):
    def setUp(self):
        self.asm = SweepAssembler(num_bins=360, include_intensities=True)

    def test_first_partial_dropped_then_full_sweep_at_rear_seam(self):
        # Start at device 0° (forward): the first sweep (0..180 device) is partial.
        n = int(2.6 * 360 / (12 * STEP))
        sweeps = _stream(self.asm, 0.0, n)
        self.assertGreaterEqual(len(sweeps), 1)
        self.assertGreaterEqual(self.asm.dropped, 1)
        sw = sweeps[0]
        # Seam at the rear: forward bins are mid-sweep, so ts_start/ts_end straddle them.
        self.assertAlmostEqual(sw.ts_end - sw.ts_start, 0.1, delta=0.01)
        self.assertAlmostEqual(sw.ts, 0.5 * (sw.ts_start + sw.ts_end))
        self.assertEqual(sw.unknown_bins, [])
        self.assertTrue(all(r == 2.0 for r in sw.ranges))

    def test_forward_point_timed_mid_sweep(self):
        # A sweep from the rear seam: the forward sample (device 0°) is half a revolution in.
        stamps = {}

        def dist_fn(deg):
            return 1500 if abs(((deg + 180) % 360) - 180) < 0.5 else 2000

        asm = SweepAssembler(num_bins=360)
        dt_pkt = 12 * STEP / SPEED
        t0 = 50.0
        n = int(2.2 * 360 / (12 * STEP))
        for k in range(n):
            a0 = 180.0 + k * 12 * STEP
            pkt = _packet(a0, int(k * dt_pkt * 1000),
                          dists=[dist_fn((a0 + i * STEP) % 360) for i in range(12)])
            for p in packet_points_timed(pkt, t0 + k * dt_pkt):
                if p[1] == 1.5:
                    stamps.setdefault("fwd", []).append(p[3])
            for sw in asm.add_points(packet_points_timed(pkt, t0 + k * dt_pkt)):
                stamps.setdefault("sweeps", []).append(sw)
        sw = stamps["sweeps"][0]
        fwd_t = [t for t in stamps["fwd"] if sw.ts_start <= t <= sw.ts_end][0]
        self.assertAlmostEqual(fwd_t, sw.ts, delta=0.003)
        self.assertEqual(sw.ranges[0], 1.5)

    def test_dropped_packet_bins_are_unknown(self):
        n = int(2.6 * 360 / (12 * STEP))
        per_rev = 360 / (12 * STEP)
        # Drop one packet in the middle of the second revolution (device ~0..9°).
        k_drop = int(per_rev) + 1
        sweeps = _stream(self.asm, 0.0, n, skip={k_drop})
        sw = sweeps[0]
        self.assertTrue(len(sw.unknown_bins) >= 8)
        for b in sw.unknown_bins:
            self.assertIsNone(sw.ranges[b])

    def test_per_point_self_mask_keeps_real_return_in_same_bin(self):
        asm = SweepAssembler(num_bins=360, self_mask_sectors_deg=[[80, 170, 0.2]])
        pts = [
            (360.0 - 120.0, 0.15, 50, 0.0),     # antenna return (published 120°) → masked
            (360.0 - 120.2, 1.30, 50, 0.001),   # real wall in the same bin
            (360.0 - 130.0, 0.12, 50, 0.002),   # antenna only → unknown
        ]
        sw = asm.bin_points(pts, 0.0, 0.1)
        self.assertEqual(sw.ranges[120], 1.30)
        self.assertIn(130, sw.unknown_bins)
        self.assertNotIn(120, sw.unknown_bins)
        self.assertEqual(sw.stats["masked"], 2)

    def test_no_return_is_not_unknown_but_too_near_is(self):
        pts = [(360.0 - 10.0, 0.0, 0, 0.0), (360.0 - 20.0, 0.02, 30, 0.001)]
        sw = self.asm.bin_points(pts, 0.0, 0.1)
        self.assertIsNone(sw.ranges[10])
        self.assertNotIn(10, sw.unknown_bins)      # observed, no echo → free
        self.assertIn(20, sw.unknown_bins)         # 2 cm: something too near → unknown
        self.assertIn(30, sw.unknown_bins)         # never sampled

    def test_mixed_pixel_filter(self):
        asm = SweepAssembler(num_bins=360, mixed_pixel_max_intensity=15, mixed_pixel_jump_m=0.15)
        pts = [
            (360.0 - 40.0, 0.60, 200, 0.000),   # jamb
            (360.0 - 41.0, 1.70, 8, 0.001),     # veil point between jamb and background
            (360.0 - 42.0, 3.00, 200, 0.002),   # background
            (360.0 - 43.0, 3.00, 200, 0.003),
            (360.0 - 44.0, 3.00, 10, 0.004),    # weak but consistent with neighbours → kept
            (360.0 - 45.0, 3.00, 200, 0.005),
        ]
        sw = asm.bin_points(pts, 0.0, 0.1)
        self.assertIsNone(sw.ranges[41])
        self.assertIn(41, sw.unknown_bins)
        self.assertEqual(sw.ranges[44], 3.0)
        off = SweepAssembler(num_bins=360).bin_points(pts, 0.0, 0.1)
        self.assertEqual(off.ranges[41], 1.70)

    def test_overlong_sweep_dropped(self):
        asm = SweepAssembler(num_bins=360, nominal_period_s=0.1)
        # Full-coverage revolution that took 0.3 s (stalled motor / merged).
        pts = [((180.0 + a) % 360.0, 2.0, 100, 10.0 + 0.3 * a / 360.0) for a in range(0, 360)]
        pts += [(180.5, 2.0, 100, 10.31)]       # wraps → closes the sweep
        out = asm.add_points(pts)
        self.assertEqual(out, [])
        self.assertEqual(asm.dropped, 1)

    def test_deskew_rotates_points_to_mid_time(self):
        pts = [(360.0 - 90.0, 2.0, 100, 0.0), (360.0 - 90.0, 2.0, 100, 0.1)]
        sw = self.asm.bin_points(pts, 0.0, 0.1, yaw_rate_radps=math.radians(100.0))
        # ±0.05 s at 100°/s → ±5°.
        self.assertEqual(sw.ranges[85], 2.0)
        self.assertEqual(sw.ranges[95], 2.0)


class _Done(Exception):
    pass


class _FakeSerial:
    """Delivers pre-built frame bytes in small chunks, then raises _Done."""

    SerialException = OSError

    def __init__(self, data):
        self._data = bytearray(data)
        self.clock = 5000.0          # fake wall clock: advances at the UART byte rate

    def time(self):
        return self.clock

    def Serial(self, *_a, **_k):  # noqa: N802 - mimics the pyserial module API
        return self

    @property
    def in_waiting(self):
        return min(len(self._data), 47 * 3)

    def read(self, n):
        if not self._data:
            raise _Done()
        out = bytes(self._data[:n])
        del self._data[:n]
        self.clock += len(out) * 10.0 / 230400.0
        return out

    def close(self):
        pass


class TestSerialLoop(unittest.TestCase):
    def test_publishes_timed_sweeps_with_seq_and_unknown_bins(self):
        import sys
        import body.lidar_driver as ld

        dt_pkt = 12 * STEP / SPEED
        data = bytearray()
        for k in range(int(3.2 * 360 / (12 * STEP))):
            data += _frame_bytes(_packet(k * 12 * STEP, int(1000 + k * dt_pkt * 1000)))
        fake = _FakeSerial(data)
        sent = []
        orig_pub = ld.zenoh_helpers.publish_json
        old_serial = sys.modules.get("serial")
        sys.modules["serial"] = fake
        ld.zenoh_helpers.publish_json = lambda _s, key, msg: sent.append((key, msg))
        import types
        orig_time = ld.time
        ld.time = types.SimpleNamespace(time=fake.time, monotonic=orig_time.monotonic,
                                        sleep=orig_time.sleep)
        try:
            with self.assertRaises(_Done):
                ld._run_serial(None, {"self_mask_sectors_deg": [[80, 170, 0.2]]})
        finally:
            ld.time = orig_time
            ld.zenoh_helpers.publish_json = orig_pub
            if old_serial is not None:
                sys.modules["serial"] = old_serial
            else:
                sys.modules.pop("serial", None)
        self.assertGreaterEqual(len(sent), 2)
        msgs = [m for _, m in sent]
        self.assertEqual([m["seq"] for m in msgs], list(range(len(msgs))))
        for m in msgs:
            self.assertLess(m["ts_start"], m["ts"])
            self.assertLess(m["ts"], m["ts_end"])
            self.assertLess(m["ts_end"], fake.clock + 1e-6)     # never in the future
            self.assertEqual(m["unknown_bins"], [])
            self.assertEqual(len(m["ranges"]), 360)


if __name__ == "__main__":
    unittest.main()

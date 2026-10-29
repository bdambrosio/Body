"""LIDAR driver: LDROBOT STL-19P / LD19-class UART scans, or synthetic stub.

Timing model (docs/body_project_spec.md §5.5): every point is stamped from the
packet's device ``timestamp_ms`` mapped to host wall clock (``LidarClock``),
plus its in-packet angular offset. A sweep runs seam-to-seam with the seam at
the *rear* of the robot (published bearing ``lidar.seam_bearing_deg``, default
180°), so the forward sector is temporally contiguous. The published ``ts`` is
the sweep mid-point; ``ts_start``/``ts_end`` bracket it.

Bins with no information this sweep (self-masked, mixed-pixel filtered, a
too-near return, or never sampled — e.g. CRC-dropped packets) are listed in
``unknown_bins`` so consumers never mistake them for a no-return (free) beam.
"""

from __future__ import annotations

import math
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from body.lib import schemas, zenoh_helpers
from body.lib.ldrobot_ldpacket import LdLidarPacket, LdPacketDecoder, packet_to_points_deg


def _run_stub(session: Any, lidar_cfg: dict[str, Any]) -> None:
    hz = float(lidar_cfg.get("publish_hz", 10))
    period = 1.0 / max(1.0, hz)
    stop = threading.Event()

    def handle_sigterm(_sig: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    next_tick = time.monotonic()
    while not stop.is_set():
        zenoh_helpers.publish_json(
            session,
            "body/lidar/scan",
            schemas.lidar_scan(scan_time_ms=int(round(period * 1000))),
        )
        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()


def _bearing_bin(bearing_deg: float, num_bins: int) -> int:
    """Nearest bin for a published (CCW) bearing; bin i is centred on i*360/num_bins."""
    return int(math.floor(bearing_deg / 360.0 * num_bins + 0.5)) % num_bins


def _bin_revolution(
    points: list[tuple[float, float, int]],
    num_bins: int,
    range_min_m: float,
    range_max_m: float,
    include_intensities: bool,
) -> tuple[list[float | None], list[int] | None]:
    """Min-range per bin for device-frame (CW) points. Kept for tools/tests."""
    ranges: list[float | None] = [None] * num_bins
    intensities: list[int] | None = [0] * num_bins if include_intensities else None
    for deg, dist_m, inte in points:
        if dist_m <= 0.0 or dist_m < range_min_m or dist_m > range_max_m:
            continue
        bi = _bearing_bin((360.0 - deg) % 360.0, num_bins)
        prev = ranges[bi]
        if prev is None or dist_m < prev:
            ranges[bi] = dist_m
            if intensities is not None:
                intensities[bi] = inte
    return ranges, intensities


def _parse_self_mask(
    sectors_deg: Any, num_bins: int,
) -> list[tuple[set[int], float]]:
    """Parse self-occlusion sectors into ``[(bin_indices, max_range_m), ...]``.

    Each sector is ``[lo_deg, hi_deg]`` or ``[lo_deg, hi_deg, max_range_m]``
    in the *published* scan frame (bin i is at bearing ``i * 360/num_bins``
    degrees, 0 = +x forward, CCW; ``lo > hi`` wraps through 0). Without a
    range, the whole sector is dropped; with one, only returns within
    ``max_range_m`` are dropped — the right tool for a fixed near-body
    return (e.g. the WiFi antenna), whose *bearing* is unstable at short
    range but whose *range* is small and stable. Masking happens before
    publish, so every consumer (Tier-3, local_map, PF, mapping) ignores it.
    """
    parsed: list[tuple[set[int], float]] = []
    for lo, hi, rng in _parse_mask_sectors(sectors_deg):
        bins = set()
        for bi in range(num_bins):
            ang = bi * 360.0 / num_bins
            if _in_sector(ang, lo, hi):
                bins.add(bi)
        parsed.append((bins, rng))
    return parsed


def _parse_mask_sectors(sectors_deg: Any) -> list[tuple[float, float, float]]:
    """``[[lo, hi(, max_range)], ...]`` → ``[(lo_deg, hi_deg, max_range_m|inf), ...]``."""
    out: list[tuple[float, float, float]] = []
    for sec in sectors_deg or []:
        try:
            lo, hi = float(sec[0]), float(sec[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        rng = float("inf")
        if len(sec) >= 3:
            try:
                v = float(sec[2])
                if v > 0:
                    rng = v
            except (TypeError, ValueError):
                pass
        out.append((lo, hi, rng))
    return out


def _in_sector(ang_deg: float, lo: float, hi: float) -> bool:
    return (lo <= ang_deg <= hi) if lo <= hi else (ang_deg >= lo or ang_deg <= hi)


class LidarClock:
    """Map LD19 packet ``timestamp_ms`` (device clock, wraps) to host wall clock.

    The host arrival time of a packet is always *after* its capture, so the
    smallest ``arrival - device_time`` over a sliding window is the best
    estimate of the clock offset (plus the minimum transport latency, which
    ``latency_s`` removes). Returns ``None`` while the device clock is not
    trustworthy (not advancing); callers then fall back to arrival time.
    """

    def __init__(
        self,
        *,
        wrap_ms: int = 30000,
        window_s: float = 2.0,
        latency_s: float = 0.002,
        max_gap_ms: float = 1000.0,
        stall_packets: int = 8,
    ) -> None:
        self.wrap_ms = max(1, int(wrap_ms))
        self.window_s = max(0.05, float(window_s))
        self.latency_s = float(latency_s)
        self.max_gap_ms = float(max_gap_ms)
        self.stall_packets = max(1, int(stall_packets))
        self._last_dev_ms: int | None = None
        self._dev_s = 0.0
        self._stalled = 0
        self._win: deque[tuple[float, float]] = deque()   # (arrival, offset), offsets ascending
        self.resyncs = 0

    def reset(self) -> None:
        self._last_dev_ms = None
        self._dev_s = 0.0
        self._stalled = 0
        self._win.clear()

    def stamp(self, device_ms: int, arrival_wall: float) -> float | None:
        dev = int(device_ms)
        if dev >= self.wrap_ms:
            # Device counts past the configured wrap: it must be a 16-bit counter.
            self.wrap_ms = 65536
        if self._last_dev_ms is None:
            self._last_dev_ms = dev
            self._dev_s = 0.0
            self._win.clear()
        else:
            delta = (dev - self._last_dev_ms) % self.wrap_ms
            self._last_dev_ms = dev
            if delta > self.max_gap_ms:
                # Gap/garbage: start a new epoch rather than trust the unwrap.
                self.resyncs += 1
                self._dev_s = 0.0
                self._win.clear()
                self._stalled = 0
            else:
                if delta == 0:
                    self._stalled += 1
                else:
                    self._stalled = 0
                self._dev_s += delta / 1000.0
        off = arrival_wall - self._dev_s
        while self._win and self._win[-1][1] >= off:
            self._win.pop()
        self._win.append((arrival_wall, off))
        while self._win and self._win[0][0] < arrival_wall - self.window_s:
            self._win.popleft()
        if self._stalled >= self.stall_packets:
            return None
        return self._dev_s + self._win[0][1] - self.latency_s


@dataclass
class Sweep:
    ranges: list[float | None]
    intensities: list[int] | None
    unknown_bins: list[int]
    ts_start: float
    ts_end: float
    n_points: int
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ts(self) -> float:
        return 0.5 * (self.ts_start + self.ts_end)


class SweepAssembler:
    """Accumulate timed points into seam-to-seam sweeps and bin them.

    Points are device-frame ``(device_deg CW, dist_m, intensity, t_wall)``.
    A sweep closes when the angle relative to the seam wraps (drops by more
    than 180°), so merged double revolutions are not possible from wrap
    detection; a sweep is still dropped if it lasts longer than
    ``max_period_factor`` × the nominal period, or if it does not cover the
    seam at both ends (the first partial revolution after start).
    """

    def __init__(
        self,
        *,
        num_bins: int = 360,
        range_min_m: float = 0.05,
        range_max_m: float = 12.0,
        include_intensities: bool = True,
        self_mask_sectors_deg: Any = None,
        seam_bearing_deg: float = 180.0,
        nominal_period_s: float = 0.1,
        max_period_factor: float = 1.5,
        edge_tolerance_deg: float = 15.0,
        mixed_pixel_max_intensity: int = 0,
        mixed_pixel_jump_m: float = 0.15,
    ) -> None:
        self.num_bins = max(1, int(num_bins))
        self.range_min_m = float(range_min_m)
        self.range_max_m = float(range_max_m)
        self.include_intensities = bool(include_intensities)
        self.mask = _parse_mask_sectors(self_mask_sectors_deg)
        # Published bearing b = 360 - device_deg, so the seam in device degrees:
        self.seam_device_deg = (360.0 - float(seam_bearing_deg)) % 360.0
        self.nominal_period_s = max(1e-3, float(nominal_period_s))
        self.max_period_factor = float(max_period_factor)
        self.edge_tolerance_deg = float(edge_tolerance_deg)
        self.mixed_pixel_max_intensity = int(mixed_pixel_max_intensity)
        self.mixed_pixel_jump_m = float(mixed_pixel_jump_m)
        self._pts: list[tuple[float, float, int, float]] = []
        self._last_rel: float | None = None
        self.dropped = 0

    def set_nominal_period(self, period_s: float) -> None:
        if period_s > 0 and math.isfinite(period_s):
            self.nominal_period_s = float(period_s)

    def _rel(self, device_deg: float) -> float:
        return (device_deg - self.seam_device_deg) % 360.0

    def add_points(
        self,
        points: list[tuple[float, float, int, float]],
        yaw_rate_radps: float | None = None,
    ) -> list[Sweep]:
        out: list[Sweep] = []
        for p in points:
            rel = self._rel(p[0])
            if self._last_rel is not None and rel < self._last_rel - 180.0:
                sw = self._finish(yaw_rate_radps)
                if sw is not None:
                    out.append(sw)
                self._pts = []
            self._pts.append(p)
            self._last_rel = rel
        return out

    def _finish(self, yaw_rate_radps: float | None) -> Sweep | None:
        pts = self._pts
        if len(pts) < 2:
            self.dropped += 1
            return None
        first_rel = self._rel(pts[0][0])
        last_rel = self._rel(pts[-1][0])
        tol = self.edge_tolerance_deg
        if first_rel > tol or last_rel < 360.0 - tol:
            self.dropped += 1      # partial revolution (start-up, or a long CRC gap at the seam)
            return None
        ts_start = pts[0][3]
        ts_end = pts[-1][3]
        if ts_end - ts_start > self.max_period_factor * self.nominal_period_s:
            self.dropped += 1
            return None
        return self.bin_points(pts, ts_start, ts_end, yaw_rate_radps)

    def _mixed_pixel_flags(self, pts: list[tuple[float, float, int, float]]) -> list[bool]:
        """Flag low-intensity returns floating between a near and a far neighbour
        (the classic jamb-edge "veil" point). Conservative: both neighbour jumps
        must exceed ``mixed_pixel_jump_m`` and the point must lie strictly between."""
        n = len(pts)
        flags = [False] * n
        thr = self.mixed_pixel_max_intensity
        if thr <= 0 or n < 3:
            return flags
        jump = self.mixed_pixel_jump_m
        for i in range(1, n - 1):
            d = pts[i][1]
            if d <= 0.0 or pts[i][2] > thr:
                continue
            a = pts[i - 1][1]
            b = pts[i + 1][1]
            if a <= 0.0 or b <= 0.0:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            if lo + jump < d < hi - jump:
                flags[i] = True
        return flags

    def bin_points(
        self,
        pts: list[tuple[float, float, int, float]],
        ts_start: float,
        ts_end: float,
        yaw_rate_radps: float | None = None,
    ) -> Sweep:
        nb = self.num_bins
        ranges: list[float | None] = [None] * nb
        intens: list[int] | None = [0] * nb if self.include_intensities else None
        observed = [False] * nb
        no_info = [False] * nb
        t_mid = 0.5 * (ts_start + ts_end)
        mixed = self._mixed_pixel_flags(pts)
        stats = {"masked": 0, "mixed_pixel": 0, "too_near": 0}
        for k, (dev_deg, dist_m, inte, t) in enumerate(pts):
            bearing = (360.0 - dev_deg) % 360.0
            if yaw_rate_radps:
                # Express the point in the body frame at t_mid (rotation only).
                bearing = (bearing + math.degrees(yaw_rate_radps * (t - t_mid))) % 360.0
            bi = _bearing_bin(bearing, nb)
            observed[bi] = True
            masked = False
            for lo, hi, rng in self.mask:
                if _in_sector(bearing, lo, hi) and (math.isinf(rng) or 0.0 < dist_m <= rng):
                    masked = True
                    break
            if masked:
                no_info[bi] = True
                stats["masked"] += 1
                continue
            if mixed[k]:
                no_info[bi] = True
                stats["mixed_pixel"] += 1
                continue
            if dist_m <= 0.0 or dist_m > self.range_max_m:
                continue                     # genuine no-return sample
            if dist_m < self.range_min_m:
                no_info[bi] = True           # something too close to range — not "free"
                stats["too_near"] += 1
                continue
            prev = ranges[bi]
            if prev is None or dist_m < prev:
                ranges[bi] = dist_m
                if intens is not None:
                    intens[bi] = int(inte)
        unknown = [
            i for i in range(nb)
            if ranges[i] is None and (no_info[i] or not observed[i])
        ]
        stats["unobserved"] = sum(1 for o in observed if not o)
        return Sweep(
            ranges=ranges,
            intensities=intens,
            unknown_bins=unknown,
            ts_start=ts_start,
            ts_end=ts_end,
            n_points=len(pts),
            stats=stats,
        )


def packet_points_timed(
    pkt: LdLidarPacket, t_first: float,
) -> list[tuple[float, float, int, float]]:
    """``(device_deg, dist_m, intensity, t_wall)`` for each of the packet's 12 samples.

    Samples are spaced in time by their angular step over the reported rotation
    speed, starting at ``t_first`` (the packet timestamp, mapped to wall clock).
    """
    pts = packet_to_points_deg(pkt)
    dt = point_spacing_s(pkt)
    return [(d, r, i, t_first + k * dt) for k, (d, r, i) in enumerate(pts)]


def point_spacing_s(pkt: LdLidarPacket) -> float:
    diff = (pkt.end_angle_0p01 + 36000 - pkt.start_angle_0p01) % 36000
    step_deg = diff / (len(pkt.distances_mm) - 1) / 100.0 if len(pkt.distances_mm) > 1 else 0.0
    if pkt.speed_deg_s <= 0 or step_deg <= 0.0 or step_deg > 30.0:
        return 0.0
    return step_deg / float(pkt.speed_deg_s)


def _run_serial(session: Any, lidar_cfg: dict[str, Any]) -> None:
    import serial  # Pi-only dep; imported lazily so the module loads for tests

    port = str(lidar_cfg.get("serial_port", "/dev/ttyUSB0"))
    baud = int(lidar_cfg.get("baud_rate", 230400))
    num_bins = int(lidar_cfg.get("num_points", 360))
    range_min_m = float(lidar_cfg.get("range_min_m", 0.05))
    range_max_m = float(lidar_cfg.get("range_max_m", 12.0))
    include_intensities = bool(lidar_cfg.get("include_intensities", True))
    # Small reads: block for at most this long waiting for the first byte, then
    # take whatever has arrived, so packets are timestamped within a few ms.
    timeout_s = float(lidar_cfg.get("serial_timeout_s", 0.005))
    read_max = int(lidar_cfg.get("serial_read_size", 4096))
    publish_hz = float(lidar_cfg.get("publish_hz", 10))
    use_device_ts = bool(lidar_cfg.get("device_timestamps_enabled", True))
    deskew = bool(lidar_cfg.get("deskew_enabled", False))
    clock = LidarClock(
        wrap_ms=int(lidar_cfg.get("device_ts_wrap_ms", 30000)),
        latency_s=float(lidar_cfg.get("device_latency_ms", 2.0)) / 1000.0,
    )
    ts_offset_s = float(lidar_cfg.get("device_ts_offset_ms", 0.0)) / 1000.0
    asm = SweepAssembler(
        num_bins=num_bins,
        range_min_m=range_min_m,
        range_max_m=range_max_m,
        include_intensities=include_intensities,
        # Fixed body-mounted occlusions (e.g. WiFi antenna), applied per point.
        self_mask_sectors_deg=lidar_cfg.get("self_mask_sectors_deg", []),
        seam_bearing_deg=float(lidar_cfg.get("seam_bearing_deg", 180.0)),
        nominal_period_s=1.0 / max(1.0, publish_hz),
        max_period_factor=float(lidar_cfg.get("max_sweep_period_factor", 1.5)),
        mixed_pixel_max_intensity=int(lidar_cfg.get("mixed_pixel_max_intensity", 0)),
        mixed_pixel_jump_m=float(lidar_cfg.get("mixed_pixel_jump_m", 0.15)),
    )
    if asm.mask:
        print(f"[lidar] self-mask: {len(asm.mask)} sector(s), per-point", flush=True)

    stop = threading.Event()

    def handle_sigterm(_sig: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    yaw_rate: dict[str, float] = {"w": 0.0, "mono": 0.0}
    if deskew:
        def on_odom(_k: str, msg: dict[str, Any]) -> None:
            try:
                yaw_rate["w"] = float(msg.get("vtheta", 0.0))
                yaw_rate["mono"] = time.monotonic()
            except (TypeError, ValueError):
                pass

        zenoh_helpers.declare_subscriber_json(session, "body/odom", on_odom)
        print("[lidar] deskew enabled (odom vtheta, rotation only)", flush=True)

    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout_s)
    except serial.SerialException as e:
        print(f"[lidar] cannot open {port} @ {baud}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    decoder = LdPacketDecoder()
    seq = 0
    fallback_logged = False
    last_stats_log = time.monotonic()

    try:
        while not stop.is_set():
            waiting = ser.in_waiting
            chunk = ser.read(min(read_max, waiting) if waiting > 0 else 1)
            if not chunk:
                continue
            arrival = time.time()
            for pkt in decoder.feed(chunk):
                dt_pt = point_spacing_s(pkt)
                t_first: float | None = None
                if use_device_ts:
                    t_dev = clock.stamp(pkt.timestamp_ms, arrival)
                    if t_dev is not None:
                        t_first = t_dev + ts_offset_s
                if t_first is None:
                    if use_device_ts and not fallback_logged:
                        print("[lidar] device timestamps not advancing; using arrival time",
                              file=sys.stderr, flush=True)
                        fallback_logged = True
                    t_first = arrival - clock.latency_s - (len(pkt.distances_mm) - 1) * dt_pt
                if pkt.speed_deg_s > 0:
                    asm.set_nominal_period(360.0 / float(pkt.speed_deg_s))
                w = None
                if deskew and time.monotonic() - yaw_rate["mono"] < 0.2:
                    w = yaw_rate["w"]
                for sw in asm.add_points(packet_points_timed(pkt, t_first), w):
                    zenoh_helpers.publish_json(
                        session,
                        "body/lidar/scan",
                        schemas.lidar_scan_from_bins(
                            sw.ranges,
                            intensities=sw.intensities,
                            range_min_m=range_min_m,
                            range_max_m=range_max_m,
                            scan_time_ms=max(1, int(round((sw.ts_end - sw.ts_start) * 1000))),
                            ts=sw.ts,
                            ts_start=sw.ts_start,
                            ts_end=sw.ts_end,
                            seq=seq,
                            unknown_bins=sw.unknown_bins,
                        ),
                    )
                    seq += 1
            now_m = time.monotonic()
            if now_m - last_stats_log > 60.0:
                last_stats_log = now_m
                print(
                    f"[lidar] seq={seq} dropped_sweeps={asm.dropped} clock_resyncs={clock.resyncs}",
                    flush=True,
                )
    finally:
        ser.close()


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    lidar_cfg = body_cfg.get("lidar", {})
    session = zenoh_helpers.open_session(body_cfg)

    if bool(lidar_cfg.get("stub_only", False)):
        _run_stub(session, lidar_cfg)
    else:
        _run_serial(session, lidar_cfg)

    session.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

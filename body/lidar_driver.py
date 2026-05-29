"""LIDAR driver: LDROBOT STL-19P / LD19-class UART scans, or synthetic stub."""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Any

from body.lib import schemas, zenoh_helpers
from body.lib.ldrobot_ldpacket import LdPacketDecoder, packet_to_points_deg


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


def _bin_revolution(
    points: list[tuple[float, float, int]],
    num_bins: int,
    range_min_m: float,
    range_max_m: float,
    include_intensities: bool,
) -> tuple[list[float | None], list[int] | None]:
    ranges: list[float | None] = [None] * num_bins
    intensities: list[int] | None = [0] * num_bins if include_intensities else None
    for deg, dist_m, inte in points:
        if dist_m <= 0.0 or dist_m < range_min_m or dist_m > range_max_m:
            continue
        bi = int(((360.0 - deg) / 360.0) * num_bins) % num_bins
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
    for sec in sectors_deg or []:
        try:
            lo, hi = float(sec[0]), float(sec[1])
        except (TypeError, ValueError, IndexError):
            continue
        rng = float("inf")
        if len(sec) >= 3:
            try:
                v = float(sec[2])
                if v > 0:
                    rng = v
            except (TypeError, ValueError):
                pass
        bins = set()
        for bi in range(num_bins):
            ang = bi * 360.0 / num_bins
            inside = (lo <= ang <= hi) if lo <= hi else (ang >= lo or ang <= hi)
            if inside:
                bins.add(bi)
        parsed.append((bins, rng))
    return parsed


def _run_serial(session: Any, lidar_cfg: dict[str, Any]) -> None:
    import serial  # Pi-only dep; imported lazily so the module loads for tests

    port = str(lidar_cfg.get("serial_port", "/dev/ttyUSB0"))
    baud = int(lidar_cfg.get("baud_rate", 230400))
    num_bins = int(lidar_cfg.get("num_points", 360))
    range_min_m = float(lidar_cfg.get("range_min_m", 0.05))
    range_max_m = float(lidar_cfg.get("range_max_m", 12.0))
    include_intensities = bool(lidar_cfg.get("include_intensities", True))
    read_chunk = int(lidar_cfg.get("serial_read_size", 4096))
    timeout_s = float(lidar_cfg.get("serial_timeout_s", 0.05))
    # Fixed body-mounted occlusions (e.g. WiFi antenna): drop these returns
    # so they never reach any consumer. Empty by default (no masking).
    self_mask = _parse_self_mask(lidar_cfg.get("self_mask_sectors_deg", []), num_bins)
    if self_mask:
        n_bins = sum(len(b) for b, _ in self_mask)
        print(f"[lidar] self-mask: {len(self_mask)} sector(s), {n_bins} bins", flush=True)

    stop = threading.Event()

    def handle_sigterm(_sig: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout_s)
    except serial.SerialException as e:
        print(f"[lidar] cannot open {port} @ {baud}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    decoder = LdPacketDecoder()
    acc: list[tuple[float, float, int]] = []
    last_deg: float | None = None
    t_rev = time.monotonic()

    try:
        while not stop.is_set():
            chunk = ser.read(read_chunk)
            if not chunk:
                continue
            for pkt in decoder.feed(chunk):
                for deg, dist_m, inte in packet_to_points_deg(pkt):
                    if last_deg is not None and last_deg > 340.0 and deg < 50.0:
                        t_now = time.monotonic()
                        scan_ms = max(1, int((t_now - t_rev) * 1000))
                        ranges, intens = _bin_revolution(
                            acc, num_bins, range_min_m, range_max_m, include_intensities
                        )
                        for bins, rng in self_mask:
                            for bi in bins:
                                r = ranges[bi]
                                if r is not None and r <= rng:
                                    ranges[bi] = None
                                    if intens is not None:
                                        intens[bi] = 0
                        zenoh_helpers.publish_json(
                            session,
                            "body/lidar/scan",
                            schemas.lidar_scan_from_bins(
                                ranges,
                                intensities=intens,
                                range_min_m=range_min_m,
                                range_max_m=range_max_m,
                                scan_time_ms=scan_ms,
                            ),
                        )
                        acc = []
                        t_rev = t_now
                    acc.append((deg, dist_m, inte))
                    last_deg = deg
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

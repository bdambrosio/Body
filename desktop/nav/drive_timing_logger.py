"""Tier-3 drive-timing capture + analysis (standalone, read-only).

The handoff records (drive/handoff/t{1,2,3}) are zenoh-live-only and the
mission JSONL tracer is wired only to the old reactive path, so neither
captures what we need to tune the sub-goal slowdown/re-pick: the wall-clock
gap between Tier-3 publishing ``state=ARRIVED`` for sub-goal N and the next
``goto`` (N+1) landing, plus the per-sub-goal ``dist`` trajectory (where the
ramp actually bites).

This subscribes the two production topics already on the wire and timestamps
every message on receipt. Decoupled: no map, no patrol, no production
coupling — same shape as ``desktop.handoff_inspector``.

  Capture (run on the desktop alongside desktop.nav, then drive a patrol):
      python -m desktop.nav.drive_timing_logger [--router tcp/HOST:PORT] [--out FILE]
      # Ctrl-C to stop; prints a summary and writes JSONL.

  Re-analyze a capture without a live session:
      python -m desktop.nav.drive_timing_logger --analyze FILE.jsonl

Topics:
  body/drive/goto    {cmd_id, x_m, y_m, arrival_tol_m, v_max, ts, frame}
  body/drive/status  {cmd_id, state, dist, goal_body, omega, ts, ...}  (every Tier-3 tick)
"""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from typing import Any, Dict, List, Optional

from body.lib import zenoh_helpers
from desktop.chassis.config import resolve_router
from desktop.chassis.transport import open_session

GOTO_KEY = "body/drive/goto"
STATUS_KEY = "body/drive/status"


class _Capture:
    """Stashes every goto/status with a monotonic receipt time, mirrors to JSONL."""

    def __init__(self, fh) -> None:
        self._fh = fh
        self._lock = threading.Lock()
        self.events: List[Dict[str, Any]] = []
        self._t0 = time.monotonic()

    def _record(self, key: str, msg: Dict[str, Any]) -> None:
        # recv_t (monotonic) drives the skew-free gap math. recv_wall (system
        # clock) lets us compare against embedded msg["ts"] values that share
        # this desktop's wall clock (the goto's ts is desktop.nav's send time),
        # so we can split the gap into desktop-side vs downstream.
        ev = {"recv_t": time.monotonic() - self._t0, "recv_wall": time.time(),
              "key": key, "msg": msg}
        with self._lock:
            self.events.append(ev)
            if self._fh is not None:
                self._fh.write(json.dumps(ev) + "\n")

    def on_goto(self, _key, msg) -> None:
        self._record(GOTO_KEY, msg)

    def on_status(self, _key, msg) -> None:
        self._record(STATUS_KEY, msg)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.events)


def analyze(events: List[Dict[str, Any]]) -> str:
    """Per-sub-goal timing: the ARRIVED->next-goto gap and the dist trajectory.

    The 'command gap' is ARRIVED(cmd N) -> goto(N+1) received: the desktop
    re-pick + link latency. The 'motion gap' is ARRIVED(cmd N) -> first
    DRIVING(N+1): the actual zero-velocity dwell the wheels see. D_margin
    should cover the motion gap at v_max so the next goal supersedes before
    the robot ramps/stops.
    """
    gotos = [e for e in events if e["key"] == GOTO_KEY]
    status = [e for e in events if e["key"] == STATUS_KEY]
    if not gotos and not status:
        return "no goto/status events captured."

    # First goto (by recv order) per cmd_id, and first ARRIVED / first DRIVING.
    # *_wall are system-clock stamps for the desktop-vs-downstream split;
    # goto_ts is the goto's embedded send time (desktop.nav's wall clock).
    first_goto: Dict[int, float] = {}
    first_goto_ts: Dict[int, float] = {}
    for e in gotos:
        cid = int(e["msg"].get("cmd_id", -1))
        first_goto.setdefault(cid, e["recv_t"])
        if cid not in first_goto_ts and e["msg"].get("ts") is not None:
            first_goto_ts[cid] = float(e["msg"]["ts"])

    first_arrived: Dict[int, float] = {}
    first_arrived_wall: Dict[int, float] = {}
    first_driving: Dict[int, float] = {}
    first_driving_wall: Dict[int, float] = {}
    # Per-cmd_id leg length (first dist_remaining seen) and the (dist, v) cloud.
    leg_len: Dict[int, float] = {}
    dv: List = []   # (dist_remaining_m, v_mps) over all DRIVING ticks
    for e in status:
        m = e["msg"]
        cid = int(m.get("cmd_id", -1))
        st = m.get("state")
        if st == "ARRIVED":
            first_arrived.setdefault(cid, e["recv_t"])
            if e.get("recv_wall") is not None:
                first_arrived_wall.setdefault(cid, e["recv_wall"])
        if st == "DRIVING":
            first_driving.setdefault(cid, e["recv_t"])
            if e.get("recv_wall") is not None:
                first_driving_wall.setdefault(cid, e["recv_wall"])
            d = m.get("dist_remaining_m")
            v = m.get("v_mps")
            if d is not None:
                leg_len.setdefault(cid, float(d))
                if v is not None:
                    dv.append((float(d), float(v)))

    out: List[str] = []
    out.append(f"captured: {len(gotos)} gotos, {len(status)} status, "
               f"{len(first_goto)} distinct cmd_ids")

    # Gaps: for each arrived cmd N, find the next cmd_id that has a goto after it.
    cmd_ids = sorted(first_goto.keys())
    cmd_gaps: List[float] = []
    motion_gaps: List[float] = []
    desk_splits: List[float] = []   # received ARRIVED -> sent next goto (desktop-side)
    down_splits: List[float] = []   # sent goto -> robot driving again (link + Pi)
    out.append("")
    out.append("sub-goal handoff gaps (ARRIVED -> next):  "
               "[desk] = recv ARRIVED -> sent goto, [down] = sent goto -> driving")
    out.append(f"  {'cmd':>6} {'arrived_t':>10} {'next_goto':>10} "
               f"{'cmd_gap':>8} {'motion':>8} {'desk':>8} {'down':>8}")
    for cid in cmd_ids:
        at = first_arrived.get(cid)
        if at is None:
            continue
        nxt = [c for c in cmd_ids if c > cid and first_goto[c] >= at]
        if not nxt:
            continue
        ncid = nxt[0]
        cg = first_goto[ncid] - at
        cmd_gaps.append(cg)
        mg_s = ""
        if ncid in first_driving:
            mg = first_driving[ncid] - at
            motion_gaps.append(mg)
            mg_s = f"{mg*1000:6.0f}ms"
        # Wall-clock split (same desktop clock for recv_wall and goto ts).
        desk_s = down_s = ""
        aw = first_arrived_wall.get(cid)
        gts = first_goto_ts.get(ncid)
        if aw is not None and gts is not None:
            dk = gts - aw
            if -0.05 < dk < 5.0:        # sane window; guards clock oddities
                desk_splits.append(dk)
                desk_s = f"{dk*1000:6.0f}ms"
            dw = first_driving_wall.get(ncid)
            if dw is not None:
                dn = dw - gts
                if -0.05 < dn < 5.0:
                    down_splits.append(dn)
                    down_s = f"{dn*1000:6.0f}ms"
        out.append(f"  {cid:>6} {at:>10.3f} {first_goto[ncid]:>10.3f} "
                   f"{cg*1000:6.0f}ms {mg_s:>8} {desk_s:>8} {down_s:>8}")

    def _stats(name: str, xs: List[float]) -> str:
        if not xs:
            return f"  {name}: (none)"
        xs2 = sorted(xs)
        p50 = xs2[len(xs2) // 2]
        p90 = xs2[min(len(xs2) - 1, int(0.9 * len(xs2)))]
        return (f"  {name}: n={len(xs)} min={min(xs)*1000:.0f} "
                f"p50={p50*1000:.0f} p90={p90*1000:.0f} "
                f"max={max(xs)*1000:.0f} ms")

    out.append("")
    out.append("gap distribution:")
    out.append(_stats("command gap (ARRIVED->next goto) ", cmd_gaps))
    out.append(_stats("motion gap (ARRIVED->driving)    ", motion_gaps))
    out.append(_stats("  split: desktop (recv->send)    ", desk_splits))
    out.append(_stats("  split: downstream (send->drive)", down_splits))
    if desk_splits and down_splits:
        import statistics as _st
        dk, dn = _st.median(desk_splits), _st.median(down_splits)
        bigger = "desktop-side (tick wait + Tier-2 compute)" if dk > dn else \
                 "downstream (link RTT + Pi processing)"
        out.append(f"  -> dominant: {bigger}")

    # Sub-goal leg length: if the median leg is shorter than slowdown_distance_m,
    # the ramp covers most of every leg and the robot never reaches v_max.
    out.append("")
    lens = sorted(leg_len.values())
    if lens:
        n = len(lens)
        out.append(f"sub-goal leg length (initial dist_remaining_m): n={n} "
                   f"min={lens[0]:.3f} p25={lens[n//4]:.3f} "
                   f"p50={lens[n//2]:.3f} p75={lens[3*n//4]:.3f} max={lens[-1]:.3f}")

    # Commanded speed vs distance-to-goal: the actual ramp. Read v_mps directly
    # (no inference). Onset = the largest dist bucket where mean v first falls
    # below 90% of v_max — i.e. how early the slowdown bites today.
    if dv:
        vmax = max(v for _, v in dv)
        below = sum(1 for _, v in dv if v < 0.8 * vmax)
        zero = sum(1 for _, v in dv if v == 0.0)
        out.append(f"commanded v_mps: max={vmax:.3f}  "
                   f"ticks <80% v_max: {100*below/len(dv):.0f}%  "
                   f"ticks at v=0 (rotate-in-place): {100*zero/len(dv):.0f}%")
        buck: Dict[float, List[float]] = {}
        for d, v in dv:
            buck.setdefault(round(d * 20) / 20, []).append(v)
        onset = None
        for b in sorted(buck):
            mean_v = sum(buck[b]) / len(buck[b])
            if mean_v < 0.9 * vmax and (onset is None or b > onset):
                onset = b
        out.append("  v_mps vs dist_remaining (0.05 m buckets):")
        out.append(f"    {'dist':>6} {'n':>4} {'v_mean':>7} {'%vmax':>6}")
        for b in sorted(buck):
            if b > 1.0:
                continue
            mv = sum(buck[b]) / len(buck[b])
            out.append(f"    {b:>6.2f} {len(buck[b]):>4} {mv:>7.3f} "
                       f"{100*mv/vmax:>5.0f}%")
        if onset is not None:
            out.append(f"  -> ramp bites by ~{onset:.2f} m. Set "
                       f"slowdown_distance_m <= leg p50; set "
                       f"D_margin >= v_max * motion_gap(p90).")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router", default=None,
                   help="Zenoh router endpoint (overrides $ZENOH_CONNECT)")
    p.add_argument("--out", default=None,
                   help="JSONL output path (default: drive_timing_<mono>.jsonl)")
    p.add_argument("--analyze", default=None, metavar="FILE",
                   help="Re-analyze an existing capture; no live session.")
    args = p.parse_args()

    if args.analyze:
        events = []
        with open(args.analyze, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        print(analyze(events))
        return 0

    out_path = args.out or f"drive_timing_{int(time.monotonic())}.jsonl"
    router = resolve_router(args.router)
    print(f"drive_timing_logger: router={router} out={out_path}")
    session = open_session(router)
    fh = open(out_path, "w", buffering=1, encoding="utf-8")
    cap = _Capture(fh)
    subs = [
        zenoh_helpers.declare_subscriber_json(session, GOTO_KEY, cap.on_goto),
        zenoh_helpers.declare_subscriber_json(session, STATUS_KEY, cap.on_status),
    ]
    print("capturing… drive a patrol, then Ctrl-C to stop and summarize.")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        while not stop.wait(1.0):
            pass
    finally:
        for s in subs:
            try:
                s.undeclare()
            except Exception:
                pass
        fh.close()
        try:
            session.close()
        except Exception:
            pass
    print("\n" + analyze(cap.snapshot()))
    print(f"\nraw capture: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

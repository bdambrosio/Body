# Pi-side spec for nav Phases 0–3

Status: spec only. Per Bruce's instruction (2026-04-26), no Pi-side changes will be made by the desktop side. This document captures what would be wanted on the Pi *eventually*; desktop implementations route around the absence in v1.

Phases covered:

- Phase 0 — test plan (no Pi changes).
- Phase 1 — mission robustness (pose-age guard, replan cadence, recovery scaffold).
- Phase 2 — motion primitives (360-rotate, back-up).
- Phase 3 — streaming RGB.

## Phase 1 — mission robustness

### 1a. Pose-age guard

**Desktop solution today:** Reuses `FuserController.status_summary()["ages"]["odom"]` (local receive time of last odom). When `> pose_age_threshold_s` (default 0.50 s, configurable), mission transitions to PAUSED("no_pose") and zeroes cmd_vel.

**No Pi change required for v1.** The age is measured in desktop wall-clock from the local arrival time of `body/odom`, so Pi clock drift doesn't matter.

**Spec for future improvement (not required):** Pi-side `body/health` topic (or extended fields on existing `body/heartbeat`) carrying per-stream "last published at" Pi-monotonic timestamps. Lets desktop distinguish "Pi is publishing but Zenoh is delayed" from "Pi is silent." Low priority.

### 1b. Replan cadence

**Desktop-only.** Replan already runs every redraw tick (`main_window.py`'s `_replan` call). Phase 1b refactors the failure path (don't immediately fail mission; emit event) but does not need Pi.

### 1c. Recovery scaffold

**Desktop-only.** Recovery actions consume Pi data (pose, costmap) but don't write back to Pi state.

## Phase 2 — motion primitives

### 2a. 360-rotate primitive

Drives via the existing `chassis.set_cmd_vel(0, omega)` interface. Yaw-integration uses the same pose source the follower uses. **No Pi change required.**

**Caveat to flag back to the Pi side:** at default omega 0.30 rad/s, the IMU + scan-match should keep lock comfortably. If the Pi team's scan-match validation (per `project_body_slam_promotion.md`) reveals a higher-omega ceiling, document it so we can raise the primitive's default if it would speed up explore-mode init.

### 2b. Back-up primitive

Drives via `chassis.set_cmd_vel(-v, 0)`. **No Pi change required for v1.**

**Stall detection — already shipped on the Pi.** Pi note 2026-04-26: `body/motor_state` publishes at 50 Hz with `stall_detected`, `left_pwm` / `right_pwm`, `left_dir` / `right_dir`, `e_stop_active`, and `cmd_timeout_active`. Desktop already subscribes (`StubController._on_motor_state` → `state.motor_state`). For test plan B4 and stall-aware recovery, the desktop side just needs to read `state.motor_state["stall_detected"]` and act on it — no new topic, no rename. The earlier "body/motor_status" reference in this spec was wrong; no Pi change wanted on this front.

### 2c. Wire primitives into recovery

**Desktop-only.**

## Phase 3 — streaming RGB

This is the main place Pi *could* help, but v1 routes around it.

### v1 approach (no Pi changes)

Desktop-side timer at the configured rate (default 2 Hz) calls a new
`StubController.request_rgb_streaming()` method that publishes the same
`{action: capture_rgb, request_id: ...}` payload as `request_rgb()` but
**does not** touch `pending_rgb_request_id`. The reply handler
(`_on_oakd_rgb`) accepts replies whenever pending is None, so streaming
replies all flow through to `state.rgb_jpeg` / `rgb_ts`. On-demand
"Request RGB" continues to use the pending-tracking path so the user
sees "awaiting RGB reply…" while a manual capture is in flight.

Per-frame cost on the Pi: one `_process_oakd_config_queue` iteration —
queue read, drain-latest from the RGB queue, `cv2.imencode`,
`zenoh.put`. The IMU loop already calls `_process_oakd_config_queue`
on every iteration (every ~10 ms based on `interval_s`); a request
queued at 2 Hz hits at most one capture per 500 ms.

**Bandwidth:** 320×240 JPEG q=85 ≈ 12–25 KB; at 2 Hz that's 25–50 KB/s,
small fraction of WiFi.

**Throughput risk:** if Pi processes config every 10 ms but we only
queue 1 request per 500 ms, per-frame jitter is small. If desktop ever
queues a *backlog* (e.g. operator toggles streaming on while a slow
network has a queue of pending requests), the Pi will drain them
back-to-back. To prevent this, desktop drops a streaming-tick request
if a streaming request is already in-flight (controller-side flag,
separate from the on-demand pending field). Spec'd in the desktop
implementation.

### Optional Pi-side improvement (not required for Phase 3)

If desktop polling proves to be too noisy (capture_rgb log spam, queue
churn, or Pi can't keep up), add a streaming-mode action:

```
body/oakd/config:
  { "action": "stream_rgb_start", "rate_hz": 2.0, "width": 320, "height": 240 }
  { "action": "stream_rgb_stop" }
```

Pi behavior:

- On `stream_rgb_start`: spawn or signal a streaming worker that
  publishes `body/oakd/rgb` at `rate_hz`, encoding from the same
  `rgb_queue` used by `capture_rgb`. Each frame uses a synthetic
  request_id of form `stream:<seq>`.
- On `stream_rgb_stop`: halt the streaming worker.
- The streaming worker shares `rgb_queue` with on-demand `capture_rgb`;
  on-demand requests interleave naturally (the queue holds the latest
  frame; capture_rgb takes whichever is latest at call time).
- Re-issue of `stream_rgb_start` updates rate without restart.
- If the Pi process restarts, streaming is implicitly off until desktop
  re-issues `stream_rgb_start` (no persistent state).

Schema additions in `body/lib/schemas.py`:

```python
def oakd_rgb_stream_start(rate_hz: float, width: int, height: int) -> dict: ...
def oakd_rgb_stream_stop() -> dict: ...
```

Reply payloads on `body/oakd/rgb` are the same shape as today's
`oakd_rgb_capture_ok` — desktop doesn't need a new code path.

**Why this is optional, not required:** the desktop polling approach
delivers the same operator UX (a 2 Hz feed when out of sight) with no
Pi changes. The Pi-side improvement saves per-frame request-publish
overhead and removes per-capture log lines, which is a quality-of-life
win on the Pi but does not change desktop capability. Re-evaluate after
Phase 6 testing if the Pi cost shows up as a measurable problem.

## Summary

| Phase | Pi change required for v1? | Pi-side want list |
|-------|---------------------------|-------------------|
| 0 — test plan | No | — |
| 1a — pose-age guard | No | (low) `body/health` per-stream Pi-monotonic timestamps |
| 1b — replan cadence | No | — |
| 1c — recovery scaffold | No | — |
| 2a — 360-rotate | No | (info) Pi confirms scan-match omega ceiling |
| 2b — back-up | No | already shipped: `body/motor_state` at 50 Hz with `stall_detected` (consume on desktop side) |
| 2c — wire to recovery | No | — |
| 3 — streaming RGB | No | (low) `stream_rgb_start/stop` actions for log-quietness |

No item is a blocker. The Pi-side want list is a backlog for future Pi sessions, not a precondition for landing Phases 0–3 on the desktop.

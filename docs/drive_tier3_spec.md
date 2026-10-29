# Tier-3 reactive drive (Pi-side) — interface spec

Tier-3 is the lowest tier of the hierarchical navigator (see
`docs/tier_contract.md`): given **one subgoal**, drive there on the live
**rasterized lidar scan** (not the fused `local_map`, which lags while
moving), avoiding what is seen, with no dependence on the global map or
particle-filter pose. It runs **on the Pi** (`body.local_drive`) so the
reactive loop sits next to the freshest scan with no network round-trip.

Production sender is the desktop hierarchical drive
(`desktop/nav/hierarchical_drive.py`); the `desktop.pi_drive` debug consoles
emit the identical `body/drive/goto` command.

## Topics

### `body/drive/goto` (sender → Pi) — `schemas.drive_goto`
| field | type | meaning |
|---|---|---|
| `ts` | float | send time (wall clock) |
| `cmd_id` | int | monotonic; a higher id supersedes a lower one. A **cancel advances the watermark** to its own id, so a delayed duplicate goto cannot re-arm a revoked goal. A goto with id **< watermark** → one-shot `REJECTED` status (carrying `watermark`); id **== watermark** is a resend of the command already taken → absorbed idempotently (no restart, no REJECTED) |
| `frame` | str | `"odom"` or `"body"`; anything else → `REJECTED` |
| `x_m`, `y_m` | float | goal point in the odom frame (`frame:"odom"`), or in the robot body frame at `anchor_ts` (`frame:"body"`) |
| `anchor_ts` | float? | `frame:"body"` only: wall-clock time the body-frame point refers to (normally the `scan.ts` it was computed from). The Pi anchors the goal in odom with **its own odom ring buffer interpolated at `anchor_ts`**; outside the ~3 s buffer it falls back to the latest odom and logs |
| `final_heading_rad` | float? | face this on arrival; omit = don't care |
| `arrival_tol_m` | float? | per-command override of config |
| `v_max` | float? | per-command speed override |
| `kind` | str | `"goto"` \| `"cancel"` \| `"stop"` |

**Why odom frame.** A body-frame point is stale by the time the Pi acts;
a world/global point would reintroduce the dependence on the global map
we are trying to escape. The sender converts a body-frame point to odom
using the live odom pose at send time (`DriveClient.send_goto_from_body`),
so a constant world↔odom offset cancels. The Pi then tracks the fixed odom
point as the robot moves, using its own odom — only a *coarse direction*
ever crosses from any higher tier into the metric drive loop.

**IMU yaw correction.** Wheel odom is blind to externally-forced rotation
(a floor ridge kicking the chassis, wheel slip), which would rotate the
goal's body-frame bearing by exactly the missed angle. The heading used for
the goal transform is therefore wheel θ plus the IMU-vs-wheel yaw divergence
accumulated since the goal started (`ImuYawCorrector`, fed by `body/imu`).
The baseline resets per goal so long-term IMU drift never enters, and the
published `body/odom` contract is untouched. No IMU (or a stale one: `ts`
older than `imu_stale_s` 0.3 s or `seq` not advancing) → zero correction,
wheel-only behavior. The baseline is not captured while |vtheta| > 0.15 rad/s
(unpaired samples), an IMU `session_id` change / `seq` decrease re-baselines
(no yaw step), and a sliding-window guard (`local_drive.imu_guard`: wheels
turned > 0.35 rad in 1 s while the IMU moved < 0.02 rad) falls back to wheel
heading until the two agree again.

**Odom resets.** `body/odom` carries `session_id`/`seq`; a `session_id`
change or `seq` decrease (motor_controller restart) drops the active goal →
`FAULT:odom_reset`. Odom with `source:"none"` is unusable → `FAULT:odom_stale`.

**Time-consistent transforms.** Each scan is planned in its own body frame,
using the odom pose interpolated at the scan's mid-sweep `ts`; the robot's
current pose is expressed in that frame (A\* start, swept-arc start), and the
path is re-expressed in the live body frame for following.

### `body/drive/status` (Pi → sender) — `schemas.drive_status`
Published every control tick.
| field | type | meaning |
|---|---|---|
| `cmd_id` | int | the goal being serviced (0 = none) |
| `state` | str | `IDLE`\|`DRIVING`\|`ARRIVED`\|`BLOCKED`\|`CANCELED`\|`FAULT`\|`REJECTED` |
| `watermark` | int? | `REJECTED` only: the current supersede watermark (seed the next id above it) |

Every accepted goto/cancel is acknowledged by the **next tick's** status
carrying its `cmd_id` (≤ 100 ms). `REJECTED` is published once, under the
rejected id; it says nothing about the active goal.
`ARRIVED` is published for one tick, then the goal drops → `IDLE`.
`CANCELED` is published for one tick after `cancel`/`stop` clears an **active**
goal (under the revoked goal's `cmd_id`), then `IDLE`. Senders that treat
`IDLE@cmd_id` as "sub-goal done" must handle `CANCELED` as revoke, not success.
A goal dropped by the Pi (e-stop, odom reset, internal error) — or a goto
refused during e-stop — is reported as `FAULT:<reason>` under its `cmd_id`
**until the next goto/cancel**; it never decays to `IDLE`.
| `goal_body_xy` | [float,float]? | active goal in the *live* body frame (for display) |
| `dist_remaining_m` | float | range to goal (on a best-effort `ARRIVED`: the residual) |
| `v_mps`, `omega_radps` | float | commanded velocity this tick |
| `blocked_reason` | str? | BLOCKED: `boxed_in`\|`start_blocked`\|`goal_out_of_map`\|`swept_block`\|`depth_block`\|`no_progress`\|`deadline`\|`rotate_timeout`\|`no_scan`; FAULT: `odom_stale`\|`odom_reset`\|`e_stop`\|`internal_error` |
| `mode` | str? | `follow`\|`rotate`\|`realign`\|`held`\|`plan` — `rotate`/`realign` = turning in place (the desktop must not re-pick on bearing drift then) |
| `plan_reason` | str? | `ok`\|`snapped`\|`frontier`; on `ARRIVED`, `snapped`/`frontier` = reached the best reachable stand-in for an unreachable goal |
| `path_body_xy` | list? | the A\* path being followed (display/inspector) |
| `build` | str? | Pi git sha — the desktop flags a stale deploy |

## cmd_vel ownership (arbitration)

Exactly one producer of `body/cmd_vel` at a time:

- While a `goto` is active, **Tier-3 owns `body/cmd_vel`.** The desktop
  keeps `StubController.live_command` **off**, so it publishes no cmd_vel
  and cannot fight the Pi. The desktop still publishes **heartbeat**
  (always, while connected) — the watchdog e-stops without it, so heartbeat
  is required for *any* motion including Pi-initiated drives.
- On `cancel`/`stop` (or goal drop), Tier-3 **publishes a zero cmd_vel
  immediately** — the motors never coast on the last command waiting for
  the 500 ms cmd timeout.
- Manual teleop (live_command on) is mutually exclusive with an active
  goto. The Pi watchdog / e-stop / motor timeout remain supreme over both.
- After a **terminal** block (`deadline`, `rotate_timeout`) Tier-3 sends a
  few zero commands and then goes **silent** on cmd_vel (status keeps
  reporting BLOCKED), so it never fights operator teleop with zeros.
- **E-stop visibility.** local_drive subscribes `body/emergency_stop`,
  `body/status` (`e_stop_active`) and `body/motor_state` (`e_stop_active`,
  `stall_detected`). While any is active it drops the goal, publishes
  `FAULT:e_stop` every tick and refuses gotos; it clears as soon as the
  reports go false (the robot boots e-stop-latched until `body/estop_rearm`,
  so `FAULT:e_stop` at startup is normal).
- Every tick runs under try/except: an exception commands zero, drops the
  goal with `FAULT:internal_error` and the loop keeps running.

## Behaviour

Each control tick (10 Hz):

The per-goal logic is `body/lib/tier3_controller.py` (`Tier3Controller`,
closed-loop tested in `body/test_tier3_controller.py`); `body/local_drive.py`
is transport + lifecycle.

1. Rasterize the latest lidar scan into a body-frame int8 grid
   (`scan_raster.rasterize_scan_msg` — `unknown_bins` stay unknown) and seed
   the robot's footprint disk (at the scan origin and at the current pose)
   clear.
2. **One obstacle model** for planner and vetoes: build the costmap (no
   speckle removal by default; lethal radius = footprint + res/√2 so cell
   quantization can't put the body inside the veto footprint; unknown cells
   cost ≥ their halo; unknown cells the veto would refuse — unexplained,
   within `unknown_block_range_m` — and unknown cells touching a hit are
   impassable) and run **local A\*** — `plan_local` → `astar_toward`. If the
   goal cell is lethal/unreachable, A\* ends at the reachable cell nearest
   to it (ties → cheapest path), reported as `snapped`/`frontier`; that
   endpoint is latched in odom while it stays reachable. Standing at it →
   `ARRIVED` (plan_reason `snapped`/`frontier`, residual in
   `dist_remaining_m`). `boxed_in` only when the robot cannot move at all.
   **Path hysteresis**: the previous path is kept while collision-free and
   ≤ (1+`path_hysteresis`)× the fresh plan's cost, and always while turning
   in place to line up with it.
3. Pure-pursuit follow: steer to a lookahead point on the path
   (`steer_to_body_point`, rotate-in-place hysteresis for large bearings,
   with a **direction latch** so a target near ±π does not flip ω).
4. **Swept-footprint veto as last resort** (`body/lib/drive_safety.py`,
   effective radius == the A\* footprint, preview arc capped at the
   remaining path length, `preview_time_s` 0.5 s, unknown within
   `unknown_explained_cells` of a hit ignored): on a block, retry the
   pure-pursuit arc, then `v_min`, then a half lookahead; only if all are
   vetoed, re-aim in place toward the lookahead (`mode=realign`,
   cumulative `swept_realign_timeout_s` between translations), else stop +
   `BLOCKED:swept_block`. The veto only stops, never steers.
5. **Depth near-field veto** (`body/lib/depth_veto.py`, config
   `local_drive.depth_veto`): while translating, if fresh `body/oakd/depth`
   shows enough obstacle hits within the footprint half-width of the
   **commanded arc** (default ~0.08–0.80 m arc length), stop with
   `BLOCKED:depth_block` (no realign). Points below
   `structure_free_height_m` count only where they show vertical structure
   (≥ `min_vertical_extent_m` height span in a `structure_cell_m` floor
   cell), so pitch error / ramps do not fire it; the hit streak advances
   only on a new depth frame (`seq`, else `ts`). Fail-open when depth is
   missing/stale/rotating fast; lidar still owns planning. This is **not**
   consumption of `body/map/local_2p5d`.

Terminal conditions:

- Within `arrival_tol_m` → `ARRIVED` for one tick (after the optional
  `final_heading_rad` rotate), then the goal drops → `IDLE`.
- No decrease of the **remaining path length** (not straight-line distance,
  so detours count) while translating for `no_progress_timeout_s` →
  `BLOCKED:no_progress`.
- **Rotation budget** `rotate_timeout_s` (default 10 s, 0 disables): time
  since the last real translation (odom displacement ≥ `translate_eps_m`)
  while turning in place (rotate, realign, final-heading) → terminal
  `BLOCKED:rotate_timeout`. Rotation never resets this clock.
- **Per-goal deadline** `goal_deadline_s` (default 30 s, 0 disables) →
  stop + terminal `BLOCKED:deadline`, evaluated before the final-heading
  rotate. The goal stays active (so the status doesn't decay to IDLE,
  which senders read as success) until a superseding goto or cancel
  clears it.
- HO-3 breakpoint arms carry a lease (`handoff_arm_lease_s`, default 5 s;
  the inspector re-sends `arm` ~1 Hz) so a closed inspector cannot leave
  Tier-3 `held` forever.
- Stale odom/scan (0.5 s) → `FAULT:odom_stale` / `BLOCKED:no_scan`, motors
  zeroed.

Config lives under `config.json` → `local_drive`. The planner/raster
sections are built through `body/lib/drive_config.py` — the **same builders
the desktop uses** to model Tier-3 (contract I8 in `docs/tier_contract.md`).

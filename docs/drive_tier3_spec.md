# Tier-3 reactive drive (Pi-side) — interface spec

Tier-3 is the lowest tier of the hierarchical navigator (see the nav
roadmap): given **one subgoal that is currently observable**, drive there
on the live body-frame `local_map`, avoiding what is seen, with no
dependence on the global map or particle-filter pose. It runs **on the
Pi** (`body.local_drive`) so the reactive loop sits next to the freshest
`local_map` with no network round-trip.

In Stage A the *operator* (via the `desktop.pi_drive` UI) plays the role
of the upper tiers by clicking the next subgoal. Later, the Tier-2
visible-waypoint stepper emits the identical `body/drive/goto` command —
the contract below does not change.

## Topics

### `body/drive/goto` (sender → Pi) — `schemas.drive_goto`
| field | type | meaning |
|---|---|---|
| `ts` | float | send time (wall clock) |
| `cmd_id` | int | monotonic; a higher id supersedes a lower one |
| `frame` | str | `"odom"` (only value in v1) |
| `x_m`, `y_m` | float | goal point in the odom frame |
| `final_heading_rad` | float? | face this on arrival; omit = don't care |
| `arrival_tol_m` | float? | per-command override of config |
| `v_max` | float? | per-command speed override |
| `kind` | str | `"goto"` \| `"cancel"` \| `"stop"` |

**Why odom frame.** A body-frame point is stale by the time the Pi acts;
a world/global point would reintroduce the dependence on the global map
we are trying to escape. The sender converts a body-frame click to odom
using the displayed `local_map` message's `anchor_pose`. The Pi then
tracks the fixed odom point as the robot moves, using its own odom — so
only a *coarse direction* ever crosses from any higher tier into the
metric drive loop; the point itself was observed live.

### `body/drive/status` (Pi → sender) — `schemas.drive_status`
Published every control tick.
| field | type | meaning |
|---|---|---|
| `cmd_id` | int | the goal being serviced (0 = none) |
| `state` | str | `IDLE`\|`DRIVING`\|`ARRIVED`\|`BLOCKED`\|`CANCELED`\|`FAULT` |
| `goal_body_xy` | [float,float]? | active goal in the *live* body frame (for display) |
| `dist_remaining_m` | float | range to goal |
| `v_mps`, `omega_radps` | float | commanded velocity this tick |
| `blocked_reason` | str? | `swept_block`\|`no_progress`\|`odom_stale`\|`out_of_local_map` |

## cmd_vel ownership (arbitration)

Exactly one producer of `body/cmd_vel` at a time:

- While a `goto` is active, **Tier-3 owns `body/cmd_vel`.** The
  `desktop.pi_drive` UI keeps `StubController.live_command` **off**, so
  the desktop publishes no cmd_vel and cannot fight the Pi. The desktop
  still publishes **heartbeat** (always, while connected) — the watchdog
  e-stops without it, so heartbeat is required for *any* motion including
  Pi-initiated drives.
- Manual teleop (live_command on) is mutually exclusive with an active
  goto: enabling teleop cancels the active goto; issuing a goto requires
  teleop off. The Pi watchdog / e-stop / motor timeout remain supreme
  over both.

## Behaviour (v1, Stage A)

- Steering is straight-line pure-pursuit toward the body-frame goal
  (rotate-in-place when the bearing is large, else arc), gated by the
  swept-footprint check on the live `local_map`.
- **No dynamic avoidance yet** (Stage D): if the swept check fires, the
  driver stops and reports `BLOCKED:swept_block`; the sender re-picks a
  visible subgoal. This is correct because every subgoal is line-of-sight
  clear when chosen.
- Arrival: within `arrival_tol_m` → `ARRIVED`, stop (and rotate to
  `final_heading_rad` if given). No goal progress for
  `no_progress_timeout_s` → `BLOCKED:no_progress`. Stale odom → `FAULT`.
  Goal outside local_map coverage → `BLOCKED:out_of_local_map`.

Config lives under `config.json` → `local_drive` (see `body/local_drive.py`).

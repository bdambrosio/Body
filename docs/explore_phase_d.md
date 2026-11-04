# Explore Phase D: explore the mapped house (plan)

*2026-11-03. Approved by the owner 11-03 with all five recommendations (see Decisions). This doc is the source of truth for Phase D. The
original explore plan file in `~/.claude/plans/` was overwritten, and its
Phase D/E text survives only in the session transcript; it is summarised here.*

## Goal

**Explore** means: go look at the whole mapped house, report what changed,
and go and look more closely at anything new. **Patrol** means: drive the
route I authored.

Today's patrol only sees what is visible from the `hallway_loop` route.
Explore differs in three ways:

1. **It picks its own viewpoints**, choosing the parts of the house it has
   seen least recently, so rooms off the patrol route get looked at.
2. **It investigates.** When something new turns up, it drives over for a
   second viewpoint and a camera look, which confirms or dismisses it.
3. **It ends with a summary**: coverage, plus the changes found, split into
   "new and confirmed", "dismissed" and "expected".

Unchanged from the original plan:

* The reference map is the anchor. Explore reports and never edits the map.
* The VLM only labels. It never gates motion.
* Phase D stays inside the mapped area. Unmapped space is Phase E.

## What already exists (and Phase D reuses unchanged)

| need | have |
|---|---|
| Drive to a point, around whatever is there today | `HierarchicalDrive` (path mode): live route on reference prior + session grid, Tier-3 path streaming, route watchdog, escalation ladder, pose doubt → suspend, self-recovery |
| No-go respected | `nogo` is lethal in `world_map/costmap.build_costmap` |
| What was seen, and when | `SessionGrid.last_hit_ts` / `last_free_ts` per cell (this session, good-support scans only) |
| Change reports | `ChangeDetector` → `ChangeReporter` (events.jsonl) → `ChangeLabeler` (crop + closed class list) |
| Is this news? | `ChangeStore` verdicts `novel` / `candidate` / `expected` / `flagged` |
| Final heading at a goal | `final_heading_rad` already passes through to Tier-3 |

The drive takes a "runner" (`PatrolRunner`: `current_target`, `on_arrived`,
`is_terminal_leg`, `next_target_xy_after`, `n`, `wp_index`, `patrol`).
Phase D adds a task layer **above** the drive and makes no structural change
to the drive itself.

## Design

### 1. Task layer (`desktop/nav/tasks.py`, new)

```python
class Task(Protocol):
    name: str
    def next_leg(self, ctx: TaskContext) -> Optional[Leg]   # None = done
    def on_leg_done(self, leg: Leg, outcome: LegOutcome) -> None
    def on_change(self, rec: dict) -> None                 # reporter record (with verdict)
    def status(self) -> dict                               # for the UI / log
    def summary(self) -> dict                              # written at the end

@dataclass
class Leg:
    xy: tuple; purpose: str            # "viewpoint" | "investigate" | "home"
    final_heading: Optional[float]; arrival_tol_m: float
    dwell_s: float = 0.0               # stand still after arrival (look)
    give_up_after: Optional[int] = 1   # ladder cycles before abandoning; None = never
```

* **`TaskRunner`** (owned by `main_window`, ticked from `_on_hier_tick`):
  1. Asks the task for a leg.
  2. Builds a one-waypoint `Patrol` (`loop=False`) and starts a
     `HierarchicalDrive` for that leg.
  3. Watches for `ARRIVED`, `FAILED` or abandon.
  4. Runs the dwell, then reports the `LegOutcome` (`arrived`, `unreachable`
     or `aborted`, plus the time taken) back to the task.

  A fresh drive per leg costs nothing. The drive's `start()` already resets
  per-leg state, and path mode sends one path goal per leg anyway.
* **`PatrolTask`**: today's Go, unchanged. The existing button keeps the
  existing code path; `PatrolTask` exists only so the Task interface has a
  second implementation. Folding Go into the runner is a later cleanup, not
  part of Phase D.
* **Abandoning a leg (the only drive change).** `HierConfig.give_up_after_cycles`
  defaults to `None`, so patrol behaves exactly as today. For explore legs it
  is 1: after verify → unstick → replan has failed once, the drive enters
  `FAILED("abandoned")` instead of the `wait` rung (15–300 s).
  * A patrol marker has to be reached.
  * An explore viewpoint does not: the task marks the target unreachable for
    this run and picks another.

### 2. Viewpoint selection (`desktop/nav/explore.py`, new; pure numpy, unit-testable)

* **Candidates.** Free cells of the planning costmap (no-go and clearance
  already applied), spaced about 0.75 m apart. They must be reachable, which
  one Dijkstra from the robot over the costmap checks; that run also gives
  the travel cost.
* **Visibility.** For each candidate, cast 180 rays out to 4 m on the
  reference map, counting the cells a scan from there would touch. This is
  precomputed once per run, takes about 1 s on the real map, and is cached
  by map key.
* **Stale cells.** A cell is stale if it is observable (on or next to a
  reference wall or obstacle, or free within 4 m of a candidate) and
  `max(last_hit_ts, last_free_ts)` is older than `stale_s`. Never seen this
  session counts as stale.
* **Score.** `gain / (travel_m + c)`, where `gain` is the number of stale
  cells visible from the candidate. It's a greedy next-best-view, re-scored
  after every leg. No tour optimisation: the house is small, and the greedy
  choice self-corrects as the grid fills.
* **Coverage** is the fraction of observable cells that are not stale. It is
  shown in the UI and logged.
* **Done** when any of these holds:
  * coverage ≥ `target_coverage`;
  * no candidate gains ≥ `min_gain_cells`;
  * the time budget runs out.

  Then a `home` leg to `boot_pose`.

Camera coverage is not modelled. The lidar sees 360°, and a change is
labelled when the camera points at it, which is what an investigate leg is
for.

### 3. Investigate legs

These are triggered from `on_change` (the reporter's record, with its store
verdict).

| report | investigate? |
|---|---|
| `new`, verdict `novel` or `flagged` | **yes** |
| `new`, verdict `candidate` (seen before, still learning) | no, report only (the 91 candidates are mostly chair legs) |
| `expected` | no (silent) |
| `missing` | no by default (glass, doors) — see decision 2 |
| `moving` | no (it will have moved; the labeller already tries while in view) |

* **Approach.** Pick a free cell 1.0–1.5 m from the change. Its bearing to
  the change should differ by at least 30° from the robot's bearing when it
  first reported it, so the second viewpoint adds new information. Arrive
  with `final_heading` facing the change, then dwell 3 s. The dwell gives
  the lidar several scans from the new viewpoint and gives the labeller a
  centred frame.
* **Outcome** (logged and written to `events.jsonl` as `investigated`):
  * `confirmed`: the session grid still holds the blob, plus the label.
  * `dismissed`: the cells cleared from the new viewpoint (a phantom or
    viewing-angle artifact). This also goes to the store as negative
    evidence, so a dismissed place does not become a proposal.
  * `gone`: the detector already reported it gone.
  * `unreachable`.
* **Limits.** At most one investigation per change and at most
  `max_investigations` per run. Investigate legs pre-empt viewpoint legs, but
  not a leg in progress: the next leg is chosen after the current one ends.

### 4. Run summary

`~/body-logs/explore/<ts>/summary.json` and a block in the nav log:

* Duration, distance and coverage, plus the viewpoints visited, and any
  abandoned, with reasons.
* Changes by kind × verdict, each investigation with its outcome, and the
  labels.
* The store's proposals from this session.

### 5. UI

* An **Explore** action next to Go, with a budget field (minutes).
* While running, the map shows:
  * the candidate viewpoints, with the chosen one highlighted;
  * investigate targets;
  * a coverage figure in the status bar.
* Stop and Resume work as they do for patrol (same drive, same suspend and
  recovery).

Explore needs no Pi changes.

## Configuration (`ExploreConfig`; defaults are first guesses, tuned on the robot)

| | default | why |
|---|---|---|
| `budget_min` | 15 | a two-lap patrol is about 6 min; the whole house about twice that |
| `stale_s` | session (∞) | "not yet seen this run"; see decision 1 |
| `target_coverage` | 0.90 | the last 10 % are corners behind furniture |
| `min_gain_cells` | 40 (≈ 0.1 m²) | stop chasing slivers |
| `viewpoint_spacing_m` | 0.75 | |
| `investigate_range_m` | 1.0–1.5 | the camera crop resolves a box or a pet well at this range |
| `investigate_dwell_s` | 3.0 | about 20 scans and 3 camera frames |
| `max_investigations` | 6 | |
| `v_max` | as patrol (0.18) | inside the map the drive is proven; Phase E slows to 0.10 |
| `give_up_after_cycles` | 1 | |

## Decisions (owner, 11-03: all as recommended)

1. **Staleness across runs.** Should explore remember what it saw last
   time?
   * *Recommended: no, for now.* Each run aims to cover the house once,
     which is simple and fits "go look at the house".
   * *Alternative:* persist a coarse (0.5 m) last-seen map per map dir, so
     consecutive runs look first where the last run didn't. This is easy to
     add later. It isn't needed until runs are cut short by the budget.
2. **Investigate "missing"?**
   * *Recommended: no.* Today's "missing" reports are glass, doors and
     grazing walls, and investigating them would spend the budget there.
     Revisit after the store has a few more proposals approved.
   * *Alternative:* investigate a missing report only if it is larger than
     0.5 m (a moved bookcase).
3. **Budget and finish.** The default is 15 min, then return to
   `boot_pose`.
4. **Abandon rule.** For explore legs: one full ladder cycle, then give up
   on that target. Patrol stays "never give up".
5. **Start point.** From home only (recommended for the first runs, since
   home is where the boot pose is known), or from anywhere with good pose
   support?

## Build order

1. **Pure modules plus tests.**
   * `explore.py`: candidates, visibility, staleness, scoring, coverage,
     investigate-spot choice.
   * Run the planner offline on the real map with a synthetic session grid
     ("fully seen" against "only the hallway seen") and check the
     viewpoints make sense on a picture.
2. **Task layer.** Add `tasks.py` (`Task`, `Leg`, `TaskRunner`) and
   `ExploreTask`. Test them in a sim with the existing fake DriveIO from
   `test_hierarchical_drive.py` / `test_path_mode.py`.
3. **Drive.** Add `give_up_after_cycles` and `FAILED("abandoned")`, with
   tests that patrol behaviour is unchanged.
4. **Wiring.** Explore action and overlay in `main_window`, summary writer,
   `investigated` event records, and store negative evidence.
5. **Robot.**
   * **Run 1.** An explore run with nothing planted. Check that coverage
     climbs, the viewpoints are sensible, it returns home, and nothing
     regresses in the drive.
   * **Run 2.** A box in a room off the patrol route. It should be found,
     investigated, confirmed and labelled `box or bag`.
   * **Run 3.** A small stuffed animal, plus a walk-past. The animal should
     be labelled `small stuffed animal` and the walk-past should not trigger
     an investigate leg.

## Not in Phase D

* **Unmapped space**, frontiers and map growth by proposal: Phase E.
* **Object commands** ("go to the box"): `ExploreTask`'s investigate leg is
  the building block, but the commands themselves are separate.
* **Folding Go into `TaskRunner`**: a later cleanup.

## As built (11-03, not yet run on the robot)

* **Files.**
  * `desktop/nav/explore.py`: `ExploreMap`, `ExplorePlanner`, `reachable_distances`,
    `investigate_spot`.
  * `desktop/nav/tasks.py`: `Leg`, `TaskRunner`, `ExploreTask`.
  * Explore toolbar action and budget spin box in `desktop/nav/main_window.py`.
  * Tests: `desktop/nav/test_explore.py`.
* **Offline check on the real map** (`map_20261102_140056`, no live
  observations, so only "looked at" counts):
  * Preparing the map takes 0.06 s and finds 88 viewpoints; each next-leg
    Dijkstra takes about 27 ms.
  * Greedy coverage reaches 90 % after 15 legs (~55 m of driving). On the
    robot, coverage should climb faster, because the session grid also
    fills in transit.
  * Added `min_leg_m` 0.5 after the planner chose a 10 cm hop.
* **Drive.**
  * `HierConfig.give_up_after_cycles` makes the drive end in
    `FAILED`, with `fail_reason() == "abandoned"`.
  * `Waypoint.final_heading_rad` is a world heading, honoured in path mode
    on the terminal marker only. Sub-goal mode ignores it.
  * **Frame catch:** Tier-3 compares `final_heading_rad` with its odom
    heading as-is, even for a body-frame goal (`body/local_drive.py`), so
    the desktop sends world θ + (odom θ − world θ) and no Pi change is
    needed. Nothing sent a heading before this.
* **Change pipeline.**
  * `ChangeReporter.add_listener` / `investigated()` (`events.jsonl` records
    of `type: investigated`).
  * `ChangeStore.on_dismissed`: a dismissed sighting gives no session
    credit, and drops an entry it created.
  * `SessionGrid.seen_mask` / `occupied_near`.
* **Summary file:** `~/body-logs/explore/<ts>/explore_summary_<t>.json`,
  plus `explore: …` lines in the nav log.
* **Not built yet:** the map overlay of candidate viewpoints. The status
  line shows the leg purpose, coverage, viewpoints visited and
  investigations.

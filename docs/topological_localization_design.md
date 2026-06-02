# Localization for a Topological (Non-Metric) Tier-1 Map

**Status:** Design note, 2026-06-02. No code yet. Captures a decision and
three design directions; supersedes the "re-capture a metric map" instinct.
**Owner:** Bruce + Claude.
**Why this exists:** the hierarchical planner assumes the Tier-1 map is
**topologically correct but not metrically accurate**. We have been
localizing against that same map with a **metric** scan-correlation MCL
(the particle filter). That is a category error, and it is the root of the
"match on the wrong side / forward-scale mismatch" symptoms seen on
2026-06-02. This note records why, and what to do instead.

---

## 1. The core finding

The new live-scan overlay (cyan scan transformed into the believed pose,
drawn over `clear/blocked/unknown`) showed the live lidar at a **larger
forward scale than the map** in corridors.

Two facts make this decisive:

1. The believed pose only **rigidly** translates/rotates the scan. A rigid
   transform cannot stretch a scan. So a *scale* mismatch is not a
   localization error — it is a genuine **metric disagreement between the
   lidar geometry and the map geometry**. Assuming the lidar reads true
   metres (0.30 % straight-drive cal + depth cam corroborate), the **map**
   is the distorted artifact.
2. The map was built by stamping live scans at the estimator's pose. In a
   **corridor, scan-matching cannot observe along-track (forward) position**
   (documented "along-track unobservability"). So corridor length in the
   map = the odom-integrated distance travelled at capture, uncorrected.
   The map came out forward-compressed; the true-scaled live scan overshoots
   it.

This is the **Tier-1 premise made visible**: the map is topologically right,
metrically loose — *as designed*. The defect is not the map; it is running a
metric global localizer against it.

### 1.1 What scan-matching can and cannot give on this map

The map is true scans stamped at distorted poses, therefore:

- **Locally** (around any point) the map ≈ a real scan → **local/relative
  scan-matching is fine** for tracking motion.
- **Globally** the relative placement of those stamps is distorted → there
  is **no consistent global metric frame**. A global (x, y) in map coords is
  not a meaningful quantity, and a global scan-match search lands in the
  wrong self-similar basin (corridors). This is exactly what `relocate()`
  and the PF wide-search do wrong here.

**The map supports relative pose, not absolute metric pose.** Every design
choice below follows from that one sentence.

---

## 2. What the hierarchical planner actually needs from pose

The drive consumes the global pose in only two places:

- **Bearing to the next waypoint** (`bearing_to_waypoint`).
- **Arrival** (`_dist(pose, wp) <= waypoint_tol_m`).

Everything metric below that — Tier-2 visibility (furthest live-visible free
point along the bearing) and Tier-3 (A*/follow on live odom) — already runs
on **live scan + live odom**, which are metrically true and **map-independent**.
The architecture deliberately pushed the metric work into the map-free
layers. The two leaks above are the only places a global metric pose is
assumed, and both can be reframed.

---

## 3. Direction A — LPR-backed / locally-metric `world_pose()` provider

`PFPoseProvider` is documented in `hierarchical_drive.py` as the *"re-align
seam — swap for an LPR-backed provider later; the orchestrator only knows
`world_pose()`."* This is that swap.

**Model:** the map frame is a chart that is only *locally* Euclidean — a
topological graph with an approximate metric embedding, re-zeroed at each
landmark. Two clocks under `world_pose()`:

- **Continuous (every tick):** integrate odom + IMU yaw to propagate the
  pose *in the map frame*. Metrically true locally, smooth, drifts globally —
  fine. The PF can remain this **local** filter; what we drop is its
  **global wide-area correction**, which the non-metric map breaks.
- **Discrete (at nodes):** **LPR.** Each distinctive waypoint (junction,
  doorway, room) carries a stored **local signature** (scan/geometry captured
  there). When the live scan confidently matches a node's signature,
  **re-anchor**: set the pose to that node's map coords + a *local* yaw/xy
  fine fit (which the map supports locally). That discrete snap corrects
  accumulated odom drift — instead of continuous global correlation.

`world_pose()` returns the anchored-dead-reckoned pose. Extend health: None
on stale odom (already done) + a low-confidence flag when drift-since-anchor
grows large (let the drive slow/hold rather than trust a far-extrapolated
bearing).

### 3.1 Arrival-by-node replaces arrival-by-distance

`_dist(pose, wp) <= tol` assumes a metric pose in a metric frame. Split by
waypoint type:

- **Distinctive node:** arrival = LPR recognizes the place (confidence high),
  optionally AND a loose dead-reckon gate. You don't need corridor length
  right; you need to recognize the junction at its end.
- **Mid-corridor pass-through point** (nothing to recognize): arrival = loose
  dead-reckon distance + Tier-2 still making progress along the bearing —
  "ride to the next distinctive node." This *is* the corridor reality.

Bearing stays computed in the map frame, but because the pose is re-anchored
at every node passed, it is locally valid and Tier-2 does the precise aiming
as it already does.

---

## 4. Implications for `.nav` Re-localize / Set location

- **Global `relocate()` (Re-localize) is the manual twin of the broken
  global MCL.** A wide xy/θ search for best scan-correlation against a
  distorted, self-similar map snaps to the wrong basin; its
  `min_improvement` gate is meaningless when the field itself is distorted.
  **Demote it.** Recast "Re-localize" as **"recognize which node I'm at"** —
  LPR over the discrete landmark signatures, not a dense grid sweep.
- **"Set location" (`relocate_at`) is already the right shape — keep and lean
  on it.** The operator supplies the global/topological anchor (you-are-here);
  the scan-match does only a *local* yaw/fine-xy fit, which the map supports.
  It is the manual version of LPR re-anchoring. Flip the UI emphasis away
  from "search the whole map" and toward "assert/recognize the node, refine
  locally."

---

## 5. Direction C — `map_editor` "Recognize": supervised local re-metricization

Idea (Bruce, 2026-06-02): in `map_editor`, drive the bot, manually adjust the
asserted location/yaw, then a **Recognize** button edits the *local occupancy*
so the asserted pose becomes the highest-scoring pose for the live scan —
"smart editing of the map to embed LPR in the map." It is the **inverse of
Set-location**: Set-location moves the pose to fit the map; Recognize moves
the map to fit an asserted ground-truth pose.

### 5.1 Why it's attractive

- Converts the original failure into the fix: the map went bad because scans
  were stamped at a *drifting estimator's* poses; Recognize re-stamps with
  *operator-corrected* poses — same mechanism, trustworthy inputs. Wherever
  you recognize, the MCL thereafter peaks at the true pose.
- Reuses existing machinery: occupancy edit + the existing "regenerate
  likelihood/distance field on save." The **live scan overlay** (shipped
  2026-06-02) is the natural verify-then-bake UI: nudge until the cyan scan
  matches the room you *know* you're in, then commit.
- A human resolves the along-track corridor ambiguity no automatic method can.

### 5.2 Caveats — read before relying on it

1. **Local peak, not global uniqueness.** Carving local occupancy so the scan
   peaks at the asserted pose works *within a window*; it cannot make that
   pose the *global* argmax in a featureless corridor. So this hardens
   **local tracking**, not **kidnapped-robot global recovery** — the latter
   still wants distinctive nodes / descriptor-LPR. It embeds *correct local
   minima*, not *place identity*.
2. **It's manual SLAM → overlap-consistency is the limiter.** Recognize at A
   then nearby B describe overlapping geometry; if the asserted A/B poses
   aren't mutually metrically consistent, the edits fight. The operator is
   supplying pose-graph constraints by hand; the tool should surface overlap
   and warn/blend on conflict or it will thrash.
3. **Powerful and dangerous.** A wrong assertion bakes a wall in the wrong
   place → the MCL then *confidently* mislocalizes there. The overlay is the
   safety interlock, not polish. Verify-then-bake.
4. **What it edits is clean:** stamp scan endpoints into occupancy + clear
   along rays + regenerate the local likelihood field — fits the existing
   save path.

### 5.3 Nice consequence

If the map is healed locally everywhere you drive, the metric distances
*near each waypoint* become trustworthy again — so **arrival-by-distance keeps
working** and the §3.1 arrival rework may be unnecessary. Directions A and C
**compose**: heal the map (C) + keep junction recognition for global recovery
(A). For a house with distinctive rooms/junctions, **C + junction-based
recognition is likely the lowest-effort route that actually works**, leaning
entirely on existing infrastructure + the overlay.

### 5.4 Resolved: local clear-and-restamp from odom-stitched scans

Neither (a) add-evidence (thickening — **rejected**, smears walls) nor (b) a
local optimizer. **Recognize = replace the observed occupancy within a fixed
radius from the asserted pose, rebuilt from 2-3 odom-stitched scans:**

- **Replace, observed-cells-only.** Within radius R (default ~2 m, the dense
  reliable lidar zone), run the standard occupancy stamp in **set (not
  accumulate)** mode: ray-trace free up to each endpoint, mark the endpoint
  occupied, leave cells no beam touched alone. Keeps walls one cell thin *and*
  never erases out-of-view geometry behind a wall (rays stop at the first hit).
- **Stitch 2-3 scans with raw odom — never the PF pose.** Anchor scan #1 at
  the operator-asserted pose; place #2/#3 by **raw odom delta** from #1 (over
  ~5 cm odom is ~0.3 mm accurate). Union the ray-stamps. Payoff: fills
  occlusion shadows and gives the likelihood a broader, smoother basin around
  the true pose instead of a knife-edge.
- **No undo needed.** Recognize is manual: the operator asserts the pose while
  seeing the old map + the live scan overlay (verify-before-bake), and
  replace-semantics make Recognize **self-correcting** — a subtly-wrong spot
  is fixed by driving back and Recognizing again (last-write-wins, no
  residue). "Re-Recognize" *is* the undo. (The editor's `snapshot_state()`
  stack still covers accidental clicks.)

### 5.5 The "small move"

Straight **forward** (diff-drive can't strafe; forward keeps odom cleanest),
**~5 cm** per step, 2-3 scans over ~10 cm total. At ~1°/beam the endpoints are
~3.5 cm apart at 2 m, so ~5 cm interleaves new endpoints between the old ones
and clears a few shadow cells past near occluders. Keep it small so the union
stays self-consistent and tightly tied to the one pose the operator verified;
returns diminish fast — a single +5 cm second scan captures most of it.

---

## 6. The radius-limited checkpoint — the unifying primitive

Directions A, B and C collapse into **one operation: a radius-limited local
patch match.** Each Recognize spot is stored as an **LPR checkpoint** = the
baked local occupancy patch (cells within R) + its asserted pose. The same
radius limit that makes the *edit* clean makes the *match* robust: at runtime
you score only the beams inside the certified patch and **ignore the distorted
far field** — exactly the part that poisons global MCL. So limiting to the
radius isn't just faster, it's *more correct*.

**Runtime loop (the "pre-MCL fast pose"):**

1. **Propagate odom** continuously (locally true) in the map frame.
2. **Fast checkpoint test:** use the dead-reckoned pose to pick the 1-3
   checkpoints within ~R; run a small-window `ScanMatcher.search()` against
   each checkpoint's patch. Above a confidence gate → **snap / re-anchor**.
   Cheap, O(1), no particle cloud.
3. **Full MCL demoted to fallback:** only when no checkpoint is in range and a
   global fix is genuinely needed (cold start / lost). Testing *all*
   checkpoints with no odom prior *is* the relocalization primitive — i.e.
   `.nav` **Re-localize = "which checkpoint am I at"** (Direction B). One code
   path serves tracking re-anchor, Re-localize, and cold start.

In a tiled house you are almost always near a checkpoint, so the particle
filter rarely runs.

**The one honest limit:** a **mid-corridor** checkpoint's within-radius patch
is **along-track-blind** (slide forward/back, the patch looks the same), so it
pins cross-track + yaw but not forward position — along-track still rides odom
through the corridor. Checkpoints buy the most at **distinctive geometry**
(junctions, doorways, corners); put one at the end of each corridor to
re-anchor the along-track drift accumulated across it.

---

## 7. Recommendation

1. The metric MCL is the right tool only for **local/relative** matching; stop
   using it as the global localizer for the hierarchical stack.
2. Build the **radius-limited patch match** (§6) as the one shared primitive:
   used by `map_editor` Recognize to bake checkpoints, and by a runtime
   checkpoint provider for fast pose + re-anchor + Re-localize.
3. Keep **Set-location** as the manual anchor and **full MCL** as the
   cold-start / lost fallback.

---

## 8. Implementation plan

Phased; each phase is independently testable and everything runtime lands
behind the existing `PoseProvider` seam, so the production drive is unaffected
until an explicit cut-over.

**Phase 1 — `map_editor` Recognize (the bake).** Reuses `EditorMap`,
`LiveLink`, `_overlay_pose` (the operator-asserted pose from the existing
align / Set-location / rotate tools).
- Add a ray-trace occupancy stamp to `EditorMap` (DDA/Bresenham): free along
  each ray, occupied at endpoint, **set-mode**, bounded to radius R; operates
  on the existing `log_odds` grid. Push `snapshot_state()` first.
- Keep a short ring of recent `(scan_body, odom_pose)` in the live tick
  (already cached at 5 Hz) so Recognize can stitch the last 2-3 scans by raw
  odom delta, anchored at `_overlay_pose`.
- Add a **Recognize** toolbar button (row 2, by Stamp scan→wall) →
  `_on_recognize()`: snapshot, restamp within R from the stitched scans at
  `_overlay_pose`, mark dirty, re-render. Keep additive Stamp as a manual
  touch-up.
- Save path unchanged: `save_npz` → `build_reference_map_from_log_odds`
  already regenerates the **global** likelihood/distance fields from the
  edited occupancy. (No local field regen — sidesteps the whole-map-only
  field constraint entirely.)

**Phase 2 — Checkpoint persistence.** Lightweight; the patch is *derived*, not
stored.
- A checkpoint = `{id, pose:[x,y,θ], radius_m, created_ts}`. Its patch is the
  occupancy disk around `pose` sliced at load time — no blob to store, and it
  always reflects the current (healed) map.
- Store as `ReferenceMap.metadata["checkpoints"]` (→ `meta_json`,
  backward-compatible). Recognize adds/updates the checkpoint at
  `_overlay_pose`; save persists it.

**Phase 3 — Runtime checkpoint localizer (fast pose).** New module, behind a
flag.
- `CheckpointMatcher` (desktop/localization): reference_map + checkpoints +
  prior pose + live scan → pick nearby checkpoint(s), slice the patch from
  occupancy, `ScanMatcher.search()` with a small xy window + yaw sweep, return
  best pose + score; gate on score.
- `CheckpointPoseProvider` implementing the `PoseProvider.world_pose()` seam:
  odom dead-reckon in the map frame + re-anchor on a confident checkpoint
  match; drop-in for `PFPoseProvider` (reuse the PF as the local filter, or
  plain odom+IMU).
- Wire into `HierarchicalDrive` via the seam, selectable by flag; validate
  against the live scan overlay before making it default.

**Phase 4 — Recast `.nav` Re-localize + cold start.**
- Re-localize button → checkpoint recognition (nearest / test-all); demote
  global `relocate()` to an explicit fallback. Keep `relocate_at`
  ("Set location") as the manual anchor.

**Phase 5 (optional) — arrival-by-node.** Only if local healing proves
insufficient: judge arrival at distinctive waypoints by checkpoint
recognition; keep dead-reckon distance for mid-corridor pass-throughs. Likely
unnecessary if §5.3 holds (healed map → metric distance trustworthy near
waypoints).

See also: `bayesian_localization_redesign.md` (PF production stack),
`tier_contract.md` / `drive_tier3_spec.md` (the hierarchical tiers).

# Pi change request — local_map: stop body-frame "smear" of clear verdicts

**Date:** 2026-05-13
**Audience:** Pi-side developer (Bruce, via Cursor Remote-SSH)
**Scope:** one-line edit to `body/local_map.py`. No config / schema / topic changes. Downstream wire format unchanged.
**Status:** request — not yet applied. Desktop has confirmed the bug from the world_map view.

---

## Problem

The desktop world map's `clear / blocked / unknown` view (and, downstream, the costmap used by the A* planner) shows a broad **green halo** extending well past where the bot's depth camera and lidar have actually observed. The halo covers cells the bot has driven *near* but not directly through, and on the outer edges of the map it produces phantom clear cells that the planner then tries to route through — sometimes wandering into a different room.

## Root cause

`body/local_map.py` keeps three persistent body-frame buffers allocated once before the main loop:

```python
# body/local_map.py:323-325
driveable_prev = np.full((nx, ny), _D_NONE, dtype=np.int8)
clear_streak  = np.zeros((nx, ny), dtype=np.int32)
lidar_streak  = np.zeros((nx, ny), dtype=np.int32)
```

These are indexed by body-frame cell coordinates and **never shifted for robot motion**. The verdict mux at the bottom of each frame falls back to `driveable_prev` when a cell is unobserved-this-frame but the clear streak hasn't fully drained:

```python
# body/local_map.py:541-549
d_now = np.where(
    instant_block, _D_BLOCK,
    np.where(observed,
             np.where(ok_mask, _D_OK, _D_BLOCK),
             np.where(faded, _D_NONE, driveable_prev)),   # ← smear source
)
```

With the default `driveable_unobs_decay_frames=1` and `driveable_clear_frames=4`, an unobserved cell continues to publish its previous verdict for up to four ticks (~2 s at `publish_hz=2`). Meanwhile the bot has moved, so each of those four ticks the same body-frame cell now corresponds to a different world cell. The desktop fuser transforms body→world using the current pose and lays a clear vote at each of those new world cells.

Net effect: a **smear** of clear evidence in the direction of robot motion (worst on rotations — every body-frame cell suddenly points somewhere new). The clear votes are laid in cells the depth camera never actually saw.

This is consistent with the observed green halo extending ~0.5–1 m past genuine observations on translations, and producing wide arcs on rotations.

## Fix

One-line change in `body/local_map.py:547`: replace the `driveable_prev` fallback in the unobserved branch with `_D_NONE`. Unobserved cells now publish `null`, regardless of any lingering clear streak.

### Diff

```diff
@@ body/local_map.py
         d_now = np.where(
             instant_block,
             _D_BLOCK,
             np.where(
                 observed,
                 np.where(ok_mask, _D_OK, _D_BLOCK),
-                np.where(faded, _D_NONE, driveable_prev),
+                _D_NONE,
             ),
         )
```

The outer `where` becomes effectively `observed ? (ok_mask ? OK : BLOCK) : NONE`.

### Why this is the right fix (not just turning `unobs_decay` up)

- **It directly removes the smear**, instead of just shortening it. Bumping `driveable_unobs_decay_frames` from 1 to 4 would mostly hide the symptom but still emit one frame of stale verdict per cell.
- **It preserves the streak logic.** `clear_streak` continues to require `driveable_clear_frames` consecutive observed-no-slab frames before a cell publishes `True`. So this isn't reducing the temporal averaging that keeps speckle out of `clear` — just stopping the buffer from publishing values for cells we aren't currently observing.
- **`driveable_prev` becomes dead.** After this change the variable is read only by the spec'd "previous verdict" path that never fires. It can stay (harmless) or be deleted in a follow-up; leaving it minimises the diff for this change.
- **Spec wording stays accurate.** `local_map_spec.md`'s description ("Unobserved cells keep the previous driveable verdict") was the *intent* under a stationary-grid assumption that does not hold once the robot moves. The spec should be updated to "Unobserved cells publish `null`" in a follow-up edit, but the wire format and behaviour for stationary operation are unchanged.

## Expected effect on the desktop side

- The `clear / blocked / unknown` map's outer halo collapses back toward the actual observation footprint.
- The lethal/halo costmap (which feeds A*) gets sharper room boundaries; phantom corridors past walls disappear.
- No change to how genuinely observed cells are classified — the bot still publishes `True` for cells it can currently see as clear floor.

## Validation steps after applying

1. Start `desktop.nav --slam` and drive the bot in a small loop in one room.
2. Inspect the `clear / blocked / unknown` panel. The green should track the bot's depth FOV cone and lidar reach, not extend ~1 m past it.
3. Rotate in place. The green should not "rotate with" the bot — cells leaving the FOV revert to gray within one publish tick rather than persisting for ~2 s.
4. Drive past a doorway. Cells beyond the door should not bloom green until the bot has actually pointed its sensors through.
5. Optional: capture a snapshot bundle before and after, compare cell counts (`driveable_clear` in the bundle's `meta.json`). Expect a meaningful drop with no loss of clears along the actual trail.

## Risk / rollback

- **Risk:** none functional. Cells that genuinely should be clear continue to be classified clear via the streak. Cells that are off-FOV become gray on the desktop map sooner than today; the planner already handles unknown cells (`unknown_cost=25.0` in the costmap), so route quality near the bot is unaffected.
- **Rollback:** revert the single line.

## Downstream coordination

- **Desktop side:** no change needed to consume this. The fuser already treats `null` as "no vote" (skips both clear and block tallies in `world_grid.fuse_local_map`).
- **Spec doc update (separate PR, not blocking):** `docs/local_map_spec.md` paragraph "Unobserved cells keep the previous driveable verdict" should be reworded to reflect the new behaviour.

---

## Optional companion change — symmetric depth-slab N-frame sticky

This is **not required** for the smear fix above, and not required to land any of the uncommitted desktop changes. It's a clean-up that would let us remove a desktop-side band-aid.

### Background

`body/local_map.py` classifies a cell as `instant_block` if **either** sensor sees an obstacle in the obstacle slab:

```python
# body/local_map.py:513-526
lidar_raw     = lidar_slab_count >= lidar_slab_min_hits
lidar_streak  = np.where(lidar_raw, lidar_streak + 1, 0)
lidar_blocked = lidar_streak >= lidar_slab_block_frames   # ← 2-frame sticky
slab_hit      = slab_count >= slab_min_pixels             # ← 1-frame (instant!)
instant_block = slab_hit | lidar_blocked
```

Lidar already requires `lidar_slab_block_frames=2` consecutive frames of slab hits before declaring blocked. **Depth has no such filter** — a single frame with `slab_count >= slab_min_pixels=2` flips the cell to block.

Meanwhile the *clear* side requires `driveable_clear_frames=4` consecutive observed-no-slab frames. So depth block is 1 frame, clear is 4 frames — heavily biased toward declaring "blocked" under depth noise (dust speckle, motion blur on edges of real obstacles, a single-frame depth artifact at a wall corner).

### Desktop-side workaround currently in place

The desktop fuser weights block votes at 0.5 of clear votes (`world_grid.WorldGrid` with `block_vote_weight=0.5`, set in `config.py`). This is a band-aid that approximately restores symmetry at the world map level without touching the Pi.

### Proposed Pi-side root fix

Add a `depth_slab_block_frames` config knob mirroring the existing `lidar_slab_block_frames`, and apply the same streak-based stickiness to depth:

```diff
@@ body/local_map.py (config parse, near line 299-303)
     slab_min_pixels = max(1, int(lm.get("driveable_slab_min_pixels", 2)))
     floor_seen_min_pixels = max(1, int(lm.get("driveable_floor_min_pixels", 2)))
     unobs_decay = max(0, int(lm.get("driveable_unobs_decay_frames", 1)))
     lidar_slab_min_range = float(lm.get("lidar_slab_min_range_m", 0.15))
     lidar_slab_block_frames = max(1, int(lm.get("lidar_slab_block_frames", 2)))
     lidar_slab_min_hits = max(1, int(lm.get("lidar_slab_min_hits", 1)))
+    depth_slab_block_frames = max(1, int(lm.get("depth_slab_block_frames", 2)))
```

```diff
@@ body/local_map.py (state allocation, near line 323-325)
     driveable_prev = np.full((nx, ny), _D_NONE, dtype=np.int8)
     clear_streak   = np.zeros((nx, ny), dtype=np.int32)
     lidar_streak   = np.zeros((nx, ny), dtype=np.int32)
+    depth_slab_streak = np.zeros((nx, ny), dtype=np.int32)
```

```diff
@@ body/local_map.py (classification, near line 513-526)
     lidar_raw     = lidar_slab_count >= lidar_slab_min_hits
     lidar_streak  = np.where(lidar_raw, lidar_streak + 1, 0)
     lidar_blocked = lidar_streak >= lidar_slab_block_frames
-    slab_hit      = (
-        slab_count >= slab_min_pixels
-        if slab_count is not None
-        else np.zeros((nx, ny), dtype=bool)
-    )
+    depth_raw = (
+        slab_count >= slab_min_pixels
+        if slab_count is not None
+        else np.zeros((nx, ny), dtype=bool)
+    )
+    depth_slab_streak = np.where(depth_raw, depth_slab_streak + 1, 0)
+    slab_hit = depth_slab_streak >= depth_slab_block_frames
     ...
     instant_block = slab_hit | lidar_blocked
```

The exact location of the new streak update depends on the broader loop structure — same pattern as `lidar_streak`, just for depth slab.

### Default value

`depth_slab_block_frames=2` is a sensible default — matches `lidar_slab_block_frames` and adds only ~0.5 s of latency at `publish_hz=2` before a genuine obstacle is reported. Real obstacles persist across many frames, so 2 frames is cheap insurance. If false-positive block reports are still common at 2, can be raised to 3.

### Why this is the *right* place to fix the bias

- It treats the two sensors symmetrically (both lidar and depth now N-frame-sticky on block).
- It treats block and clear symmetrically *per sensor* (both require persistence before flipping).
- It removes the world-map FIFO's dependency on the desktop's `block_vote_weight` knob — `block_vote_weight=1.0` becomes safe again, simplifying the desktop config.

### Desktop coordination after this lands

If/when the Pi adds `depth_slab_block_frames`, the desktop should revert `block_vote_weight` from `0.5` → `1.0` in `desktop/world_map/config.py:FuserConfig`. Not before — premature revert would leave the bias unmitigated. Coordinate via commit ordering: Pi change merged + verified on a real drive, *then* desktop revert.

### Risk / rollback

- **Risk:** 0.5 s slower to flag a new obstacle (at default `publish_hz=2`). The desktop pose-age guard and pure-pursuit follower already operate at much shorter horizons; a half-second of "is that thing actually there?" is well within safety margins.
- **Rollback:** set `depth_slab_block_frames=1` in Pi `config.json` to restore old behaviour without code revert.

### Validation steps after applying

1. Stand a person in front of the bot. Confirm `body/map/local_2p5d` flags the cells as blocked within ~1 s, not 0.5 s sooner.
2. Wave a dark object briefly across the depth FOV (1 frame). Confirm cells do **not** flip to blocked from that single frame.
3. Repeat the drive scenarios that produced the original "red between bot and waypoint" complaint on the desktop. Expect the spurious red blobs to be substantially reduced even before reverting `block_vote_weight`.


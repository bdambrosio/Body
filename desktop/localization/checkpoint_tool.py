"""List / delete LPR checkpoints stored in a reference map's metadata.

Checkpoints live in ``ReferenceMap.metadata["checkpoints"]`` (see
``checkpoints.py``). There is no operator-facing delete in the map editor, so
this is the CLI for inspecting and erasing them.

    # show what's stored
    python -m desktop.localization.checkpoint_tool list  MAP.npz

    # erase one (writes a .bak alongside unless --no-backup)
    python -m desktop.localization.checkpoint_tool delete MAP.npz --id cp_003

    # erase all
    python -m desktop.localization.checkpoint_tool clear  MAP.npz

Saving writes the loaded map straight back with only the checkpoint metadata
changed — occupancy and the derived fields round-trip untouched. If this tool
ever grows an occupancy edit, the save must go through
``build_reference_map_from_log_odds`` (like the editor's Save) or the stored
fields go stale.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys

from desktop.localization.checkpoints import (
    checkpoints_from_metadata,
    write_checkpoints_to_metadata,
)
from desktop.reference_map.reference_map import (
    load_reference_map,
    save_reference_map,
)


def _load(path):
    rm = load_reference_map(path)
    return rm, checkpoints_from_metadata(rm.metadata)


def _print(cps):
    if not cps:
        print("  (no checkpoints)")
        return
    for c in cps:
        print(f"  {c.id:>8}  x={c.x_m:+.3f} y={c.y_m:+.3f} "
              f"θ={math.degrees(c.theta_rad):+6.1f}°  r={c.radius_m:.2f}m")


def _save(path, rm, cps, backup):
    write_checkpoints_to_metadata(rm.metadata, cps)
    if backup:
        shutil.copy2(path, path + ".bak")
        print(f"backup → {path}.bak")
    save_reference_map(path, rm)
    print(f"saved {len(cps)} checkpoint(s) → {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["list", "delete", "clear"])
    ap.add_argument("map", help="path to reference_map.npz")
    ap.add_argument("--id", help="checkpoint id to delete (delete action)")
    ap.add_argument("--no-backup", action="store_true",
                    help="overwrite without writing a .bak copy first")
    args = ap.parse_args(argv)

    rm, cps = _load(args.map)
    print(f"{len(cps)} checkpoint(s) in {args.map}:")
    _print(cps)

    if args.action == "list":
        return 0

    if args.action == "delete":
        if not args.id:
            print("delete needs --id", file=sys.stderr)
            return 2
        kept = [c for c in cps if c.id != args.id]
        if len(kept) == len(cps):
            print(f"no checkpoint with id {args.id!r}", file=sys.stderr)
            return 1
        print(f"\ndeleting {args.id}")
        _save(args.map, rm, kept, not args.no_backup)
        return 0

    # clear
    if not cps:
        print("\nnothing to clear")
        return 0
    print(f"\nclearing all {len(cps)}")
    _save(args.map, rm, [], not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Owner tool for the explore change store (Phase C).

    PYTHONPATH=. .venv/bin/python scripts/change_store.py STORE list [--all]
    PYTHONPATH=. .venv/bin/python scripts/change_store.py STORE approve ID [ID ...]
    PYTHONPATH=. .venv/bin/python scripts/change_store.py STORE reject ID [ID ...]
    PYTHONPATH=. .venv/bin/python scripts/change_store.py STORE forget ID [ID ...]

STORE is ``change_store.json`` next to the map (nav creates it), e.g.
~/Body/maps/<sid>/map_<ts>/change_store.json.

* approve — this place routinely differs from the map (a door, chair legs):
  stop announcing matching reports (still logged, verdict "expected").
* reject  — always announce changes here, even if they recur.
* forget  — drop the entry; it is learned again from scratch.

``list`` shows proposed and background entries (``--all``: candidates and
rejected too). Nothing becomes background without ``approve``.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desktop.perception.change_store import (  # noqa: E402
    ST_BACKGROUND, ST_CANDIDATE, ST_PROPOSED, ST_REJECTED, ChangeStore,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("action", choices=("list", "approve", "reject", "forget"))
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    path = os.path.expanduser(args.store)
    if not os.path.exists(path):
        print(f"no store at {path}", file=sys.stderr)
        return 1
    st = ChangeStore(path)
    if args.action == "list":
        show = (ST_PROPOSED, ST_BACKGROUND) if not args.all else (
            ST_PROPOSED, ST_BACKGROUND, ST_CANDIDATE, ST_REJECTED)
        rows = sorted((e for e in st.entries.values() if e.state in show),
                      key=lambda e: (show.index(e.state), -len(e.sessions)))
        print(f"{'id':>4} {'state':10s} {'kind':7s} {'x':>6} {'y':>6} {'size':>5} "
              f"{'patrols':>7} {'reports':>7} {'misses':>6}  label / last seen")
        for e in rows:
            last = datetime.datetime.fromtimestamp(e.last_seen).strftime("%m-%d %H:%M")
            print(f"{e.eid:4d} {e.state:10s} {e.kind:7s} {e.x:6.2f} {e.y:6.2f} "
                  f"{e.extent_m:5.2f} {len(e.sessions):7d} {e.n_reports:7d} {e.misses:6d}  "
                  f"{e.label or '-'} / {last}")
        print(st.summary())
        return 0
    state = {"approve": ST_BACKGROUND, "reject": ST_REJECTED}.get(args.action)
    for eid in args.ids:
        ok = st.forget(eid) if args.action == "forget" else st.set_state(eid, state)
        print(f"{eid}: {'ok' if ok else 'no such entry'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

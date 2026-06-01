"""Standalone world-map editor entry point.

Phase 1 (offline):
    QT_QPA_PLATFORM=xcb python -m desktop.map_editor \
        --map ~/Body/maps/<sid>/map_<ts>/reference_map.npz

Phase 2 (live overlay, bot reachable) will add --router / --pf-* and a
read-only lidar overlay; those flags are accepted now but unused until
Phase 2 lands.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Wayland/Mutter friction — force xcb (house style; see feedback memory).
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python -m desktop.map_editor",
        description="Edit a saved world-map snapshot (layers.npz).",
    )
    p.add_argument("--map", "--load-map", dest="map_path", metavar="PATH",
                   default=None,
                   help="reference_map.npz (from a mapping run) to open")
    # Live-overlay flags. With --router, a Connect action enables the
    # read-only MCL pose + lidar overlay; without it the editor is
    # purely offline.
    p.add_argument("--router", default=None,
                   help="Zenoh router endpoint to enable the live overlay "
                        "(e.g. tcp/192.168.8.60:7447)")
    p.add_argument("--pf-particles", type=int, default=5000,
                   help="particle count for the live MCL pose")
    p.add_argument("--pf-device", choices=("auto", "cpu", "cuda"),
                   default="auto", help="particle filter device")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    router = None
    if args.router:
        from desktop.localization.config import resolve_router
        router = resolve_router(args.router)
    from .main_window import run
    return run(map_path=args.map_path, router=router,
               pf_device=args.pf_device, pf_particles=args.pf_particles)


if __name__ == "__main__":
    # Hard-exit once the Qt loop returns. zenoh/torch spawn native threads the
    # interpreter would otherwise block on at shutdown — that's why closing the
    # window used to hang until ^C. The window closeEvent already tore down the
    # live link before app.exec() returned.
    os._exit(main())

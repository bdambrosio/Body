"""Entry point: python -m desktop.handoff_inspector [--router tcp/HOST:PORT]

A standalone window that subscribes the three tier-handoff record topics and
arms/single-steps the breakpoints over zenoh. Run it alongside `desktop.nav`
(Tier-1/Tier-2) and the Pi `body.local_drive` (Tier-3); it needs only the same
zenoh router — no map, no patrol, no production coupling.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt6.QtWidgets import QApplication

from desktop.chassis.config import resolve_router
from desktop.chassis.transport import open_session
from desktop.handoff_inspector.window import HandoffInspectorWindow


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="handoff_inspector",
        description="Standalone Tier-1/2/3 handoff breakpoint inspector.",
    )
    p.add_argument("--router", default=None,
                   help="Zenoh router endpoint (overrides $ZENOH_CONNECT)")
    p.add_argument("--map", dest="map_path", default=None, metavar="PATH",
                   help="Reference map (reference_map.npz) for the Tier-1 "
                        "global-map panel. Optional — without it HO-1 stays "
                        "body-frame.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _load_reference_map(map_path):
    """Return (drive_int8, meta) for the T1 global layer, or None."""
    import os
    from desktop.reference_map.legacy_convert import load_map_auto
    ref = load_map_auto(os.path.expanduser(map_path))
    meta = {"resolution_m": ref.resolution_m, "origin_x_m": ref.origin_x_m,
            "origin_y_m": ref.origin_y_m, "nx": ref.nx, "ny": ref.ny}
    return ref.driveable_int8(), meta


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    router = resolve_router(args.router)
    log = logging.getLogger(__name__)
    log.info("handoff inspector router=%s", router)
    reference_map = None
    if args.map_path:
        try:
            reference_map = _load_reference_map(args.map_path)
        except Exception:
            log.exception("failed to load --map %s (T1 stays body-frame)",
                          args.map_path)
    session = open_session(router)

    app = QApplication.instance() or QApplication(sys.argv)
    win = HandoffInspectorWindow(session, reference_map=reference_map)
    win.resize(1320, 580)
    win.show()
    try:
        return app.exec()
    finally:
        try:
            session.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Hard-exit once the Qt loop returns (zenoh native threads otherwise block).
    os._exit(main())

"""Entry point: python -m desktop.pi_drive [--router tcp/HOST:PORT]

Tier-3 drive debug console. Connects to the Pi, shows the live body-frame
local_map, lets the operator click a goal, and drives there via the
Pi-side body.local_drive. Run with QT_QPA_PLATFORM=xcb on Wayland.

Precedence for router: --router > $ZENOH_CONNECT > tcp/127.0.0.1:7447.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_DESKTOP = Path(__file__).resolve().parents[1]
if str(_DESKTOP) not in sys.path:
    sys.path.insert(0, str(_DESKTOP))

from PyQt6.QtWidgets import QApplication

from desktop.chassis.config import ENV_VAR, StubConfig, resolve_router
from desktop.chassis.controller import StubController

from .drive_client import DriveClient
from .main_window import PiDriveWindow


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="pi_drive",
        description="Tier-3 reactive-drive debug console for the Body robot.",
    )
    p.add_argument(
        "--router", default=None,
        help=f"Zenoh router endpoint (overrides ${ENV_VAR}); e.g. tcp/192.168.1.50:7447",
    )
    p.add_argument(
        "--trace", default=None,
        help="append drive/status JSONL to this path for offline leg review",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    router = resolve_router(args.router)
    controller = StubController(StubConfig(router=router))
    drive = DriveClient(router, trace_path=args.trace)

    app = QApplication.instance() or QApplication(sys.argv)
    win = PiDriveWindow(controller, drive)
    win.resize(900, 720)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

"""Entry point: python -m desktop.pi_drive [--router tcp/HOST:PORT]

Tier-3 drive debug console. Connects to the Pi, shows the live body-frame
local_map, lets the operator click a goal, and drives there via the
Pi-side body.local_drive. Run with QT_QPA_PLATFORM=xcb on Wayland.

Precedence for router: --router > $ZENOH_CONNECT > tcp/127.0.0.1:7447.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

_DESKTOP = Path(__file__).resolve().parents[1]
if str(_DESKTOP) not in sys.path:
    sys.path.insert(0, str(_DESKTOP))

from PyQt6.QtWidgets import QApplication

from desktop.chassis.config import ENV_VAR, StubConfig, resolve_router
from desktop.chassis.controller import StubController

from .drive_client import DriveClient
from .main_window import PiDriveWindow
from .tier2_window import Tier2Window


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
        help="append JSONL to this path for offline review (drive/status, or "
             "Tier-2 decisions with --tier2)",
    )
    p.add_argument(
        "--tier2", action="store_true",
        help="launch the Tier-2 debug console (manual target → sub-goal → "
             "Tier-3) instead of the Tier-3 manual console",
    )
    p.add_argument(
        "--load-map", dest="map_path", default=None, metavar="PATH",
        help="(with --tier2) reference map to localize against; enables the "
             "PF + world-map panel so you can set a true world-frame target",
    )
    p.add_argument(
        "--relocate-on-load", action="store_true",
        help="run global relocalization once the first scan arrives",
    )
    p.add_argument("--pf-particles", type=int, default=5000)
    p.add_argument("--pf-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def _resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _build_localizer(router: str, args):
    """Build + connect the PF localizer for the Tier-2 map console (or None)."""
    from desktop.localization.config import LocalizationConfig
    from desktop.localization.controller import LocalizationController
    from desktop.reference_map.legacy_convert import load_map_auto

    log = logging.getLogger(__name__)
    map_path = os.path.expanduser(args.map_path)
    reference_map = load_map_auto(map_path)
    loc_cfg = LocalizationConfig(
        router=router, map_path=map_path,
        pf_device=_resolve_device(args.pf_device),
        pf_n_particles=args.pf_particles,
    )
    localizer = LocalizationController(loc_cfg, reference_map)
    ok, err = localizer.connect()
    if not ok:
        log.warning("localizer connect failed (%s)", err)
    if args.relocate_on_load and localizer.connected:
        def _deferred():
            ps = localizer.pose_source
            deadline = time.time() + 5.0
            while time.time() < deadline and getattr(ps, "_last_scan_ts", 0.0) <= 0.0:
                time.sleep(0.1)
            try:
                res = localizer.request_relocate(reason="relocate_on_load")
                log.info("relocate-on-load: %s", res.get("success"))
            except Exception:
                log.exception("relocate-on-load failed")
        threading.Thread(target=_deferred, name="relocate-on-load", daemon=True).start()
    return localizer


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    router = resolve_router(args.router)
    controller = StubController(StubConfig(router=router))
    # In --tier2 the window writes the richer Tier-2 decision trace; don't also
    # let DriveClient write raw status to the same file.
    drive = DriveClient(router, trace_path=None if args.tier2 else args.trace)

    app = QApplication.instance() or QApplication(sys.argv)
    if args.tier2:
        localizer = _build_localizer(router, args) if args.map_path else None
        win = Tier2Window(controller, drive, localizer=localizer, trace_path=args.trace)
    else:
        win = PiDriveWindow(controller, drive)
    win.resize(1280 if (args.tier2 and args.map_path) else 980, 760)
    win.show()
    return app.exec()


if __name__ == "__main__":
    # Hard-exit once the Qt loop returns. zenoh/torch spawn native threads the
    # interpreter would otherwise block on at shutdown — that's why closing the
    # window used to hang until ^C. The window closeEvent already shut down the
    # drive/controller/localizer before app.exec() returned.
    os._exit(main())

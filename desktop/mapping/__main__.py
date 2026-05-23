"""Mapping session entry: python -m desktop.mapping"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

_NAV = Path(__file__).resolve().parent.parent / "nav"
_DESKTOP = Path(__file__).resolve().parent.parent
_BODY = _DESKTOP.parent
for _p in (_BODY, _DESKTOP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.localization.config import resolve_router
from desktop.mapping.controller import MappingConfig, MappingController
from desktop.mapping.ui_qt import run_mapping_app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Body mapping session")
    p.add_argument("--router", default=None)
    p.add_argument("--extent-m", type=float, default=40.0)
    p.add_argument("--resolution-m", type=float, default=0.05)
    p.add_argument(
        "--heartbeat-hz", type=float, default=5.0,
        help="Chassis heartbeat rate.",
    )
    p.add_argument(
        "--map-stale-s", type=float, default=2.0,
        help="Local map staleness threshold (safety).",
    )
    p.add_argument("--no-autoconnect", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)
    router = resolve_router(args.router)
    config = MappingConfig(
        router=router,
        extent_m=args.extent_m,
        resolution_m=args.resolution_m,
        map_stale_s=args.map_stale_s,
    )
    chassis_config = StubConfig(
        router=router,
        heartbeat_hz=args.heartbeat_hz,
        map_stale_s=args.map_stale_s,
    )
    controller = MappingController(config)
    chassis = StubController(chassis_config)

    if not args.no_autoconnect:
        for name, ctrl in (("mapping", controller), ("chassis", chassis)):
            ok, err = ctrl.connect()
            if not ok:
                log.warning("%s autoconnect failed (%s)", name, err)

    try:
        return run_mapping_app(controller, config, chassis, chassis_config)
    finally:
        try:
            chassis.shutdown()
        except Exception:
            log.exception("chassis shutdown failed")
        try:
            controller.shutdown()
        except Exception:
            log.exception("mapping shutdown failed")


if __name__ == "__main__":
    sys.exit(main())

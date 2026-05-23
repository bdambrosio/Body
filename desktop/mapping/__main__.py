"""Mapping session entry: python -m desktop.mapping"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_NAV = Path(__file__).resolve().parent.parent / "nav"
_DESKTOP = Path(__file__).resolve().parent.parent
_BODY = _DESKTOP.parent
for _p in (_BODY, _DESKTOP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from desktop.localization.config import resolve_router
from desktop.mapping.controller import MappingConfig, MappingController
from desktop.mapping.ui_qt import run_mapping_app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Body mapping session")
    p.add_argument("--router", default=None)
    p.add_argument("--extent-m", type=float, default=40.0)
    p.add_argument("--resolution-m", type=float, default=0.05)
    p.add_argument("--no-autoconnect", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    router = resolve_router(args.router)
    config = MappingConfig(
        router=router,
        extent_m=args.extent_m,
        resolution_m=args.resolution_m,
    )
    controller = MappingController(config)
    if not args.no_autoconnect:
        ok, err = controller.connect()
        if not ok:
            logging.getLogger(__name__).warning("autoconnect failed: %s", err)
    return run_mapping_app(controller, config)


if __name__ == "__main__":
    sys.exit(main())

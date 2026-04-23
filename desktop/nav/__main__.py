"""Entry point: python -m desktop.nav [--router tcp/HOST:PORT]

Launches the nav shell: one Qt process, two controllers (world_map
fuser + chassis driver) each owning their own Zenoh session against the
same router.

Precedence for router: --router > $ZENOH_CONNECT > tcp/127.0.0.1:7447.
"""
from __future__ import annotations

import argparse
import logging
import sys

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.world_map.config import ENV_VAR, FuserConfig, resolve_router
from desktop.world_map.controller import FuserController

from .main_window import run_app


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="nav",
        description="Body nav shell: composes world_map (fusion + map "
                    "views) and chassis (Pi driver + teleop).",
    )
    p.add_argument(
        "--router", default=None,
        help=f"Zenoh router endpoint (overrides ${ENV_VAR}); "
             f"e.g. tcp/192.168.1.50:7447",
    )
    p.add_argument(
        "--world-extent-m", type=float, default=40.0,
        help="Square world side length in meters (default 40).",
    )
    p.add_argument(
        "--world-resolution-m", type=float, default=0.08,
        help="Cell size; must match Pi local_map.resolution_m (default 0.08).",
    )
    p.add_argument(
        "--publish-hz", type=float, default=2.0,
        help="World-driveable publish rate cap (default 2 Hz).",
    )
    p.add_argument(
        "--heartbeat-hz", type=float, default=5.0,
        help="chassis heartbeat rate (default 5 Hz).",
    )
    p.add_argument(
        "--map-stale-s", type=float, default=2.0,
        help="local_map staleness threshold in seconds (default 2.0).",
    )
    p.add_argument(
        "--no-autoconnect", action="store_true",
        help="Don't connect on startup; wait for the user to click Connect.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="debug logging",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    router = resolve_router(args.router)
    log = logging.getLogger(__name__)
    log.info(f"nav starting; router={router} (override via --router or ${ENV_VAR})")

    fuser_config = FuserConfig(
        router=router,
        world_extent_m=args.world_extent_m,
        world_resolution_m=args.world_resolution_m,
        publish_hz=args.publish_hz,
        map_stale_s=args.map_stale_s,
    )
    chassis_config = StubConfig(
        router=router,
        heartbeat_hz=args.heartbeat_hz,
        map_stale_s=args.map_stale_s,
    )

    fuser = FuserController(fuser_config)
    chassis = StubController(chassis_config)

    if not args.no_autoconnect:
        for name, ctrl in (("fuser", fuser), ("chassis", chassis)):
            ok, err = ctrl.connect()
            if not ok:
                log.warning(f"{name} autoconnect failed ({err}); retry via UI")

    try:
        return run_app(fuser, fuser_config, chassis, chassis_config)
    finally:
        # Order matters: chassis publishes zero commands on disconnect,
        # fuser doesn't touch motors, so tear chassis down first.
        try:
            chassis.shutdown()
        except Exception:
            log.exception("chassis shutdown raised")
        try:
            fuser.shutdown()
        except Exception:
            log.exception("fuser shutdown raised")


if __name__ == "__main__":
    sys.exit(main())

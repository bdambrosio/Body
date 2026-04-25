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
from pathlib import Path
from typing import Optional

# Support both `python -m desktop.nav` (from Body/) and `python -m nav`
# (from Body/desktop/, where the .venv lives). The second form puts
# Body/desktop/ on sys.path but not Body/, so `from desktop.*` would
# fail. Also make sure desktop/ itself is on sys.path so bare imports
# like `import vision_service` (used from chassis.ui_qt._VisionWorker)
# resolve — chassis.__main__ does the same.
_NAV_DIR = Path(__file__).resolve().parent
_DESKTOP = _NAV_DIR.parent
_BODY_ROOT = _DESKTOP.parent
for _p in (_BODY_ROOT, _DESKTOP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.world_map.config import ENV_VAR, FuserConfig, resolve_router
from desktop.world_map.controller import FuserController

from .main_window import run_app
from .slam.shadow_driver import ShadowSlamDriver


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
        "--shadow-slam", action="store_true",
        help="Enable the shadow SLAM driver: subscribes to body/imu + "
             "body/lidar/scan and logs candidate pose corrections. Does "
             "not write to the fuser's pose — purely observational.",
    )
    p.add_argument(
        "--slam", action="store_true",
        help="Promote SLAM to the production pose source. Replaces "
             "OdomPose with ImuPlusScanMatchPose: encoder translation "
             "+ BNO085 yaw + lidar scan-match corrections against the "
             "world grid. See docs/slam_pi_contract.md.",
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
        slam_enabled=args.slam,
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

    # Shadow SLAM: only wires if the fuser already has a live session
    # (i.e. autoconnect succeeded). Post-launch reconnects via the
    # safety toolbar don't re-install the driver; restart nav if the
    # fuser was reconnected and you want shadow SLAM going again.
    shadow: Optional[ShadowSlamDriver] = None
    if args.shadow_slam and args.slam:
        log.info(
            "--slam already promotes the SLAM pose source; "
            "ignoring redundant --shadow-slam.",
        )
    elif args.shadow_slam:
        if fuser.connected:
            shadow = ShadowSlamDriver(
                session=fuser.session,
                grid=fuser.grid,
                pose_source=fuser.pose_source,
            )
            try:
                shadow.connect()
            except Exception:
                log.exception("shadow_slam connect failed; continuing without it")
                shadow = None
        else:
            log.warning(
                "shadow_slam requested but fuser not connected; "
                "driver not installed. Launch with --router pointing at a "
                "live Pi, or restart after connecting via the UI.",
            )

    try:
        return run_app(fuser, fuser_config, chassis, chassis_config)
    finally:
        # Shadow first — its subscribers live on the fuser's session.
        if shadow is not None:
            try:
                shadow.disconnect()
            except Exception:
                log.exception("shadow_slam disconnect raised")
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

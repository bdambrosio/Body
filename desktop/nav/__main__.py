"""Entry point: python -m desktop.nav [--router tcp/HOST:PORT] --map PATH

Launches the nav shell with MCL localization against a frozen reference
map plus chassis teleop on the same Zenoh router.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_NAV_DIR = Path(__file__).resolve().parent
_DESKTOP = _NAV_DIR.parent
_BODY_ROOT = _DESKTOP.parent
for _p in (_BODY_ROOT, _DESKTOP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.localization.config import LocalizationConfig, resolve_router
from desktop.localization.controller import LocalizationController
from desktop.reference_map.legacy_convert import load_map_auto

from .main_window import run_app


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="nav",
        description="Body nav: MCL localization + chassis teleop.",
    )
    p.add_argument(
        "--router", default=None,
        help="Zenoh router endpoint (overrides $ZENOH_CONNECT)",
    )
    p.add_argument(
        "--map", "--load-map", dest="map_path", required=True,
        metavar="PATH",
        help="Reference map (reference_map.npz or legacy layers.npz). Required.",
    )
    p.add_argument(
        "--relocate-on-load", action="store_true",
        help="Run global localization once the first scan arrives.",
    )
    p.add_argument(
        "--pf-device", choices=("auto", "cpu", "cuda"), default="auto",
        help="MCL particle filter device.",
    )
    p.add_argument(
        "--pf-particles", type=int, default=5000,
        help="MCL particle count (default 5000).",
    )
    p.add_argument(
        "--pf-imu-obs-hz", type=float, default=5.0,
        help="IMU yaw observation rate cap (Hz).",
    )
    p.add_argument(
        "--publish-hz", type=float, default=2.0,
        help="World map publish rate cap.",
    )
    p.add_argument(
        "--heartbeat-hz", type=float, default=5.0,
        help="Chassis heartbeat rate.",
    )
    p.add_argument(
        "--map-stale-s", type=float, default=2.0,
        help="Local map staleness threshold (safety).",
    )
    p.add_argument(
        "--no-autoconnect", action="store_true",
        help="Wait for Connect in the UI.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)
    router = resolve_router(args.router)
    map_path = os.path.expanduser(args.map_path)
    log.info("nav starting router=%s map=%s", router, map_path)

    try:
        reference_map = load_map_auto(map_path)
    except Exception:
        log.exception("failed to load map %s", map_path)
        return 1

    pf_device = _resolve_device(args.pf_device)
    loc_config = LocalizationConfig(
        router=router,
        map_path=map_path,
        publish_hz=args.publish_hz,
        map_stale_s=args.map_stale_s,
        pf_device=pf_device,
        pf_n_particles=args.pf_particles,
        pf_imu_obs_hz=args.pf_imu_obs_hz,
    )
    chassis_config = StubConfig(
        router=router,
        heartbeat_hz=args.heartbeat_hz,
        map_stale_s=args.map_stale_s,
    )

    localizer = LocalizationController(loc_config, reference_map)
    chassis = StubController(chassis_config)

    if not args.no_autoconnect:
        for name, ctrl in (("localizer", localizer), ("chassis", chassis)):
            ok, err = ctrl.connect()
            if not ok:
                log.warning("%s autoconnect failed (%s)", name, err)

    if args.relocate_on_load and localizer.connected:
        def _deferred_relocate():
            deadline = time.time() + 5.0
            ps = localizer.pose_source
            while time.time() < deadline:
                if getattr(ps, "_last_scan_ts", 0.0) > 0.0:
                    break
                time.sleep(0.1)
            try:
                result = ps.relocate()
            except Exception:
                log.exception("relocate-on-load failed")
                return
            if result.get("success"):
                log.info(
                    "relocate-on-load ok dx=%+.2f dy=%+.2f",
                    result["dx"], result["dy"],
                )
            else:
                log.warning("relocate-on-load failed: %s", result)

        threading.Thread(
            target=_deferred_relocate, name="relocate-on-load", daemon=True,
        ).start()

    try:
        return run_app(localizer, loc_config, chassis, chassis_config)
    finally:
        try:
            chassis.shutdown()
        except Exception:
            log.exception("chassis shutdown failed")
        try:
            localizer.shutdown()
        except Exception:
            log.exception("localizer shutdown failed")


if __name__ == "__main__":
    sys.exit(main())

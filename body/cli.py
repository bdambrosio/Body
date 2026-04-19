"""Pi-local CLI for Body Zenoh topics (same wire protocol as the desktop agent).

Requires `zenohd` and the relevant Body processes (e.g. oakd_driver for OAK-D capture).
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
import uuid

from body.lib import schemas, zenoh_helpers


def _cmd_oakd_capture(args: argparse.Namespace) -> int:
    request_id = str(uuid.uuid4())
    body_cfg = zenoh_helpers.load_body_config()
    session = zenoh_helpers.open_session(body_cfg)
    result: dict | None = None

    def on_rgb(_key: str, msg: dict) -> None:
        nonlocal result
        if msg.get("request_id") == request_id:
            result = msg

    zenoh_helpers.declare_subscriber_json(session, "body/oakd/rgb", on_rgb)
    zenoh_helpers.publish_json(
        session,
        "body/oakd/config",
        schemas.oakd_config_capture_rgb(request_id),
    )
    deadline = time.time() + float(args.timeout)
    while time.time() < deadline and result is None:
        time.sleep(0.02)
    session.close()

    if result is None:
        print("oakd capture: timeout waiting for body/oakd/rgb", file=sys.stderr)
        return 1
    if not result.get("ok"):
        err = result.get("error", "unknown")
        print(f"oakd capture failed: {err}", file=sys.stderr)
        return 2
    data = result.get("data")
    if not isinstance(data, str):
        print("oakd capture: missing base64 data in response", file=sys.stderr)
        return 3
    try:
        raw = base64.standard_b64decode(data)
    except (ValueError, TypeError) as e:
        print(f"oakd capture: invalid base64: {e}", file=sys.stderr)
        return 4
    path = args.output
    with open(path, "wb") as f:
        f.write(raw)
    w = result.get("width")
    h = result.get("height")
    print(f"Wrote {path} ({w}x{h} JPEG)", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Body stack — local Zenoh tools (see docs/body_project_spec.md topics).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    oakd_p = sub.add_parser("oakd", help="OAK-D on-request actions via Zenoh")
    oakd_sub = oakd_p.add_subparsers(dest="oakd_action", required=True)
    cap_p = oakd_sub.add_parser(
        "capture",
        help="Request one RGB JPEG from oakd_driver (set oakd.rgb_enabled, restart driver)",
    )
    cap_p.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        required=True,
        help="Output JPEG path",
    )
    cap_p.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for body/oakd/rgb (default: 15)",
    )
    cap_p.set_defaults(_handler=_cmd_oakd_capture)

    args = parser.parse_args()
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        raise SystemExit(2)
    raise SystemExit(handler(args))


if __name__ == "__main__":
    main()

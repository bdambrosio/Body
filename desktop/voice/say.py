"""python -m desktop.voice.say "text" [--kind alert] — make the robot say something.

Needs only zenoh (works from the main .venv); the voice service must be running.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from body.lib import schemas  # noqa: E402
from desktop.chassis.config import Topics, resolve_router  # noqa: E402
from desktop.chassis.transport import open_session  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="desktop.voice.say")
    p.add_argument("text")
    p.add_argument("--kind", choices=("alert", "chat", "info"), default="chat")
    p.add_argument("--router", default=None)
    args = p.parse_args(argv)
    session = open_session(resolve_router(args.router))
    time.sleep(0.5)  # let the session reach the router before the one-shot put
    session.put(Topics().voice_say, json.dumps(schemas.voice_say(args.text, kind=args.kind)))
    time.sleep(0.2)
    session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

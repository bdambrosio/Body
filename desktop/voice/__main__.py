"""python -m desktop.voice [--router tcp/HOST:PORT] — the robot's voice (TTS).

Run with the voice venv (Python 3.12 + Kokoro), see desktop/voice/README.md:
  .venv-voice/bin/python -m desktop.voice --router tcp/192.168.8.60:7447
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from body.lib.zenoh_helpers import load_body_config  # noqa: E402
from desktop.chassis.config import resolve_router  # noqa: E402
from desktop.chassis.transport import open_session  # noqa: E402
from desktop.voice.service import VoiceService  # noqa: E402
from desktop.voice.tts import KokoroTts  # noqa: E402


def main(argv=None) -> int:
    cfg = load_body_config().get("voice", {})
    p = argparse.ArgumentParser(prog="desktop.voice")
    p.add_argument("--router", default=None, help="zenoh router (default $ZENOH_CONNECT / localhost)")
    p.add_argument("--voice", default=cfg.get("voice", "af_heart"))
    p.add_argument("--pitch", type=float, default=float(cfg.get("pitch", 1.0)))
    p.add_argument("--speed", type=float, default=float(cfg.get("speed", 1.0)))
    p.add_argument("--gpu-match", default=cfg.get("gpu_match", "5060"),
                   help="use the CUDA device whose name contains this ('' = CPU)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tts = KokoroTts(voice=args.voice, pitch=args.pitch, speed=args.speed, gpu_match=args.gpu_match)
    tts.warm_up()
    session = open_session(resolve_router(args.router))
    svc = VoiceService(session, tts, max_queue=int(cfg.get("max_queue", 6)),
                       info_max_age_s=float(cfg.get("info_max_age_s", 10.0)))
    svc.start()
    logging.getLogger("desktop.voice").info("voice service ready (%s)", tts.device)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    svc.shutdown()
    session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

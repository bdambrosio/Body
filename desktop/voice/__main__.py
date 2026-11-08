"""python -m desktop.voice [--router tcp/HOST:PORT] — the robot's voice (TTS) and ears (chat).

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
if str(_ROOT / "desktop") not in sys.path:   # vision_service imports utils.* bare
    sys.path.append(str(_ROOT / "desktop"))

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
    p.add_argument("--no-listen", action="store_true", help="speak only: no mic / chat")
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
    listener = None
    lcfg = cfg.get("listen", {})
    if not args.no_listen and lcfg.get("enabled", True):
        listener = _start_listener(session, svc, cfg, lcfg, args.gpu_match)
    logging.getLogger("desktop.voice").info(
        "voice service ready (%s)%s", tts.device,
        f"; listening for {cfg.get('robot_name', 'Inky')!r}" if listener else "")

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    if listener is not None:
        listener.shutdown()
    svc.shutdown()
    session.close()
    return 0


def _start_listener(session, svc, cfg: dict, lcfg: dict, gpu_match: str):
    from desktop import vision_service
    from desktop.voice.chat import DEFAULT_PERSONA, Conversation
    from desktop.voice.listener import Listener
    from desktop.voice.segmenter import Segmenter
    from desktop.voice.stt import SileroVad, WhisperStt

    stt = WhisperStt(lcfg.get("stt_model", "openai/whisper-large-v3-turbo"), gpu_match=gpu_match)
    stt.warm_up()
    max_tokens = int(lcfg.get("max_tokens", 160))
    temperature = float(lcfg.get("temperature", 0.6))

    def llm(messages):
        return vision_service.chat(messages, temperature=temperature, max_tokens=max_tokens)

    conv = Conversation(
        name=cfg.get("robot_name", "Inky"), llm=llm,
        persona=lcfg.get("persona") or DEFAULT_PERSONA,
        follow_up_s=float(lcfg.get("follow_up_s", 8.0)),
        ack_text=lcfg.get("ack_text", "Yes?"))
    seg = Segmenter(end_silence_s=float(lcfg.get("end_silence_s", 0.7)),
                    min_speech_s=float(lcfg.get("min_speech_s", 0.3)),
                    max_s=float(lcfg.get("max_segment_s", 15.0)))
    listener = Listener(session, svc, conv, stt=stt, vad=SileroVad(), segmenter=seg)
    listener.start()
    return listener


if __name__ == "__main__":
    sys.exit(main())

"""Inky listens: Pi mic → VAD segments → Whisper → wake word / follow-up → LLM → speech.

  body/audio/mic    (sub)  schemas.audio_mic_chunk from body/audio_bridge (muted while playing)
  body/voice/heard  (pub)  {ts, text, action, reason, reply, stt_s, llm_s, audio_s}

Runs inside the voice service process (same venv, same GPU, and it needs to
know when Inky is speaking). Every transcribed segment is logged to
~/body-logs/voice/heard_YYYYMMDD.jsonl — text only, no audio.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from desktop.chassis.config import Topics
from desktop.voice.chat import Conversation
from desktop.voice.segmenter import Segmenter
from desktop.voice.speech_queue import SpeechRequest

logger = logging.getLogger(__name__)

LOG_DIR = Path.home() / "body-logs" / "voice"
REPLY_KEY = "chat_reply"
ACK_KEY = "chat_ack"
SORRY = "Sorry, my thinking part isn't answering right now."


class Listener:
    def __init__(self, session: Any, service: Any, conv: Conversation, *,
                 stt: Callable[[np.ndarray], str], vad: Any, segmenter: Optional[Segmenter] = None,
                 topics: Topics = Topics(), log_dir: Optional[Path] = LOG_DIR,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._session = session
        self._svc = service
        self._conv = conv
        self._stt = stt
        self._vad = vad
        self._seg = segmenter or Segmenter()
        self._topics = topics
        self._log_dir = log_dir
        self._clock = clock
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_seq: Optional[int] = None
        self._jobs: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=4)
        self._thread: Optional[threading.Thread] = None
        self._sub: Any = None
        self.last_mic = 0.0

    def start(self, *, worker: bool = True) -> None:
        prev = self._svc.on_done

        def on_done(req: SpeechRequest, error: Optional[str]) -> None:
            if prev is not None:
                prev(req, error)
            if req.key in (REPLY_KEY, ACK_KEY) and error is None:
                self._conv.open_window(self._clock())

        self._svc.on_done = on_done
        if worker:  # tests drain the job queue themselves
            self._thread = threading.Thread(target=self._work, name="listen", daemon=True)
            self._thread.start()
        self._sub = self._session.declare_subscriber(self._topics.audio_mic, self._on_mic)

    def shutdown(self) -> None:
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ── mic stream (zenoh callback thread) ──────────────────────────

    def _on_mic(self, sample: Any) -> None:
        try:
            msg = json.loads(sample.payload.to_bytes())
        except (ValueError, UnicodeDecodeError):
            return
        self.on_mic_msg(msg)

    def on_mic_msg(self, msg: dict) -> None:
        self.last_mic = self._clock()
        seq = msg.get("seq")
        gap = self._last_seq is not None and isinstance(seq, int) and seq != self._last_seq + 1
        self._last_seq = seq if isinstance(seq, int) else None
        if msg.get("muted") or "data" not in msg or gap or self._svc.speaking:
            self._end_segment()
            if msg.get("muted") or "data" not in msg or self._svc.speaking:
                return
        try:
            pcm = np.frombuffer(base64.standard_b64decode(msg["data"]), dtype="<i2")
        except (ValueError, TypeError):
            return
        self._buf = np.concatenate([self._buf, pcm.astype(np.float32) / 32768.0])
        f = self._seg.frame
        while len(self._buf) >= f:
            frame, self._buf = self._buf[:f], self._buf[f:]
            seg = self._seg.feed(frame, self._vad(frame))
            if seg is not None:
                self._submit(seg)

    def _end_segment(self) -> None:
        seg = self._seg.flush()
        self._buf = np.zeros(0, dtype=np.float32)
        self._vad.reset()
        if seg is not None:
            self._submit(seg)

    def _submit(self, seg: np.ndarray) -> None:
        try:
            self._jobs.put_nowait(seg)
        except queue.Full:
            logger.warning("listener busy: dropped a %.1f s segment", len(seg) / 16000)

    # ── worker ──────────────────────────────────────────────────────

    def _work(self) -> None:
        while True:
            seg = self._jobs.get()
            if seg is None:
                return
            try:
                self.handle_segment(seg)
            except Exception:
                logger.exception("segment handling failed")

    def handle_segment(self, seg: np.ndarray) -> dict:
        t0 = time.monotonic()
        text = self._stt(seg)
        stt_s = time.monotonic() - t0
        d = self._conv.decide(text, self._clock())
        rec: dict = {"text": text, "action": d.action, "reason": d.reason,
                     "audio_s": round(len(seg) / 16000, 2), "stt_s": round(stt_s, 3)}
        if d.wake is not None and d.wake.matched:
            rec["heard_as"], rec["wake_score"] = d.wake.heard_as, d.wake.score
        if d.action == "ack":
            rec["reply"] = self._conv.ack_text
            self._say(self._conv.ack_text, ACK_KEY)
        elif d.action == "reply":
            t1 = time.monotonic()
            try:
                answer = self._conv.reply(d.prompt, self._clock())
            except Exception as e:
                logger.warning("LLM failed: %s", e)
                answer, rec["error"] = SORRY, str(e)
            rec["llm_s"] = round(time.monotonic() - t1, 3)
            rec["reply"] = answer
            if answer:
                self._say(answer, REPLY_KEY)
        logger.info("heard %r -> %s (%s)%s", text, d.action, d.reason,
                    f" reply {rec['reply']!r}" if "reply" in rec else "")
        self._publish(rec)
        return rec

    def _say(self, text: str, key: str) -> None:
        self._svc.offer(SpeechRequest(text=text, kind="chat", key=key))

    def _publish(self, rec: dict) -> None:
        rec = {"ts": time.time(), **rec}
        try:
            self._session.put(self._topics.voice_heard, json.dumps(rec))
        except Exception:
            logger.exception("voice_heard publish failed")
        if self._log_dir is None:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            with open(self._log_dir / time.strftime("heard_%Y%m%d.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            logger.exception("heard log write failed")

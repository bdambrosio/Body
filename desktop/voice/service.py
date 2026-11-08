"""Desktop voice service: ``body/voice/say`` requests → Kokoro → Pi speaker.

  body/voice/say    (sub)  schemas.voice_say — from the nav UI, desktop.voice.say, ...
  body/voice/state  (pub)  {ts, state: idle|speaking, text, kind, bridge_ok}
  body/audio/play   (pub)  schemas.audio_play_chunk, one chunk per synthesized segment
  body/audio/stop   (pub)  when an alert pre-empts what's playing
  body/audio/status (sub)  Pi bridge edges: utt_started / utt_done

One utterance is on the Pi at a time: the next is sent only after the bridge
reports ``utt_done`` (or a timeout), so the queue policy here decides order.
Timings per utterance go to ~/body-logs/voice/voice_YYYYMMDD.jsonl.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from body.lib import schemas
from desktop.chassis.config import Topics
from desktop.voice.speech_queue import SpeechQueue, SpeechRequest

logger = logging.getLogger(__name__)

LOG_DIR = Path.home() / "body-logs" / "voice"
DONE_TIMEOUT_PAD_S = 5.0
BRIDGE_STALE_S = 3.0
BRIDGE_WAIT_S = 5.0     # hold an utterance this long for a silent bridge to appear
LEAD_S = 1.0            # send audio at most this far ahead of real-time playback
CHUNK_S = 0.5           # max audio per body/audio/play message: an 8 s sentence in one
                        # 515 KB message never reached the Pi (11-08); ~24 KB pieces do


def _block_put_kwargs() -> dict:
    """Audio chunks must not be dropped: a burst of DROP-QoS chunks lost pieces (11-08)."""
    try:
        import zenoh
        return {"congestion_control": zenoh.CongestionControl.BLOCK}
    except Exception:
        return {}


class VoiceService:
    def __init__(self, session: Any, tts: Any, *, topics: Topics = Topics(),
                 max_queue: int = 6, info_max_age_s: float = 10.0,
                 log_dir: Optional[Path] = LOG_DIR) -> None:
        self._session = session
        self._tts = tts
        self._topics = topics
        self._cv = threading.Condition()
        self._queue = SpeechQueue(max_len=max_queue, info_max_age_s=info_max_age_s)
        self._cut = False
        self._stopping = False
        self._done: dict[str, tuple[float, Optional[str]]] = {}
        self._started: dict[str, float] = {}
        self._bridge_seen = 0.0
        self._log_dir = log_dir
        self._thread: Optional[threading.Thread] = None
        self._subs: list[Any] = []
        self._speaking = False
        self._put_kw = _block_put_kwargs()
        # Called on the worker thread after each utterance: (request, error or None).
        self.on_done: Optional[Callable[[SpeechRequest, Optional[str]], None]] = None

    @property
    def speaking(self) -> bool:
        """True from the moment an utterance is taken off the queue until it's done."""
        return self._speaking

    # ── zenoh inputs ────────────────────────────────────────────────

    def start(self) -> None:
        self._subs.append(self._session.declare_subscriber(self._topics.voice_say, self._on_say))
        self._subs.append(self._session.declare_subscriber(self._topics.audio_status, self._on_status))
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self._thread.start()
        self._publish_state("idle")

    def shutdown(self) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _decode(sample: Any) -> Optional[dict]:
        try:
            obj = json.loads(sample.payload.to_bytes())
        except (ValueError, UnicodeDecodeError):
            return None
        return obj if isinstance(obj, dict) else None

    def _on_say(self, sample: Any) -> None:
        msg = self._decode(sample)
        if msg is None:
            return
        try:
            req = SpeechRequest(text=str(msg.get("text", "")), kind=str(msg.get("kind", "info")),
                                key=msg.get("key"), cooldown_s=float(msg.get("cooldown_s", 0.0)))
        except (TypeError, ValueError):
            return
        self.offer(req)

    def offer(self, req: SpeechRequest) -> bool:
        with self._cv:
            off = self._queue.offer(req, time.monotonic())
            if off.preempt:
                self._cut = True
            self._cv.notify_all()
        if off.preempt:
            logger.info("alert pre-empts current utterance")
            self._session.put(self._topics.audio_stop, json.dumps({"ts": time.time()}))
        if not off.accepted:
            logger.debug("say rejected (%s): %r", off.reason, req.text)
        return off.accepted

    def _on_status(self, sample: Any) -> None:
        msg = self._decode(sample)
        if msg is None:
            return
        now = time.monotonic()
        with self._cv:
            self._bridge_seen = now
            ev, uid = msg.get("event"), msg.get("event_utt_id")
            if ev == "utt_started" and uid:
                self._started[uid] = now
            elif ev == "utt_done" and uid:
                self._done[uid] = (now, msg.get("error"))
                self._cv.notify_all()

    def bridge_ok(self) -> bool:
        return self._bridge_seen > 0 and time.monotonic() - self._bridge_seen < BRIDGE_STALE_S

    # ── worker ──────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            with self._cv:
                req = None
                while not self._stopping:
                    req = self._queue.pop(time.monotonic())
                    if req is not None:
                        break
                    self._cv.wait(timeout=1.0)
                if self._stopping:
                    return
                self._cut = False
                self._speaking = True
            error: Optional[str] = "exception"
            try:
                error = self._speak(req)
            except Exception:
                logger.exception("speak failed: %r", req.text)
            finally:
                with self._cv:
                    self._queue.done()
                    self._speaking = False
                self._publish_state("idle")
                if self.on_done is not None:
                    try:
                        self.on_done(req, error)
                    except Exception:
                        logger.exception("on_done hook failed")

    def _speak(self, req: SpeechRequest) -> Optional[str]:
        utt_id = uuid.uuid4().hex[:12]
        t0 = time.monotonic()
        # A freshly (re)started bridge may not be subscribed yet: chunks sent
        # before its first status are lost (seen 11-06). Wait briefly for it.
        with self._cv:
            wait_end = t0 + BRIDGE_WAIT_S
            while not self.bridge_ok() and not self._stopping:
                left = wait_end - time.monotonic()
                if left <= 0:
                    break
                self._cv.wait(timeout=min(left, 0.2))
        self._publish_state("speaking", req)
        rate = self._tts.sample_rate
        seq = 0
        total_bytes = 0
        t_first: Optional[float] = None
        pending: Optional[bytes] = None

        sent_s = 0.0

        def send(pcm: bytes, final: bool) -> None:
            nonlocal seq, t_first, sent_s
            msg = schemas.audio_play_chunk(
                utt_id=utt_id, seq=seq, final=final, sample_rate=rate,
                pcm_base64=base64.standard_b64encode(pcm).decode("ascii"),
                text=req.text if seq == 0 else None)
            if t_first is None:
                t_first = time.monotonic()
            else:  # pace: stay at most LEAD_S ahead of playback
                ahead = sent_s - (time.monotonic() - t_first) - LEAD_S
                if ahead > 0:
                    time.sleep(ahead)
            self._session.put(self._topics.audio_play, json.dumps(msg), **self._put_kw)
            sent_s += len(pcm) / 2 / rate
            seq += 1

        cut = False
        max_bytes = max(2, int(rate * CHUNK_S)) * 2
        for seg_pcm in self._tts.synth(req.text):
            for i in range(0, len(seg_pcm), max_bytes):
                pcm = seg_pcm[i:i + max_bytes]
                with self._cv:
                    cut = self._cut or self._stopping
                if cut:
                    break
                total_bytes += len(pcm)
                if pending is not None:
                    send(pending, final=False)
                pending = pcm
            if cut:
                break
        if pending is not None and not cut:
            send(pending, final=True)

        error: Optional[str] = None
        t_done: Optional[float] = None
        if seq == 0:
            error = "stopped" if cut else "no_audio"
        else:
            audio_s = total_bytes / 2 / rate
            deadline = time.monotonic() + audio_s + DONE_TIMEOUT_PAD_S
            with self._cv:
                while utt_id not in self._done and not self._stopping:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self._cv.wait(timeout=left)
                done = self._done.pop(utt_id, None)
                t_started = self._started.pop(utt_id, None)
            if done is None:
                error = "no_ack"  # bridge down / link lost
            else:
                t_done, error = done
            self._log({
                "utt_id": utt_id, "text": req.text, "kind": req.kind, "key": req.key,
                "queued_s": round(t0 - req.t_rx, 3),
                "first_chunk_s": None if t_first is None else round(t_first - t0, 3),
                "started_s": None if t_started is None else round(t_started - t0, 3),
                "done_s": None if t_done is None else round(t_done - t0, 3),
                "audio_s": round(audio_s, 2), "chunks": seq, "error": error,
            })
        if error:
            logger.info("utterance %s %r: %s", utt_id, req.text, error)
        return error

    # ── outputs ─────────────────────────────────────────────────────

    def _publish_state(self, state: str, req: Optional[SpeechRequest] = None) -> None:
        msg = {"ts": time.time(), "state": state, "bridge_ok": self.bridge_ok(),
               "text": req.text if req else None, "kind": req.kind if req else None}
        try:
            self._session.put(self._topics.voice_state, json.dumps(msg))
        except Exception:
            logger.exception("voice state publish failed")

    def _log(self, rec: dict) -> None:
        rec = {"ts": time.time(), **rec}
        logger.info("said %r: %s", rec["text"], {k: rec[k] for k in ("first_chunk_s", "started_s", "done_s", "error")})
        if self._log_dir is None:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            path = self._log_dir / time.strftime("voice_%Y%m%d.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            logger.exception("voice log write failed")

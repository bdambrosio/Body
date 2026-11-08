"""Pi audio bridge: play PCM utterances from the desktop on the USB speaker.

The desktop voice service (desktop/voice) does the text-to-speech and sends
utterances as chunked PCM on ``body/audio/play``; this process only plays them.
Each utterance gets one ``pw-cat`` playback process fed through stdin, so the
chunks of one utterance play gaplessly and playback can start as soon as the
first chunk arrives. Utterances play strictly in arrival order.

  body/audio/play   (sub)  schemas.audio_play_chunk
  body/audio/stop   (sub)  {} — cut the current utterance, drop the queue
  body/audio/status (pub)  schemas.audio_status, 2 Hz + on every edge
  body/audio/mic    (pub)  schemas.audio_mic_chunk, 16 kHz mono blocks (Phase 2 chat)

The mic is half-duplex: while an utterance plays, and for ``mute_tail_s``
after it ends, blocks go out as ``muted`` with no audio, so the desktop's
speech recognizer never hears the robot's own voice.

PipeWire runs in bruce's user session (linger on, XDG_RUNTIME_DIR set in
deploy/body-launcher.service). With no sound device the bridge keeps running,
reports ``device_ok: false``, answers every utterance with an ``utt_done``
error so the desktop never waits forever, and re-probes every
``probe_interval_s``.
"""

from __future__ import annotations

import base64
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from body.lib import schemas, zenoh_helpers

PLAY_TOPIC = "body/audio/play"
STOP_TOPIC = "body/audio/stop"
STATUS_TOPIC = "body/audio/status"
MIC_TOPIC = "body/audio/mic"


class Player(Protocol):
    """One utterance's output stream."""

    def write(self, pcm: bytes) -> None: ...
    def finish(self) -> None: ...  # close input, block until playback has drained
    def kill(self) -> None: ...


class PwCatPlayer:
    """Raw mono s16le PCM piped into ``pw-cat --playback``."""

    def __init__(self, sample_rate: int) -> None:
        self._proc = subprocess.Popen(
            ["pw-cat", "--playback", "--raw", "--rate", str(int(sample_rate)),
             "--channels", "1", "--format", "s16", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def write(self, pcm: bytes) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(pcm)  # blocks at real-time pace once the pipe is full
        self._proc.stdin.flush()

    def finish(self) -> None:
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except OSError:
            pass
        self._proc.wait()

    def kill(self) -> None:
        try:
            self._proc.kill()
        except OSError:
            pass


def probe_device() -> bool:
    """True when pw-cat exists and PipeWire has a default sink."""
    if shutil.which("pw-cat") is None or shutil.which("wpctl") is None:
        return False
    try:
        r = subprocess.run(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@dataclass
class _Utterance:
    utt_id: str
    sample_rate: int
    chunks: dict[int, bytes] = field(default_factory=dict)
    final_seq: Optional[int] = None
    last_rx: float = 0.0


class AudioBridge:
    """Utterance queue + playback thread. Transport-free (tests drive it directly).

    ``emit(event, utt_id, error)`` is called on every edge: event is
    "utt_started" or "utt_done" (error None on a clean finish).
    """

    def __init__(
        self,
        player_factory: Callable[[int], Player],
        emit: Callable[[str, str, Optional[str]], None],
        *,
        gap_timeout_s: float = 2.0,
        max_queue: int = 8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._player_factory = player_factory
        self._emit = emit
        self._gap_timeout_s = gap_timeout_s
        self._max_queue = max_queue
        self._clock = clock
        self._cv = threading.Condition()
        self._queue: deque[_Utterance] = deque()
        self._by_id: dict[str, _Utterance] = {}
        self._done_ids: deque[str] = deque(maxlen=64)  # late chunks of finished/cut utterances
        self._current: Optional[_Utterance] = None
        self._player: Optional[Player] = None
        self._cut = False
        self._stopping = False
        self.device_ok = True
        self._thread: Optional[threading.Thread] = None

    # ── inputs ──────────────────────────────────────────────────────

    def on_chunk(self, msg: dict[str, Any]) -> None:
        try:
            utt_id = str(msg["utt_id"])
            seq = int(msg["seq"])
            rate = int(msg["sample_rate"])
            pcm = base64.standard_b64decode(msg["data"])
        except (KeyError, TypeError, ValueError):
            return
        dropped: list[tuple[str, str]] = []
        with self._cv:
            if utt_id in self._done_ids:
                return
            u = self._by_id.get(utt_id)
            if u is None and self._current is not None and self._current.utt_id == utt_id:
                u = self._current
            if u is None:
                if not self.device_ok:
                    self._done_ids.append(utt_id)
                    dropped.append((utt_id, "no_device"))
                elif len(self._queue) >= self._max_queue:
                    self._done_ids.append(utt_id)
                    dropped.append((utt_id, "queue_full"))
                else:
                    u = _Utterance(utt_id, rate)
                    self._by_id[utt_id] = u
                    self._queue.append(u)
            if u is not None:
                u.chunks[seq] = pcm
                if msg.get("final"):
                    u.final_seq = seq
                u.last_rx = self._clock()
                self._cv.notify_all()
        for uid, err in dropped:
            self._emit("utt_done", uid, err)

    def stop_all(self) -> None:
        """Cut the playing utterance and drop everything queued."""
        with self._cv:
            dropped = [u.utt_id for u in self._queue]
            for uid in dropped:
                self._done_ids.append(uid)
            self._queue.clear()
            self._by_id.clear()
            if self._current is not None:
                self._cut = True
                if self._player is not None:
                    self._player.kill()
            self._cv.notify_all()
        for uid in dropped:
            self._emit("utt_done", uid, "stopped")

    # ── state ───────────────────────────────────────────────────────

    def snapshot(self) -> tuple[str, Optional[str], int]:
        with self._cv:
            if self._current is not None:
                return "playing", self._current.utt_id, len(self._queue)
            return "idle", None, len(self._queue)

    # ── playback thread ─────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="audio-play", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        with self._cv:
            self._stopping = True
        self.stop_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stopping and not self._queue:
                    self._cv.wait()
                if self._stopping:
                    return
                u = self._queue.popleft()
                self._by_id.pop(u.utt_id, None)
                self._current = u
                self._cut = False
            self.play_one(u)

    def play_one(self, u: _Utterance) -> None:
        """Play ``u`` to completion (or cut / gap timeout). Runs on the playback thread."""
        error: Optional[str] = None
        player: Optional[Player] = None
        next_seq = 0
        try:
            player = self._player_factory(u.sample_rate)
            with self._cv:
                self._player = player
            self._emit("utt_started", u.utt_id, None)
            while True:
                with self._cv:
                    while True:
                        if self._cut or self._stopping:
                            break
                        if next_seq in u.chunks:
                            break
                        if u.final_seq is not None and next_seq > u.final_seq:
                            break
                        waited = self._clock() - u.last_rx
                        if waited >= self._gap_timeout_s:
                            break
                        self._cv.wait(timeout=self._gap_timeout_s - waited)
                    if self._cut or self._stopping:
                        error = "stopped"
                        break
                    if u.final_seq is not None and next_seq > u.final_seq:
                        break
                    pcm = u.chunks.pop(next_seq, None)
                    if pcm is None:
                        error = "gap_timeout"
                        break
                next_seq += 1
                player.write(pcm)  # outside the lock: blocks at playback pace
            if error is None:
                player.finish()
            else:
                player.kill()
        except (OSError, ValueError) as e:  # BrokenPipe when killed mid-write, spawn failure
            with self._cv:
                cut = self._cut or self._stopping
            error = "stopped" if cut else f"player_error: {e}"
            if player is not None:
                player.kill()
        finally:
            with self._cv:
                self._player = None
                self._current = None
                self._done_ids.append(u.utt_id)
            self._emit("utt_done", u.utt_id, error)


class MicMute:
    """Half-duplex gate: muted while playing and for ``tail_s`` after."""

    def __init__(self, tail_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._tail_s = tail_s
        self._clock = clock
        self._playing = False
        self._until = 0.0
        self._lock = threading.Lock()

    def on_event(self, event: str) -> None:
        with self._lock:
            if event == "utt_started":
                self._playing = True
            elif event == "utt_done":
                self._playing = False
                self._until = self._clock() + self._tail_s

    def muted(self) -> bool:
        with self._lock:
            return self._playing or self._clock() < self._until


class MicStreamer:
    """``pw-record`` → fixed-size blocks → ``publish(seq, pcm | None)``.

    ``pcm`` is None for a muted block. Restarts the recorder if it exits.
    """

    def __init__(self, publish: Callable[[int, Optional[bytes]], None], mute: MicMute,
                 *, rate: int = 16000, block_ms: int = 100) -> None:
        self._publish = publish
        self._mute = mute
        self.rate = int(rate)
        self._block_bytes = int(self.rate * block_ms / 1000) * 2
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._seq = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="audio-mic", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    ["pw-record", "--raw", "--rate", str(self.rate), "--channels", "1",
                     "--format", "s16", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
            except OSError as e:
                print(f"[audio_bridge] mic: pw-record failed to start: {e}", flush=True)
                self._stop.wait(10.0)
                continue
            self.pump(self._proc.stdout)
            try:
                self._proc.kill()
                self._proc.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if not self._stop.is_set():
                print("[audio_bridge] mic: recorder exited; restarting in 2 s", flush=True)
                self._stop.wait(2.0)

    def pump(self, stream: Any) -> None:
        """Read blocks from ``stream`` until EOF/stop and publish them."""
        buf = b""
        while not self._stop.is_set():
            data = stream.read(self._block_bytes - len(buf))
            if not data:
                return
            buf += data
            if len(buf) < self._block_bytes:
                continue
            block, buf = buf, b""
            try:
                self._publish(self._seq, None if self._mute.muted() else block)
            except Exception as e:  # transport hiccup: drop the block
                print(f"[audio_bridge] mic publish failed: {e}", flush=True)
            self._seq += 1


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    cfg = body_cfg.get("audio", {})
    status_hz = float(cfg.get("status_hz", 2.0))
    probe_interval_s = float(cfg.get("probe_interval_s", 10.0))

    session = zenoh_helpers.open_session(body_cfg)
    pub_lock = threading.Lock()
    bridge_ref: list[AudioBridge] = []

    def publish_status(event: Optional[str] = None, event_utt_id: Optional[str] = None,
                       error: Optional[str] = None) -> None:
        b = bridge_ref[0]
        state, utt_id, queued = b.snapshot()
        msg = schemas.audio_status(state=state, device_ok=b.device_ok, utt_id=utt_id,
                                   queued=queued, event=event, event_utt_id=event_utt_id,
                                   error=error)
        with pub_lock:
            zenoh_helpers.publish_json(session, STATUS_TOPIC, msg)

    mute = MicMute(float(cfg.get("mute_tail_s", 0.4)))

    def emit(event: str, utt_id: str, error: Optional[str]) -> None:
        mute.on_event(event)
        if error and error != "stopped":
            print(f"[audio_bridge] {event} {utt_id}: {error}", flush=True)
        publish_status(event, utt_id, error)

    bridge = AudioBridge(
        PwCatPlayer, emit,
        gap_timeout_s=float(cfg.get("gap_timeout_s", 2.0)),
        max_queue=int(cfg.get("max_queue", 8)),
    )
    bridge_ref.append(bridge)
    bridge.device_ok = probe_device()
    print(f"[audio_bridge] started, device_ok={bridge.device_ok}", flush=True)

    zenoh_helpers.declare_subscriber_json(session, PLAY_TOPIC, lambda _k, m: bridge.on_chunk(m))
    zenoh_helpers.declare_subscriber_json(session, STOP_TOPIC, lambda _k, _m: bridge.stop_all())
    bridge.start()

    mic: Optional[MicStreamer] = None
    if cfg.get("mic_enabled", True) and shutil.which("pw-record") is not None:
        mic_rate = int(cfg.get("mic_rate", 16000))

        def publish_mic(seq: int, pcm: Optional[bytes]) -> None:
            msg = schemas.audio_mic_chunk(
                seq=seq, sample_rate=mic_rate, muted=pcm is None,
                pcm_base64=None if pcm is None else base64.standard_b64encode(pcm).decode("ascii"))
            with pub_lock:
                zenoh_helpers.publish_json(session, MIC_TOPIC, msg)

        mic = MicStreamer(publish_mic, mute, rate=mic_rate, block_ms=int(cfg.get("mic_block_ms", 100)))
        mic.start()
        print(f"[audio_bridge] mic streaming {mic_rate} Hz on {MIC_TOPIC}", flush=True)

    stop = threading.Event()

    def handle_signal(_sig: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGHUP, handle_signal)

    next_probe = time.monotonic() + probe_interval_s
    period = 1.0 / max(0.5, status_hz)
    while not stop.is_set():
        now = time.monotonic()
        if now >= next_probe:
            next_probe = now + probe_interval_s
            ok = probe_device()
            if ok != bridge.device_ok:
                print(f"[audio_bridge] device_ok -> {ok}", flush=True)
                bridge.device_ok = ok
        try:
            publish_status()
        except Exception as e:  # never let a transport hiccup kill the process
            print(f"[audio_bridge] status publish failed: {e}", flush=True)
        stop.wait(period)

    if mic is not None:
        mic.shutdown()
    bridge.shutdown()
    session.close()


if __name__ == "__main__":
    main()

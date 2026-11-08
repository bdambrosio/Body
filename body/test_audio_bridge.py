"""Tests for body.audio_bridge.AudioBridge with a fake player (no PipeWire needed)."""
import base64
import threading
import time
import unittest

import io

from body.audio_bridge import AudioBridge, MicMute, MicStreamer
from body.lib import schemas


def chunk(utt, seq, data=b"\x01\x00", final=False, rate=24000):
    return schemas.audio_play_chunk(utt_id=utt, seq=seq, final=final, sample_rate=rate,
                                    pcm_base64=base64.standard_b64encode(data).decode())


class FakePlayer:
    def __init__(self, log, rate, block=None):
        self.log, self.rate, self.block = log, rate, block
        self.data = b""
        self.killed = False
        self.finished = False
        log.append(self)

    def write(self, pcm):
        if self.block is not None:
            self.block.wait(2.0)
        if self.killed:
            raise BrokenPipeError("killed")
        self.data += pcm

    def finish(self):
        self.finished = True

    def kill(self):
        self.killed = True


class Harness:
    def __init__(self, gap_timeout_s=0.3, max_queue=8, block=None):
        self.players = []
        self.events = []
        self.cv = threading.Condition()

        def emit(ev, uid, err):
            with self.cv:
                self.events.append((ev, uid, err))
                self.cv.notify_all()

        self.bridge = AudioBridge(lambda r: FakePlayer(self.players, r, block), emit,
                                  gap_timeout_s=gap_timeout_s, max_queue=max_queue)
        self.bridge.start()

    def wait_done(self, n, timeout=3.0):
        end = time.monotonic() + timeout
        with self.cv:
            while sum(1 for e in self.events if e[0] == "utt_done") < n:
                left = end - time.monotonic()
                if left <= 0:
                    raise AssertionError(f"timeout; events={self.events}")
                self.cv.wait(left)
        return [e for e in self.events if e[0] == "utt_done"]


class TestAudioBridge(unittest.TestCase):
    def test_chunks_play_in_seq_order(self):
        h = Harness()
        h.bridge.on_chunk(chunk("a", 1, b"BB"))
        h.bridge.on_chunk(chunk("a", 0, b"AA"))
        h.bridge.on_chunk(chunk("a", 2, b"CC", final=True))
        done = h.wait_done(1)
        self.assertEqual(done, [("utt_done", "a", None)])
        self.assertEqual(h.players[0].data, b"AABBCC")
        self.assertTrue(h.players[0].finished)
        self.assertEqual(h.players[0].rate, 24000)
        h.bridge.shutdown()

    def test_chunks_arriving_while_playing_join_the_same_utterance(self):
        h = Harness()
        h.bridge.on_chunk(chunk("a", 0, b"AA"))
        time.sleep(0.05)  # utterance is now current
        h.bridge.on_chunk(chunk("a", 1, b"BB", final=True))
        h.wait_done(1)
        self.assertEqual(len(h.players), 1)
        self.assertEqual(h.players[0].data, b"AABB")
        h.bridge.shutdown()

    def test_utterances_play_in_arrival_order(self):
        h = Harness()
        h.bridge.on_chunk(chunk("a", 0, b"A", final=True))
        h.bridge.on_chunk(chunk("b", 0, b"B", final=True))
        done = h.wait_done(2)
        self.assertEqual([d[1] for d in done], ["a", "b"])
        starts = [e[1] for e in h.events if e[0] == "utt_started"]
        self.assertEqual(starts, ["a", "b"])
        h.bridge.shutdown()

    def test_missing_final_ends_on_gap_timeout(self):
        h = Harness(gap_timeout_s=0.2)
        h.bridge.on_chunk(chunk("a", 0, b"AA"))
        done = h.wait_done(1)
        self.assertEqual(done, [("utt_done", "a", "gap_timeout")])
        self.assertTrue(h.players[0].killed)
        h.bridge.shutdown()

    def test_stop_cuts_current_and_drops_queue(self):
        block = threading.Event()
        h = Harness(block=block)
        h.bridge.on_chunk(chunk("a", 0, b"AA", final=True))
        h.bridge.on_chunk(chunk("b", 0, b"BB", final=True))
        time.sleep(0.05)
        h.bridge.stop_all()
        block.set()
        done = h.wait_done(2)
        self.assertEqual(sorted(done), [("utt_done", "a", "stopped"), ("utt_done", "b", "stopped")])
        # late chunk of a cut utterance is ignored
        h.bridge.on_chunk(chunk("a", 1, b"XX", final=True))
        time.sleep(0.1)
        self.assertEqual(len(h.players), 1)
        h.bridge.shutdown()

    def test_no_device_answers_immediately(self):
        h = Harness()
        h.bridge.device_ok = False
        h.bridge.on_chunk(chunk("a", 0, final=True))
        self.assertEqual(h.wait_done(1), [("utt_done", "a", "no_device")])
        self.assertEqual(h.players, [])
        h.bridge.shutdown()

    def test_queue_full_rejects(self):
        block = threading.Event()
        h = Harness(max_queue=1, block=block)
        h.bridge.on_chunk(chunk("a", 0, final=True))
        time.sleep(0.05)  # a playing, queue empty
        h.bridge.on_chunk(chunk("b", 0, final=True))
        h.bridge.on_chunk(chunk("c", 0, final=True))
        self.assertIn(("utt_done", "c", "queue_full"), h.events)
        block.set()
        h.wait_done(3)
        h.bridge.shutdown()

    def test_malformed_chunk_ignored(self):
        h = Harness()
        h.bridge.on_chunk({"utt_id": "a"})
        time.sleep(0.05)
        self.assertEqual(h.events, [])
        h.bridge.shutdown()


class TestMic(unittest.TestCase):
    def test_mute_while_playing_and_for_tail(self):
        now = [100.0]
        m = MicMute(0.4, clock=lambda: now[0])
        self.assertFalse(m.muted())
        m.on_event("utt_started")
        now[0] += 5.0
        self.assertTrue(m.muted())
        m.on_event("utt_done")
        now[0] += 0.3
        self.assertTrue(m.muted())
        now[0] += 0.2
        self.assertFalse(m.muted())

    def test_pump_publishes_whole_blocks_and_mutes(self):
        out = []
        mute = MicMute(0.0)
        mic = MicStreamer(lambda seq, pcm: out.append((seq, pcm)), mute, rate=1000, block_ms=10)
        # 10 ms at 1 kHz = 10 samples = 20 bytes per block; 50 bytes -> 2 blocks + 10 left over
        mic.pump(io.BytesIO(bytes(range(50))))
        self.assertEqual([s for s, _ in out], [0, 1])
        self.assertEqual(out[0][1], bytes(range(20)))
        mute.on_event("utt_started")
        mic.pump(io.BytesIO(bytes(20)))
        self.assertEqual(out[-1], (2, None))

    def test_mic_chunk_schema(self):
        m = schemas.audio_mic_chunk(seq=3, sample_rate=16000, pcm_base64="AAA=")
        self.assertEqual(m["data"], "AAA=")
        self.assertFalse(m["muted"])
        m = schemas.audio_mic_chunk(seq=4, sample_rate=16000, pcm_base64=None, muted=True)
        self.assertNotIn("data", m)


if __name__ == "__main__":
    unittest.main()

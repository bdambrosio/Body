"""Tests for the chat path: wake match, segmenter, conversation, listener wiring (no GPU)."""
import base64
import unittest

import numpy as np

from desktop.voice.chat import Conversation, clean_reply
from desktop.voice.listener import ACK_KEY, REPLY_KEY, Listener
from desktop.voice.segmenter import Segmenter
from desktop.voice.wake import match_wake


class TestWake(unittest.TestCase):
    def test_spellings_whisper_produces(self):
        for t in ["Inky, what time is it?", "Inkie what's up", "Hey Inky, hello",
                  "OK, inkey. Tell me a joke.", "Hi, in key, how are you?", "Hinky, hi"]:
            self.assertTrue(match_wake(t, "Inky").matched, t)

    def test_rest_is_after_the_name(self):
        m = match_wake("Hey Inky, what's your favorite color?", "Inky")
        self.assertEqual(m.rest, "what's your favorite color?")
        self.assertEqual(m.heard_as.lower(), "inky")

    def test_not_a_wake(self):
        for t in ["I think the ink is dry", "Pass the milk please", "Where is Inky?",
                  "the inky black night", "Thank you."]:
            self.assertFalse(match_wake(t, "Inky").matched, t)


def tone_frames(seg, pattern, frame=512):
    """pattern: list of (p, n_frames). Returns the segments emitted."""
    out = []
    for p, n in pattern:
        for _ in range(n):
            s = seg.feed(np.full(frame, p, dtype=np.float32), p)
            if s is not None:
                out.append(s)
    return out


class TestSegmenter(unittest.TestCase):
    def test_segment_with_pre_roll_closes_after_silence(self):
        seg = Segmenter(end_silence_s=0.32, pre_roll_s=0.096, min_speech_s=0.1)  # 10 / 3 / 3 frames
        out = tone_frames(seg, [(0.0, 20), (0.9, 30), (0.1, 10)])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), (3 + 30 + 10) * 512)
        self.assertAlmostEqual(float(out[0][0]), 0.0)  # pre-roll is the quiet audio

    def test_short_blip_dropped(self):
        seg = Segmenter(end_silence_s=0.32, min_speech_s=0.3)
        self.assertEqual(tone_frames(seg, [(0.9, 2), (0.0, 20)]), [])

    def test_max_length_cuts(self):
        seg = Segmenter(max_s=1.0)
        self.assertEqual(len(tone_frames(seg, [(0.9, 70)])), 2)

    def test_flush(self):
        seg = Segmenter(min_speech_s=0.1)
        tone_frames(seg, [(0.9, 10)])
        self.assertIsNotNone(seg.flush())
        self.assertFalse(seg.active)


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def llm(msgs):
            self.calls.append(msgs)
            return "**Hi!** I'm Inky 🤖."
        self.conv = Conversation(name="Inky", llm=llm, follow_up_s=8.0)

    def test_wake_reply_and_follow_up_window(self):
        d = self.conv.decide("Inky, who are you?", 0.0)
        self.assertEqual((d.action, d.prompt), ("reply", "who are you?"))
        self.assertEqual(self.conv.reply(d.prompt, 0.0), "Hi! I'm Inky .".replace(" .", "."))
        self.assertEqual(self.conv.decide("what do you like?", 1.0).action, "ignore")
        self.conv.open_window(3.0)   # reply finished playing
        self.assertEqual(self.conv.decide("what do you like?", 5.0).action, "reply")
        self.assertEqual(self.conv.decide("thank you", 5.0).action, "ignore")
        self.assertEqual(self.conv.decide("what do you like?", 12.0).action, "ignore")

    def test_name_only_acks(self):
        self.assertEqual(self.conv.decide("Inky?", 0.0).action, "ack")
        self.assertEqual(self.conv.decide("Hey, Inky. Thank you.", 0.0).action, "ack")

    def test_history_kept_and_reset(self):
        self.conv.reply("one", 0.0)
        self.conv.reply("two", 10.0)
        self.assertEqual(len(self.calls[-1]), 1 + 2 + 1)   # system + 1 pair + new
        self.conv.messages("three", 10.0 + 1000)
        self.assertEqual(self.conv.history, [])

    def test_clean_reply(self):
        self.assertEqual(clean_reply("<think>x</think> Hello *there*"), "Hello there")


class FakeVad:
    def __init__(self):
        self.resets = 0

    def __call__(self, frame):
        return 1.0 if float(np.abs(frame).max()) > 0.1 else 0.0

    def reset(self):
        self.resets += 1


class FakeSvc:
    def __init__(self):
        self.speaking = False
        self.on_done = None
        self.said = []

    def offer(self, req):
        self.said.append(req)
        return True


class FakeSession:
    def __init__(self):
        self.puts = []

    def declare_subscriber(self, *_a):
        return object()

    def put(self, topic, payload):
        self.puts.append(topic)


def mic_msg(seq, level, n=1600, muted=False):
    if muted:
        return {"seq": seq, "muted": True}
    pcm = np.full(n, int(level * 32767), dtype="<i2").tobytes()
    return {"seq": seq, "muted": False, "data": base64.standard_b64encode(pcm).decode()}


class TestListener(unittest.TestCase):
    def make(self, text):
        self.svc = FakeSvc()
        conv = Conversation(name="Inky", llm=lambda m: "I'm fine.")
        self.lis = Listener(FakeSession(), self.svc, conv, stt=lambda a: text, vad=FakeVad(),
                            segmenter=Segmenter(end_silence_s=0.2, min_speech_s=0.1), log_dir=None)
        self.lis.start(worker=False)
        return conv

    def tearDown(self):
        self.lis.shutdown()

    def feed(self, msgs):
        for m in msgs:
            self.lis.on_mic_msg(m)
        segs = []
        while not self.lis._jobs.empty():
            segs.append(self.lis._jobs.get_nowait())
        return segs

    def test_speech_then_silence_makes_one_segment(self):
        self.make("Inky, how are you?")
        segs = self.feed([mic_msg(i, 0.5) for i in range(5)] + [mic_msg(5 + i, 0.0) for i in range(3)])
        self.assertEqual(len(segs), 1)
        rec = self.lis.handle_segment(segs[0])
        self.assertEqual((rec["action"], rec["reply"]), ("reply", "I'm fine."))
        self.assertEqual(self.svc.said[-1].key, REPLY_KEY)

    def test_mute_ends_segment_and_drops_audio(self):
        self.make("x")
        segs = self.feed([mic_msg(0, 0.5), mic_msg(1, 0.5), mic_msg(2, 0.5, muted=True),
                          mic_msg(3, 0.5, muted=True)])
        self.assertEqual(len(segs), 1)

    def test_reply_done_opens_window(self):
        conv = self.make("Inky")
        rec = self.lis.handle_segment(np.zeros(16000, dtype=np.float32))
        self.assertEqual(rec["action"], "ack")
        self.assertEqual(self.svc.said[-1].key, ACK_KEY)
        self.svc.on_done(self.svc.said[-1], None)
        self.assertTrue(conv.in_window(self.lis._clock()))


if __name__ == "__main__":
    unittest.main()

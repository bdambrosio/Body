import unittest

from desktop.voice.speech_queue import SpeechQueue, SpeechRequest


def R(text, kind="info", key=None, cooldown_s=0.0):
    return SpeechRequest(text=text, kind=kind, key=key, cooldown_s=cooldown_s)


class TestSpeechQueue(unittest.TestCase):
    def test_priority_then_fifo(self):
        q = SpeechQueue()
        q.offer(R("i1"), 0)
        q.offer(R("c1", "chat"), 0)
        q.offer(R("a1", "alert"), 0)
        q.offer(R("i2"), 0)
        self.assertEqual([q.pop(1).text for _ in range(4)], ["a1", "c1", "i1", "i2"])
        self.assertIsNone(q.pop(1))

    def test_cooldown_per_key(self):
        q = SpeechQueue()
        self.assertTrue(q.offer(R("person", key="person", cooldown_s=60), 0).accepted)
        off = q.offer(R("person", key="person", cooldown_s=60), 30)
        self.assertEqual((off.accepted, off.reason), (False, "cooldown"))
        self.assertTrue(q.offer(R("pet", key="pet", cooldown_s=60), 30).accepted)
        self.assertTrue(q.offer(R("person", key="person", cooldown_s=60), 61).accepted)

    def test_stale_info_dropped_alert_kept(self):
        q = SpeechQueue(info_max_age_s=10)
        q.offer(R("old info"), 0)
        q.offer(R("old alert", "alert"), 0)
        self.assertEqual(q.pop(20).text, "old alert")
        self.assertIsNone(q.pop(20))

    def test_alert_preempts_non_alert(self):
        q = SpeechQueue()
        q.offer(R("info"), 0)
        q.pop(0)
        self.assertTrue(q.offer(R("stuck", "alert"), 1).preempt)
        q.done()
        q.pop(1)  # now an alert is current
        self.assertFalse(q.offer(R("estop", "alert"), 2).preempt)
        self.assertFalse(q.offer(R("info2"), 2).preempt)

    def test_queue_full_drops_lowest(self):
        q = SpeechQueue(max_len=2)
        q.offer(R("i1"), 0)
        q.offer(R("i2"), 0)
        self.assertEqual(q.offer(R("i3"), 0).reason, "queue_full")
        self.assertTrue(q.offer(R("a", "alert"), 0).accepted)  # evicts i2
        self.assertEqual([q.pop(0).text, q.pop(0).text], ["a", "i1"])

    def test_empty_text_rejected(self):
        q = SpeechQueue()
        self.assertFalse(q.offer(R("   "), 0).accepted)

    def test_unknown_kind_is_info(self):
        q = SpeechQueue()
        q.offer(R("x", "shout"), 0)
        self.assertEqual(q.pop(0).kind, "info")


if __name__ == "__main__":
    unittest.main()

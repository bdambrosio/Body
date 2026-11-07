import unittest
from types import SimpleNamespace

from desktop.nav.voice_client import STUCK_AFTER_S, VoiceAnnouncer, stuck_phrase


def det(cid, label=None, verdict="novel"):
    return SimpleNamespace(change_id=cid, label=label, verdict=verdict)


class TestVoiceAnnouncer(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.a = VoiceAnnouncer(self.sent.append)

    def texts(self):
        return [m["text"] for m in self.sent]

    def test_sighting_once_per_change(self):
        self.a.on_detection(det(1))                 # unlabeled yet
        self.a.on_detection(det(1, "person"))
        self.a.on_detection(det(1, "person"))
        self.assertEqual(self.texts(), ["Hello! I see a person."])
        self.assertEqual(self.sent[0]["key"], "sight:person")
        self.assertEqual(self.sent[0]["kind"], "info")

    def test_routine_and_uninteresting_skipped(self):
        self.a.on_detection(det(2, "person", verdict="expected"))
        self.a.on_detection(det(3, "furniture"))
        self.assertEqual(self.sent, [])

    def test_stuck_only_after_persisting(self):
        self.a.on_tick("BLOCKED", "depth_block", None, False, 0.0)
        self.a.on_tick("BLOCKED", "depth_block", None, False, STUCK_AFTER_S - 0.1)
        self.assertEqual(self.sent, [])
        self.a.on_tick("BLOCKED", "depth_block", None, False, STUCK_AFTER_S + 0.1)
        self.a.on_tick("BLOCKED", "depth_block", None, False, STUCK_AFTER_S + 5)
        self.assertEqual(self.texts(), ["I'm stuck. Something is in my way."])
        self.assertEqual(self.sent[0]["kind"], "alert")

    def test_short_block_not_announced_and_resets(self):
        self.a.on_tick("BLOCKED", "depth_block", None, False, 0.0)
        self.a.on_tick("DRIVING_SUBGOAL", None, None, False, 1.0)
        self.a.on_tick("BLOCKED", "depth_block", None, False, 2.0)
        self.a.on_tick("BLOCKED", "depth_block", None, False, 2.0 + STUCK_AFTER_S - 0.5)
        self.assertEqual(self.sent, [])

    def test_estop_latched_at_startup_is_quiet(self):
        self.a.on_tick(None, None, None, True, 0.0)
        self.a.on_tick(None, None, None, False, 20.0)
        self.assertEqual(self.sent, [])
        self.a.on_tick(None, None, None, True, 21.0)
        self.assertEqual(self.texts(), ["Emergency stop."])

    def test_failed_and_estop_edges(self):
        self.a.on_tick(None, None, None, False, 0.0)
        self.a.on_tick("FAILED", None, "abandoned", True, 11.0)
        self.a.on_tick("FAILED", None, "abandoned", True, 12.0)
        self.assertEqual(sorted(self.texts()),
                         ["Emergency stop.", "I give up. I couldn't get there."])

    def test_mute(self):
        self.a.muted = True
        self.a.on_detection(det(5, "pet"))
        self.a.on_task_finished("done")
        self.assertEqual(self.sent, [])

    def test_task_finished(self):
        self.a.on_task_finished("done")
        self.a.on_task_finished("stopped")
        self.assertEqual(self.texts(), ["Exploration finished.", "Exploration stopped."])

    def test_stuck_phrase_unknown_reason(self):
        self.assertEqual(stuck_phrase("weird"), "I'm stuck.")


if __name__ == "__main__":
    unittest.main()

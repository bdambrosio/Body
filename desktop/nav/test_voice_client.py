import unittest
from types import SimpleNamespace

from desktop.nav.voice_client import (STUCK_AFTER_S, VoiceAnnouncer, explore_summary_phrase,
                                      is_find, stuck_phrase)


def det(cid, label=None, verdict="novel", kind="new", confidence=0.8, what=None,
        investigated=None):
    return SimpleNamespace(change_id=cid, label=label, verdict=verdict, kind=kind,
                           confidence=confidence, what=what, investigated=investigated)


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
        self.assertEqual(self.texts(), ["Exploration finished. I found nothing new.",
                                        "Exploration stopped."])

    def test_explore_start_leg_and_home(self):
        self.a.on_explore_start()
        self.a.on_explore_leg("viewpoint")
        self.a.on_explore_leg("investigate", det(7, "box or bag"))
        self.a.on_explore_leg("investigate", det(7, "box or bag"))   # retried leg: once
        self.a.on_explore_leg("investigate", det(8))
        self.a.on_explore_leg("home")
        self.assertEqual(self.texts(), [
            "Okay, I'm going exploring.",
            "I'm going to take a closer look at that box.",
            "Something changed over there. Let me take a closer look.",
            "Heading home now."])

    def test_found_something_new_only_while_exploring(self):
        self.a.on_detection(det(1, "other", what="Blue laundry basket"))
        self.assertEqual(self.sent, [])
        self.a.on_explore_start()
        self.sent.clear()
        self.a.on_detection(det(1, "other", what="Blue laundry basket."))
        self.a.on_detection(det(2, "other", what="chair", confidence=0.4))   # unsure
        self.a.on_detection(det(3, "door"))
        self.a.on_detection(det(4, "other", what="umbrella", verdict="candidate"))
        self.assertEqual(self.texts(), ["I found something new: a blue laundry basket."])
        self.a.on_detection(det(5, "other", what="umbrella"))
        self.assertEqual(self.texts()[-1], "I found something new: an umbrella.")
        self.assertEqual(self.sent[-1]["key"], "explore_find")

    def test_investigation_results(self):
        self.a.on_explore_start()
        self.sent.clear()
        self.a.on_detection(det(1, "small stuffed animal", investigated="confirmed"))
        self.a.on_detection(det(1, "small stuffed animal", investigated="confirmed"))
        self.a.on_detection(det(2, investigated="gone"))
        self.a.on_detection(det(3, investigated="unseen"))
        self.assertEqual(self.texts(), [
            "Yes, there really is a stuffed toy there.",   # result
            "I found a stuffed toy.",                     # first sighting of #1
            "Whatever it was, it's gone now."])

    def test_summary_phrase(self):
        finds = [det(1, "box or bag"), det(2, "box or bag"), det(3, "small stuffed animal"),
                 det(4, "other", what="laundry basket")]
        self.assertEqual(explore_summary_phrase("done", 0.724, finds),
                         "Exploration finished. I looked over 72 percent of the house and found "
                         "2 boxes, a stuffed toy and a laundry basket.")
        self.assertEqual(explore_summary_phrase("done", 0.5, []),
                         "Exploration finished. I looked over 50 percent of the house, and "
                         "found nothing new.")
        self.assertEqual(explore_summary_phrase("stopped", 0.5, finds), "Exploration stopped.")
        self.a.on_explore_start()
        self.a.on_task_finished("done", 0.6, finds[:1])
        self.assertFalse(self.a.exploring)
        self.assertIn("found a box", self.texts()[-1])

    def test_is_find(self):
        self.assertTrue(is_find(det(1, "pet")))
        self.assertFalse(is_find(det(1, "pet", investigated="dismissed")))
        self.assertFalse(is_find(det(1, "nothing there")))
        self.assertFalse(is_find(det(1, "other", kind="missing")))

    def test_stuck_phrase_unknown_reason(self):
        self.assertEqual(stuck_phrase("weird"), "I'm stuck.")


if __name__ == "__main__":
    unittest.main()

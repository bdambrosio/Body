"""ChangeReporter / ChangeLabeler (explore plan B2/B3)."""
import json
import math
import os
import tempfile
import unittest

from desktop.perception.change_detector import (
    KIND_MISSING, KIND_MOVING, KIND_NEW, STATUS_APPEARED, STATUS_GONE, ChangeEvent,
)
from desktop.perception.change_report import (
    CHANGE_CLASSES, ChangeLabeler, ChangeReporter, classify_crop, crop_box_px, crop_jpeg,
    expected_bearing,
)
from desktop.perception.frames import PosedFrame
from desktop.perception.vlm_worker import VlmResult
from desktop.tasks.locate import CameraModel

CAM = CameraModel(558.3, 558.3, 320.0, 200.0, 640, 400)


def _jpeg(w=640, h=400):
    import io
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (w, h), (120, 100, 80)).save(out, format="JPEG")
    return out.getvalue()


JPEG = _jpeg()


class SyncWorker:
    def submit(self, job):
        v = job.query(job.frame.jpeg)
        job.on_result(VlmResult(kind=job.kind, request_id=job.frame.request_id,
                                frame_ts=job.frame.recv_ts, pose=job.frame.pose,
                                verdict=v, latency_s=0.1, frame=job.frame))
        return True


def grab_at(pose):
    def grab(on_frame, on_fail=None, **kw):
        on_frame(PosedFrame("r1", JPEG, 640, 400, 1.0, 1.0, pose))
        return "r1"
    return grab


def _read(d):
    with open(os.path.join(d, "events.jsonl")) as f:
        return [json.loads(l) for l in f]


class TestReporter(unittest.TestCase):
    def test_writes_jsonl_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            rep = ChangeReporter(d, clock=lambda: 123.0)
            rep.report(ChangeEvent(KIND_NEW, STATUS_APPEARED, 7, 10.0, 1.0, 2.0, n_cells=4),
                       robot_pose=(0.0, 0.0, 0.0))
            rep.label(7, "backpack", range_m=1.2)
            lines = _read(d)
            self.assertEqual(lines[0]["kind"], "new")
            self.assertEqual(lines[0]["robot"], [0.0, 0.0, 0.0])
            self.assertEqual(lines[1], {"type": "label", "wall_ts": 123.0, "change_id": 7,
                                        "label": "backpack", "range_m": 1.2})
            self.assertEqual(rep.recent()[0]["label"], "backpack")
            self.assertEqual(rep.counts, {"new/appeared": 1})

    def test_no_dir_is_memory_only(self):
        rep = ChangeReporter(None)
        rep.report(ChangeEvent(KIND_MISSING, STATUS_GONE, 1, 0.0, 0.0, 0.0))
        self.assertEqual(len(rep.recent()), 1)
        self.assertIsNone(rep.save_jpeg("x.jpg", b"x"))


class TestLabeler(unittest.TestCase):
    def _lab(self, d, answer, pose=(0.0, 0.0, 0.0), seen=None):
        rep = ChangeReporter(d)

        def classify(crop, *, kind, range_m):
            if seen is not None:
                seen.append((crop, kind, range_m))
            return answer
        lab = ChangeLabeler(grab_at(pose), SyncWorker(), CAM, rep,
                            detect_fn=classify, clock=lambda: 100.0)
        return rep, lab

    def test_bearing(self):
        b, r = expected_bearing((1.0, 1.0, math.pi / 2), 1.0, 3.0)
        self.assertAlmostEqual(b, 0.0)
        self.assertAlmostEqual(r, 2.0)

    def test_classifies_a_crop_at_the_change_bearing(self):
        seen = []
        with tempfile.TemporaryDirectory() as d:
            # Change 2 m ahead, 0.3 m left → bearing +8.5° → u ≈ 320 - 83 = 237.
            rep, lab = self._lab(d, {"label": "person", "confidence": 0.8, "what": "legs"},
                                 seen=seen)
            ev = ChangeEvent(KIND_MOVING, STATUS_APPEARED, 3, 1.0, 2.0, 0.3)
            self.assertTrue(lab.maybe_label(ev, (0.0, 0.0, 0.0)))
            self.assertEqual(rep.recent(), [])      # label record only, not a change record
            last = _read(d)[-1]
            self.assertEqual((last["label"], last["confidence"], last["what"]),
                             ("person", 0.8, "legs"))
            u0, u1 = last["crop_px"]
            self.assertLess(u0, 237)
            self.assertGreater(u1, 237)
            self.assertLess(u1 - u0, 640)            # a crop, not the frame
            self.assertTrue(os.path.exists(last["jpeg_path"]))
            crop, kind, rng = seen[0]
            self.assertEqual(kind, KIND_MOVING)
            self.assertAlmostEqual(rng, math.hypot(2.0, 0.3), places=3)

    def test_no_answer_is_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            rep, lab = self._lab(d, None)
            lab.maybe_label(ChangeEvent(KIND_NEW, STATUS_APPEARED, 4, 1.0, 2.0, 0.0),
                            (0.0, 0.0, 0.0))
            self.assertEqual(_read(d)[-1]["label"], "unknown")

    def test_gates(self):
        rep, lab = self._lab(None, None)
        pose = (0.0, 0.0, 0.0)
        behind = ChangeEvent(KIND_NEW, STATUS_APPEARED, 1, 0.0, -2.0, 0.0)
        far = ChangeEvent(KIND_NEW, STATUS_APPEARED, 2, 0.0, 5.0, 0.0)
        gone = ChangeEvent(KIND_MOVING, STATUS_GONE, 3, 0.0, 2.0, 0.0)
        missing = ChangeEvent(KIND_MISSING, STATUS_APPEARED, 4, 0.0, 2.0, 0.0)
        for ev in (behind, far, gone, missing):
            self.assertFalse(lab.maybe_label(ev, pose), ev)
        self.assertFalse(lab.maybe_label(ChangeEvent(KIND_NEW, STATUS_APPEARED, 5, 0, 2, 0), None))

    def test_once_per_change_and_rate_limited(self):
        rep, lab = self._lab(None, None)
        ev = ChangeEvent(KIND_NEW, STATUS_APPEARED, 9, 0.0, 2.0, 0.0)
        self.assertTrue(lab.maybe_label(ev, (0.0, 0.0, 0.0)))
        self.assertFalse(lab.maybe_label(ev, (0.0, 0.0, 0.0)))     # done
        ev2 = ChangeEvent(KIND_NEW, STATUS_APPEARED, 10, 0.0, 2.0, 0.0)
        self.assertFalse(lab.maybe_label(ev2, (0.0, 0.0, 0.0)))    # < 1 s later



class TestLabelWhilePresent(unittest.TestCase):
    """A change reported out of view is labelled later, when it comes into
    view while still present (11-02 black cat)."""

    def _lab(self, clock):
        rep = ChangeReporter(None)
        grabs = []

        def grab(on_frame, on_fail=None, **kw):
            grabs.append(1)
            on_frame(PosedFrame("r", JPEG, 640, 400, 1.0, 1.0, (0.0, 0.0, 0.0)))
            return "r"
        lab = ChangeLabeler(grab, SyncWorker(), CAM, rep,
                            detect_fn=lambda crop, **kw: {"label": "pet", "confidence": 0.9,
                                                          "what": "cat"},
                            clock=lambda: clock[0])
        return rep, lab, grabs

    def test_behind_then_in_view(self):
        clock = [100.0]
        rep, lab, grabs = self._lab(clock)
        ev = ChangeEvent(KIND_NEW, STATUS_APPEARED, 54, 0.0, 2.0, 0.0)
        lab.track(ev)
        self.assertFalse(lab.poll((4.0, 0.0, 0.0)))          # change is behind the robot
        self.assertEqual(lab.pending(), [54])
        clock[0] += 5
        self.assertTrue(lab.poll((0.0, 0.0, 0.0)))           # now dead ahead, 2 m
        self.assertEqual(lab.pending(), [])
        self.assertEqual(len(grabs), 1)
        clock[0] += 5
        self.assertFalse(lab.poll((0.0, 0.0, 0.0)))          # labelled once

    def test_gone_before_in_view_is_dropped(self):
        clock = [100.0]
        rep, lab, grabs = self._lab(clock)
        lab.track(ChangeEvent(KIND_NEW, STATUS_APPEARED, 7, 0.0, 2.0, 0.0))
        lab.track(ChangeEvent(KIND_NEW, STATUS_GONE, 7, 1.0, 2.0, 0.0))
        self.assertFalse(lab.poll((0.0, 0.0, 0.0)))
        self.assertEqual(grabs, [])

    def test_missing_not_tracked(self):
        clock = [100.0]
        rep, lab, grabs = self._lab(clock)
        lab.track(ChangeEvent(KIND_MISSING, STATUS_APPEARED, 8, 0.0, 2.0, 0.0))
        self.assertEqual(lab.pending(), [])


class TestCropAndClassify(unittest.TestCase):
    def test_crop_box_centres_on_bearing_and_widens_up_close(self):
        u0, u1 = crop_box_px(CAM, 0.0, 3.0)
        self.assertAlmostEqual((u0 + u1) / 2, 320, delta=2)
        near = crop_box_px(CAM, 0.0, 0.8)
        self.assertGreater(near[1] - near[0], u1 - u0)     # ±0.5 m is wider at 0.8 m
        left = crop_box_px(CAM, math.radians(15), 2.0)
        self.assertLess((left[0] + left[1]) / 2, 320)      # + bearing = left = smaller u
        self.assertIsNone(crop_box_px(CAM, math.radians(60), 2.0))   # out of view

    def test_crop_jpeg_is_a_full_height_strip(self):
        import io
        from PIL import Image
        with Image.open(io.BytesIO(crop_jpeg(JPEG, 100, 260))) as im:
            self.assertEqual(im.size, (160, 400))

    def test_classify_uses_the_closed_list(self):
        calls = []

        def fake_chat_json(messages, schema, **kw):
            calls.append((messages, schema))
            return {"label": "small stuffed animal", "confidence": 0.7, "what": "plush dog"}
        ans = classify_crop(JPEG, kind=KIND_NEW, range_m=1.2, chat_json=fake_chat_json)
        self.assertEqual(ans["label"], "small stuffed animal")
        msgs, schema = calls[0]
        self.assertEqual(schema["properties"]["label"]["enum"], list(CHANGE_CLASSES))
        self.assertIn("person", msgs[0]["content"])
        self.assertIsNone(classify_crop(JPEG, kind=KIND_NEW, range_m=1.0,
                                        chat_json=lambda *a, **k: {"label": "latch"}))


if __name__ == "__main__":
    unittest.main()

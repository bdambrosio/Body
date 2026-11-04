from __future__ import annotations

import unittest

from desktop.nav.detections import DetectionModel


def ch(cid, kind="new", status="appeared", verdict=None, ts=100.0, **kw):
    r = {"type": "change", "change_id": cid, "kind": kind, "status": status,
         "x": 1.0, "y": 2.0, "extent_m": 0.3, "wall_ts": ts, **kw}
    if verdict:
        r["verdict"] = verdict
    return r


class TestDetectionModel(unittest.TestCase):
    def test_folds_change_label_and_investigation(self):
        m = DetectionModel()
        m.ingest(ch(73, verdict="novel", store_entry=9))
        m.ingest({"type": "label", "change_id": 73, "label": "box or bag",
                  "confidence": 0.98, "jpeg_path": "/x/change_00073.jpg", "wall_ts": 101})
        m.ingest({"type": "investigated", "change_id": 73, "outcome": "confirmed",
                  "wall_ts": 102})
        d = m.get(73)
        self.assertEqual((d.label, d.investigated, d.store_entry), ("box or bag", "confirmed", 9))
        self.assertEqual(d.jpeg_path, "/x/change_00073.jpg")
        self.assertTrue(d.notable and d.alert and d.present)
        m.ingest(ch(73, status="gone", ts=110))
        self.assertFalse(m.get(73).present)
        self.assertEqual(m.get(73).first_ts, 100.0)

    def test_default_view_hides_routine_places_and_missing(self):
        m = DetectionModel()
        m.ingest(ch(1, verdict="candidate"))
        m.ingest(ch(2, verdict="expected"))
        m.ingest(ch(3, kind="missing", verdict="novel"))
        m.ingest(ch(4, verdict="novel", ts=104))
        m.ingest(ch(5, kind="moving", ts=105))                 # no store
        m.ingest(ch(6, verdict="candidate", ts=106))
        m.ingest({"type": "label", "change_id": 6, "label": "pet", "wall_ts": 107})
        self.assertEqual([d.change_id for d in m.rows()], [6, 5, 4])
        self.assertEqual(len(m.rows(show_all=True)), 6)
        self.assertTrue(m.get(6).alert)
        self.assertFalse(m.get(4).alert)

    def test_since_filters_by_last_activity(self):
        m = DetectionModel()
        m.ingest(ch(1, ts=50))
        m.ingest(ch(2, ts=150))
        m.ingest(ch(1, status="gone", ts=160))
        self.assertEqual({d.change_id for d in m.rows(since=100)}, {1, 2})
        self.assertEqual([d.change_id for d in m.rows(since=155)], [1])

    def test_unknown_ids_and_stubs_are_ignored(self):
        m = DetectionModel()
        self.assertIsNone(m.ingest({"type": "label", "change_id": 9, "label": "pet"}))
        self.assertIsNone(m.ingest({"type": "change", "change_id": 9, "label": "pet",
                                    "verdict": "novel"}))
        self.assertEqual(m.rows(show_all=True), [])

    def test_by_store_entry_prefers_a_photo(self):
        m = DetectionModel()
        m.ingest(ch(1, store_entry=4, ts=1))
        m.ingest({"type": "label", "change_id": 1, "label": "door", "jpeg_path": "/a.jpg",
                  "wall_ts": 2})
        m.ingest(ch(2, store_entry=4, ts=3))
        self.assertEqual(m.by_store_entry()[4].change_id, 1)


if __name__ == "__main__":
    unittest.main()

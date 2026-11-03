"""Explore Phase C change store: learn recurring places, background only with
approval, label/size mismatches still report, drift guards."""
import os
import tempfile
import unittest

from desktop.perception.change_detector import (
    KIND_MISSING, KIND_MOVING, KIND_NEW, STATUS_APPEARED, STATUS_GONE, ChangeEvent,
)
from desktop.perception.change_report import ChangeReporter
from desktop.perception.change_store import (
    CANDIDATE, EXPECTED, FLAGGED, NOVEL, ST_BACKGROUND, ST_PROPOSED, ST_REJECTED,
    ChangeStore, ChangeStoreConfig,
)

_cid = [0]


def ev(kind=KIND_NEW, x=1.0, y=2.0, extent=0.2, status=STATUS_APPEARED, cid=None):
    if cid is None:
        _cid[0] += 1
        cid = _cid[0]
    return ChangeEvent(kind, status, cid, 0.0, x, y, extent_m=extent)


def patrol(st, name, events, path=((1.0, 0.0),)):
    st.begin_session(name)
    for x, y in path:
        st.note_pose(x, y)
    out = [st.classify(e) for e in events]
    return out, st.end_session()


class TestChangeStore(unittest.TestCase):
    def test_recurring_place_is_proposed_never_silently_background(self):
        st = ChangeStore(None)
        (v1, _), = patrol(st, "s1", [ev()])[0]
        self.assertEqual(v1, NOVEL)
        (v2, _), = patrol(st, "s2", [ev(x=1.1)])[0]
        self.assertEqual(v2, CANDIDATE)
        (v3, eid), = patrol(st, "s3", [ev(y=2.1)])[0]
        self.assertEqual(st.entries[eid].state, ST_PROPOSED)
        (v4, _), = patrol(st, "s4", [ev()])[0]
        self.assertEqual(v4, CANDIDATE)          # proposed is still reported
        st.set_state(eid, ST_BACKGROUND)
        (v5, _), = patrol(st, "s5", [ev()])[0]
        self.assertEqual(v5, EXPECTED)

    def test_box_in_a_doors_place_is_still_reported(self):
        # 11-03: the box sat where a door's edge is reported every patrol.
        st = ChangeStore(None)
        for i in range(3):
            st.begin_session(f"s{i}")
            e = ev(extent=0.15)
            v, eid = st.classify(e)
            st.on_label(e.change_id, "door")
            st.end_session()
        st.set_state(eid, ST_BACKGROUND)
        st.begin_session("box")
        box = ev(extent=0.15)
        self.assertEqual(st.classify(box)[0], EXPECTED)        # same place and size …
        self.assertEqual(st.on_label(box.change_id, "box or bag"), NOVEL)   # … but not a door
        door = ev(extent=0.15)
        st.classify(door)
        self.assertIsNone(st.on_label(door.change_id, "door"))
        self.assertIsNone(st.on_label(door.change_id, "unknown"))

    def test_different_size_or_kind_does_not_match(self):
        st = ChangeStore(None)
        st.begin_session("a")
        _, eid = st.classify(ev(extent=0.15))
        self.assertNotEqual(st.classify(ev(extent=0.9))[1], eid)          # much bigger
        self.assertNotEqual(st.classify(ev(kind=KIND_MISSING))[1], eid)
        self.assertNotEqual(st.classify(ev(x=1.6))[1], eid)               # 0.6 m away

    def test_moving_is_always_novel_and_gone_inherits(self):
        st = ChangeStore(None)
        st.begin_session("a")
        self.assertEqual(st.classify(ev(kind=KIND_MOVING)), (NOVEL, None))
        a = ev()
        st.classify(a)
        gone = ev(status=STATUS_GONE, cid=a.change_id)
        self.assertEqual(st.classify(gone), st.classify(gone))
        self.assertEqual(len(st.entries), 1)

    def test_rejected_place_is_flagged(self):
        st = ChangeStore(None)
        st.begin_session("a")
        _, eid = st.classify(ev())
        st.set_state(eid, ST_REJECTED)
        self.assertEqual(st.classify(ev())[0], FLAGGED)

    def test_entry_expires_only_when_looked_at_and_not_seen(self):
        st = ChangeStore(None, ChangeStoreConfig(expire_after_misses=2))
        _, _ = patrol(st, "s0", [ev()])
        eid = next(iter(st.entries))
        patrol(st, "away1", [], path=((9.0, 9.0),))          # never looked: no miss
        patrol(st, "away2", [], path=((9.0, 9.0),))
        self.assertIn(eid, st.entries)
        patrol(st, "near1", [], path=((1.0, 1.0),))          # looked, not seen
        self.assertEqual(st.entries[eid].misses, 1)
        patrol(st, "near2", [], path=((1.0, 1.0),))
        self.assertNotIn(eid, st.entries)

    def test_persists_and_reporter_hides_expected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "change_store.json")
            st = ChangeStore(path)
            for i in range(3):
                patrol(st, f"s{i}", [ev()])
            eid = next(iter(st.entries))
            st.set_state(eid, ST_BACKGROUND)
            st2 = ChangeStore(path)                           # reload
            self.assertEqual(st2.entries[eid].state, ST_BACKGROUND)
            st2.begin_session("live")
            rep = ChangeReporter(d, store=st2)
            rec = rep.report(ev())
            self.assertEqual(rec["verdict"], EXPECTED)
            self.assertEqual(rep.recent(), [])                # not announced
            self.assertEqual(rep.report(ev(x=-3.0))["verdict"], NOVEL)
            self.assertEqual(len(rep.recent()), 1)


if __name__ == "__main__":
    unittest.main()

"""Dangerous objects (Phase 47 step 1, labeler/weapons.py): held at a wrist on 2 of the last 3 looks raises once per
person; not in a cooking / food preparation area; again after a while without one; the camera's look (counts and
objects from one pass) and its check."""
from datetime import datetime

import numpy as np

from labeler.pipeline import TrackState
from labeler.weapons import WeaponWatch, at_wrist
from labeler.zones import ZoneCounter
from test_pipeline_logic import make_pipeline

PERSON = (100.0, 100.0, 200.0, 400.0)   # 300 px tall


def kp(wrist=(190.0, 250.0)):
    k = np.zeros((17, 3), np.float32)
    k[10] = [*wrist, 0.9]   # right wrist
    return k


KNIFE_AT_HAND = (185.0, 240.0, 230.0, 260.0)
KNIFE_ON_TABLE = (400.0, 300.0, 450.0, 320.0)


def test_at_wrist():
    assert at_wrist(KNIFE_AT_HAND, kp(), PERSON, 0.15)
    assert at_wrist((230.0, 240.0, 260.0, 260.0), kp(), PERSON, 0.15)       # 40 px off: within 15% of 300 px
    assert not at_wrist((240.0, 240.0, 270.0, 260.0), kp(), PERSON, 0.15)   # 50 px off: too far
    assert not at_wrist(KNIFE_ON_TABLE, kp(), PERSON, 0.15)
    assert not at_wrist(KNIFE_AT_HAND, None, PERSON, 0.15)
    weak = kp()
    weak[10, 2] = 0.1
    assert not at_wrist(KNIFE_AT_HAND, weak, PERSON, 0.15)


def looks(w, pattern, role="", t0=0.0, obj=("knife", 0.8, KNIFE_AT_HAND)):
    out = []
    for i, c in enumerate(pattern):
        out += [(t0 + i, *x) for x in w.update(t0 + i, [obj] if c == "x" else [], {7: (PERSON, kp(), role)})]
    return out


def test_held_twice_in_three_looks_raises_once():
    w = WeaponWatch({})
    assert looks(w, "x..x..") == []
    assert looks(w, "x.x", t0=10) == [(12.0, 7, "knife")]
    assert looks(w, "xxx", t0=13) == []                                   # once
    assert looks(w, "." * 31 + "xx", t0=16) == [(48.0, 7, "knife")]      # after 30 s without one: again
    assert looks(WeaponWatch({}), "xxx", role="cooking") == []            # a kitchen knife in the kitchen
    assert looks(WeaponWatch({}), "xxx", obj=("knife", 0.8, KNIFE_ON_TABLE)) == []
    assert looks(WeaponWatch({}), "xx", obj=("handgun", 0.6, KNIFE_AT_HAND)) == [(1.0, 7, "handgun")]


class Boxes:
    def __init__(self, dets):
        self.cls = np.array([d[0] for d in dets], float)
        self.conf = np.array([d[1] for d in dets], float)
        self.xyxy = np.array([d[2] for d in dets], float).reshape(-1, 4)


class Model:
    def __init__(self, dets):
        self.dets, self.asked = dets, []

    def predict(self, frame, **kw):
        self.asked.append(kw.get("classes"))
        return [type("R", (), {"boxes": Boxes(self.dets)})()]


class Queue:
    def __init__(self):
        self.raised = []

    def suspect(self, camera, category, reason, **kw):
        self.raised.append((category, reason, kw.get("people")))
        return "queued"


def test_the_camera_looks_and_checks(db, camera):
    p = make_pipeline(db)
    p.mode, p.det_cfg = "security", {"device": "cpu"}
    hall = {"id": 4, "kind": "area", "name": "hall", "points": [(0, 0), (1, 0), (1, 1), (0, 1)]}
    p.zone_counter = ZoneCounter([hall])
    p.counter_model = Model([(0, 0.9, PERSON), (43, 0.8, KNIFE_AT_HAND), (43, 0.2, KNIFE_ON_TABLE)])
    frame = np.zeros((480, 640, 3), np.uint8)
    counts, objects = p._look_objects(frame, [hall], True)
    assert counts == {4: 1} and objects == [("knife", 0.8, KNIFE_AT_HAND)]       # the weak one left out
    assert p.counter_model.asked[-1] == [0, 34, 43, 76]
    assert p._look_objects(frame, [hall], False)[1] == [] and p.counter_model.asked[-1] == [0]
    p.weapon_watch, p.incidents, p.memory = WeaponWatch({}), Queue(), object()
    st = TrackState(7, datetime(2026, 10, 21, 12), 0.0, db_id=db.create_track(camera, 7, datetime(2026, 10, 21, 12)))
    st.box, st.kpts = PERSON, kp()
    p.tracks[7] = st
    for t in (1.0, 2.0):
        st.last_seen = t
        p._check_weapons(frame, t, objects)
    assert p.incidents.raised == [("weapon", "knife held", [st.db_id])]

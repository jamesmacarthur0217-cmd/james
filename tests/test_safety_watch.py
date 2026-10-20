"""Quick checks for people (Phase 45 step 1, labeler/safety.py) and falls checked by the vision model with a
fail-safe: lying still, slumped, not moved for long (the slow round), nothing on a bed, moving again re-arms; a fall
on a safety camera alerts on "yes", "unsure" or no answer, never on a clear "no"."""
from datetime import datetime

import numpy as np

from labeler.falls import FallWatch
from labeler.geometry import L_EYE, L_HIP, L_SHOULDER, NOSE, R_EYE, R_HIP, R_SHOULDER
from labeler.pipeline import TrackState
from labeler.safety import DISTRESS_QUESTION, SafetyWatch, head_low
from labeler.zones import role_at
from test_falls import CFG, lying, standing
from test_pipeline_logic import make_pipeline

FPS = 5


def sitting(x=300.0, top=200.0, h=200.0, head_down=False):
    """Someone sitting upright (shoulders above hips); head_down: the head below the shoulders."""
    k = np.zeros((17, 3), np.float32)
    k[[L_SHOULDER, R_SHOULDER], :] = [[x - 30, top + 0.3 * h, 0.9], [x + 30, top + 0.3 * h, 0.9]]
    k[[L_HIP, R_HIP], :] = [[x - 20, top + 0.75 * h, 0.9], [x + 20, top + 0.75 * h, 0.9]]
    head_y = top + 0.4 * h if head_down else top + 0.1 * h
    k[[NOSE, L_EYE, R_EYE], :] = [[x, head_y, 0.9], [x - 8, head_y - 4, 0.9], [x + 8, head_y - 4, 0.9]]
    return (x - 50, top, x + 50, top + h), k


def run(watch, frames, seconds, role="", key=1, t0=0.0):
    """Feed one pose for `seconds` at FPS; returns every sign raised."""
    out = []
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        box, k = frames(i) if callable(frames) else frames
        watch.update(key, t, box, k, role)
        out += [(round(t, 1), *s[1:]) for s in watch.tick(t)]
    return out


def test_head_low():
    assert head_low(sitting(head_down=True)[1]) is True and head_low(sitting()[1]) is False
    assert head_low(None) is None and head_low(np.zeros((17, 3), np.float32)) is None


def test_lying_still_once_then_again_after_moving():
    w = SafetyWatch({"down_s": 20, "slumped_s": 30, "still_s": 300})
    got = run(w, lying(), 60)
    assert len(got) == 1 and got[0][0] == 20.0 and got[0][1:3] == ("person_down", "lying still 20 s")
    moved = run(w, lambda i: lying(x=300 + 40 * (i % 2)), 3, t0=60)   # moving about: re-armed, nothing raised
    assert moved == []
    assert len(run(w, lying(), 25, t0=63)) == 1


def test_nothing_on_a_bed_and_nothing_short():
    assert run(SafetyWatch({}), lying(), 60, role="bed") == []
    assert run(SafetyWatch({}), lying(), 15) == []


def test_slumped_and_the_slow_round():
    w = SafetyWatch({"down_s": 20, "slumped_s": 30, "still_s": 300})
    got = run(w, sitting(head_down=True), 40)
    assert [g[0] for g in got] == [30.0] and "slumped" in got[0][2]
    upright = SafetyWatch({"down_s": 20, "slumped_s": 30, "still_s": 120})
    got = run(upright, sitting(), 130)
    assert len(got) == 1 and got[0][0] == 120.0 and got[0][2] == "not moved for 2 min"
    assert got[0][3] == "round" and got[0][4] == DISTRESS_QUESTION


def feed_many(watch, people, seconds, t0=0.0):
    """people: {key: (pose or callable(i) -> pose, role, age)}; returns every sign raised."""
    out = []
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        for key, (pose, role, age) in people.items():
            box, k = pose(i) if callable(pose) else pose
            watch.update(key, t, box, k, role, age)
        out += [(round(t, 1), s[0], s[2]) for s in watch.tick(t)]
    return out


def test_a_child_alone():
    w = SafetyWatch({"child_alone_s": 120})
    adult, child = standing(x=300, top=100, h=300), standing(x=340, top=250, h=150)
    assert feed_many(w, {1: (adult, "", "adult"), 2: (child, "", "child")}, 130) == []    # with an adult near
    got = feed_many(w, {2: (child, "", "child")}, 125, t0=130)                            # the adult left
    assert [g[1:] for g in got] == [(2, "child with no adult near for 120 s")]
    far = standing(x=2000, top=100, h=300)                                                # an adult far away
    assert len(feed_many(SafetyWatch({"child_alone_s": 120}), {1: (far, "", "adult"), 2: (child, "", "child")}, 125)) == 1
    assert feed_many(SafetyWatch({"child_alone_s": 0}), {2: (child, "", "child")}, 125) == []   # off


def test_a_small_body_next_to_an_adult_in_the_kitchen():
    w = SafetyWatch({})
    adult, child = standing(x=300, top=100, h=300), standing(x=420, top=250, h=150)   # same feet line, half as tall
    got = feed_many(w, {1: (adult, "", None), 2: (child, "cooking", None)}, 10)
    assert [g[1:] for g in got] == [(2, "child in the cooking area")] and got[0][0] <= 4.0
    labeled = SafetyWatch({})   # the label wins: a short adult is no child
    assert feed_many(labeled, {1: (adult, "", None), 2: (child, "cooking", "adult")}, 10) == []
    deeper = SafetyWatch({})    # smaller because further away (feet higher in the picture): no child
    assert feed_many(deeper, {1: (adult, "", None), 2: (standing(x=420, top=150, h=150), "cooking", None)}, 10) == []


def test_a_child_climbing():
    w = SafetyWatch({"child_alone_s": 0})
    base = standing(x=300, top=250, h=150)
    up = standing(x=300, top=190, h=150)         # the feet 60 px (40% of the height) higher, the same size
    away = standing(x=300, top=200, h=110)       # feet higher but smaller: walking away
    assert feed_many(w, {2: (base, "", "child")}, 10) == []
    got = feed_many(w, {2: (up, "", "child")}, 4, t0=10)
    assert [g[1:] for g in got] == [(2, "child climbing (feet raised)")]
    w2 = SafetyWatch({"child_alone_s": 0})
    feed_many(w2, {2: (base, "", "child")}, 10)
    assert feed_many(w2, {2: (away, "", "child")}, 4, t0=10) == []


def test_role_at():
    zones = [{"kind": "area", "role": "", "points": [(0, 0), (1, 0), (1, 1), (0, 1)]},
             {"kind": "area", "role": "bed", "points": [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]}]
    assert role_at(zones, (0.2, 0.2)) == "bed" and role_at(zones, (0.8, 0.8)) == ""


class Queue:
    def __init__(self, status="queued"):
        self.status, self.raised = status, []

    def suspect(self, camera, category, reason, **kw):
        self.raised.append((category, reason, kw))
        return self.status


def _fall_pipeline(db, camera, queue):
    p = make_pipeline(db)
    p.falls = FallWatch(CFG)
    p._fall_db = {}
    p.memory, p.incidents = object(), queue
    fired = []
    p._check_alerts = lambda db_id, fallen=False, **kw: fired.append((db_id, fallen))
    st = TrackState(7, datetime.now(), 0.0, db_id=db.create_track(camera, 7, datetime.now()))
    st.box = (100, 300, 400, 380)
    p.tracks[7] = st
    return p, st, fired


def test_a_fall_is_checked_and_only_a_clear_no_stops_it(db, camera):
    q = Queue()
    p, st, fired = _fall_pipeline(db, camera, q)
    p._on_fall(7, (480, 640, 3))
    category, reason, kw = q.raised[0]
    assert category == "fall" and fired == [] and kw["people"] == [st.db_id]
    assert kw["crop"] == (100 / 640, 300 / 480, 400 / 640, 380 / 480) and "lying or sitting on the floor" in kw["question"]
    kw["on_done"]({"answer": "no", "what": "a person sitting"})
    assert fired == []                                     # a clear no: no alert
    for answer in ({"answer": "yes"}, {"answer": "unsure"}, None):   # yes, unsure, no answer: the alert
        kw["on_done"](answer)
    assert fired == [(st.db_id, True)] * 3


def test_a_fall_is_not_raised_again_as_lying_still(db, camera):
    p, st, fired = _fall_pipeline(db, camera, Queue())
    p.safety_watch = SafetyWatch({"down_s": 20})
    run(p.safety_watch, lying(), 10, key=7)
    p._on_fall(7, (480, 640, 3))
    assert run(p.safety_watch, lying(), 30, key=7, t0=10) == []


def test_a_fall_alerts_at_once_without_the_check(db, camera):
    for status in ("cooldown", "no pictures", "failed"):
        p, st, fired = _fall_pipeline(db, camera, Queue(status))
        p._on_fall(7, (480, 640, 3))
        assert fired == [(st.db_id, True)], status
    p, st, fired = _fall_pipeline(db, camera, None)          # a kiosk camera: no confirm queue
    p._on_fall(7)
    assert fired == [(st.db_id, True)]

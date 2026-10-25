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


HALL = {"id": 5, "name": "hall", "max_people": 3, "points": [(0, 0), (0.5, 0), (0.5, 1), (0, 1)]}


def crowd_run(w, n, seconds, t0=0.0, x_of=lambda k, i: 100 + 60 * k, inside=True):
    """n people standing (or moving: x_of) for `seconds`; returns the crowd signs."""
    out = []
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        for k in range(n):
            x = x_of(k, i)
            box, kp = standing(x=x, top=100, h=300)
            w.update(k, t, box, kp, anchor=((0.25 if inside else 0.75), 0.5))
        out += [(round(t, 1), (a or {}).get("name"), r) for a, r in w.crowd(t, [HALL])]
    return out


def test_too_many_people_in_an_area():
    w = SafetyWatch({"crowd_s": 30})
    assert crowd_run(w, 3, 40) == []                                      # at the limit: fine
    got = crowd_run(w, 5, 40, t0=40)
    assert len(got) == 1 and got[0][1] == "hall" and got[0][2].startswith("5 people in hall (at most 3) for 30 s")
    assert crowd_run(w, 5, 40, t0=80) == []                               # once
    crowd_run(w, 2, 33, t0=120)                                            # calm again 30 s: re-armed
    assert len(crowd_run(w, 5, 31, t0=153)) == 1
    assert crowd_run(SafetyWatch({"crowd_s": 30}), 5, 40, inside=False) == []   # outside the area


def test_the_cameras_own_count_wins():
    w = SafetyWatch({"crowd_s": 5})
    got = [r for i in range(40) for r in w.crowd(i / FPS, [HALL], {5: 9})]   # nobody tracked, 9 counted
    assert len(got) == 1 and got[0][1].startswith("9 people in hall (at most 3)")
    assert [r for i in range(40) for r in SafetyWatch({"crowd_s": 5}).crowd(i / FPS, [HALL], {5: 3})] == []


def test_people_running_the_same_way():
    fast = lambda k, i: 100 + 60 * k + i * 120   # noqa: E731  (2 body heights a second at 5 frames a second)
    got = crowd_run(SafetyWatch({}), 8, 4, x_of=fast, inside=False)
    assert [g[1:] for g in got] == [(None, "8 people moving fast the same way")] and got[0][0] >= 2.0
    walking = lambda k, i: 100 + 60 * k + i * 30   # noqa: E731  (half a body height a second)
    assert crowd_run(SafetyWatch({}), 8, 4, x_of=walking, inside=False) == []
    both_ways = lambda k, i: 100 + 60 * k + (1 if k % 2 else -1) * i * 120   # noqa: E731
    assert crowd_run(SafetyWatch({}), 8, 4, x_of=both_ways, inside=False) == []
    assert crowd_run(SafetyWatch({}), 4, 4, x_of=fast, inside=False) == []    # too few people


def test_the_people_limit_on_areas(db, tmp_path):
    import copy

    import pytest
    from fastapi.testclient import TestClient

    from labeler.config import DEFAULTS
    from labeler.web import create_app
    z = db.add_zone("cam1", "hall", "area", [(0, 0), (1, 0), (1, 1)], max_people=40)
    assert z["max_people"] == 40
    assert db.update_zone(z["id"], max_people=0)["max_people"] is None and db.update_zone(z["id"], role="exit")["role"] == "exit"
    with pytest.raises(ValueError):
        db.add_zone("cam1", "door", "line", [(0, 0.5), (1, 0.5)], max_people=3)
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    client = TestClient(create_app(cfg, db, None), client=("127.0.0.1", 5000))
    r = client.patch(f"/api/zones/{z['id']}", json={"max_people": 12})
    assert r.status_code == 200 and r.json()["max_people"] == 12 and r.json()["role"] == "exit"   # the role kept
    assert client.patch(f"/api/zones/{z['id']}", json={"max_people": -1}).status_code == 400
    assert "at most 12 people" in client.get("/api/audit").json()["entries"][0]["detail"]


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


def test_each_kind_can_be_turned_off():
    """Phase 51 (the Settings page): a kind turned off raises nothing; the others still do."""
    assert run(SafetyWatch({"down_enabled": False}), lying(), 60) == []
    assert run(SafetyWatch({"down_enabled": False}), sitting(head_down=True), 40) == []
    child = standing(x=340, top=250, h=150)
    assert feed_many(SafetyWatch({"children_enabled": False, "child_alone_s": 120}), {2: (child, "", "child")}, 125) == []
    got = feed_many(SafetyWatch({"down_enabled": False, "child_alone_s": 120}), {2: (child, "", "child")}, 125)
    assert [g[1:] for g in got] == [(2, "child with no adult near for 120 s")]   # children still on
    assert crowd_run(SafetyWatch({"crowd_enabled": False, "crowd_s": 30}), 5, 40) == []
    fast = lambda k, i: 100 + 60 * k + i * 120   # noqa: E731
    assert crowd_run(SafetyWatch({"crowd_enabled": False}), 8, 4, x_of=fast, inside=False) == []   # running too

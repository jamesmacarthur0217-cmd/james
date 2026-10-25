"""Fights (Phase 48 step 1, labeler/safety.py SafetyWatch.fights): two people close together with a fast wrist touching or
reaching the other raise one check; hugs (slow), two people walking side by side (the wrists move with the body) and
someone far away swinging their arms do not; again only after a calm while."""
import numpy as np

from labeler.safety import SafetyWatch

A = (100, 100, 200, 400)   # 300 px tall
B = (180, 100, 280, 400)   # touching A
FAR = (900, 100, 1000, 400)


def body(box, wrist=None):
    """Body points: shoulders, hips, ankles where an upright person has them; the right wrist at `wrist` (pixels)."""
    x1, y1, x2, y2 = box
    k = np.zeros((17, 3), np.float32)
    cx, h = (x1 + x2) / 2, y2 - y1
    for i, (x, y) in {0: (cx, y1 + 0.08 * h), 5: (cx - 20, y1 + 0.2 * h), 6: (cx + 20, y1 + 0.2 * h),
                      11: (cx - 15, y1 + 0.55 * h), 12: (cx + 15, y1 + 0.55 * h), 15: (cx - 15, y2 - 5),
                      16: (cx + 15, y2 - 5), 9: (cx - 30, y1 + 0.5 * h)}.items():
        k[i] = (x, y, 0.9)
    k[10] = (*(wrist or (cx + 30, y1 + 0.5 * h)), 0.9)
    return k


def run(w, frames, people, t0=0.0, fps=10):
    """people(i) -> {key: (box, kpts)}; returns [(t, a, b, reason)]."""
    out = []
    for i in range(frames):
        t = t0 + i / fps
        for key, (box, kpts) in people(i).items():
            w.update(key, t, box, kpts)
        out += [(round(t, 1), *f) for f in w.fights(t)]
    return out


def punching(i):
    """A's right wrist swings between its side and B's face every frame (about 3 body heights a second)."""
    fist = (160, 250) if i % 2 == 0 else (240, 140)
    return {"a": (A, body(A, fist)), "b": (B, body(B))}


def test_a_punch_raises_one_check_for_the_two():
    got = run(SafetyWatch({}), 20, punching)
    assert len(got) == 1
    t, a, b, reason = got[0]
    assert {a, b} == {"a", "b"} and t <= 0.5 and "fast arm" in reason


def test_a_hug_and_people_walking_together_do_not():
    hug = lambda i: {"a": (A, body(A, (215, 160))), "b": (B, body(B, (165, 160)))}   # noqa: E731
    assert run(SafetyWatch({}), 30, hug) == []

    def walking(i):   # both 0.5 body heights a second to the right, arms with the body
        dx = 15 * i
        a, b = (A[0] + dx, A[1], A[2] + dx, A[3]), (B[0] + dx, B[1], B[2] + dx, B[3])
        return {"a": (a, body(a)), "b": (b, body(b))}
    assert run(SafetyWatch({}), 30, walking) == []


def test_far_away_arms_do_not():
    far = lambda i: {"a": (FAR, body(FAR, (910, 250) if i % 2 else (990, 140))), "b": (B, body(B))}   # noqa: E731
    assert run(SafetyWatch({}), 30, far) == []


def test_again_only_after_a_calm_while():
    w = SafetyWatch({"fight_rearm_s": 10})
    calm = lambda i: {"a": (A, body(A)), "b": (B, body(B))}   # noqa: E731
    assert len(run(w, 20, punching)) == 1
    assert run(w, 50, calm, t0=2.0) == [] and run(w, 20, punching, t0=7.0) == []   # calm only 5 s: still the same
    assert run(w, 110, calm, t0=9.0) == []
    assert len(run(w, 20, punching, t0=20.0)) == 1


def test_forgetting_a_person_forgets_their_pairs():
    w = SafetyWatch({})
    run(w, 20, punching)
    w.forget("a")
    assert w.pairs == {}


# --- collisions through the fall check ---

def test_touched_finds_who_touched_around_a_time():
    w = SafetyWatch({})
    for i in range(30):
        t = i / 10
        w.update("a", t, A, body(A))
        w.update("b", t, B if t < 1.0 else FAR, body(B))    # touching A until 1 s, then far away
        w.update("c", t, FAR, body(FAR))
    assert w.touched("a", 0.0, 3.0) == ["b"]
    assert w.touched("a", 1.5, 3.0) == [] and w.touched("x", 0.0, 3.0) == []


def test_a_fall_after_touching_someone_says_so(db, camera):
    from datetime import datetime

    from labeler.config import DEFAULTS
    from labeler.falls import FallWatch
    from labeler.pipeline import TrackState, VideoClock
    from test_pipeline_logic import make_pipeline

    p = make_pipeline(db)
    p.falls, p._fall_db = FallWatch({**DEFAULTS["falls"], "after_s": 10}), {}
    p.clock = VideoClock(datetime(2026, 10, 23, 12))
    p.safety_watch = SafetyWatch({})
    p.memory, p.incidents = object(), None   # no confirm queue: alerts at once
    notes = []
    p._check_alerts = lambda db_id, fallen=False, note="", **kw: notes.append((db_id, note))
    for key, box in ((7, A), (8, B)):
        p.tracks[key] = TrackState(key, datetime.now(), 0.0, db_id=db.create_track(camera, key, datetime.now()))
        p.tracks[key].box = box
    for i in range(140):   # B touches A until the fall at 2 s, then A lies there; confirmed 10 s later
        t = i / 10
        p.safety_watch.update(7, t, A, body(A))
        p.safety_watch.update(8, t, B if t < 2.5 else FAR, body(B))
    p.clock.t = 12.0
    p._on_fall(7, (480, 640, 3))
    assert notes == [(p.tracks[7].db_id, f"after touching person #{p.tracks[8].db_id}")]
    p.safety_watch = SafetyWatch({})   # nobody near: no note
    notes.clear()
    p._on_fall(7, (480, 640, 3))
    assert notes == [(p.tracks[7].db_id, "")]


def test_the_note_goes_into_the_fall_alert(db, camera):
    from datetime import datetime

    from labeler.alerts import AlertEngine
    engine = AlertEngine(db)
    engine.ensure_fall_rule()
    t = db.create_track(camera, 1, datetime.now())
    assert [e["message"] for e in engine.check(t, "test-cam", fallen=True, note="after touching person #3")] == \
        ["Fall: fell; after touching person #3"]


def test_the_camera_raises_one_check_about_both(db, camera):
    from datetime import datetime

    from labeler.pipeline import TrackState
    from test_pipeline_logic import make_pipeline

    class Queue:
        def __init__(self):
            self.raised = []

        def suspect(self, camera, category, reason, **kw):
            self.raised.append((category, reason, kw))
            return "queued"

    p = make_pipeline(db)
    p.incidents = Queue()
    for key, box in ((7, A), (8, B)):
        p.tracks[key] = TrackState(key, datetime.now(), 0.0, db_id=db.create_track(camera, key, datetime.now()))
        p.tracks[key].box = box
    p._raise_fight(7, 8, "two people close", (480, 640, 3))
    [(category, reason, kw)] = p.incidents.raised
    # one check per camera a minute: no people for the queue's cooldown, their numbers in the reason
    assert category == "fight" and kw["people"] == [] and kw["span_s"] == 2
    assert reason == f"two people close (people #{p.tracks[7].db_id} and #{p.tracks[8].db_id})"
    assert kw["crop"] == (100 / 640, 100 / 480, 280 / 640, 400 / 480)   # around both


def test_fights_turned_off_raise_nothing_but_touches_are_still_known():
    """Phase 51: fights off; a fall's "after touching person #12" still works (people are still followed)."""
    w = SafetyWatch({"fights_enabled": False})
    assert run(w, 20, punching) == []
    assert w.touched("a", 0.0, 2.0) == ["b"]

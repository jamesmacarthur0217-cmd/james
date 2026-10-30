"""Suspicious behaviour (Phase 49, labeler/behaviour.py): climbing at a fence or wall, in a restricted area at night or
when Away, pacing back and forth, the camera covered or turned; and what must not count (walking away from the
camera, walking straight through, a light switched off, someone passing the lens)."""
from datetime import datetime

import cv2
import numpy as np

from labeler.behaviour import BehaviourWatch

SHAPE = (480, 640, 3)
FENCE = {"id": 1, "kind": "area", "role": "fence", "name": "back wall",
         "points": [(0.0, 0.3), (0.5, 0.3), (0.5, 1.0), (0.0, 1.0)]}
STORE = {"id": 2, "kind": "area", "role": "restricted", "name": "store room",
         "points": [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]}
YARD = {"id": 3, "kind": "area", "role": "", "name": "yard", "points": [(0, 0), (1, 0), (1, 1), (0, 1)]}


def feed(w, frames, where, zones, t0=0.0, fps=10, watchful=True, key="p"):
    """where(i) -> box; returns [(t, sign, reason)]."""
    out = []
    for i in range(frames):
        t = t0 + i / fps
        w.update(key, t, where(i), SHAPE, list(zones))
        out += [(round(t, 1), s, r) for _, s, r, _ in w.tick(t, list(zones), SHAPE, watchful)]
    return out


def test_climbing_at_a_fence():
    standing = (100, 200, 160, 400)
    def climb(i):   # standing 2 s, then the feet go up 40% of the height and stay up
        up = 0 if i < 20 else 80
        return (100, 200 - up, 160, 400 - up)
    got = feed(BehaviourWatch({}), 50, climb, [FENCE])
    assert [s for _, s, _ in got] == ["climb"] and 3.0 <= got[0][0] <= 3.2 and "back wall" in got[0][2]
    assert feed(BehaviourWatch({}), 50, climb, [YARD]) == []        # no fence there
    def away(i):   # walking away from the camera: the feet go up in the picture, but the body shrinks
        k = min(i, 40) / 40
        return (100, 200 - 60 * k, 160 - 20 * k, 400 - 120 * k)
    assert feed(BehaviourWatch({}), 60, away, [FENCE]) == []
    assert feed(BehaviourWatch({}), 30, lambda i: standing, [FENCE]) == []


def test_restricted_only_at_night_or_away_and_once_per_visit():
    inside = lambda i: (450, 200, 510, 400)   # noqa: E731
    assert feed(BehaviourWatch({}), 100, inside, [STORE], watchful=False) == []   # day at Home: normal
    w = BehaviourWatch({})
    got = feed(w, 100, inside, [STORE])
    assert [(t, s) for t, s, _ in got] == [(3.0, "restricted")] and "store room for 3 s" in got[0][2]
    outside = lambda i: (100, 200, 160, 400)   # noqa: E731
    assert feed(w, 50, outside, [STORE], t0=10) == []
    assert feed(w, 50, inside, [STORE], t0=15) == []                  # back after 5 s: the same visit
    feed(w, 650, outside, [STORE], t0=20)                             # away 65 s: a new visit next time
    assert len(feed(w, 40, inside, [STORE], t0=85)) == 1


def walker(legs: int, leg_s: float = 8.0, span: float = 400.0):
    """Walking left and right `legs` times across the yard (each leg span pixels, a person 200 px tall)."""
    def where(i):
        t = i / 10
        k = t / leg_s
        leg, part = int(k), k - int(k)
        if leg >= legs:
            leg, part = legs - 1, 1.0
        x = 20 + span * (part if leg % 2 == 0 else 1 - part)
        return (x, 200, x + 60, 400)
    return where


def test_pacing_back_and_forth():
    w = BehaviourWatch({})
    got = feed(w, int(6 * 8 * 10), walker(6), [YARD])
    assert [s for _, s, _ in got] == ["pacing"] and "4 turns" in got[0][2]
    assert feed(BehaviourWatch({}), 80, walker(1), [YARD]) == []                       # straight through
    assert feed(BehaviourWatch({}), 480, walker(6), [YARD], watchful=False) == []       # day at Home
    assert feed(BehaviourWatch({}), 480, walker(6, span=150), [YARD]) == []            # small steps: not pacing


def test_turns_count_legs_only():
    w = BehaviourWatch({})
    walk = [(i, x, 300, 100) for i, x in enumerate([0, 200, 0, 200, 0, 190, 210, 0])]
    assert w.turns(walk) == 5


_rng = np.random.default_rng(3)
SCENE = np.clip(_rng.normal(120, 50, (360, 640, 3)), 0, 255).astype(np.uint8)
SCENE = np.ascontiguousarray(np.repeat(np.repeat(SCENE[::8, ::8], 8, 0), 8, 1)[:360, :640])   # blocks: edges


def watch_camera(w, seconds, picture, t0=0.0):
    out = []
    for i in range(seconds):
        got = w.camera(t0 + i, picture(i))
        if got:
            out.append((t0 + i, got[0]))
    return out


def test_the_camera_covered_and_turned():
    w = BehaviourWatch({})
    assert watch_camera(w, 15, lambda i: SCENE) == []
    got = watch_camera(w, 12, lambda i: np.full_like(SCENE, 12), t0=15)
    assert got == [(20, "covered")]   # the middle of the last 5 looks is black from the 3rd black one, then held 3 s
    w = BehaviourWatch({})
    watch_camera(w, 15, lambda i: SCENE)
    turned = np.roll(SCENE, 200, axis=1)
    assert watch_camera(w, 10, lambda i: turned, t0=15)[0][1] == "turned"


def test_a_light_or_someone_passing_is_not_the_camera():
    w = BehaviourWatch({})
    watch_camera(w, 15, lambda i: SCENE)
    dim = (SCENE * 0.4).astype(np.uint8)                              # the light switched off, the room still seen
    assert watch_camera(w, 15, lambda i: dim, t0=15) == []
    # Phase 53: SCENE's small gray picture has a contrast of about 20, like a dark night picture; dimmed to a quarter it
    # falls below covered_std (6), flat by the old rule, but its edges stay where they were: not covered (black is,
    # test_the_camera_covered_and_turned).
    w = BehaviourWatch({})
    watch_camera(w, 15, lambda i: SCENE)
    darker = (SCENE.astype(np.float32) * 0.25).astype(np.uint8)
    small = cv2.resize(cv2.cvtColor(darker, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
    assert small.std() < 6 and watch_camera(w, 15, lambda i: darker, t0=15) == []

    def passing(i):   # people walk close past the lens: a big dark shape crossing the picture in about 3 s, again and again
        f = SCENE.copy()
        x = (200 * i) % 840 - 200
        f[:, max(0, x):max(0, x + 250)] = 30
        return f
    w = BehaviourWatch({})
    watch_camera(w, 15, lambda i: SCENE)
    assert watch_camera(w, 12, passing, t0=15) == []


def test_night_hours():
    w = BehaviourWatch({"night_hours": "22-06"})
    assert w.night(datetime(2026, 10, 23, 23)) and w.night(datetime(2026, 10, 23, 5))
    assert not w.night(datetime(2026, 10, 23, 12))


class Queue:
    def __init__(self):
        self.raised, self.direct_ = [], []

    def suspect(self, camera, category, reason, **kw):
        self.raised.append((category, reason, kw))
        return "queued"

    def direct(self, camera, category, reason, **kw):
        self.direct_.append((category, reason, kw))
        return []


class Arming:
    def __init__(self, state):
        self.state = state

    def current(self):
        return self.state


def camera_pipe(db, camera, hour=12, away=False):
    from types import SimpleNamespace

    from labeler.incidents import FrameMemory
    from labeler.pipeline import TrackState, VideoClock
    from labeler.zones import ZoneCounter
    from test_pipeline_logic import make_pipeline
    p = make_pipeline(db)
    p.clock = VideoClock(datetime(2026, 10, 23, hour))
    p.zone_counter = ZoneCounter([FENCE, STORE])
    p.incidents, p.memory = Queue(), FrameMemory()
    p.alerts = SimpleNamespace(arming=Arming("away" if away else "home"))
    p.behaviour = BehaviourWatch({})
    p.tracks[5] = TrackState(5, datetime.now(), 0.0, db_id=db.create_track(camera, 5, datetime.now()))
    p.person_seen_wall = None
    return p


def test_the_camera_raises_climbing_and_alerts_patterns(db, camera):
    p = camera_pipe(db, camera, hour=12, away=True)
    frame = np.zeros(SHAPE, np.uint8)
    for i in range(40):   # standing at the back wall, then climbing
        t = i / 10
        up = 0 if i < 20 else 80
        p.tracks[5].box = (100, 200 - up, 160, 400 - up)
        p.clock.t = t
        p.behaviour.update(5, t, p.tracks[5].box, SHAPE, p.zone_counter.zones)
        p._look_for_behaviour(frame, t)
    [(category, reason, kw)] = p.incidents.raised
    assert category == "suspicious" and kw["level"] == "critical" and "climbing over" in kw["question"]
    assert reason.endswith("when Away")
    p = camera_pipe(db, camera, hour=23)   # at night, at Home
    for i in range(40):
        t = i / 10
        p.tracks[5].box = (450, 200, 510, 400)
        p.behaviour.update(5, t, p.tracks[5].box, SHAPE, p.zone_counter.zones)
        p._look_for_behaviour(frame, t)
    [(category, reason, kw)] = p.incidents.direct_
    assert reason == "in the restricted area store room for 3 s at night" and kw["people"] == [p.tracks[5].db_id]
    assert "never how they look" in kw["describe"]


def test_the_camera_covered_alerts_only_after_someone_was_seen(db, camera):
    from datetime import timedelta
    p = camera_pipe(db, camera)
    scene = SCENE.copy()
    for i in range(15):
        p.clock.t = float(i)
        p.memory.add(scene, p.clock.wall(), float(i))
        p._look_for_behaviour(scene, float(i))
    p.person_seen_wall = p.clock.wall() - timedelta(seconds=2)
    black = np.full_like(scene, 10)
    for i in range(15, 24):
        p.clock.t = float(i)
        p._look_for_behaviour(black, float(i))
    [(category, reason, kw)] = p.incidents.direct_
    assert category == "suspicious" and reason.startswith("the camera was covered") and len(kw["pictures"]) == 1
    p = camera_pipe(db, camera)   # nobody seen: a camera problem only (the picture check)
    for i in range(15):
        p._look_for_behaviour(scene, float(i))
    for i in range(15, 24):
        p._look_for_behaviour(black, float(i))
    assert p.incidents.direct_ == []


def test_a_direct_alert_and_the_words_added_after(db, camera, tmp_path):
    import copy

    from labeler.config import DEFAULTS
    from labeler.incidents import ConfirmQueue, FrameMemory, IncidentAlerts

    class Alerts:
        def __init__(self, rule):
            self.rule = rule

        def check_incident(self, camera, category, area):
            return [self.rule]

    rule = db.add_rule("Suspicious behaviour", [["incident", "suspicious"]])
    rule = rule if isinstance(rule, dict) else {"id": rule, "name": "Suspicious behaviour"}
    mem = FrameMemory()
    mem.add(SCENE, datetime(2026, 10, 23, 23), 0.0)
    q = ConfirmQueue(copy.deepcopy(DEFAULTS), None, {"cam1": mem}, clock=lambda: 0.0)
    q.listeners.append(IncidentAlerts(db, Alerts(rule), [], tmp_path, clips_on=False))
    made = q.direct("cam1", "suspicious", "walking back and forth in yard (4 turns in 2 min) at night",
                    describe="what is the person doing?")
    assert len(made) == 1 and made[0]["level"] == "warning"
    assert made[0]["message"] == "Suspicious behaviour: walking back and forth in yard (4 turns in 2 min) at night"
    [check] = q.waiting                                  # then the vision model is asked what the person does
    assert check.alert_if_yes is False and check.question == "what is the person doing?"
    check.on_done({"answer": "no", "what": "a man looking into a car"})   # its words are added; it never stops it
    msg = [a["message"] for a in db.list_alerts() if a["id"] == made[0]["id"]][0]
    assert msg.endswith("; the vision model: a man looking into a car")
    assert q.state()["direct"] == 1
    # once a minute per camera, area and sign: the next person walking in a second later makes no new alert
    assert q.direct("cam1", "suspicious", "walking back and forth in yard at night", describe="what?") == []
    q.waiting.clear()
    q.last_sent.clear()   # a minute later: the next one, and its question is asked though one was just asked
    q.last_sent[("cam1", "suspicious", None, ())] = -1.0
    assert len(q.direct("cam1", "suspicious", "walking back and forth in yard at night", describe="what?")) == 1
    assert len(q.waiting) == 1
    climb = q.direct("cam1", "suspicious", "climbing at back wall when Away", level="critical", sign="climb")
    assert climb[0]["level"] == "critical"


DOOR = {"id": 9, "name": "front door", "kind": "line", "role": "door", "points": [(0.5, 0.0), (0.5, 1.0)]}


def test_following_someone_through_a_door():
    """Phase 54: on a "door" line, two people crossing the same way within follow_s (2 s); not one person twice, not
    the other way, not later, not a plain counting line."""
    w = BehaviourWatch({})
    assert w.crossed(10.0, 1, DOOR, "in") is None
    other, reason = w.crossed(11.4, 2, DOOR, "in")
    assert other == 1 and reason == ("two people went in through front door 1.4 s apart (someone following another "
                                     "through)")
    assert w.crossed(12.0, 3, DOOR, "out") is None                 # the other way: meeting, not following
    assert w.crossed(12.5, 3, DOOR, "in")[0] == 2                  # within 2 s of person 2
    assert w.crossed(20.0, 4, DOOR, "in") is None                  # 7.5 s later: alone
    w2 = BehaviourWatch({})
    w2.crossed(0.0, 1, DOOR, "in")
    assert w2.crossed(1.0, 1, DOOR, "in") is None                  # the same person (back and forth on the line)
    plain = {**DOOR, "id": 10, "role": ""}
    w3 = BehaviourWatch({})
    w3.crossed(0.0, 1, plain, "in")
    assert w3.crossed(0.5, 2, plain, "in") is None                 # a plain counting line


def test_the_camera_alerts_following_only_at_night_or_when_away(db, camera):
    from labeler.pipeline import TrackState
    for hour, away, alerts in ((12, False, 0), (23, False, 1), (12, True, 1)):
        p = camera_pipe(db, camera, hour=hour, away=away)
        p.zone_counter.zones = [DOOR]
        p.tracks[5].box = (300, 200, 360, 400)
        p.tracks[6] = TrackState(6, datetime.now(), 0.0, db_id=db.create_track(camera, 6, datetime.now()))
        p.tracks[6].box = (380, 200, 440, 400)
        p._check_doors(5, [{"zone_id": 9, "kind": "in"}], 10.0, SHAPE)
        p._check_doors(6, [{"zone_id": 9, "kind": "in"}], 11.0, SHAPE)
        assert len(p.incidents.direct_) == alerts, (hour, away)
        if alerts:
            [(category, reason, kw)] = p.incidents.direct_
            assert category == "suspicious" and reason.startswith("two people went in through front door 1.0 s apart")
            assert reason.endswith("when Away" if away else "at night") and kw["sign"] == "following 9"
            assert sorted(kw["people"]) == sorted([p.tracks[5].db_id, p.tracks[6].db_id]) and kw["crop"][0] < 0.6


CAR_A = (100, 250, 300, 380)       # a parked car, pixels
CAR_B = (400, 250, 600, 380)


def _at(x, wrist=None):
    """A person standing with their feet at y 390 (beside a car's bottom), 200 px tall, centred at x; a wrist if given."""
    k = np.zeros((17, 3), np.float32)
    if wrist:
        k[10] = (*wrist, 0.9)
    return (x - 30, 190, x + 30, 390), k


def _cars_run(w, frames, seconds, t0=0.0, parked=((1, CAR_A, "car"), (2, CAR_B, "car")), watchful=True, fps=5):
    """frames(t) -> (box, kpts); returns [(t, sign, reason)] of the car signs."""
    out = []
    for i in range(int(seconds * fps)):
        t = t0 + i / fps
        box, k = frames(t)
        w.update(5, t, box, SHAPE, [], k)
        out += [(round(t, 1), s, r) for _, s, r, _ in w.cars(t, list(parked), watchful)]
    return out


def test_a_hand_at_a_parked_car_for_5_s():
    """Phase 54: standing at a parked car, a wrist in it, hardly moving, 5 s: one sign; not by day; not walking past."""
    got = _cars_run(BehaviourWatch({}), lambda t: _at(200, (220, 300)), 8)
    assert [(s, r) for _, s, r in got] == [("car", "a hand at a parked car for 5 s")] and 5.0 <= got[0][0] <= 6.2
    assert _cars_run(BehaviourWatch({}), lambda t: _at(200, (220, 300)), 8, watchful=False) == []   # by day
    walking = lambda t: _at(60 + 40 * t, (80 + 40 * t, 300))   # noqa: E731  (a wrist over the car, walking past)
    assert [s for _, s, _ in _cars_run(BehaviourWatch({}), walking, 8)] == []
    assert _cars_run(BehaviourWatch({}), lambda t: _at(200, (220, 120)), 8) == []   # the hand above the car, not in it


def test_stopping_at_two_parked_cars_in_a_row():
    w = BehaviourWatch({})
    a = _cars_run(w, lambda t: _at(200), 4)                         # stops 4 s at car A
    walk = _cars_run(w, lambda t: _at(200 + 50 * (t - 4)), 4, t0=4)  # walks on to car B
    b = _cars_run(w, lambda t: _at(500), 4, t0=8)                   # stops 4 s at car B
    assert a == [] and walk == [] and [(s, r) for _, s, r in b] == [("cars", "stopped at 2 parked vehicles in a row")]
    w2 = BehaviourWatch({})
    _cars_run(w2, lambda t: _at(200), 4)
    assert _cars_run(w2, lambda t: _at(200), 10, t0=4) == []        # one car, however long: not a pattern


def test_one_car_check_a_minute_per_camera():
    w = BehaviourWatch({})
    assert len(_cars_run(w, lambda t: _at(200, (220, 300)), 8)) == 1
    assert _cars_run(w, lambda t: _at(500, (520, 300)), 8, t0=10) == []      # 18 s later, another car: held back
    assert len(_cars_run(w, lambda t: _at(500, (520, 300)), 8, t0=70)) == 1  # a minute on: again


def test_the_camera_asks_the_vision_model_about_someone_at_a_car(db, camera):
    from types import SimpleNamespace

    from labeler.behaviour import CAR_QUESTION
    car = SimpleNamespace(key=1, box=CAR_A, kind="car", parked=True, left=False)
    for hour, checks in ((12, 0), (23, 1)):
        p = camera_pipe(db, camera, hour=hour)
        p.vehicles = SimpleNamespace(tracks={1: car})
        frame = np.zeros(SHAPE, np.uint8)
        for i in range(40):
            t = i / 5
            box, k = _at(200, (220, 300))
            p.tracks[5].box = box
            p.behaviour.update(5, t, box, SHAPE, [], k)
            p._look_for_behaviour(frame, t)
        cars = [r for r in p.incidents.raised if "parked car" in r[1]]
        assert len(cars) == checks, hour
        if checks:
            category, reason, kw = cars[0]
            assert category == "suspicious" and reason == "a hand at a parked car for 5 s at night"
            assert kw["question"] == CAR_QUESTION and kw["people"] == [p.tracks[5].db_id]
            assert kw["crop"][0] == CAR_A[0] / SHAPE[1] and kw["crop"][1] == 190 / SHAPE[0]   # around both


YARD_AREA = {"id": 3, "name": "yard", "kind": "area", "role": "", "points": [(0, 0), (1, 0), (1, 1), (0, 1)]}


def _look(i):
    """A unit look vector: the same i is the same person (similarity 1), different i are far apart (0)."""
    v = np.zeros(8, np.float32)
    v[i] = 1.0
    return v


def test_coming_back_again_and_again():
    """Phase 54: the 3rd visit to an area within 30 min, visits a minute apart, the same person by look (another
    tracker number each time) or the same number; not someone else, not a quick step out and back, not after 30 min."""
    w = BehaviourWatch({})
    assert w.entered(0.0, 1, YARD_AREA, _look(0), ("a", None)) is None
    assert w.entered(30.0, 1, YARD_AREA, _look(0)) is None                 # back in after 30 s: the same visit
    assert w.entered(300.0, 2, YARD_AREA, _look(1), ("b", None)) is None   # someone else
    assert w.entered(400.0, 3, YARD_AREA, _look(0), ("c", None)) is None   # 2nd visit (new number, same look)
    reason, pictures = w.entered(900.0, 4, YARD_AREA, _look(0), ("d", None))
    assert reason == "the same person (by look) came into yard 3 times in 15 min" and [pic for pic, _ in pictures] == ["a", "c", "d"]
    assert w.entered(1000.0, 5, YARD_AREA, _look(0)) is None               # once an area every 10 min
    w2 = BehaviourWatch({})
    for i, t in enumerate((0.0, 1000.0, 2000.0)):                          # 3 visits, but over 33 min
        got = w2.entered(t, 10 + i, YARD_AREA, _look(0))
    assert got is None
    w3 = BehaviourWatch({})
    w3.entered(0.0, 7, YARD_AREA, None)
    w3.entered(100.0, 7, YARD_AREA, None)                                  # no look: the same number still counts
    assert w3.entered(200.0, 7, YARD_AREA, None)[0].endswith("3 times in 3 min")


def test_the_camera_alerts_a_comeback_at_night_with_the_visits_pictures(db, camera):
    from labeler.pipeline import Candidate, TrackState
    for hour, alerts in ((12, 0), (23, 1)):
        p = camera_pipe(db, camera, hour=hour)
        p.zone_counter.zones = [YARD_AREA]
        p.embedder = None
        for i, t in enumerate((0.0, 200.0, 400.0)):
            st = TrackState(20 + i, datetime.now(), t, db_id=db.create_track(camera, 20 + i, datetime.now()))
            st.vector = _look(0)
            st.candidates = [Candidate(0.9, np.full((320, 120, 3), 100 + i, np.uint8), None, (20, 0, 100, 60))]
            p.tracks[st.tracker_id] = st
            p._check_visits(st, [{"zone_id": 3, "kind": "enter"}], t)
        assert len(p.incidents.direct_) == alerts, hour
        if alerts:
            [(category, reason, kw)] = p.incidents.direct_
            assert category == "suspicious" and reason.endswith("3 times in 7 min at night")
            assert kw["sign"] == "came back" and kw["people"] == [p.tracks[22].db_id]
            [(_, strip, heads)] = kw["pictures"]
            assert strip.shape[0] == 160 and strip.shape[1] == 3 * 60 + 2 * 6 and len(heads) == 3
            assert heads[1][0] == 66 + 10 and heads[0][3] == 30                # the 2nd head moved along the strip

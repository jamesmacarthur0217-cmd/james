"""Floor hazards and blocked exits (Phase 47 step 2, labeler/floor.py): the empty picture is learned around people,
a change held 30 s on a walkway (2 min in an exit) is raised once, two "yes" a minute apart alert, a "no" goes into the
empty picture, the lights changing is not a hazard, and the empty picture is saved and read back."""
import numpy as np

from labeler.floor import FloorWatch

H, W = 270, 480
WALK = {"id": 1, "kind": "area", "role": "walkway", "name": "aisle",
        "points": [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]}
EXIT = {**WALK, "id": 2, "role": "exit", "name": "door"}
PLAIN = {**WALK, "id": 3, "role": "", "name": "counted"}
BAG = (200, 200, 230, 230)   # pixels: x1, y1, x2, y2, in the walkway (lower half)
_rng = np.random.default_rng(7)
FLOOR = np.clip(_rng.normal(130, 30, (H, W, 3)), 0, 255).astype(np.uint8)


NOISE = [np.random.default_rng(s).normal(0, 3, FLOOR.shape).astype(np.float32) for s in range(10)]


def scene(bag=False, gain=1.0, seed=0, box=BAG):
    """The floor with a camera's noise (one of 10 patterns), brighter or darker by gain, with or without the bag."""
    f = FLOOR.astype(np.float32) * gain + NOISE[seed % 10]
    if bag:
        x1, y1, x2, y2 = box
        f[y1:y2, x1:x2] = (40, 30, 160)
    return np.clip(f, 0, 255).astype(np.uint8)


def run(w, seconds, t0=0.0, zones=(WALK,), people=lambda t: [], **kw):
    """One look a second; returns [(time, suspect)]."""
    out = []
    for i in range(int(seconds)):
        t = t0 + i
        for s in w.update(t, scene(seed=int(t), **kw), people(t), list(zones)):
            out.append((t, s))
    return out


def learned(**cfg):
    w = FloorWatch({"floor_learn_n": 5, "floor_learn_every_s": 1, "floor_learn_min": 3, **cfg})
    assert run(w, 6) == [] and w.ready_share() == 1.0
    return w


def test_learns_then_a_bag_held_30_s_is_raised_once():
    w = learned()
    got = run(w, 40, t0=6, bag=True)
    assert len(got) == 1
    t, s = got[0]
    assert t == 6 + 30 and s["category"] == "floor" and s["first_look"] and s["area"]["name"] == "aisle"
    assert s["reason"].startswith("something new on the floor of aisle for 30 s")
    x1, y1, x2, y2 = s["crop"]
    assert abs(x1 - 200 / W) < 0.02 and abs(y2 - 230 / H) < 0.02   # around the bag
    assert run(w, 30, t0=46, bag=True) == []   # waiting for the answer: not raised again


def test_someone_missed_while_learning_is_not_learned():
    """A passer-by the detectors miss crosses during 2 of the 5 learning looks: the median leaves them out."""
    w = FloorWatch({"floor_learn_n": 5, "floor_learn_every_s": 1, "floor_learn_min": 3})
    for t in range(5):
        frame = scene(seed=t, bag=t in (1, 3), box=(100 + 40 * t, 150, 160 + 40 * t, 260))
        assert w.update(t, frame, [], [WALK]) == []
    assert w.learned and np.abs(w.picture().astype(int) - FLOOR.astype(int)).max() < 20
    assert run(w, 60, t0=5) == []


def test_exit_waits_two_minutes_and_plain_areas_are_not_watched():
    w = learned()
    got = run(w, 130, t0=6, zones=(EXIT,), bag=True)
    assert [(t, s["category"]) for t, s in got] == [(126, "exit")]
    assert run(FloorWatch({"floor_learn_n": 5, "floor_learn_every_s": 1, "floor_learn_min": 3}), 60, zones=(PLAIN,), bag=True) == []


def test_people_are_left_out():
    w = FloorWatch({"floor_learn_n": 5, "floor_learn_every_s": 1, "floor_learn_min": 3})
    # someone stands where the bag will be the whole time the empty picture is learned: that spot waits
    stand = lambda t: [(190, 120, 240, 235)]   # noqa: E731
    assert run(w, 6, people=stand, bag=True) == []
    assert w.ready_share() < 1.0
    # someone walking across the floor is never a hazard
    walker = lambda t: [(20 + 10 * (t % 40), 120, 60 + 10 * (t % 40), 250)]   # noqa: E731
    w = learned()
    assert run(w, 120, t0=6, people=walker) == []
    # a person standing on the bag hides it, but the time it has been there goes on
    w = learned()
    on_it = lambda t: [(190, 120, 240, 235)] if 15 <= t < 30 else []   # noqa: E731
    got = run(w, 40, t0=6, people=on_it, bag=True)
    assert [t for t, _ in got] == [36]


def test_two_yes_a_minute_apart_then_reported():
    w = learned()
    (t, s), = run(w, 31, t0=6, bag=True)
    w.answered(1, {"answer": "yes", "what": "a bag"})
    assert run(w, 59, t0=37, bag=True) == []          # the second look waits floor_again_s
    (t2, s2), = run(w, 5, t0=96, bag=True)
    assert t2 == 97 and not s2["first_look"]           # this one alerts on "yes"
    w.answered(1, {"answer": "yes", "what": "a bag"})
    assert run(w, 300, t0=101, bag=True) == []         # reported: not again while it stays
    assert w.state()["reported"] > 0
    assert run(w, 5, t0=401) == []                      # taken away: no longer reported
    assert w.state()["reported"] == 0
    got = run(w, 32, t0=406, bag=True)                 # put back: a new first look
    assert len(got) == 1 and got[0][1]["first_look"]


def test_one_alert_for_one_bag_even_when_part_of_it_was_hidden():
    """Someone stands on the bag's right half until t=80: those spots begin to differ only then, too late to be part of
    the second look at t=97. After the alert the whole bag is reported, so its right half is not raised again."""
    w = learned()
    on_half = lambda t: [(212, 120, 250, 235)] if t < 80 else []   # noqa: E731
    (t, s), = run(w, 31, t0=6, people=on_half, bag=True)
    w.answered(1, {"answer": "yes", "what": "a bag"})
    (t2, s2), = run(w, 65, t0=37, people=on_half, bag=True)
    assert not s2["first_look"]
    w.answered(1, {"answer": "yes", "what": "a bag"})
    assert run(w, 150, t0=102, bag=True) == []


def test_no_puts_the_change_into_the_empty_picture():
    w = learned()
    run(w, 31, t0=6, bag=True)
    w.answered(1, {"answer": "no", "what": "a chair"})
    assert run(w, 200, t0=37, bag=True) == []
    # unsure or no answer: asked again later, not every second
    w = learned()
    run(w, 31, t0=6, bag=True)
    w.answered(1, None)
    got = run(w, 40, t0=37, bag=True)
    assert [t for t, _ in got] == [67]


def test_the_lights_are_not_a_hazard():
    w = learned(floor_light_s=10)
    assert run(w, 60, t0=6, gain=0.85) == []           # the camera's exposure: matched
    assert run(w, 15, t0=66, gain=2.5) == []           # the lights went on: everything changed
    assert w.relearned == 1 and w.ready_share() < 1.0
    assert run(w, 60, t0=81, gain=2.5) == []           # learned again, nothing raised


def test_the_empty_picture_is_saved_and_read_back(tmp_path):
    path = tmp_path / "empty" / "cam1.png"
    w = FloorWatch({"floor_learn_n": 5, "floor_learn_every_s": 1, "floor_learn_min": 3, "floor_save_s": 0}, path)
    run(w, 6)
    assert path.is_file() and w.state()["saved"]
    again = FloorWatch({"floor_learn_n": 5, "floor_learn_every_s": 1, "floor_learn_min": 3}, path)
    got = run(again, 31, bag=True)                     # ready at once: no learning first
    assert [t for t, _ in got] == [30]
    other = FloorWatch({"floor_learn_n": 5, "floor_learn_every_s": 1, "floor_learn_min": 3}, path)
    other.update(0, np.zeros((480, 480, 3), np.uint8), [], [WALK])   # another camera size: learned afresh
    assert other.ready_share() < 1.0


def test_no_watched_area_does_nothing():
    w = FloorWatch({})
    assert w.update(0, scene(), [], [PLAIN]) == [] and w.empty is None
    assert w.state()["areas"] == []


class Queue:
    def __init__(self):
        self.raised = []

    def suspect(self, camera, category, reason, **kw):
        self.raised.append((category, reason, kw))
        return "queued"


def test_the_camera_sends_the_empty_picture_and_now(db):
    from datetime import datetime

    from labeler.incidents import FrameMemory
    from labeler.pipeline import VideoClock
    from labeler.zones import ZoneCounter
    from test_pipeline_logic import make_pipeline

    p = make_pipeline(db)
    p.zone_counter = ZoneCounter([WALK])
    p.clock = VideoClock(datetime(2026, 10, 21, 12))
    p.memory, p.incidents = FrameMemory(keep_s=10), Queue()
    p.floor_watch, p._counted_people = FloorWatch({"floor_learn_n": 5, "floor_learn_every_s": 1, "floor_learn_min": 3}), (-1e9, [])
    for t in range(100):
        p.clock.t = float(t)
        frame = scene(seed=t, bag=t >= 6)
        p.memory.add(frame, p.clock.wall(), float(t))
        p._look_at_floor(frame, float(t), set())
        if len(p.incidents.raised) == 1 and t == 36:
            category, reason, kw = p.incidents.raised[0]
            assert category == "floor" and kw["alert_if_yes"] is False and kw["question"].startswith("Picture 1")
            assert kw["cut"] == (0.6, 0.26)
            (before, empty, _), (now, pic, _) = kw["pictures"]
            assert (now - before).total_seconds() == 30   # the empty picture is dated when the change began
            assert abs(int(empty[215, 215, 0]) - int(FLOOR[215, 215, 0])) < 15 and tuple(pic[215, 215]) == (40, 30, 160)
            kw["on_done"]({"answer": "yes", "what": "a bag"})
    assert [kw["alert_if_yes"] for _, _, kw in p.incidents.raised] == [False, True]   # the second look alerts


def test_only_the_second_look_alerts(tmp_path):
    from labeler.incidents import IncidentAlerts, Suspect

    class Alerts:
        asked = 0

        def check_incident(self, camera, category, area):
            self.asked += 1
            return []

    alerts = Alerts()
    listener = IncidentAlerts(None, alerts, [], tmp_path)
    yes = {"answer": "yes", "what": "a bag"}
    listener(Suspect("cam1", "floor", "new", alert_if_yes=False, on_done=print), yes)
    assert alerts.asked == 0
    listener(Suspect("cam1", "floor", "new", alert_if_yes=True, on_done=print), yes)
    listener(Suspect("cam1", "fall", "fell", on_done=print), yes)   # falls alert themselves, as before
    listener(Suspect("cam1", "fire", "flames"), yes)
    assert alerts.asked == 2


def test_health_line():
    from types import SimpleNamespace

    from labeler.healthcheck import check_floor

    assert check_floor([SimpleNamespace(name="cam1", floor_watch=None)]) is None   # no safety camera
    w = FloorWatch({"floor_learn_n": 5, "floor_learn_every_s": 1, "floor_learn_min": 3})
    cams = [SimpleNamespace(name="cam1", floor_watch=w)]
    assert check_floor(cams)["level"] == "off"                                    # no walkway or exit area yet
    run(w, 2)
    assert "learning the empty picture (2 of 5 looks)" in check_floor(cams)["detail"]
    run(w, 5, t0=2)
    assert check_floor(cams)["detail"] == "cam1: aisle watched"

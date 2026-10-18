"""Unusual quiet (Phase 39 step 2): the built-in rule fires at the end of a usual hour with nobody seen, once a day,
only for hours the camera watched while Home; the message, "seen at", the pop-up flag and the rule's checks."""
import copy
from datetime import datetime, time, timedelta

from fastapi.testclient import TestClient

from labeler.alerts import UNUSUAL_RULE, AlertEngine, describe, is_unusual_rule, rule_kind
from labeler.checks import alert_what
from labeler.config import DEFAULTS
from labeler.routine import Routine
from labeler.unusual import UnusualWatch
from labeler.web import create_app
from test_routine import TODAY, _people, _regular


class Cam:
    name, mode, running = "test-cam", "kiosk", True

    def __init__(self):
        self.person_seen_wall = None


class Arming:
    def __init__(self, mode="home", since=None):
        self.mode, self.since = mode, since

    def state(self):
        return {"mode": self.mode, "since": self.since.isoformat() if self.since else None}


def at(h, m=0):
    return datetime.combine(TODAY, time(h, m))   # TODAY is a Monday; usual: 07, 08, 12, 18-21


def _setup(db, camera, history=True, arming=None, first_tick=at(6, 0)):
    if history:
        _people(db, camera, _regular())
    engine = AlertEngine(db)
    engine.ensure_unusual_rule()
    clock = [first_tick]
    routine = Routine(db, copy.deepcopy(DEFAULTS), ["test-cam"], clock=lambda: clock[0])
    cam = Cam()
    watch = UnusualWatch(db, engine, [cam], routine, arming or Arming(), clock=lambda: clock[0])
    watch.tick()   # the first tick: the watch sees the camera working from now
    return watch, cam, clock


def _run(watch, clock, until, step=timedelta(minutes=1)):
    made = []
    while clock[0] < until:
        clock[0] += step
        made += watch.tick()
    return made


def test_the_rule_is_made_once_and_described(db):
    engine = AlertEngine(db)
    engine.ensure_unusual_rule(), engine.ensure_unusual_rule()
    rules = [r for r in db.list_rules() if is_unusual_rule(r["conditions"])]
    assert len(rules) == 1 and rules[0]["name"] == "Unusual quiet" and rules[0]["conditions"] == UNUSUAL_RULE
    assert rule_kind(UNUSUAL_RULE) == "quiet" and describe(UNUSUAL_RULE).startswith("nobody seen when someone usually is")


def test_nobody_up_by_the_end_of_the_usual_hour(db, camera):
    watch, cam, clock = _setup(db, camera)
    assert _run(watch, clock, at(7, 59)) == []
    made = _run(watch, clock, at(8, 0))
    assert len(made) == 1
    assert made[0]["message"] == ("Unusual quiet: nobody seen on test-cam between 07:00 and 08:00; on weekdays someone "
                                  "usually is (on 100% of 40 days)")
    assert _run(watch, clock, at(22, 0), step=timedelta(minutes=10)) == []   # once a day
    alert = db.list_alerts()[0]
    assert alert["unusual"] and not alert["quiet"] and alert["track_id"] is None
    assert alert_what(alert, UNUSUAL_RULE) == "unusual"
    cam.person_seen_wall = at(22, 5)   # someone again: the alert says when
    _run(watch, clock, at(22, 6))
    assert db.list_alerts()[0]["message"].endswith("; seen at 22:05")
    clock[0] = datetime.combine(TODAY + timedelta(days=1), time(6))   # the next day can alert again
    assert len(_run(watch, clock, clock[0] + timedelta(hours=2))) == 1


def test_someone_seen_in_the_hour_means_no_alert(db, camera):
    watch, cam, clock = _setup(db, camera)
    _people(db, camera, [(at(7, 10), at(8, 40)), (at(12, 20), at(12, 50))])
    assert _run(watch, clock, at(17, 0), step=timedelta(minutes=5)) == []
    cam.person_seen_wall = at(18, 30)   # in view now (not in the database yet): seen too
    assert _run(watch, clock, at(19, 0)) == []
    cam.person_seen_wall = at(19, 50)   # newer sightings do not make 18:00-19:00 (judged fine) look empty again
    assert _run(watch, clock, at(20, 0)) == []
    cam.person_seen_wall = at(20, 40)
    assert _run(watch, clock, at(21, 0)) == []
    made = _run(watch, clock, at(22, 0))   # nobody 21:00-22:00
    assert len(made) == 1 and "between 21:00 and 22:00" in made[0]["message"]


def test_hours_the_camera_did_not_watch_do_not_count(db, camera):
    watch, cam, clock = _setup(db, camera, first_tick=at(8, 30))   # the app started at 08:30
    assert _run(watch, clock, at(12, 0), step=timedelta(minutes=5)) == []   # 07 and 08 were not watched whole
    watch.problem_end["test-cam"] = at(12, 10)   # a camera problem during 12:00-13:00
    assert _run(watch, clock, at(13, 5)) == []
    assert len(_run(watch, clock, at(19, 0), step=timedelta(minutes=5))) == 1   # 18:00-19:00 watched, nobody


def test_only_when_home(db, camera):
    away = Arming("away", at(5))
    watch, cam, clock = _setup(db, camera, arming=away)
    assert _run(watch, clock, at(7, 30)) == []   # Away: nothing
    away.mode, away.since = "home", at(7, 30)    # back home at 07:30
    assert _run(watch, clock, at(8, 5)) == []    # the hour from 07:00 began while Away: not counted
    made = _run(watch, clock, at(9, 0))          # 08:00-09:00 was all Home, and nobody was seen
    assert len(made) == 1 and "between 08:00 and 09:00" in made[0]["message"]


def test_still_learning_never_fires(db, camera):
    _people(db, camera, _regular(days=10))
    watch, cam, clock = _setup(db, camera, history=False)
    assert _run(watch, clock, at(22, 0), step=timedelta(minutes=10)) == []


def test_rule_api_accepts_it_with_home_away_only(db, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    client = TestClient(create_app(cfg, db, None), client=("127.0.0.1", 5000))
    ok = client.post("/api/alerts/rules", json={"name": "Mum's room", "conditions": [["unusual", "yes"], ["armed", "home"]]})
    assert ok.status_code == 200
    bad = client.post("/api/alerts/rules", json={"name": "x", "conditions": [["unusual", "yes"], ["hours", "08-22"]]})
    assert bad.status_code == 400 and "Home / Away" in bad.json()["detail"]

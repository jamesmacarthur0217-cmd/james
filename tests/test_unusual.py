"""Unusual quiet (Phase 39): the built-in rule fires when today's quiet stretch (since the last sighting) is at least
3 whole hours in which, in the history, someone was seen on 97% of days; once a day; only hours watched while Home;
the message, "seen at", the pop-up flag, the rule's checks, the Health line and the day summaries."""
import copy
from datetime import datetime, time, timedelta

from fastapi.testclient import TestClient

from labeler.alerts import UNUSUAL_RULE, AlertEngine, describe, is_unusual_rule, rule_kind
from labeler.checks import alert_what
from labeler.config import DEFAULTS
from labeler.routine import Routine, hour_ranges, learn, store_days
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
    return datetime.combine(TODAY, time(h, m))   # TODAY is a Monday; regular: up 07:10, lunch 12:20, evening 18:00


def _setup(db, camera, history=True, arming=None, first_tick=at(0, 0), spans=None):
    if history:
        _people(db, camera, spans or _regular())
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
        clock[0] = min(clock[0] + step, until)   # the last tick exactly at `until`
        made += watch.tick()
    return made


def test_the_rule_is_made_once_and_described(db):
    engine = AlertEngine(db)
    engine.ensure_unusual_rule(), engine.ensure_unusual_rule()
    rules = [r for r in db.list_rules() if is_unusual_rule(r["conditions"])]
    assert len(rules) == 1 and rules[0]["name"] == "Unusual quiet" and rules[0]["conditions"] == UNUSUAL_RULE
    assert rule_kind(UNUSUAL_RULE) == "quiet" and describe(UNUSUAL_RULE).startswith("nobody seen when someone usually is")


def test_nobody_up_by_an_hour_after_the_usual_time(db, camera):
    watch, cam, clock = _setup(db, camera)
    assert _run(watch, clock, at(7, 59), step=timedelta(minutes=5)) == []
    made = _run(watch, clock, at(8, 0))
    assert len(made) == 1
    assert made[0]["message"] == ("Unusual quiet: nobody seen on test-cam since 00:00; on weekdays someone is seen "
                                  "between 00:00 and 08:00 on 40 of 40 days")
    assert _run(watch, clock, at(22, 0), step=timedelta(minutes=10)) == []   # once a day
    alert = db.list_alerts()[0]
    assert alert["unusual"] and not alert["quiet"] and alert["track_id"] is None
    assert alert_what(alert, UNUSUAL_RULE) == "unusual"
    cam.person_seen_wall = at(22, 5)   # someone again: the alert says when
    _run(watch, clock, at(22, 6))
    assert db.list_alerts()[0]["message"].endswith("; seen at 22:05")
    clock[0] = datetime.combine(TODAY + timedelta(days=1), time(0))   # the next day can alert again
    assert len(_run(watch, clock, clock[0] + timedelta(hours=9), step=timedelta(minutes=5))) == 1


def test_a_normal_day_and_a_missed_evening_give_no_alert(db, camera):
    watch, cam, clock = _setup(db, camera)
    _people(db, camera, [(at(7, 10), at(8, 40)), (at(12, 20), at(12, 50))])
    # Nobody after lunch, no evening: evenings are missed on some days (1 in 20), so this is not unusual.
    assert _run(watch, clock, at(23, 55), step=timedelta(minutes=5)) == []


def test_a_short_quiet_stretch_is_not_enough(db, camera):
    # Someone in every hour 07-22 on every day: then quiet from 13:05.
    days = [TODAY - timedelta(days=n) for n in range(1, 57)]
    history = [(datetime.combine(d, time(h, 10)), datetime.combine(d, time(h, 20))) for d in days for h in range(7, 23)]
    watch, cam, clock = _setup(db, camera, spans=history)
    _people(db, camera, [(at(h, 10), at(h, 20)) for h in range(7, 13)] + [(at(13, 0), at(13, 5))])
    assert _run(watch, clock, at(16, 59), step=timedelta(minutes=5)) == []   # 14:00-16:00: 2 whole hours, not 3
    made = _run(watch, clock, at(17, 0))
    assert len(made) == 1 and "since 13:05" in made[0]["message"] and "between 14:00 and 17:00 on 40 of 40" in \
        made[0]["message"]   # a Monday: the 40 weekdays of the 8 weeks


def test_hours_the_camera_did_not_watch_do_not_count(db, camera):
    watch, cam, clock = _setup(db, camera, first_tick=at(8, 30))   # the app started at 08:30
    assert _run(watch, clock, at(12, 59), step=timedelta(minutes=5)) == []   # 09-12: nobody usually there
    made = _run(watch, clock, at(13, 0))    # 09:00-13:00 includes lunch, which had someone every weekday
    assert len(made) == 1 and "since 08:30" in made[0]["message"]


def test_a_camera_problem_starts_the_stretch_again(db, camera):
    watch, cam, clock = _setup(db, camera)
    _run(watch, clock, at(7, 0), step=timedelta(minutes=5))
    watch.problem_end["test-cam"] = at(7, 50)   # no picture until 07:50: nothing known before
    assert _run(watch, clock, at(10, 59), step=timedelta(minutes=5)) == []   # not at 08:00: the stretch starts again
    made = _run(watch, clock, at(11, 0))         # 08:00-11:00 had someone every weekday (until 08:40)
    assert len(made) == 1 and "since 07:50" in made[0]["message"]


def test_only_when_home(db, camera):
    away = Arming("away", at(0))
    watch, cam, clock = _setup(db, camera, arming=away)
    assert _run(watch, clock, at(7, 30), step=timedelta(minutes=5)) == []   # Away: nothing
    away.mode, away.since = "home", at(7, 30)    # back home at 07:30, nobody seen since
    assert _run(watch, clock, at(10, 59), step=timedelta(minutes=5)) == []
    made = _run(watch, clock, at(11, 0))         # 08:00-11:00 (3 h) had someone every weekday (until 08:40)
    assert len(made) == 1 and "since 07:30" in made[0]["message"]


def test_still_learning_never_fires(db, camera):
    _people(db, camera, _regular(days=10))
    watch, cam, clock = _setup(db, camera, history=False)
    assert _run(watch, clock, at(22, 0), step=timedelta(minutes=10)) == []


def test_day_summaries_keep_the_usual_day_after_people_are_deleted(db, camera):
    _people(db, camera, _regular())
    now = datetime.combine(TODAY, time(10))
    assert store_days(db, ["test-cam"], now) == 56 and store_days(db, ["test-cam"], now) == 0   # once each
    before = learn(db, "test-cam", now)
    db.cleanup(30, {})   # retention: people older than 30 days are deleted
    after = learn(db, "test-cam", now)
    assert after["days"] == before["days"] == {"weekday": 40, "weekend": 16}
    assert after["share"] == before["share"]
    db.cleanup_routine_days(datetime.combine(TODAY, time()))   # without summaries: only what is left of the people
    assert learn(db, "test-cam", now)["total_days"] < 32


def test_health_line(db, camera):
    from labeler.healthcheck import check_routine
    _people(db, camera, _regular())
    routine = Routine(db, copy.deepcopy(DEFAULTS), ["test-cam"], clock=lambda: at(10))
    assert check_routine(routine)["detail"] == ("test-cam: usual on weekdays 07:00-09:00, 12:00-13:00, 18:00-22:00 "
                                                "(from 56 days)")
    learning = Routine(db, copy.deepcopy(DEFAULTS), ["other"], clock=lambda: at(10))
    assert check_routine(learning)["detail"] == "other: still learning (0 of 14 days with people)"
    assert check_routine(None) is None and hour_ranges([23]) == "23:00-00:00"


def test_rule_api_accepts_it_with_home_away_only(db, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    client = TestClient(create_app(cfg, db, None), client=("127.0.0.1", 5000))
    ok = client.post("/api/alerts/rules", json={"name": "Mum's room", "conditions": [["unusual", "yes"], ["armed", "home"]]})
    assert ok.status_code == 200
    bad = client.post("/api/alerts/rules", json={"name": "x", "conditions": [["unusual", "yes"], ["hours", "08-22"]]})
    assert bad.status_code == 400 and "Home / Away" in bad.json()["detail"]


def test_the_measuring_tool_runs():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import unusual_check
    usual, false, normal, bad = unusual_check.run("regular", unusual_check.regular, 3, 7, 1, 0.97, 3)
    assert usual.startswith("07:00-09:00") and normal == 6 and len(bad) == 1 and bad[0][1] is not None

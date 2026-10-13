"""Home / Away (labeler/arming.py): the weekly schedule, button presses, the API and who may use it."""
import copy
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from labeler import accounts  # noqa: E402
from labeler.alerts import AlertEngine, describe, matches  # noqa: E402
from labeler.arming import (AWAY, HOME, Arming, last_change, next_change, parse_entry, parse_schedule,  # noqa: E402
                            scheduled_mode)
from labeler.config import DEFAULTS, load_config  # noqa: E402
from labeler.vehicles import VehicleStore  # noqa: E402
from labeler.web import create_app  # noqa: E402

# 2026-10-12 is a Monday.
MON, TUE, FRI, SAT, SUN = 12, 13, 16, 17, 18


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute)


class Clock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _arming(db, schedule, now):
    clock = Clock(now)
    return Arming(db, {"arming": {"schedule": schedule}}, clock=clock), clock


def test_parse_days_and_times():
    assert parse_entry("mon-fri 08:00-18:00").days == {0, 1, 2, 3, 4}
    assert parse_entry("Sat,Sun 9:00-13:30").days == {5, 6}
    assert parse_entry("fri-mon 22:00-06:00").days == {4, 5, 6, 0}    # a range over the weekend
    assert parse_entry("daily 00:00-24:00").days == set(range(7))
    for bad in ["weekdays 08:00-18:00", "mon 8-18", "mon 25:00-26:00", "mon 08:00-08:00", "mon 00:00-24:00 x",
                "mon 10:60-11:00", "mon 00:00-24:30"]:
        with pytest.raises(ValueError):
            parse_entry(bad)
    with pytest.raises(ValueError):
        parse_schedule("mon-fri 08:00-18:00")    # a list, not one text
    assert parse_schedule(None) == []


def test_schedule_edges_and_midnight():
    work = parse_schedule(["mon-fri 08:00-18:00"])
    assert scheduled_mode(work, at(MON, 7, 59)) == HOME
    assert scheduled_mode(work, at(MON, 8)) == AWAY          # from 08:00 ...
    assert scheduled_mode(work, at(MON, 18)) == HOME         # ... until 18:00
    assert scheduled_mode(work, at(SAT, 12)) == HOME
    night = parse_schedule(["fri 22:00-06:00"])
    assert scheduled_mode(night, at(FRI, 23)) == AWAY
    assert scheduled_mode(night, at(SAT, 5, 59)) == AWAY     # still Friday's night
    assert scheduled_mode(night, at(SAT, 6)) == HOME
    assert scheduled_mode(night, at(SAT, 23)) == HOME
    assert last_change(work, at(MON, 12)) == at(MON, 8)
    assert last_change(work, at(SAT, 12)) == at(FRI, 18)
    assert last_change(work, at(MON, 7)) == at(FRI - 7, 18)   # the Friday before
    assert next_change(work, at(FRI, 19)) == (at(MON + 7, 8), AWAY)
    assert next_change(night, at(FRI, 23)) == (at(SAT, 6), HOME)


def test_back_to_back_stretches_are_one_change():
    weekend = parse_schedule(["sat 00:00-24:00", "sun 00:00-24:00"])
    assert last_change(weekend, at(SUN, 12)) == at(SAT, 0)    # Saturday midnight -> Sunday: no change
    assert next_change(weekend, at(SAT, 12)) == (at(SUN + 1, 0), HOME)
    always = parse_schedule(["daily 00:00-24:00"])
    assert last_change(always, at(MON, 12)) is None and next_change(always, at(MON, 12)) is None


def test_button_only_without_schedule(db):
    arming, _ = _arming(db, [], at(MON, 12))
    assert arming.current() == HOME and db.latest_arming() is None    # Home until someone presses
    assert arming.set(AWAY, "james") is True
    assert arming.set(AWAY, "james") is False                         # already Away: nothing written
    assert arming.current() == AWAY
    assert [(h["mode"], h["by"], h["why"]) for h in db.arming_history()] == [(AWAY, "james", "button")]
    with pytest.raises(ValueError):
        arming.set("holiday", "james")


def test_schedule_switches_and_logs_when_it_was_due(db):
    arming, clock = _arming(db, ["mon-fri 08:00-18:00"], at(MON, 7))
    assert arming.current() == HOME
    assert db.latest_arming() is None          # Home already: nothing to write
    clock.now = at(MON, 9, 30)                 # nobody asked at 08:00 (e.g. the app was off)
    assert arming.current() == AWAY
    last = db.latest_arming()
    assert (last["mode"], last["at"], last["by"], last["why"]) == (AWAY, at(MON, 8), "schedule", "schedule")
    clock.now = at(MON, 18, 1)
    assert arming.current() == HOME


def test_button_holds_until_the_next_scheduled_change(db):
    arming, clock = _arming(db, ["mon-fri 08:00-18:00"], at(MON, 7))
    arming.set(AWAY, "tixuan")                 # leaving early
    clock.now = at(MON, 12)
    assert arming.current() == AWAY
    arming.set(HOME, "james")                  # came home at noon: holds ...
    clock.now = at(MON, 17, 59)
    assert arming.current() == HOME
    clock.now = at(MON, 18)                    # ... the 18:00 change is to Home anyway
    assert arming.current() == HOME
    clock.now = at(MON, 20)
    arming.set(AWAY, "james")                  # out for the evening
    clock.now = at(TUE, 7, 59)
    assert arming.current() == AWAY            # holds over night
    clock.now = at(TUE, 18, 0)
    assert arming.current() == HOME            # the next change wins (08:00 Away, then 18:00 Home)
    state = arming.state()
    assert state["why"] == "schedule" and state["since"] == "2026-10-13T18:00:00"
    assert state["next"] == {"mode": AWAY, "at": "2026-10-14T08:00:00"}


def test_state_survives_a_restart(db):
    arming, _ = _arming(db, ["mon-fri 08:00-18:00"], at(SAT, 10))
    arming.set(AWAY, "james")
    again, _ = _arming(db, ["mon-fri 08:00-18:00"], at(SAT, 22))
    assert again.current() == AWAY             # the press still holds after the restart
    later, _ = _arming(db, ["mon-fri 08:00-18:00"], at(MON + 7, 19))
    assert later.current() == HOME             # two changes (08:00, 18:00) since: the schedule decides


def test_always_away_schedule_starts_away(db):
    arming, _ = _arming(db, ["daily 00:00-24:00"], at(MON, 12))
    assert arming.current() == AWAY
    arming.set(HOME, "james")
    assert arming.current() == HOME            # never changes again by itself


def test_bad_schedule_in_config_stops_the_start(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text('arming:\n  schedule: ["weekdays 08:00-18:00"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="arming.schedule"):
        load_config(cfg)
    cfg.write_text('arming:\n  schedule: ["mon-fri 08:00-18:00"]\n', encoding="utf-8")
    assert load_config(cfg)["arming"]["schedule"] == ["mon-fri 08:00-18:00"]


def _app(db, tmp_path, address="127.0.0.1", arming=None):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, None, arming=arming), client=(address, 5000))


def test_api_and_audit(db, tmp_path):
    arming, _ = _arming(db, ["mon-fri 08:00-18:00"], at(SAT, 10))
    client = _app(db, tmp_path, arming=arming)
    got = client.get("/api/arming").json()
    assert got["mode"] == HOME and got["schedule"] == ["mon-fri 08:00-18:00"]
    assert got["next"] == {"mode": AWAY, "at": "2026-10-19T08:00:00"}
    r = client.post("/api/arming", json={"mode": "away"})
    assert r.status_code == 200 and r.json()["mode"] == AWAY and r.json()["why"] == "button"
    assert client.post("/api/arming", json={"mode": "maybe"}).status_code == 400
    entry = client.get("/api/audit").json()["entries"][0]
    assert (entry["action"], entry["detail"]) == ("home/away", "set to Away")
    client.post("/api/arming", json={"mode": "away"})     # no change: not logged again
    assert [e["action"] for e in client.get("/api/audit").json()["entries"]].count("home/away") == 1


def test_viewers_may_switch(db, tmp_path):
    db.add_user("tixuan", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _app(db, tmp_path, address="192.168.1.20")
    viewer.post("/api/login", json={"username": "tixuan", "password": "correct horse battery"})
    assert viewer.get("/api/arming").json()["mode"] == HOME
    assert viewer.post("/api/arming", json={"mode": "away"}).json()["by"] == "tixuan"
    assert db.latest_arming()["by"] == "tixuan"
    anonymous = _app(db, tmp_path, address="192.168.1.21")
    assert anonymous.post("/api/arming", json={"mode": "home"}).status_code == 401


# --- rules "only when Away / Home" (step 2) ---

def test_armed_condition_matches_and_reads():
    assert matches([["armed", "away"]], {"armed": "away"})
    assert not matches([["armed", "away"]], {"armed": "home"})
    assert describe([["upper_color", "red"], ["armed", "away"]]) == "upper color = red, only when Away"


def test_person_rules_only_when_away(db, camera):
    arming, _ = _arming(db, [], at(MON, 12))
    engine = AlertEngine(db, arming)
    engine.ensure_fall_rule()
    db.add_rule("anyone when away", [["armed", "away"]])
    db.add_rule("anyone at home", [["armed", "home"]])
    db.add_rule("anyone", [])
    engine.invalidate()
    a = db.create_track(camera, 1, datetime.now())
    assert sorted(e["message"] for e in engine.check(a, "test-cam")) == ["anyone at home: only when Home",
                                                                         "anyone: anyone"]
    arming.set(AWAY, "james")
    b = db.create_track(camera, 2, datetime.now())
    assert sorted(e["message"] for e in engine.check(b, "test-cam")) == ["anyone when away: only when Away",
                                                                         "anyone: anyone"]
    assert [e["message"] for e in engine.check(a, "test-cam")] == ["anyone when away: only when Away"]  # later, Away
    assert [e["message"] for e in engine.check(b, "test-cam", fallen=True)] == ["Fall: fell"]   # falls: always


def test_vehicle_rules_only_when_away(db, camera, tmp_path):
    arming, _ = _arming(db, ["daily 00:00-24:00"], at(MON, 12))   # Away all the time
    engine = AlertEngine(db, arming)
    db.add_rule("car at home", [["vehicle", "car"], ["armed", "home"]])
    db.add_rule("car away", [["vehicle", "car"], ["armed", "away"]])
    car = VehicleStore(db, camera, tmp_path).create("car", datetime.now(), 1)
    assert [e["message"] for e in engine.check_vehicle(car, "test-cam")] == ["car away: vehicle = car, only when Away"]


def test_engine_without_arming_is_home(db, camera):
    engine = AlertEngine(db)
    db.add_rule("away", [["armed", "away"]])
    db.add_rule("home", [["armed", "home"]])
    t = db.create_track(camera, 1, datetime.now())
    assert [e["message"] for e in engine.check(t, "test-cam")] == ["home: only when Home"]


def test_pages_switch_the_state_the_alerts_use(db, camera, tmp_path):
    arming, _ = _arming(db, [], at(MON, 12))
    engine = AlertEngine(db, arming)
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    client = TestClient(create_app(cfg, db, None, None, engine), client=("127.0.0.1", 5000))
    r = client.post("/api/alerts/rules", json={"name": "intruder", "conditions": [["armed", "away"]]})
    assert r.status_code == 200
    client.post("/api/arming", json={"mode": "away"})
    t = db.create_track(camera, 1, datetime.now())
    assert [e["message"] for e in engine.check(t, "test-cam")] == ["intruder: only when Away"]


def test_rule_api_checks_armed(db, tmp_path):
    client = _app(db, tmp_path)
    add = lambda conds: client.post("/api/alerts/rules", json={"name": "r", "conditions": conds}).status_code  # noqa: E731
    assert add([["armed", "away"]]) == 200
    assert add([["armed", "home"], ["upper_color", "red"]]) == 200
    assert add([["vehicle", "any"], ["armed", "away"], ["hours", "22-06"]]) == 200
    assert add([["fallen", "yes"], ["armed", "home"]]) == 200
    assert add([["armed", "holiday"]]) == 400

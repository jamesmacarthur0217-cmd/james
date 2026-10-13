"""Loitering alerts (Phase 20): "stayed = gate:120" = someone (or a vehicle) inside area "gate" at least 120 s."""
import copy
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from labeler import zones
from labeler.alerts import AlertEngine, describe, matches, nice_seconds, parse_stayed
from labeler.config import DEFAULTS
from labeler.pipeline import TrackState
from labeler.vehicles import VehicleState, VehicleStore
from labeler.web import create_app
from test_pipeline_logic import make_pipeline

GATE = {"id": 1, "name": "gate", "kind": "area", "points": [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]}
T0 = datetime(2026, 10, 14, 12, 0, 0)


class Clock:
    def __init__(self):
        self.now = T0

    def wall(self):
        return self.now

    def mono(self):
        return (self.now - T0).total_seconds()


def test_parse_match_and_describe():
    assert parse_stayed("gate:120") == ("gate", 120)
    assert parse_stayed("car park: B:30") == ("car park: B", 30)     # the seconds are after the last colon
    for bad in ["gate", "gate:", ":120", "gate:0", "gate:-5", "gate:2m", f"gate:{24 * 3600 + 1}"]:
        with pytest.raises(ValueError):
            parse_stayed(bad)
    rule = [["stayed", "gate:120"]]
    assert not matches(rule, {"stayed": {"gate": 119.9}})
    assert matches(rule, {"stayed": {"gate": 120.0}})
    assert not matches(rule, {"stayed": {"door": 500}})                # another area
    assert not matches(rule, {})                                        # a label change: no stay known
    assert describe([["stayed", "gate:120"], ["hours", "22-06"]]) == "stayed in gate 2 min, between 22:00 and 06:00"
    assert [nice_seconds(s) for s in (30, 60, 90, 3600, 7200)] == ["30 s", "1 min", "90 s", "1 h", "2 h"]


def test_engine_stay_limits_and_once_per_person(db, camera):
    engine = AlertEngine(db)
    db.add_rule("long at gate", [["stayed", "gate:120"]])
    db.add_rule("very long at gate", [["stayed", "gate:600"]])
    db.add_rule("other camera", [["stayed", "door:30"]], camera="elsewhere")
    db.add_rule("car at gate", [["vehicle", "any"], ["stayed", "gate:60"]])
    assert engine.stay_limits("test-cam") == {"gate": 120}
    assert engine.stay_limits("test-cam", vehicles=True) == {"gate": 60}
    t = db.create_track(camera, 1, T0)
    assert engine.check(t, "test-cam") == []                               # labels changed: no stay given
    assert engine.check(t, "test-cam", stayed={"gate": 100}) == []
    assert [e["message"] for e in engine.check(t, "test-cam", stayed={"gate": 130})] == ["long at gate: stayed in gate 2 min"]
    assert engine.check(t, "test-cam", stayed={"gate": 131}) == []        # once per rule and person
    assert [e["message"] for e in engine.check(t, "test-cam", stayed={"gate": 600})] == [
        "very long at gate: stayed in gate 10 min"]


def _pipeline(db, engine):
    p = make_pipeline(db)
    p.alerts, p.clock, p._stay_checked = engine, Clock(), 0.0
    p.zone_counter = zones.ZoneCounter([GATE], min_frames=1)
    return p


def _walk_in(p, st):
    p.zone_counter.update(st.zones, (0.3, 0.3), p.clock.now)            # inside the gate area from now


def test_pipeline_alerts_a_person_staying_in_an_area(db, camera):
    engine = AlertEngine(db)
    db.add_rule("loitering", [["stayed", "gate:120"]])
    p = _pipeline(db, engine)
    st = p.tracks[1] = TrackState(1, T0, 0.0, db_id=db.create_track(camera, 1, T0))
    _walk_in(p, st)
    p.clock.now = T0 + timedelta(seconds=119)
    p._check_stays()
    assert db.list_alerts() == []
    p.clock.now = T0 + timedelta(seconds=121)
    p._check_stays()
    p._check_stays()
    assert [(e["message"], e["track_id"]) for e in db.list_alerts()] == [("loitering: stayed in gate 2 min", st.db_id)]


def test_leaving_the_area_starts_the_time_again(db, camera):
    engine = AlertEngine(db)
    db.add_rule("loitering", [["stayed", "gate:120"]])
    p = _pipeline(db, engine)
    st = p.tracks[1] = TrackState(1, T0, 0.0, db_id=db.create_track(camera, 1, T0))
    _walk_in(p, st)
    p.clock.now = T0 + timedelta(seconds=100)
    p.zone_counter.update(st.zones, (0.9, 0.9), p.clock.now)            # steps out after 100 s ...
    p.clock.now = T0 + timedelta(seconds=110)
    _walk_in(p, st)                                                      # ... and back in
    p.clock.now = T0 + timedelta(seconds=200)                            # 90 s inside this time
    p._check_stays()
    assert db.list_alerts() == []
    p.clock.now = T0 + timedelta(seconds=231)
    p._check_stays()
    assert len(db.list_alerts()) == 1


def test_a_person_without_a_record_or_without_rules_is_not_checked(db, camera):
    engine = AlertEngine(db)
    p = _pipeline(db, engine)
    st = p.tracks[1] = TrackState(1, T0, 0.0)                           # no record yet
    _walk_in(p, st)
    asked = []
    p._check_alerts = lambda *a, **k: asked.append(k)
    p.clock.now = T0 + timedelta(seconds=999)
    p._check_stays()                                                     # no rules: nothing asked
    db.add_rule("loitering", [["stayed", "gate:120"]])
    engine.invalidate()
    p._check_stays()                                                     # no record: nothing asked
    st.db_id = db.create_track(camera, 1, T0)
    p._check_stays()
    assert asked == [{"stayed": {"gate": 999.0}}]


def test_vehicle_standing_at_the_gate(db, camera, tmp_path):
    engine = AlertEngine(db)
    db.add_rule("car at gate", [["vehicle", "car"], ["stayed", "gate:600"]])
    db.add_rule("person at gate", [["stayed", "gate:30"]])               # a person rule: never for vehicles
    p = _pipeline(db, engine)

    class Watch:
        tracks = {}
    p.vehicles = Watch()
    vid = VehicleStore(db, camera, tmp_path).create("car", T0, 1)
    v = Watch.tracks[1] = VehicleState(1, 0.0, T0)
    v.db_id, v.zones = vid, zones.PersonZones()
    _walk_in(p, v)
    p.clock.now = T0 + timedelta(seconds=599)
    p._check_stays()
    assert db.list_alerts() == []
    p.clock.now = T0 + timedelta(minutes=10, seconds=1)
    p._check_stays()
    assert [(e["message"], e["vehicle_id"]) for e in db.list_alerts()] == [
        ("car at gate: vehicle = car, stayed in gate 10 min", vid)]


def test_rule_api_checks_stayed(db):
    db.add_zone("test-cam", "gate", "area", GATE["points"])
    db.add_zone("test-cam", "door", "line", [(0.0, 0.5), (1.0, 0.5)])
    cfg = copy.deepcopy(DEFAULTS)
    client = TestClient(create_app(cfg, db, None, None, AlertEngine(db)), client=("127.0.0.1", 5000))
    add = lambda conds: client.post("/api/alerts/rules", json={"name": "r", "conditions": conds})  # noqa: E731
    assert add([["stayed", "gate:120"]]).status_code == 200
    assert add([["stayed", "gate:120"], ["hours", "22-06"], ["armed", "away"]]).status_code == 200
    assert add([["vehicle", "car"], ["stayed", "gate:600"]]).status_code == 200
    assert "no area named 'door'" in add([["stayed", "door:60"]]).json()["detail"]   # a line has no inside
    assert "no area named 'yard'" in add([["stayed", "yard:60"]]).json()["detail"]
    assert add([["stayed", "gate:soon"]]).status_code == 400
    assert add([["fallen", "yes"], ["stayed", "gate:60"]]).status_code == 400

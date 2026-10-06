"""Alert rules and reports."""
import copy
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from labeler import reports
from labeler.alerts import AlertEngine, matches
from labeler.config import DEFAULTS
from labeler.web import create_app


def _person(db, camera, **labels):
    t = db.create_track(camera, 1, datetime.now())
    db.add_attributes(t, [(n, v, 1.0) for n, v in labels.items()], "vlm", "q", threshold=0.6)
    return t


def test_matching_rules():
    attrs = {"upper_color": "red", "carrying": "backpack,phone"}
    assert matches([["upper_color", "red"], ["carrying", "backpack"]], attrs)
    assert not matches([["upper_color", "blue"]], attrs)
    assert not matches([["carrying", "luggage"]], attrs)
    assert matches([], attrs)                                    # no conditions = anyone


def test_rule_fires_once_per_person_and_respects_camera(db, camera):
    engine = AlertEngine(db)
    db.add_rule("red top", [["upper_color", "red"]])
    db.add_rule("other camera only", [["upper_color", "red"]], camera="entrance")
    red, blue = _person(db, camera, upper_color="red"), _person(db, camera, upper_color="blue")
    assert [e["track_id"] for e in engine.check(red, "test-cam")] == [red]
    assert engine.check(red, "test-cam") == []                   # already fired for this person
    assert engine.check(blue, "test-cam") == []
    assert [e["message"] for e in db.list_alerts()] == ["red top: upper color = red"]


def test_disabled_rules_and_cache_refresh(db, camera):
    engine = AlertEngine(db)
    t = _person(db, camera, upper_color="red")
    assert engine.check(t, "test-cam") == []                     # no rules yet (and cached)
    rule = db.add_rule("anyone", [])
    engine.invalidate()
    db.update_rule(rule["id"], enabled=False)
    engine.invalidate()
    assert engine.check(t, "test-cam") == []
    db.update_rule(rule["id"], enabled=True)
    engine.invalidate()
    assert len(engine.check(t, "test-cam")) == 1


def test_deleting_a_person_or_rule_removes_its_alerts(db, camera):
    engine = AlertEngine(db)
    rule = db.add_rule("anyone", [])
    a, b = _person(db, camera, upper_color="red"), _person(db, camera, upper_color="red")
    engine.check(a, "test-cam")
    engine.check(b, "test-cam")
    db.delete_tracks([a])
    assert [e["track_id"] for e in db.list_alerts()] == [b]
    db.delete_rule(rule["id"])
    assert db.list_alerts() == []


def _row(started, **attrs):
    return {"id": 1, "camera": "cam1", "started_at": started.isoformat(), "ended_at": None, "status": "finished",
            "crop": "1.jpg", "attributes": {n: {"value": v} for n, v in attrs.items()}}


def test_summary_counts_per_hour_and_attributes():
    day = datetime(2026, 10, 7)
    people = [_row(day + timedelta(hours=9, minutes=5), upper_color="red", carrying="backpack,phone"),
              _row(day + timedelta(hours=9, minutes=50), upper_color="red", carrying="none"),
              _row(day + timedelta(hours=14), upper_color="unknown", apparent_gender="woman")]
    s = reports.summarize(people, day, day + timedelta(days=1))
    assert s["unit"] == "hour" and len(s["timeline"]) == 24 and s["total"] == 3
    assert s["timeline"][9]["count"] == 2 and s["timeline"][14]["count"] == 1
    assert s["attributes"]["upper_color"] == [{"value": "red", "count": 2}]          # unknown left out
    assert {r["value"] for r in s["attributes"]["carrying"]} == {"backpack", "phone"}


def test_long_ranges_are_daily():
    day = datetime(2026, 10, 1)
    s = reports.summarize([], day, day + timedelta(days=7))
    assert s["unit"] == "day" and len(s["timeline"]) == 7


def test_csv_has_one_column_per_attribute():
    text = reports.to_csv([_row(datetime(2026, 10, 7, 9), upper_color="red")])
    header, row = text.strip().splitlines()
    assert header.startswith("id,camera,started_at") and "upper_color" in header and "age_group" in header
    assert ",red," in row


def _api(db, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, None, None, AlertEngine(db)))


def test_alert_rule_api(db, tmp_path):
    api = _api(db, tmp_path)
    assert api.post("/api/alerts/rules", json={"name": "x", "conditions": [["upper_color", "crimson"]]}).status_code == 400
    assert api.post("/api/alerts/rules", json={"name": " ", "conditions": []}).status_code == 400
    rule = api.post("/api/alerts/rules", json={"name": "Backpack", "conditions": [["carrying", "backpack"]]}).json()
    assert api.patch(f"/api/alerts/rules/{rule['id']}", json={"enabled": False}).status_code == 200
    assert api.get("/api/alerts/rules").json()[0]["enabled"] is False
    assert api.delete(f"/api/alerts/rules/{rule['id']}").status_code == 200
    assert api.delete(f"/api/alerts/rules/{rule['id']}").status_code == 404
    assert api.get("/api/alerts/events").json() == {"latest_id": 0, "events": []}


def test_report_api_and_csv(db, camera, tmp_path):
    api = _api(db, tmp_path)
    _person(db, camera, upper_color="red")
    today = datetime.now().strftime("%Y-%m-%d")
    s = api.get("/api/reports/summary", params={"start": today, "end": today}).json()
    assert s["total"] == 1 and s["attributes"]["upper_color"][0]["value"] == "red"
    assert api.get("/api/reports/summary", params={"start": "07/10/2026"}).status_code == 400
    r = api.get("/api/reports/export.csv", params={"start": today})
    assert r.headers["content-type"].startswith("text/csv") and r.text.startswith("﻿id,camera")
    assert len(r.text.strip().splitlines()) == 2


def test_answer_key_flow_and_score(db, camera, tmp_path):
    api = _api(db, tmp_path)
    a = _person(db, camera, upper_color="red", apparent_gender="man")
    b = _person(db, camera, upper_color="blue", apparent_gender="unknown")
    for t in (a, b):
        db.update_track(t, status="finished", crop_path=f"{t}.jpg")
    first = api.get("/api/answer-key/next").json()["person"]
    assert first["id"] == b                                           # newest first
    assert api.get("/api/answer-key/next", params={"skip": str(b)}).json()["person"]["id"] == a
    assert api.post(f"/api/answer-key/{b}", json={"labels": {"upper_color": "crimson"}}).status_code == 400
    assert api.post(f"/api/answer-key/{b}", json={"labels": {"upper_color": "red", "apparent_gender": "woman",
                                                             "carrying": "none"}}).status_code == 200
    assert api.post(f"/api/answer-key/{a}", json={"labels": {"upper_color": "red"}}).status_code == 200
    assert api.get("/api/answer-key/next").json()["person"] is None
    s = api.get("/api/answer-key/score").json()
    assert s["people"] == 2
    assert s["attributes"]["upper_color"] == {"n": 2, "answered": 2, "right": 1, "accuracy": 0.5, "overall": 0.5}
    assert s["attributes"]["apparent_gender"]["answered"] == 0        # the system said unknown
    db.delete_tracks([a])
    assert api.get("/api/answer-key/score").json()["people"] == 1

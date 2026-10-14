"""What happened while you were away (Phase 25): periods, the summary, the API and viewers."""
import copy
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from labeler import accounts, summary
from labeler.config import DEFAULTS
from labeler.db import AlertEvent
from labeler.web import create_app

NOW = datetime(2026, 10, 14, 9, 30)


def test_away_period_from_the_home_away_history(db):
    assert summary.away_period(db, NOW) is None                                 # never Away
    db.add_arming("away", datetime(2026, 10, 13, 22, 0), "james", "button")
    assert summary.away_period(db, NOW) == (datetime(2026, 10, 13, 22, 0), NOW)   # still Away: until now
    db.add_arming("home", datetime(2026, 10, 14, 7, 15), "schedule", "schedule")
    assert summary.away_period(db, NOW) == (datetime(2026, 10, 13, 22, 0), datetime(2026, 10, 14, 7, 15))
    db.add_arming("away", datetime(2026, 10, 14, 8, 0), "james", "button")
    assert summary.away_period(db, NOW)[0] == datetime(2026, 10, 14, 8, 0)      # the newest Away stretch


def test_night_and_today():
    assert summary.period(None, "night", NOW) == (datetime(2026, 10, 13, 18), datetime(2026, 10, 14, 8))
    early = datetime(2026, 10, 14, 2, 0)
    assert summary.period(None, "night", early) == (datetime(2026, 10, 13, 18), early)   # not 08:00 yet: until now
    assert summary.period(None, "today", NOW) == (datetime(2026, 10, 14), NOW)
    with pytest.raises(ValueError):
        summary.period(None, "week", NOW)


def _night(db, camera):
    """A night: a person at 23:10, a car at 01:00, a cat at 03:00, a fall alert, the gate opening, a camera problem;
    and things outside the night that must not show."""
    t = datetime(2026, 10, 13, 23, 10)
    person = db.create_track(camera, 1, t)
    db.add_attributes(person, [("apparent_gender", "man", 0.9), ("upper_color", "red", 0.9),
                               ("upper_type", "t-shirt", 0.9)], "vlm", "q", threshold=0.6)
    db.create_track(camera, 2, datetime(2026, 10, 13, 12, 0))                    # the afternoon before: not in it
    car = db.create_vehicle(camera, 1, "car", datetime(2026, 10, 14, 1, 0))
    db.update_vehicle(car, color="white", crop_path="vehicle_1.jpg")
    db.create_animal(camera, 1, "cat", datetime(2026, 10, 14, 3, 0))
    rule = db.add_rule("Fall", [["fallen", "yes"]])["id"]
    alert = db.record_alert(rule, person, "test-cam", "Fall: fell", repeat=True)
    with db.Session.begin() as s:
        s.get(AlertEvent, alert["id"]).created_at = datetime(2026, 10, 13, 23, 20)
    q = db.add_scene_question("test-cam", "Is the gate open?")
    for at, a in [((13, 17), "no"), ((14, 0), "no"), ((14, 2), "unsure"), ((14, 4), "yes"), ((14, 5), "yes")]:
        db.add_scene_answer(q["id"], a, datetime(2026, 10, at[0], at[1], 0))
    with db.Session.begin() as s:
        from labeler.db import CameraEvent
        s.add(CameraEvent(camera="test-cam", kind="offline", message="no picture",
                          started_at=datetime(2026, 10, 14, 2, 30), ended_at=datetime(2026, 10, 14, 2, 40)))
        s.add(CameraEvent(camera="test-cam", kind="offline", message="earlier", started_at=datetime(2026, 10, 12),
                          ended_at=datetime(2026, 10, 12, 1)))
    return person


def test_summary_of_a_night(db, camera):
    person = _night(db, camera)
    got = summary.build(db, datetime(2026, 10, 13, 18), datetime(2026, 10, 14, 8), clip_for=lambda i: f"alert_{i}.webm")
    assert got["counts"] == {"people": 1, "vehicles": 1, "animals": 1, "alerts": 1, "scene_changes": 1,
                             "camera_problems": 1}
    assert got["people"] == [{"id": person, "camera": "test-cam", "at": "2026-10-13T23:10:00",
                              "text": "man, red t-shirt", "picture": None}]
    assert got["vehicle_kinds"] == {"car": 1} and got["animal_kinds"] == {"cat": 1}
    assert got["scene_changes"] == [{"question_id": 1, "question": "Is the gate open?", "camera": "test-cam",
                                     "at": "2026-10-14T04:00:00", "before": "no", "after": "yes"}]
    assert [(x["type"], x["at"][11:16]) for x in got["timeline"]] == [
        ("person", "23:10"), ("alert", "23:20"), ("vehicle", "01:00"), ("camera", "02:30"), ("animal", "03:00"),
        ("scene", "04:00")]
    assert got["alerts"][0]["clip"] == f"alert_{got['alerts'][0]['id']}.webm"
    assert [x["text"] for x in got["timeline"] if x["type"] == "scene"] == ["Is the gate open?: no → yes"]
    viewer = summary.build(db, datetime(2026, 10, 13, 18), datetime(2026, 10, 14, 8), admin=False,
                           clip_for=lambda i: "x")
    assert viewer["camera_problems"] == [] and viewer["alerts"][0]["clip"] is None


def test_first_answer_counts_as_a_change():
    class Db:
        def scene_answer_history(self, start, end):
            return [{"id": 1, "question": "Is the gate open?", "camera": "c", "before": None,
                     "answers": [{"answer": "no", "at": "2026-10-14T01:00:00"}, {"answer": "no", "at": "2026-10-14T01:05:00"}]}]
    assert [(c["before"], c["after"]) for c in summary.scene_changes(Db(), NOW, NOW)] == [(None, "no")]


def _app(db, tmp_path, address="127.0.0.1"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, None), client=(address, 5000))


def test_api_periods_and_viewers(db, camera, tmp_path):
    _night(db, camera)
    client = _app(db, tmp_path)
    assert client.get("/api/summary").json() == {"period": "away", "none": "never switched to Away yet"}
    got = client.get("/api/summary", params={"start": "2026-10-13T18:00", "end": "2026-10-14T08:00"}).json()
    assert got["period"] == "custom" and got["counts"]["people"] == 1 and got["counts"]["camera_problems"] == 1
    assert client.get("/api/summary", params={"start": "yesterday"}).status_code == 400
    assert client.get("/api/summary", params={"start": "2026-10-14T08:00", "end": "2026-10-13T18:00"}).status_code == 400
    assert client.get("/api/summary", params={"start": "2026-08-01T00:00", "end": "2026-10-01T00:00"}).status_code == 400
    assert client.get("/api/summary", params={"period": "week"}).status_code == 400
    assert client.get("/api/summary", params={"period": "today"}).json()["period"] == "today"
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _app(db, tmp_path, address="192.168.1.20")
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    got = viewer.get("/api/summary", params={"start": "2026-10-13T18:00", "end": "2026-10-14T08:00"}).json()
    assert got["counts"]["people"] == 1 and got["camera_problems"] == [] and got["alerts"][0]["clip"] is None


# --- step 2: the page ---

def test_summary_page_for_admins_and_viewers(db, tmp_path):
    from pathlib import Path
    assert "What happened" in _app(db, tmp_path).get("/summary").text
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _app(db, tmp_path, address="192.168.1.20")
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/summary", follow_redirects=False).status_code == 200
    static = Path(__file__).resolve().parent.parent / "labeler" / "static"
    for page in static.glob("*.html"):
        if page.name != "login.html":
            assert 'href="/summary"' in page.read_text(encoding="utf-8"), page.name   # in every page's menu

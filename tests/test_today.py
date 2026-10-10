"""Today at a glance: people today against the same weekday before, the busiest hour, alerts and camera problems."""
import copy
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from labeler import accounts, today
from labeler.config import DEFAULTS
from labeler.web import create_app

NOW = datetime(2026, 10, 10, 15, 30)     # a Saturday afternoon


def _people(db, camera, times):
    for t in times:
        db.create_track(camera, 1, t)


def test_people_today_against_the_same_weekday(db, camera):
    day = NOW.replace(hour=0, minute=0)
    _people(db, camera, [day + timedelta(hours=h) for h in (9, 10, 10, 14)])            # today: 4, busiest 10:00
    week1 = day - timedelta(weeks=1)
    _people(db, camera, [week1 + timedelta(hours=h) for h in (9, 11, 16, 17, 18)])     # 2 by 15:30, 5 all day
    week2 = day - timedelta(weeks=2)
    _people(db, camera, [week2 + timedelta(hours=h) for h in (8, 12, 13, 19)])         # 3 by 15:30, 4 all day
    _people(db, camera, [day - timedelta(days=1, hours=-10)])                          # yesterday: not compared
    s = today.summary(db, now=NOW)
    assert s["people"] == 4 and s["busiest_hour"] == {"hour": 10, "people": 2}
    assert s["usual_from_days"] == 2                       # weeks 3 and 4 had nobody (the app was off): left out
    assert s["usual_by_now"] == 2.5 and s["usual_day"] == 4.5


def test_first_day_has_nothing_to_compare(db, camera):
    s = today.summary(db, now=NOW)
    assert s["people"] == 0 and s["usual_by_now"] is None and s["busiest_hour"] is None


def test_admins_also_see_alerts_and_camera_problems(db, camera, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    db.open_camera_event("cam1", "blank", "the picture is blank")
    app = create_app(cfg, db, None)
    admin = TestClient(app, client=("127.0.0.1", 5000)).get("/api/today").json()
    assert admin["alerts"] == 0 and admin["camera_problems"] == 1 and admin["camera_problems_now"] == 1
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    seen = viewer.get("/api/today").json()
    assert "people" in seen and "alerts" not in seen and "camera_problems" not in seen

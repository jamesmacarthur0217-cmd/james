"""The week (Phase 40 step 1): the last 7 days or a chosen week, one row per day, the listed alerts, recordings and
storage, the week's checks; viewers without problems, storage and checks; the API."""
import copy
from datetime import date, datetime, timedelta

from fastapi.testclient import TestClient

from labeler import accounts, summary
from labeler.config import DEFAULTS
from labeler.db import AlertEvent, CameraEvent, Check
from labeler.web import create_app

NOW = datetime(2026, 10, 18, 15, 0)   # a Sunday afternoon
MON = datetime(2026, 10, 12)


def _alert(db, rule, track, at, message):
    e = db.record_alert(rule, track, "test-cam", message, repeat=True)
    with db.Session.begin() as s:
        s.get(AlertEvent, e["id"]).created_at = at
    return e


def _week(db, camera):
    """People at noon each day (pitfall 18), more on Wednesday; a car on Tuesday; a fall, a help sign, a person rule,
    an unusual quiet; a camera problem and a PC stop; two recordings; two checks."""
    people = {}
    for n in range(7):
        d = MON + timedelta(days=n, hours=12)
        people[n] = [db.create_track(camera, 10 * n + i, d + timedelta(minutes=i)) for i in range(1 + 2 * (n == 2))]
    db.create_track(camera, 99, MON - timedelta(hours=1))   # Sunday before: not in the week
    db.create_vehicle(camera, 1, "car", MON + timedelta(days=1, hours=9))
    fall = db.add_rule("Fall", [["fallen", "yes"]])["id"]
    helped = db.add_rule("Help sign", [["help", "yes"]])["id"]
    red = db.add_rule("Red top", [["upper_color", "red"]])["id"]
    unusual = db.add_rule("Unusual quiet", [["unusual", "yes"], ["armed", "home"]])["id"]
    _alert(db, fall, people[2][0], MON + timedelta(days=2, hours=12, minutes=5), "Fall: fell")
    _alert(db, helped, people[4][0], MON + timedelta(days=4, hours=12, minutes=1), "Help sign")
    _alert(db, red, people[4][0], MON + timedelta(days=4, hours=12, minutes=2), "Red top")
    _alert(db, unusual, None, MON + timedelta(days=5, hours=9), "Unusual quiet: nobody seen since 00:00")
    with db.Session.begin() as s:
        s.add(CameraEvent(camera="test-cam", kind="blank", message="blank", started_at=MON + timedelta(days=1, hours=3),
                          ended_at=MON + timedelta(days=1, hours=4)))
        s.add(CameraEvent(camera="PC", kind="stopped", message="the PC was turned off",
                          started_at=MON + timedelta(days=3, hours=23), ended_at=MON + timedelta(days=4, hours=1)))
    r1 = db.add_recording("test-cam", "a.mp4", datetime(2026, 10, 17, 10), "x")
    db.update_recording(r1, ended_at=datetime(2026, 10, 17, 11, 30), size_bytes=2_000_000_000)
    r2 = db.add_recording("test-cam", "b.mp4", datetime(2026, 10, 18, 14), "x")   # still open: counts until now
    db.save_check("alert", 1, "fall", "test-cam", "yes", None, "admin")
    db.save_check("vehicle", 1, "car", "test-cam", "no", "van", "admin")
    with db.Session.begin() as s:
        for c in s.query(Check):
            c.created_at = MON + timedelta(days=5)
    return people, r2


def test_week_rows_and_lists(db, camera):
    _week(db, camera)
    s, e = summary.week_period(NOW, date(2026, 10, 15))   # any day of that week: Monday to Sunday
    assert (s, e) == (MON, NOW)
    got = summary.week(db, s, e, cfg=copy.deepcopy(DEFAULTS), now=NOW)
    days = {d["day"]: d for d in got["days"]}
    assert list(days) == [f"2026-10-{n}" for n in range(12, 19)]
    assert [d["people"] for d in got["days"]] == [1, 1, 3, 1, 1, 1, 1] and got["counts"]["people"] == 9
    assert days["2026-10-13"]["vehicles"] == 1 and days["2026-10-13"]["camera_problems"] == 1
    assert days["2026-10-15"]["pc_problems"] == 1 and days["2026-10-16"]["pc_problems"] == 1   # across midnight
    assert days["2026-10-14"]["alert_kinds"] == {"fall": 1}
    assert days["2026-10-16"]["alert_kinds"] == {"help": 1, "person": 1}
    assert got["alert_kinds"] == {"fall": 1, "help": 1, "person": 1, "unusual": 1}
    assert [a["message"] for a in got["listed"]] == ["Unusual quiet: nobody seen since 00:00", "Help sign", "Fall: fell"]
    # Recordings: kept 3 days, so earlier days are "not kept"; the open one counts until now.
    assert days["2026-10-14"]["recorded_h"] is None and days["2026-10-15"]["recorded_h"] == 0.0
    assert days["2026-10-17"]["recorded_h"] == 1.5 and days["2026-10-18"]["recorded_h"] == 1.0
    assert got["storage"]["recordings_gb"] == 2.0 and got["storage"]["keep_days"] == 3
    assert got["checks"]["count"] == 2
    assert {r["what"]: (r["right"], r["wrong"]) for r in got["checks"]["results"]} == {"fall alerts": (1, 0),
                                                                                      "vehicles": (0, 1)}
    assert "timeline_all" not in got and "_vehicles" not in got


def test_the_last_seven_days_and_viewers(db, camera):
    _week(db, camera)
    s, e = summary.week_period(NOW)
    assert (s, e) == (datetime(2026, 10, 12), NOW)   # today so far and the 6 days before
    viewer = summary.week(db, s, e, admin=False, cfg=copy.deepcopy(DEFAULTS), now=NOW)
    assert "camera_problems" not in viewer["days"][1] and "recorded_h" not in viewer["days"][1]
    assert "checks" not in viewer and "storage" not in viewer and viewer["camera_problems"] == []


def _client(db, tmp_path, address="127.0.0.1"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, None), client=(address, 5000))


def test_api(db, camera, tmp_path):
    _week(db, camera)
    admin = _client(db, tmp_path)
    got = admin.get("/api/summary", params={"period": "week", "week": "2026-10-14"}).json()
    assert got["period"] == "week" and got["start"] == "2026-10-12T00:00:00" and len(got["days"]) == 7
    assert got["end"] == "2026-10-19T00:00:00"   # a past week: all of it
    assert admin.get("/api/summary", params={"period": "week", "week": "14-10-2026"}).status_code == 400
    assert admin.get("/api/summary", params={"period": "week", "week": "2099-01-05"}).status_code == 400
    last = admin.get("/api/summary", params={"period": "week"}).json()
    assert len(last["days"]) == 7 and last["days"][-1]["day"] == datetime.now().date().isoformat()
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _client(db, tmp_path, address="192.168.1.20")
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    v = viewer.get("/api/summary", params={"period": "week", "week": "2026-10-14"}).json()
    assert "storage" not in v and v["counts"]["people"] == 9

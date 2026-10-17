"""First-run check (Phase 36 step 2): the server's checks, "Try the help sign" (no alert), the page and API."""
import copy
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import text

from labeler import accounts, firstrun
from labeler.alerts import AlertEngine
from labeler.arming import Arming
from labeler.config import DEFAULTS
from labeler.db import Database
from labeler.gestures import GestureWatch
from labeler.pipeline import TrackState
from labeler.web import create_app
from test_pipeline_logic import make_pipeline

NOON = datetime(2026, 10, 18, 12, 0)


def _by_key(result):
    return {i["key"]: i for i in result["items"]}


def test_database_line(tmp_path):
    url = "sqlite:///" + (tmp_path / "t.db").as_posix()
    first = Database(url)
    assert firstrun.check_database(first)["detail"] == "a new database, made at this start"
    with first.engine.begin() as conn:
        conn.execute(text("ALTER TABLE zones DROP COLUMN lying_ok"))
    gone = firstrun.check_database(first)   # a column missing now (the start did not add it)
    assert gone["ok"] is False and "zones.lying_ok" in gone["detail"] and gone["todo"]
    first.engine.dispose()
    again = Database(url)
    line = firstrun.check_database(again)
    assert line["ok"] is True and line["detail"] == "updated at this start: 1 new column"
    again.engine.dispose()
    assert firstrun.check_database(Database(url))["detail"].startswith("up to date")


def test_built_in_rules(db):
    cfg = copy.deepcopy(DEFAULTS)
    lines = {i["key"]: i for i in firstrun.check_rules(cfg, db)}
    assert [lines[k]["ok"] for k in ("rule_falls", "rule_help_sign", "rule_quiet")] == [False, False, False]
    engine = AlertEngine(db)
    engine.ensure_fall_rule(), engine.ensure_help_rule(), engine.ensure_quiet_rule()
    lines = {i["key"]: i for i in firstrun.check_rules(cfg, db)}
    assert all(lines[k]["ok"] is True for k in lines)
    help_rule = next(r for r in db.list_rules() if r["name"] == "Help sign")
    db.update_rule(help_rule["id"], False)
    cfg["quiet"]["enabled"] = False
    lines = {i["key"]: i for i in firstrun.check_rules(cfg, db)}
    assert lines["rule_help_sign"]["ok"] is None and "Alerts page" in lines["rule_help_sign"]["detail"]
    assert lines["rule_quiet"]["ok"] is None and "Settings" in lines["rule_quiet"]["detail"]


def test_recording_today(db):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["recording"]["enabled"] = True   # off by default; config.yaml turns it on
    p = make_pipeline(db)
    line = firstrun.check_recording(cfg, db, [p], NOON)
    assert line["ok"] is False and "Stand in front" in line["todo"]
    old = db.add_recording(p.name, "x.mp4", NOON - timedelta(days=1), "test")   # yesterday's does not count
    db.update_recording(old, ended_at=NOON - timedelta(days=1) + timedelta(minutes=10))
    rec = db.add_recording(p.name, "y.mp4", NOON - timedelta(hours=2), "test")
    db.update_recording(rec, ended_at=NOON - timedelta(hours=1, minutes=50))
    line = firstrun.check_recording(cfg, db, [p], NOON)
    assert line["ok"] is True and "until 10:10" in line["detail"]
    db.add_recording(p.name, "z.mp4", NOON - timedelta(minutes=5), "test")   # still open
    assert firstrun.check_recording(cfg, db, [p], NOON)["detail"] == "2 files today, recording now"
    cfg["recording"]["enabled"] = False
    assert firstrun.check_recording(cfg, db, [p], NOON)["ok"] is None


def test_arming_line(db):
    line = firstrun.check_arming(Arming(db, copy.deepcopy(DEFAULTS)))
    assert line["ok"] is None and line["detail"].startswith("Home")


def test_help_test_reports_the_sign_without_an_alert(db, camera):
    p = make_pipeline(db)
    p.gestures = GestureWatch()
    asked = []

    class Alerts:
        def check(self, *a, **k):
            asked.append(k)
            return []

    p.alerts = Alerts()
    st = TrackState(3, datetime.now(), 0.0, db_id=db.create_track(camera, 3, datetime.now()))
    p.tracks[3] = st
    assert p.help_test_state() == {"active": False, "seen": None}
    p.start_help_test(30)
    p._on_help(3, {"since": 0.0, "hand": "right"})
    state = p.help_test_state()
    assert asked == [] and state["active"] and state["seen"]["hand"] == "right" and state["seen"]["person"] == st.db_id
    p.help_test["until"] = 0   # the 30 s are over: signs make alerts again
    p._on_help(3)
    assert len(asked) == 1 and asked[0]["helped"] is True
    assert p.help_test_state()["active"] is False


def test_gesture_watch_says_how_long_a_hand_is_up():
    from test_gestures import LEFT_UP, person
    w = GestureWatch()
    assert w.holding(0.0) == 0.0
    for i in range(16):
        w.update(1, i * 0.1, LEFT_UP)
    assert abs(w.holding(1.5) - 1.5) < 1e-6
    fired = [round(i * 0.1, 1) for i in range(16, 80) if w.update(1, i * 0.1, LEFT_UP)]
    assert fired == [3.0]   # confirmed once, however long the hand stays up
    w.rearm(8.0)   # "Try the help sign" pressed with the hand still up: it counts again after hold_s
    fired = [round(i * 0.1, 1) for i in range(80, 120) if w.update(1, i * 0.1, LEFT_UP)]
    assert fired == [11.0]
    w.update(1, 12.5, person())
    assert w.holding(12.5) == 0.0   # the hand came down


def _client(db, tmp_path, pipelines, address="127.0.0.1"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, pipelines), client=(address, 5000))


def test_api_and_page(db, tmp_path):
    p = make_pipeline(db)
    p.cam, p.gestures, p.help_test = {"source": 0}, GestureWatch(), None
    client = _client(db, tmp_path, [p])
    got = client.get("/api/first-run").json()
    assert got["cameras"] == [{"name": "test-cam", "mode": "kiosk", "help_sign": True}]
    assert {"database", "rule_falls", "rule_help_sign", "rule_quiet", "recording", "arming"} <= set(_by_key(got))
    assert client.get("/first-run").status_code == 200
    started = client.post("/api/first-run/help-test", json={"camera": "test-cam"}).json()
    assert started["active"] and started["left_s"] == 30 and started["hold_s"] == 3.0
    assert client.get("/api/first-run/help-test", params={"camera": "test-cam"}).json()["active"]
    assert client.post("/api/first-run/help-test", json={"camera": "nope"}).status_code == 404
    assert any(e["action"] == "try help sign" for e in db.list_audit(limit=10))
    p.gestures = None
    assert client.post("/api/first-run/help-test", json={"camera": "test-cam"}).status_code == 409
    # Viewers: not their page.
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _client(db, tmp_path, [p], address="192.168.1.20")
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/first-run").status_code == 403
    assert viewer.post("/api/first-run/help-test", json={"camera": "test-cam"}).status_code == 403
    assert viewer.get("/first-run", follow_redirects=False).status_code in (302, 303)


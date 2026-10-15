"""No movement (Phase 31): labeler/quiet.py with a clock, the built-in rule, alerts listed and kept like others."""
import copy
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from labeler.alerts import QUIET_RULE, AlertEngine, describe, rule_kind
from labeler.arming import Arming
from labeler.config import DEFAULTS
from labeler.quiet import QuietWatch
from labeler.web import create_app

DAY = datetime(2026, 10, 15)


class Clock:
    def __init__(self, at):
        self.now = at

    def __call__(self):
        return self.now

    def step(self, **kw):
        self.now += timedelta(**kw)


class Cam:
    def __init__(self, name="cam1", mode="kiosk"):
        self.name, self.mode, self.running, self.person_seen_wall = name, mode, True, None


class Health:
    def __init__(self):
        self.problems = []

    def current(self):
        return [{"camera": c} for c in self.problems]


def _setup(db, at, cams=None):
    clock = Clock(at)
    arming = Arming(db, copy.deepcopy(DEFAULTS), clock=clock)
    alerts = AlertEngine(db, arming)
    alerts.ensure_quiet_rule()
    health = Health()
    cams = cams or [Cam()]
    return QuietWatch(db, alerts, cams, arming, health, clock=clock), clock, cams, health, arming


def _messages(db):
    return [e["message"] for e in db.list_alerts()]


def test_the_built_in_rule():
    assert rule_kind(QUIET_RULE) == "quiet"
    assert describe(QUIET_RULE) == "nobody seen for 4 h, between 08:00 and 22:00, only when Home"


def test_four_hours_quiet_in_the_day_alerts_once_and_ends_when_someone_is_seen(db):
    watch, clock, (cam,), _, _ = _setup(db, DAY.replace(hour=7))
    assert len([r for r in db.list_rules() if r["name"] == "No movement"]) == 1
    watch.alerts.ensure_quiet_rule()                                          # not twice
    assert len(db.list_rules()) == 1
    watch.tick()                                                              # the camera works from 07:00
    clock.step(hours=4, minutes=59)                                           # 11:59: counted from 08:00 only
    assert watch.tick() == []
    clock.step(minutes=1)                                                     # 12:00
    made = watch.tick()
    assert len(made) == 1 and made[0]["track_id"] is None
    assert _messages(db) == ["No movement: nobody seen on cam1 for 4 h (since 08:00)"]
    clock.step(minutes=30)
    assert watch.tick() == []                                                 # one alert per quiet stretch
    cam.person_seen_wall = clock.now                                          # 12:30: someone seen
    watch.tick()
    assert _messages(db) == ["No movement: nobody seen on cam1 for 4 h (since 08:00); seen again at 12:30, after 4 h 30 min"]
    clock.step(hours=4)
    assert len(watch.tick()) == 1                                             # quiet again 4 h: a new alert
    assert db.list_alerts()[0]["quiet"] is True and db.list_alerts()[0]["message"].endswith("(since 12:30)")


def test_not_at_night_not_when_away_and_a_new_day_starts_afresh(db):
    watch, clock, (cam,), _, arming = _setup(db, DAY.replace(hour=20))
    cam.person_seen_wall = DAY.replace(hour=19)
    watch.tick()
    clock.step(hours=5)                                                       # 01:00: outside 08-22
    assert watch.tick() == []
    clock.step(hours=10, minutes=59)                                          # 11:59: counted from 08:00
    assert watch.tick() == []
    arming.set("away", "test")                                                # the family went out
    clock.step(hours=2)
    assert watch.tick() == []
    arming.set("home", "test")                                                # 13:59 back home: counts from now
    clock.step(hours=3, minutes=59)
    assert watch.tick() == []
    clock.step(minutes=1)
    assert len(watch.tick()) == 1
    clock.step(hours=12)                                                      # the next day, nobody seen at all
    watch.tick()
    assert len(_messages(db)) == 1
    clock.step(hours=6, minutes=1)                                            # 12:00 the next day: a new stretch
    assert len(watch.tick()) == 1 and len(_messages(db)) == 2


def test_camera_problems_and_security_cameras(db):
    gate = Cam("gate", "security")
    watch, clock, (cam, _), health, _ = _setup(db, DAY.replace(hour=8), [Cam(), gate])
    watch.tick()
    clock.step(hours=2)
    health.problems = ["cam1"]                                                # lens covered: its own problem
    watch.tick()
    clock.step(hours=3)
    assert watch.tick() == []
    health.problems = []                                                      # 13:00: working again
    watch.tick()
    clock.step(hours=3, minutes=59)
    assert watch.tick() == []                                                 # counts from the problem's end
    clock.step(minutes=1)
    made = watch.tick()
    assert [e["camera"] for e in made] == ["cam1"]                            # never the gate (security camera)
    cam.running = False
    clock.step(hours=5)
    assert watch.tick() == []                                                 # not running: not watched


def test_alerts_about_nobody_are_listed_and_kept_like_the_cameras_people(db, camera):
    watch, clock, _, _, _ = _setup(db, DAY.replace(hour=8), [Cam("test-cam")])
    watch.tick()
    clock.step(hours=4)
    made = watch.tick()
    assert [e["id"] for e in db.list_alerts()] == [made[0]["id"]]
    from labeler.db import AlertEvent
    with db.Session.begin() as s:
        s.get(AlertEvent, made[0]["id"]).created_at = datetime.now() - timedelta(days=40)
    db.cleanup(30)
    assert db.list_alerts() == []


def test_api_checks_no_movement_rules(db, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    client = TestClient(create_app(cfg, db, None), client=("127.0.0.1", 5000))
    ok = client.post("/api/alerts/rules", json={"name": "Quiet 2 h", "conditions": [["quiet", "120"], ["hours", "08-22"]]})
    assert ok.status_code == 200, ok.text
    for bad in ([["quiet", "5"]], [["quiet", "x"]], [["quiet", "120"], ["upper_color", "red"]]):
        assert client.post("/api/alerts/rules", json={"name": "bad", "conditions": bad}).status_code == 400


# --- step 2: Health line, pages, Settings ---

def test_health_line_says_how_long_it_has_been_quiet(db):
    from labeler.healthcheck import check_quiet
    watch, clock, (cam, _gate), health, arming = _setup(db, DAY.replace(hour=9), [Cam(), Cam("gate", "security")])
    assert check_quiet(None) is None
    watch.tick()
    clock.step(hours=1, minutes=20)
    line = check_quiet(watch)
    assert line["level"] == "ok" and line["detail"] == "cam1: nobody seen for 1 h 20 min (alert after 4 h)"
    clock.step(hours=3)
    watch.tick()
    assert check_quiet(watch)["level"] == "warning" and check_quiet(watch)["detail"].endswith(", alerted")
    arming.set("away", "test")
    assert check_quiet(watch)["detail"] == "cam1: not counted now (only when Home)"
    health.problems = ["cam1"]
    assert "camera problem" in check_quiet(watch)["detail"]
    for r in db.list_rules():
        db.delete_rule(r["id"])
    watch.alerts.invalidate()
    assert check_quiet(watch)["level"] == "off"


def test_pages_show_no_movement():
    from pathlib import Path
    static = Path(__file__).resolve().parent.parent / "labeler" / "static"
    js = (static / "session.js").read_text(encoding="utf-8")
    assert 'e.quiet ? "no movement"' in js and 'className = "toast quiet"' in js
    alerts = (static / "alerts.html").read_text(encoding="utf-8")
    assert 'quiet: ["60", "120", "180", "240"' in alerts and "nobody seen for ${niceSeconds(v * 60)}" in alerts
    assert 'id="quiet_enabled"' in (static / "settings.html").read_text(encoding="utf-8")


def test_health_line_names_the_rule_when_there_are_several(db):
    from labeler.healthcheck import check_quiet
    watch, clock, _, _, _ = _setup(db, DAY.replace(hour=9))
    db.add_rule("Quiet 1 h", [["quiet", "60"]])
    watch.alerts.invalidate()
    watch.tick()
    clock.step(minutes=5)
    assert check_quiet(watch)["detail"] == ("cam1 (No movement): nobody seen for 5 min (alert after 4 h); "
                                            "cam1 (Quiet 1 h): nobody seen for 5 min (alert after 1 h)")

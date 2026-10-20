"""Safety incidents as alerts (Phase 44 step 3): one built-in rule per kind, a "yes" from the vision model becomes an
alert (picture, blurred copy, clip) for every enabled rule of that kind that matches (camera, area, hours, Home /
Away); "no" and "unsure" alert nothing; the pages, the Health line, the Check page and the week's summary know them."""
import copy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from labeler import checks, privacy, summary
from labeler.alerts import INCIDENT, AlertEngine, describe, rule_kind
from labeler.config import DEFAULTS
from labeler.healthcheck import check_safety
from labeler.incidents import CATEGORIES, ConfirmQueue, FrameMemory, IncidentAlerts, Suspect
from labeler.web import create_app

T0 = datetime(2026, 10, 19, 12, 0)
KITCHEN = {"id": 3, "name": "kitchen", "points": [(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)]}


class Clips:
    def __init__(self):
        self.asked = []

    def request(self, event_id):
        self.asked.append(event_id)


class Cam:
    def __init__(self, name="cam1"):
        self.name, self.clips = name, Clips()


class Arming:
    def __init__(self, mode="home"):
        self.mode = mode

    def current(self):
        return self.mode


def _suspect(category="fire", area=None, camera="cam1"):
    pic = np.full((540, 960, 3), 120, np.uint8)
    return Suspect(camera, category, "a bright patch growing", area, [], [(T0, pic, [(10, 10, 60, 60)])], at=T0, number=4)


def _setup(db, tmp_path, arming=None):
    engine = AlertEngine(db, arming)
    engine.ensure_incident_rules()
    cam = Cam()
    return engine, cam, IncidentAlerts(db, engine, [cam], tmp_path)


def test_one_built_in_rule_per_kind(db):
    engine = AlertEngine(db)
    engine.ensure_incident_rules(), engine.ensure_incident_rules()
    rules = [r for r in db.list_rules() if rule_kind(r["conditions"]) == "incident"]
    assert len(rules) == len(CATEGORIES) == 12
    fire = next(r for r in rules if r["conditions"] == [[INCIDENT, "fire"]])
    assert fire["name"] == "Fire or smoke" and describe(fire["conditions"]) == "fire or smoke (checked by the vision model)"
    db.update_rule(fire["id"], False)          # turned off: stays off
    db.delete_rule(next(r["id"] for r in rules if r["conditions"] == [[INCIDENT, "crowd"]]))   # deleted: comes back
    engine.ensure_incident_rules()
    rules = [r for r in db.list_rules() if rule_kind(r["conditions"]) == "incident"]
    assert len(rules) == 12 and not next(r for r in rules if r["conditions"] == [[INCIDENT, "fire"]])["enabled"]


def test_a_yes_becomes_an_alert_with_picture_and_clip(db, tmp_path):
    engine, cam, make = _setup(db, tmp_path)
    assert make(_suspect(), {"answer": "no", "what": "a stove"}) == []
    assert make(_suspect(), {"answer": "unsure", "what": "too dark"}) == []
    [event] = make(_suspect(), {"answer": "yes", "what": "flames on the sofa"})
    assert event["category"] == "fire" and event["level"] == "critical" and event["confirm"] == "flames on the sofa"
    assert event["message"] == "Fire or smoke: flames on the sofa; seen first: a bright patch growing"
    assert (tmp_path / event["crop"]).is_file() and privacy.blurred_path(tmp_path, event["crop"]).is_file()
    assert cam.clips.asked == [event["id"]]
    listed = db.list_alerts()[0]
    assert (listed["category"], listed["level"], listed["words"], listed["crop"]) == \
        ("fire", "critical", "fire or smoke", event["crop"])
    assert checks.alert_what(listed, [[INCIDENT, "fire"]]) == "safety: fire or smoke"
    assert checks.label({"kind": "alert", "what": "safety: fire or smoke"}) == "fire or smoke alerts"
    assert summary.alert_kind(listed) == "incident" and "incident" in summary.LISTED


def test_rules_limit_where_and_when(db, tmp_path):
    arming = Arming("home")
    engine, cam, make = _setup(db, tmp_path, arming)
    db.add_rule("Knife in the kitchen at night", [[INCIDENT, "weapon"], ["zone", "kitchen"], ["armed", "away"]])
    engine.invalidate()
    weapon = next(r for r in db.list_rules() if r["conditions"] == [[INCIDENT, "weapon"]])
    db.update_rule(weapon["id"], False)   # the built-in one off: only the limited rule is left
    engine.invalidate()
    assert make(_suspect("weapon", KITCHEN), {"answer": "yes", "what": "a raised knife"}) == []   # Home
    arming.mode = "away"
    assert make(_suspect("weapon"), {"answer": "yes", "what": "a raised knife"}) == []           # not in the kitchen
    [event] = make(_suspect("weapon", KITCHEN), {"answer": "yes", "what": "a raised knife"})
    assert event["message"].startswith("Knife in the kitchen at night: a raised knife (in kitchen)")
    other = Suspect("cam2", "weapon", "x", KITCHEN, [], _suspect().pictures, at=T0, number=5)
    db.add_rule("cam1 only", [[INCIDENT, "floor"]], "cam1")
    engine.invalidate()
    floor_rules = engine.check_incident("cam2", "floor", None)
    assert [r["name"] for r in floor_rules] == ["Floor hazard"]   # the built-in one; not the cam1 rule
    assert len(make(other, {"answer": "yes", "what": "knife"})) == 1   # no camera on the rule: any camera


def test_the_queue_and_the_alerts_together(db, tmp_path):
    engine, cam, make = _setup(db, tmp_path)
    cfg = copy.deepcopy(DEFAULTS)
    mem = FrameMemory()
    for i in range(5):
        mem.add(np.zeros((540, 960, 3), np.uint8), T0 + timedelta(seconds=i), float(i))

    class Worker:
        def submit(self, job, frames, mode=None, priority=0):
            self.job = job
            return True

    worker = Worker()
    q = ConfirmQueue(cfg, worker, {"cam1": mem}, clock=lambda: 10.0)
    q.listeners.append(make)
    q.suspect("cam1", "person_down", "lying still 20 s")
    q.tick()
    q.on_result(worker.job, [{"answer": "yes", "what": "a woman on the floor", "seconds": "1.3"}])
    alert = db.list_alerts()[0]
    assert alert["category"] == "person_down" and alert["confirm"] == "a woman on the floor"
    line = check_safety(q)
    assert line["name"] == "safety checks" and line["level"] == "ok"
    assert line["detail"].startswith("on cam1: 1 checked since the start (1 yes, 0 no, 0 unsure)")
    assert "person down yes (a woman on the floor)" in line["detail"]
    q.counts["dropped_old"] = 2
    assert check_safety(q)["level"] == "warning" and check_safety(None) is None


def _client(db, tmp_path, queue=None):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    app = create_app(cfg, db, None)
    if queue is not None:
        app.state.incidents = queue
    return TestClient(app, client=("127.0.0.1", 5000))


def test_api_and_rules(db, tmp_path):
    client = _client(db, tmp_path)
    got = client.get("/api/safety").json()
    assert got["on"] is False and got["state"] is None and len(got["kinds"]) == 12
    assert {"category": "fire", "level": "critical", "words": "fire or smoke"} in got["kinds"]
    ok = client.post("/api/alerts/rules", json={"name": "Smoke at night", "conditions": [["incident", "fire"],
                                                                                          ["hours", "22-06"]]})
    assert ok.status_code == 200
    for bad in ([["incident", "fire"], ["upper_color", "red"]], [["incident", "fire"], ["incident", "fight"]],
                [["incident", "aliens"]]):
        assert client.post("/api/alerts/rules", json={"name": "x", "conditions": bad}).status_code == 400
    q = ConfirmQueue(copy.deepcopy(DEFAULTS), None, {"cam1": FrameMemory()})
    assert _client(db, tmp_path, q).get("/api/safety").json()["state"]["cameras"] == ["cam1"]


def test_the_pages():
    static = Path(__file__).resolve().parent.parent / "labeler" / "static"
    js = (static / "session.js").read_text(encoding="utf-8")
    assert 'e.level === "critical"' in js and "toast fall incident" in js
    alerts = (static / "alerts.html").read_text(encoding="utf-8")
    for part in ['id="ev-filter"', 'fetch("/api/safety")', 'n === "incident"', "crit"]:
        assert part in alerts, part
    settings = (static / "settings.html").read_text(encoding="utf-8")
    for part in ['id="safety_enabled"', 'id="safety_confirm_per_min"', 'class="saf"']:
        assert part in settings, part

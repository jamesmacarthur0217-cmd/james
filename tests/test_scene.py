"""Scene questions (Phase 22): yes/no questions the vision model answers about a camera's whole picture."""
import copy
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from labeler import accounts
from labeler.config import DEFAULTS
from labeler.web import create_app


def _app(db, tmp_path, address="127.0.0.1"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cfg["cameras"] = [{"name": "gate", "source": 0, "mode": "security"}]
    return TestClient(create_app(cfg, db, None), client=(address, 5000))


# --- step 1: storage and the Alerts page ---

def test_questions_are_stored_and_listed(db):
    q = db.add_scene_question("gate", "Is the gate open?", 300, "yes")
    assert (q["camera"], q["question"], q["every_s"], q["alert_on"], q["enabled"], q["last_answer"]) == \
        ("gate", "Is the gate open?", 300, "yes", True, None)
    db.update_scene_question(q["id"], last_answer="no", last_at=datetime(2026, 10, 14, 14, 5))
    db.add_scene_answer(q["id"], "no", datetime(2026, 10, 14, 14, 5), 1.4)
    db.add_scene_answer(q["id"], "yes", datetime(2026, 10, 14, 14, 10), 1.2)
    assert [a["answer"] for a in db.scene_answers(q["id"])] == ["yes", "no"]          # newest first
    assert db.list_scene_questions()[0]["last_answer"] == "no"
    assert db.delete_scene_question(q["id"])["question"] == "Is the gate open?"
    assert db.list_scene_questions() == [] and db.scene_answers(q["id"]) == []
    assert db.delete_scene_question(q["id"]) is None


def test_old_answers_are_cleaned_up(db):
    q = db.add_scene_question("gate", "Is the gate open?")
    db.add_scene_answer(q["id"], "no", datetime.now() - timedelta(days=400))
    db.add_scene_answer(q["id"], "yes", datetime.now())
    assert db.cleanup_scene_answers(365) == 1 and len(db.scene_answers(q["id"])) == 1
    assert db.cleanup_scene_answers(0) == 0                                            # 0 = keep


def test_api_adds_checks_changes_and_deletes(db, tmp_path):
    client = _app(db, tmp_path)
    r = client.post("/api/scene", json={"camera": "gate", "question": "  Is the   gate open? ", "every_min": 5})
    assert r.status_code == 200 and r.json()["question"] == "Is the gate open?" and r.json()["every_s"] == 300
    qid = r.json()["id"]
    bad = [{"camera": "door", "question": "Is the gate open?"},                         # no such camera
           {"camera": "gate", "question": "ok?"},                                       # too short
           {"camera": "gate", "question": "Is the gate open?", "every_min": 0},
           {"camera": "gate", "question": "Is the gate open?", "every_min": 61},
           {"camera": "gate", "question": "Is the gate open?", "alert_on": "maybe"}]
    assert [client.post("/api/scene", json=b).status_code for b in bad] == [400] * 5
    r = client.patch(f"/api/scene/{qid}", json={"enabled": False, "every_min": 15})
    assert r.json()["enabled"] is False and r.json()["every_s"] == 900
    assert client.patch("/api/scene/999", json={"enabled": True}).status_code == 404
    assert [q["id"] for q in client.get("/api/scene").json()] == [qid]
    assert client.delete(f"/api/scene/{qid}").json() == {"ok": True}
    assert client.get("/api/scene").json() == []
    actions = [e["action"] for e in client.get("/api/audit").json()["entries"]]
    assert actions[:3] == ["scene question deleted", "scene question changed", "scene question added"]


def test_viewers_cannot_see_or_write_questions(db, tmp_path):
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _app(db, tmp_path, address="192.168.1.20")
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/scene").status_code == 403
    assert viewer.post("/api/scene", json={"camera": "gate", "question": "Is the gate open?"}).status_code == 403


# --- step 2: asking the vision model ---

import numpy as np  # noqa: E402

from labeler.scene import SCENE_BASE, SceneRunner, job_id, question_of  # noqa: E402
from labeler.vlm import VLMClient, clean_scene, scene_schema  # noqa: E402

T0 = datetime(2026, 10, 14, 12, 0, 0)


class Clock:
    def __init__(self):
        self.now, self.mono_t = T0, 0.0

    def __call__(self):
        return self.now

    def mono(self):
        return self.mono_t

    def step(self, seconds):
        self.now += timedelta(seconds=seconds)
        self.mono_t += seconds


class Worker:
    def __init__(self):
        self.idle, self.jobs = True, []

    def submit(self, job, frames, mode):
        self.jobs.append((job, frames, mode))
        return True


class Cam:
    def __init__(self, name, clock):
        self.name, self.clock = name, clock
        self.scene_frame = (np.full((40, 60, 3), 90, np.uint8), clock.now, [(5, 5, 15, 15)])


def _runner(db, tmp_path, clock, alerts=None):
    cfg = copy.deepcopy(DEFAULTS)
    cam = Cam("gate", clock)
    worker = Worker()
    runner = SceneRunner(db, [cam], worker, tmp_path, cfg, clock=clock, mono=clock.mono,
                         on_alert=(lambda q, pic: alerts.append((q["question"], pic))) if alerts is not None else None)
    return runner, worker, cam


def test_job_numbers_never_mix_with_people_or_vehicles():
    assert job_id(7) == -(SCENE_BASE + 7) and question_of(job_id(7)) == 7
    assert question_of(12) is None and question_of(-12) is None                  # a person, a vehicle


def test_answers_are_cleaned():
    assert scene_schema()["properties"]["answer"]["enum"] == ["yes", "no", "unsure"]
    assert [clean_scene({"answer": a}) for a in ("Yes", " no ", "maybe", None)] == ["yes", "no", "unsure", "unsure"]


def test_client_asks_the_question_with_the_whole_picture(monkeypatch):
    client = VLMClient.__new__(VLMClient)
    asked = {}

    def chat(prompt, images, fmt):
        asked.update(prompt=prompt, n=len(images), fmt=fmt)
        return {"answer": "yes"}

    monkeypatch.setattr(client, "_chat", chat, raising=False)
    got = client.label(np.zeros((720, 1280, 3), np.uint8), None, ("scene", "Is the gate open?"))
    assert got["answer"] == "yes" and float(got["seconds"]) >= 0
    assert "Question: Is the gate open?" in asked["prompt"] and asked["n"] == 1 and asked["fmt"] == scene_schema()


def test_due_questions_go_only_when_the_model_is_free(db, tmp_path):
    clock = Clock()
    runner, worker, cam = _runner(db, tmp_path, clock)
    q = db.add_scene_question("gate", "Is the gate open?", 300)
    db.add_scene_question("door", "Is the door open?", 300)                     # no such camera running: skipped
    off = db.add_scene_question("gate", "Are the lights on?", 300)
    db.update_scene_question(off["id"], enabled=False)
    worker.idle = False
    assert runner.tick() is None and worker.jobs == []                          # people first
    worker.idle = True
    assert runner.tick() == q["id"]
    job, frames, mode = worker.jobs[0]
    assert job == job_id(q["id"]) and mode == ("scene", "Is the gate open?") and frames[0][0].shape == (40, 60, 3)
    assert runner.tick() is None                                                # one question out at a time
    runner.on_result(job, [{"answer": "no", "seconds": "1.30"}])
    got = db.list_scene_questions()[0]
    assert got["last_answer"] == "no" and got["last_picture"] == "scene_1.jpg"
    assert (tmp_path / "scene_1.jpg").is_file() and (tmp_path / "blurred" / "scene_1.jpg").is_file()
    assert db.scene_answers(q["id"])[0] == {"answer": "no", "at": "2026-10-14T12:00:00", "seconds": 1.3}
    clock.step(299)
    cam.scene_frame = (cam.scene_frame[0], clock.now, [])
    assert runner.tick() is None                                                # not yet 5 minutes
    clock.step(1)
    assert runner.tick() == q["id"]


def test_old_pictures_and_failed_requests_wait(db, tmp_path):
    clock = Clock()
    runner, worker, cam = _runner(db, tmp_path, clock)
    q = db.add_scene_question("gate", "Is the gate open?", 300)
    clock.step(31)                                                              # the camera's picture is 31 s old
    assert runner.tick() is None
    cam.scene_frame = (cam.scene_frame[0], clock.now, [])
    assert runner.tick() == q["id"]
    runner.on_result(job_id(q["id"]), [])                                       # the model failed
    assert db.list_scene_questions()[0]["last_answer"] is None
    clock.step(59)
    cam.scene_frame = (cam.scene_frame[0], clock.now, [])
    assert runner.tick() is None                                                # tried again after a minute
    clock.step(1)
    cam.scene_frame = (cam.scene_frame[0], clock.now, [])
    assert runner.tick() == q["id"]


def test_alert_after_two_answers_in_a_row_and_not_again(db, tmp_path):
    clock = Clock()
    alerts = []
    runner, worker, cam = _runner(db, tmp_path, clock, alerts)
    q = db.add_scene_question("gate", "Is the gate open?", 300)
    closed = db.add_scene_question("gate", "Is the gate closed?", 300, alert_on="no")
    answer = lambda qid, a: runner.on_result(job_id(qid), [{"answer": a, "seconds": "1"}])  # noqa: E731
    for a in ["yes", "no", "yes", "unsure", "yes"]:                             # never two in a row
        answer(q["id"], a)
    assert alerts == []
    answer(q["id"], "yes")
    assert alerts == [("Is the gate open?", None)]                             # no picture sent in this test
    answer(q["id"], "yes")
    answer(q["id"], "yes")
    assert len(alerts) == 1                                                     # still open: no new alert
    answer(q["id"], "no")
    answer(q["id"], "yes")
    answer(q["id"], "yes")
    assert len(alerts) == 2                                                     # closed, then open again
    answer(closed["id"], "no")
    answer(closed["id"], "no")
    assert alerts[-1][0] == "Is the gate closed?"
    answer(999, "yes")                                                          # a deleted question: ignored


# --- step 3: the alert ---

def test_alert_is_stored_with_its_own_picture_and_a_clip(db, tmp_path):
    clock = Clock()
    runner, worker, cam = _runner(db, tmp_path, clock)
    asked = []
    cam.clips = type("Clips", (), {"request": lambda self, event_id, before_s=None: asked.append(event_id)})()
    q = db.add_scene_question("gate", "Is the gate open?", 300)
    for _ in range(2):
        assert runner.tick() == q["id"]
        runner.on_result(job_id(q["id"]), [{"answer": "yes", "seconds": "1"}])
        clock.step(300)
        cam.scene_frame = (cam.scene_frame[0], clock.now, [])
    events = db.list_alerts()
    assert len(events) == 1
    e = events[0]
    assert (e["scene_id"], e["camera"], e["message"], e["track_id"]) == (q["id"], "gate", "Is the gate open? → yes", None)
    assert e["crop"] == "scene_1_20261014120500.jpg" and (tmp_path / e["crop"]).is_file()
    assert (tmp_path / "blurred" / e["crop"]).is_file()                       # viewers get faces blurred
    assert asked == [e["id"]]
    tick_again = runner.tick()                                                  # the next answer replaces scene_1.jpg ...
    runner.on_result(job_id(tick_again), [{"answer": "no", "seconds": "1"}])
    assert (tmp_path / e["crop"]).is_file()                                     # ... the alert keeps its own copy


def test_deleting_the_question_deletes_its_alerts(db, tmp_path):
    q = db.add_scene_question("gate", "Is the gate open?", 300)
    db.update_scene_question(q["id"], last_picture="scene_1.jpg")
    e = db.record_scene_alert(q["id"], "gate", "Is the gate open? → yes", "scene_1_x.jpg")
    other = db.add_scene_question("gate", "Are the lights on?", 300)
    db.record_scene_alert(other["id"], "gate", "Are the lights on? → yes", None)
    gone = db.delete_scene_question(q["id"])
    assert sorted(gone["files"]) == sorted(["scene_1.jpg", "scene_1_x.jpg", f"clips/alert_{e['id']}.webm"])
    assert [a["scene_id"] for a in db.list_alerts()] == [other["id"]]
    assert db.record_scene_alert(q["id"], "gate", "late", None) is None        # deleted meanwhile


def test_old_scene_alerts_go_with_retention(db, camera):
    from labeler.db import AlertEvent
    q = db.add_scene_question("test-cam", "Is the gate open?", 300)
    old = db.record_scene_alert(q["id"], "test-cam", "old", "scene_1_old.jpg")
    db.record_scene_alert(q["id"], "test-cam", "new", "scene_1_new.jpg")
    with db.Session.begin() as s:
        s.get(AlertEvent, old["id"]).created_at = datetime.now() - timedelta(days=40)
    removed = db.cleanup(30)
    assert "scene_1_old.jpg" in removed and f"clips/alert_{old['id']}.webm" in removed
    assert [e["message"] for e in db.list_alerts()] == ["new"]


def test_pop_ups_name_scene_alerts():
    from pathlib import Path
    static = Path(__file__).resolve().parent.parent / "labeler" / "static"
    for page in ("session.js", "alerts.html"):
        assert 'e.scene_id ? "scene question"' in (static / page).read_text(encoding="utf-8")


# --- step 5: Settings, Health ---

def test_settings_save_the_scene_options(db, tmp_path):
    import yaml
    from labeler.config import load_config
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("cameras:\n  - name: gate\n    source: samples/bus.jpg\n"
                        f"settings_file: {(tmp_path / 'settings.yaml').as_posix()}\n", encoding="utf-8")
    client = TestClient(create_app(load_config(cfg_file), db, None), client=("127.0.0.1", 5000))
    form = client.get("/api/settings").json()["values"]
    assert form["scene_enabled"] is True and form["scene_answers_in_a_row"] == 2
    r = client.post("/api/settings", json={**form, "scene_answers_in_a_row": 3})
    assert r.status_code == 200, r.json()
    assert yaml.safe_load((tmp_path / "settings.yaml").read_text(encoding="utf-8")) == {"scene": {"answers_in_a_row": 3}}
    assert client.post("/api/settings", json={**form, "scene_answers_in_a_row": 9}).status_code == 400


def test_health_line(db):
    from labeler.healthcheck import check_scene
    cfg = copy.deepcopy(DEFAULTS)
    now = datetime.now()
    assert check_scene(cfg, db, now) is None                                   # no questions: no line
    q = db.add_scene_question("gate", "Is the gate open?", 300)
    db.update_scene_question(q["id"], last_answer="no", last_at=now - timedelta(minutes=3))
    line = check_scene(cfg, db, now)
    assert line["level"] == "ok" and line["detail"] == "1 question: Is the gate open? no (3 min ago)"
    db.update_scene_question(q["id"], last_at=now - timedelta(minutes=12))      # 5 min question, 12 min ago
    line = check_scene(cfg, db, now)
    assert line["level"] == "warning" and "1 not answered for a while (Is the gate open?)" in line["detail"]
    cfg["vlm"]["enabled"] = False
    assert "vision model is turned off" in check_scene(cfg, db, now)["detail"]
    cfg["scene"]["enabled"] = False
    assert check_scene(cfg, db, now)["level"] == "off"


# --- Phase 27 step 1: one area of the picture ---

from labeler.scene import AREA_WAY, area_picture  # noqa: E402
from labeler.vlm import PROMPT_SCENE_PART  # noqa: E402

SQUARE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


def test_area_picture_cuts_out_the_area_with_a_margin():
    frame = np.full((100, 200, 3), 200, np.uint8)
    part = area_picture(frame, SQUARE, "part")
    assert part.shape == (64, 124, 3)                         # 100x50 area + 10% margin + 2 px each side
    triangle = area_picture(frame, [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]], "part")
    assert triangle[2, 2].tolist() == [200] * 3               # inside (with the margin): as it was
    assert triangle[-1, -1].tolist() == [70] * 3              # outside the triangle: darkened
    outline = area_picture(frame, SQUARE, "outline")
    assert outline.shape == frame.shape and outline[25, 100].tolist() == [0, 0, 255] and outline[50, 100].tolist() == [200] * 3
    assert frame[25, 100].tolist() == [200] * 3               # the camera's own frame is not drawn on


def test_client_asks_about_the_area(monkeypatch):
    client = VLMClient.__new__(VLMClient)
    asked = {}
    monkeypatch.setattr(client, "_chat", lambda prompt, images, fmt: asked.update(prompt=prompt) or {"answer": "no"},
                        raising=False)
    assert client.label(np.zeros((50, 80, 3), np.uint8), None, ("scene", "Is the gate open?", "part"))["answer"] == "no"
    assert asked["prompt"] == PROMPT_SCENE_PART.format(question="Is the gate open?")


def test_runner_sends_only_the_area_and_keeps_that_picture(db, tmp_path):
    clock = Clock()
    runner, worker, cam = _runner(db, tmp_path, clock)
    cam.scene_frame = (np.full((100, 200, 3), 90, np.uint8), clock.now, [(60, 30, 80, 50)])
    z = db.add_zone("gate", "gate", "area", SQUARE)
    q = db.add_scene_question("gate", "Is the gate open?", 300, zone_id=z["id"])
    assert db.list_scene_questions()[0]["zone_id"] == z["id"]
    assert runner.tick() == q["id"]
    job, frames, mode = worker.jobs[0]
    assert mode == ("scene", "Is the gate open?", AREA_WAY) and frames[0][0].shape == area_picture(cam.scene_frame[0], SQUARE, AREA_WAY).shape
    runner.on_result(job, [{"answer": "yes", "seconds": "1"}])
    import cv2
    for path in (tmp_path / "scene_1.jpg", tmp_path / "blurred" / "scene_1.jpg"):
        assert cv2.imread(str(path)).shape == frames[0][0].shape   # the Check page shows what the model saw
    db.delete_zone(z["id"])                                         # the area is gone: the whole picture again
    clock.step(300)
    cam.scene_frame = (cam.scene_frame[0], clock.now, [])
    assert db.list_scene_questions()[0]["zone_id"] is None
    assert runner.tick() == q["id"] and worker.jobs[1][2] == ("scene", "Is the gate open?")
    assert worker.jobs[1][1][0][0].shape == (100, 200, 3)


def test_api_chooses_the_area(db, tmp_path):
    client = _app(db, tmp_path)
    area = db.add_zone("gate", "gate", "area", SQUARE)
    line = db.add_zone("gate", "door line", "line", [[0.1, 0.5], [0.9, 0.5]])
    other = db.add_zone("door", "porch", "area", SQUARE)
    base = {"camera": "gate", "question": "Is the gate open?"}
    assert [client.post("/api/scene", json={**base, "zone_id": z["id"]}).status_code
            for z in (line, other)] == [400, 400]                      # a line, another camera's area
    r = client.post("/api/scene", json={**base, "zone_id": area["id"]})
    assert r.status_code == 200 and r.json()["zone_id"] == area["id"]
    qid = r.json()["id"]
    assert client.patch(f"/api/scene/{qid}", json={"zone_id": 0}).json()["zone_id"] is None
    assert client.patch(f"/api/scene/{qid}", json={"zone_id": area["id"]}).json()["zone_id"] == area["id"]
    assert client.patch(f"/api/scene/{qid}", json={"zone_id": line["id"]}).status_code == 400
    assert client.patch("/api/scene/999", json={"zone_id": 0}).status_code == 404
    assert client.delete(f"/api/zones/{area['id']}").status_code == 200
    assert client.get("/api/scene").json()[0]["zone_id"] is None
    entries = client.get("/api/audit").json()["entries"]
    assert entries[0]["action"] == "zone deleted" and "Is the gate open?" in entries[0]["detail"]
    assert any(e["action"] == "scene question added" and "area gate" in e["detail"] for e in entries)

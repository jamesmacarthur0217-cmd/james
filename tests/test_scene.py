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

"""The health check (/api/health): 200 when everything works, 503 with the reason when something does not."""
import copy
import logging
import os
import time

import pytest
from fastapi.testclient import TestClient

from labeler import accounts, healthcheck, logs
from labeler.config import DEFAULTS, mask_password
from labeler.web import create_app

TOKEN = "a-long-random-monitoring-token"


class FakePipeline:
    def __init__(self, name="cam1"):
        self.name, self.mode, self.cam = name, "kiosk", {"source": "rtsp://admin:secret@192.168.1.50/stream"}
        self.running, self.crashed, self.fps, self.tracks = True, False, 24.8, {}
        self.last_frame_at = time.monotonic()
        self.vlm = object()          # the app connected to Ollama at start


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(DEFAULTS)
    c["storage"]["crop_dir"] = str(tmp_path / "crops")
    c["storage"]["min_free_gb"] = 0
    c["web"]["health_token"] = TOKEN
    return c


@pytest.fixture
def ollama_ok(monkeypatch):
    """Ollama answering with the model pulled (no real network)."""
    from labeler import vlm

    class Version:
        def json(self):
            return {"version": "0.12.3"}

    monkeypatch.setattr(vlm.VLMClient, "available", lambda self: (True, "ok"))
    monkeypatch.setattr(healthcheck.requests, "get", lambda *a, **k: Version())


@pytest.fixture(autouse=True)
def no_old_errors(monkeypatch):
    monkeypatch.setattr(logs, "recent_errors", logs.RecentErrors())


def _local(cfg, db, pipelines):
    return TestClient(create_app(cfg, db, pipelines), client=("127.0.0.1", 5000))


def _levels(body):
    return {c["name"]: c["level"] for c in body["checks"]}


def test_everything_ok_is_200(cfg, db, ollama_ok):
    r = _local(cfg, db, [FakePipeline()]).get("/api/health")
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["ok"] and body["problems"] == []
    levels = _levels(body)
    assert levels["database"] == levels["vision model (Ollama)"] == levels["camera cam1"] == levels["disk"] == "ok"
    assert levels["database backup"] == "off"                  # SQLite: backups are for PostgreSQL
    assert body["app"]["uptime_s"] >= 0
    assert "secret" not in r.text and "192.168.1.50" not in r.text   # camera addresses are never shown


def test_database_down_is_503_with_the_reason(cfg, db, ollama_ok, monkeypatch):
    def down():
        raise ConnectionRefusedError("no")
    monkeypatch.setattr(db, "ping", down)
    r = _local(cfg, db, [FakePipeline()]).get("/api/health")
    assert r.status_code == 503
    assert r.json()["problems"][0].startswith("database: the database does not answer")


def test_ollama_down_is_503(cfg, db, monkeypatch):
    from labeler import vlm
    monkeypatch.setattr(vlm.VLMClient, "available", lambda self: (False, "Ollama is not reachable at x"))
    r = _local(cfg, db, [FakePipeline()]).get("/api/health")
    assert r.status_code == 503 and "Ollama is not reachable" in r.json()["problems"][0]


def test_ollama_back_but_app_started_without_it_says_restart(cfg, db, ollama_ok):
    p = FakePipeline()
    p.vlm = None
    body = _local(cfg, db, [p]).get("/api/health").json()
    assert not body["ok"] and "restart the app" in body["problems"][0]


def test_vlm_turned_off_is_not_a_problem(cfg, db):
    cfg["vlm"]["enabled"] = False
    p = FakePipeline()
    p.vlm = None
    r = _local(cfg, db, [p]).get("/api/health")
    assert r.status_code == 200 and _levels(r.json())["vision model (Ollama)"] == "off"


def test_cameras_not_running_or_without_pictures_are_503(cfg, db, ollama_ok):
    stopped, frozen, fine = FakePipeline("door"), FakePipeline("yard"), FakePipeline("hall")
    stopped.running, stopped.crashed = False, True
    frozen.last_frame_at = time.monotonic() - 60
    body = _local(cfg, db, [stopped, frozen, fine]).get("/api/health").json()
    assert body["problems"] == ["camera door: not running (it crashed; see the log)",
                                "camera yard: no picture for 60 s"]


def test_camera_problems_from_the_health_monitor(cfg, db, ollama_ok):
    class Monitor:
        def current(self):
            return [{"camera": "cam1", "message": "the picture is blank (lens covered, or another app is using "
                                                  "the camera)"}]
    app = create_app(cfg, db, [FakePipeline()], health=Monitor())
    body = TestClient(app, client=("127.0.0.1", 5000)).get("/api/health").json()
    assert body["problems"] == ["camera cam1: the picture is blank (lens covered, or another app is using the camera)"]


def test_low_disk_is_a_problem(cfg, db, ollama_ok):
    cfg["storage"]["min_free_gb"] = 10 ** 9
    assert "disk: only" in _local(cfg, db, [FakePipeline()]).get("/api/health").json()["problems"][0]


def test_backup_age(db, tmp_path, monkeypatch):
    monkeypatch.setattr(db, "is_postgres", True)
    now = time.time()
    assert healthcheck.check_backup(db, now, tmp_path)["level"] == "warning"           # none yet
    old = tmp_path / "visionlabeler-20260101-120000.dump"
    old.write_bytes(b"x")
    os.utime(old, (now - 10 * 86400, now - 10 * 86400))
    assert "10.0 days old" in healthcheck.check_backup(db, now, tmp_path)["detail"]
    new = tmp_path / "visionlabeler-20260108-120000.dump"
    new.write_bytes(b"x")
    os.utime(new, (now - 3600, now - 3600))
    check = healthcheck.check_backup(db, now, tmp_path)
    assert check["level"] == "ok" and new.name in check["detail"]


def test_errors_in_the_log_are_a_warning_and_shown_to_admins_only(cfg, db, ollama_ok):
    logs.recent_errors.handle(logging.LogRecord("labeler.sources", logging.ERROR, __file__, 1,
                                                "cannot open rtsp://admin:secret@10.0.0.9/live", None, None))
    app = create_app(cfg, db, [FakePipeline()])
    admin = TestClient(app, client=("127.0.0.1", 5000)).get("/api/health")
    assert admin.status_code == 200                              # a warning does not fail the check
    assert admin.json()["warnings"][0].startswith("errors in the log: 1 in the last 24 hours")
    assert admin.json()["latest_errors"][0]["message"] == "labeler.sources: cannot open rtsp://admin:•••@10.0.0.9/live"
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    r = viewer.get("/api/health")
    assert r.status_code == 200 and "latest_errors" not in r.json() and "secret" not in r.text
    assert viewer.get("/health").status_code == 200


def test_monitoring_token(cfg, db, ollama_ok):
    remote = TestClient(create_app(cfg, db, [FakePipeline()]), client=("203.0.113.5", 5000))
    assert remote.get("/api/health").status_code == 401                      # no login, no token
    assert remote.get("/api/health", params={"token": "wrong"}).status_code == 401
    r = remote.get("/api/health", params={"token": TOKEN})
    assert r.status_code == 200 and set(r.json()) == {"ok", "checked_at", "problems", "warnings"}
    assert remote.get("/api/tracks", params={"token": TOKEN}).status_code == 401   # only the health check


def test_no_token_configured_refuses_any_token(cfg, db, ollama_ok):
    cfg["web"]["health_token"] = ""
    remote = TestClient(create_app(cfg, db, [FakePipeline()]), client=("203.0.113.5", 5000))
    assert remote.get("/api/health", params={"token": ""}).status_code == 401


def test_camera_list_masks_passwords(cfg, db):
    cams = _local(cfg, db, [FakePipeline()]).get("/api/cameras").json()
    assert cams[0]["source"] == "rtsp://admin:•••@192.168.1.50/stream"


@pytest.mark.parametrize("raw,shown", [
    ("rtsp://user:pa:ss@host:554/s", "rtsp://user:•••@host:554/s"),
    ("http://cam/x.mjpg", "http://cam/x.mjpg"),
    ("0", "0"),
    ("opening rtsp://a:b@h and http://c:d@e", "opening rtsp://a:•••@h and http://c:•••@e"),
])
def test_mask_password(raw, shown):
    assert mask_password(raw) == shown


class Loaded:
    """A vision model queue stand-in with a set load (VLMWorker.load())."""

    def __init__(self, **load):
        self._load = {"window_s": 600, "busy": 0.0, "answers_per_min": 0.0, "labels": 0, "label_wait_s": None,
                      "refused": 0, **load}

    def load(self):
        return self._load


class Dropping:
    def __init__(self, n):
        self.n = n

    def dropped_within(self, seconds):
        return self.n


def test_the_vision_model_line_shows_its_load():
    assert healthcheck.load_words(Loaded().load()) == ("last 10 min: idle", False)
    words, over = healthcheck.load_words(Loaded(busy=0.92, answers_per_min=38.4, labels=40, label_wait_s=6.2).load())
    assert words == "last 10 min: busy 92%, 38 answers a minute, people's labels waited 6 s" and not over
    words, over = healthcheck.load_words(Loaded(window_s=30, busy=1.0, answers_per_min=42, refused=3).load(), 2)
    assert words.startswith("since the start: busy 100%") and "3 people refused" in words
    assert "2 safety checks dropped" in words and over


def test_a_vision_model_with_more_work_than_it_can_do_is_a_warning(cfg, db, ollama_ok):
    p = FakePipeline()
    p.vlm = Loaded(busy=0.5, answers_per_min=20, label_wait_s=1.0)
    line = healthcheck.check_ollama(cfg, [p])
    assert line["level"] == "ok" and line["detail"].startswith("Ollama 0.12.3, model ")
    assert "last 10 min: busy 50%, 20 answers a minute" in line["detail"]
    assert healthcheck.check_ollama(cfg, [p], Dropping(1))["level"] == "warning"   # a safety check dropped
    p.vlm = Loaded(busy=1.0, answers_per_min=42, refused=5)
    body = _local(cfg, db, [p]).get("/api/health")
    assert body.status_code == 200 and _levels(body.json())["vision model (Ollama)"] == "warning"   # not a problem

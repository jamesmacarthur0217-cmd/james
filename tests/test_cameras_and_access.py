"""Camera list, login for other devices, config defaults, VLM retry."""
import base64
import copy

import pytest
import requests
from fastapi.testclient import TestClient

from labeler.config import DEFAULTS, _expand, normalize_cameras
from labeler.vlm import VLMClient
from labeler.web import check_access, create_app


def test_single_camera_block_still_works():
    cams = normalize_cameras({"mode": "kiosk", "camera": {"name": "cam1", "source": 0}})
    assert cams == [{"name": "cam1", "source": 0, "loop_files": True, "ignore_zones": [], "mode": "kiosk"}]


def test_camera_list_gets_names_and_modes():
    cams = normalize_cameras({"mode": "kiosk", "cameras": [{"source": 0}, {"source": "rtsp://x", "mode": "security"}]})
    assert [(c["name"], c["mode"]) for c in cams] == [("cam1", "kiosk"), ("cam2", "security")]


def test_camera_names_must_be_unique():
    with pytest.raises(ValueError):
        normalize_cameras({"mode": "kiosk", "cameras": [{"name": "a"}, {"name": "a"}]})


def test_env_default_syntax(monkeypatch):
    monkeypatch.delenv("VL_NOT_SET", raising=False)
    assert _expand("${VL_NOT_SET:-}") == ""
    assert _expand("${VL_NOT_SET:-fallback}") == "fallback"
    monkeypatch.setenv("VL_NOT_SET", "real")
    assert _expand("${VL_NOT_SET:-fallback}") == "real"


def test_network_access_needs_a_password():
    check_access({"host": "127.0.0.1", "password": ""})
    check_access({"host": "0.0.0.0", "password": "long-password"})
    with pytest.raises(SystemExit):
        check_access({"host": "0.0.0.0", "password": ""})


def _app(db, tmp_path, password=""):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cfg["web"]["password"] = password
    return create_app(cfg, db, None)


def _auth(user, pw):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


def test_other_devices_must_log_in(db, tmp_path):
    remote = TestClient(_app(db, tmp_path, password="s3cret"), client=("192.168.1.20", 5000))
    assert remote.get("/api/tracks").status_code == 401
    assert remote.get("/api/tracks", headers=_auth("admin", "wrong")).status_code == 401
    assert remote.get("/api/tracks", headers=_auth("admin", "s3cret")).status_code == 200


def test_this_pc_needs_no_login(db, tmp_path):
    local = TestClient(_app(db, tmp_path, password="s3cret"), client=("127.0.0.1", 5000))
    assert local.get("/api/tracks").status_code == 200


def test_no_password_means_no_login(db, tmp_path):
    remote = TestClient(_app(db, tmp_path), client=("192.168.1.20", 5000))
    assert remote.get("/api/tracks").status_code == 200


def test_status_reports_disk_space(db, tmp_path):
    s = TestClient(_app(db, tmp_path)).get("/api/status").json()
    assert s["disk_free_gb"] > 0 and isinstance(s["disk_low"], bool) and s["cameras"] == []


class _Resp:
    def __init__(self, code):
        self.status_code = code


def test_vlm_retries_once_after_an_ollama_error(monkeypatch):
    client = VLMClient({"host": "http://x", "model": "m", "timeout_s": 1}, "kiosk")
    calls = []

    def flaky(person, face, mode):
        calls.append(mode)
        if len(calls) == 1:
            raise requests.HTTPError(response=_Resp(500))
        return {"ok": mode}

    monkeypatch.setattr(client, "_label", flaky)
    monkeypatch.setattr("labeler.vlm.time.sleep", lambda s: None)
    assert client.label(None, None, "security") == {"ok": "security"}
    assert calls == ["security", "security"]


def test_vlm_does_not_retry_a_bad_request(monkeypatch):
    client = VLMClient({"host": "http://x", "model": "m", "timeout_s": 1}, "kiosk")

    def bad(person, face, mode):
        raise requests.HTTPError(response=_Resp(400))

    monkeypatch.setattr(client, "_label", bad)
    with pytest.raises(requests.HTTPError):
        client.label(None, None)


def test_supervisor_restarts_crashes_but_not_normal_stops(monkeypatch):
    from labeler import supervisor
    runs = iter([(3, False), (1, False), (0, False)])           # crash, crash, then a normal stop
    monkeypatch.setattr(supervisor, "run_once", lambda argv: next(runs))
    monkeypatch.setattr(supervisor.time, "sleep", lambda s: None)
    assert supervisor.main([]) == 0


def test_supervisor_gives_up_after_repeated_crashes(monkeypatch):
    from labeler import supervisor
    calls = []
    monkeypatch.setattr(supervisor, "run_once", lambda argv: calls.append(1) or (3, False))
    monkeypatch.setattr(supervisor.time, "sleep", lambda s: None)
    assert supervisor.main([]) == 3 and len(calls) == supervisor.MAX_CRASHES


def test_supervisor_does_not_restart_after_ctrl_c(monkeypatch):
    from labeler import supervisor
    monkeypatch.setattr(supervisor, "run_once", lambda argv: (1, True))
    assert supervisor.main([]) == 0

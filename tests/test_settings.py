"""Settings page: changes go to a separate file over config.yaml, every value is checked, passwords stay hidden,
and "Restart now" restarts under the supervisor without counting as a crash."""
import pytest
import yaml
from fastapi.testclient import TestClient

from labeler import accounts
from labeler import settings as settings_mod
from labeler.config import load_config
from labeler.web import create_app

PW = "correct horse battery"
CONFIG = """\
cameras:
  - name: cam1
    source: samples/bus.jpg
  - name: door
    source: rtsp://admin:s3cret@192.168.1.50:554/stream1
    mode: security
    ignore_zones: [[0.1, 0.1, 0.2, 0.2]]
storage:
  retention_days: 30
"""


@pytest.fixture
def paths(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG + f"""\
  crop_dir: {(tmp_path / 'crops').as_posix()}
settings_file: {(tmp_path / 'settings.yaml').as_posix()}
""", encoding="utf-8")
    return cfg, tmp_path / "settings.yaml"


def _app(paths, db):
    cfg = load_config(paths[0])
    return create_app(cfg, db, None), cfg


def _form(**changes):
    form = {"cameras": [{"name": "cam1", "source": "samples/bus.jpg", "mode": "kiosk"},
                        {"name": "door", "source": "rtsp://admin:•••@192.168.1.50:554/stream1", "mode": "security"}],
            "retention_days": 30, "audit_retention_days": 365, "blur_faces": False,
            "blur_faces_for_viewers": True, "https": False, "retrain_enabled": True,
            "backup_enabled": True, "backup_copy_to": "", "person_videos": True, "mobility_aids": True,
            "falls_enabled": True, "fall_after_s": 10, "vehicles_enabled": True,
            "vehicle_kinds": ["car", "motorbike", "bicycle", "bus", "lorry"], "vehicle_parked_after_s": 60,
            "animals_enabled": True, "animal_kinds": ["dog", "cat", "bird"],
            "scene_enabled": True, "scene_answers_in_a_row": 2, "quiet_enabled": True, "safety_enabled": True, "safety_confirm_per_min": 6, "help_enabled": True, "help_hold_s": 3,
            "idle_enabled": True, "idle_after_s": 60, "recording_enabled": False, "recording_when": "people",
            "recording_keep_days": 3, "recording_max_gb": 100,
            "recording_alert_keep_days": 30, "recording_alert_before_s": 60, "recording_alert_after_s": 120}
    form.update(changes)
    return form


def test_the_settings_file_wins_over_config_yaml(paths):
    cfg_path, file = paths
    file.write_text("storage:\n  retention_days: 7\nweb:\n  https: true\n"
                    "cameras:\n  - name: cam1\n    source: 1\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg["storage"]["retention_days"] == 7 and cfg["web"]["https"] is True
    assert [c["name"] for c in cfg["cameras"]] == ["cam1"] and cfg["cameras"][0]["source"] == 1
    base = load_config(cfg_path, use_settings=False)
    assert base["storage"]["retention_days"] == 30 and len(base["cameras"]) == 2


def test_only_the_projects_own_config_uses_data_settings(tmp_path):
    other = tmp_path / "test_config.yaml"
    other.write_text("mode: kiosk\n", encoding="utf-8")
    assert "_settings_path" not in load_config(other)     # a test config never reads the real data/settings.yaml


def test_a_broken_or_hand_edited_file_does_not_stop_the_start(paths):
    cfg_path, file = paths
    file.write_text("cameras: [1, 2\n", encoding="utf-8")                 # not YAML
    cfg = load_config(cfg_path)
    assert "could not be used" in cfg["_settings_error"] and len(cfg["cameras"]) == 2
    file.write_text("cameras:\n  - name: x\n    mode: flying\n", encoding="utf-8")   # invalid mode
    assert "could not be used" in load_config(cfg_path)["_settings_error"]
    file.write_text("detector:\n  conf: 0.5\nstorage:\n  min_free_gb: 1\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg["_settings_ignored"] == ["detector", "storage.min_free_gb"]
    assert cfg["detector"]["conf"] == 0.1 and cfg["storage"]["min_free_gb"] == 5   # not applied


def test_page_shows_values_with_passwords_masked(paths, db):
    app, _ = _app(paths, db)
    r = TestClient(app, client=("127.0.0.1", 5000)).get("/api/settings")
    assert r.status_code == 200
    s = r.json()
    assert s["values"]["cameras"][1]["source"] == "rtsp://admin:•••@192.168.1.50:554/stream1"
    assert "s3cret" not in r.text and s["restart_needed"] is False and s["restart_available"] is False


def test_saving_stores_only_the_changes_and_is_audited(paths, db):
    app, _ = _app(paths, db)
    local = TestClient(app, client=("127.0.0.1", 5000))
    r = local.post("/api/settings", json=_form(retention_days=14, https=True))
    assert r.status_code == 200, r.json()
    assert r.json()["changes"] == ["retention_days: 30 → 14", "https: False → True"]
    assert r.json()["restart_needed"] is True
    saved = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    assert saved == {"storage": {"retention_days": 14}, "web": {"https": True}}   # cameras unchanged: not stored
    assert local.get("/api/settings").json()["values"]["retention_days"] == 14
    entry = db.list_audit(10)[0]
    assert entry["action"] == "settings changed" and "retention_days: 30 → 14" in entry["detail"]
    # Back to config.yaml's value: nothing left to store.
    local.post("/api/settings", json=_form())
    assert yaml.safe_load(paths[1].read_text(encoding="utf-8")) is None


def test_masked_password_is_kept_and_never_logged(paths, db):
    app, _ = _app(paths, db)
    local = TestClient(app, client=("127.0.0.1", 5000))
    form = _form()
    form["cameras"][1]["mode"] = "kiosk"
    form["cameras"].append({"name": "yard", "source": "rtsp://cam:yardpass@10.0.0.7/live", "mode": "security",
                            "retention_days": 7, "imgsz": 1280})
    r = local.post("/api/settings", json=form)
    assert r.status_code == 200, r.json()
    cams = {c["name"]: c for c in yaml.safe_load(paths[1].read_text(encoding="utf-8"))["cameras"]}
    assert cams["door"]["source"] == "rtsp://admin:s3cret@192.168.1.50:554/stream1"   # the stored password
    assert cams["door"]["ignore_zones"] == [[0.1, 0.1, 0.2, 0.2]]                     # other keys kept
    assert cams["yard"]["imgsz"] == 1280 and cams["yard"]["retention_days"] == 7
    detail = db.list_audit(10)[0]["detail"]
    assert "camera door mode: security → kiosk" in detail and "camera yard added" in detail
    assert "s3cret" not in detail and "yardpass" not in detail and "rtsp://cam:•••@10.0.0.7/live" in detail
    # The mask with a different address: the password must be typed again.
    form["cameras"][1]["source"] = "rtsp://admin:•••@192.168.1.99:554/stream1"
    r = local.post("/api/settings", json=form)
    assert r.status_code == 400 and "type its password again" in r.json()["detail"]


@pytest.mark.parametrize("change,message", [
    ({"cameras": [{"name": "cam1", "source": "samples/nothing.mp4"}]}, "no file"),
    ({"cameras": [{"name": "cam1", "source": 12}]}, "webcam number is 0-9"),
    ({"cameras": [{"name": "cam1", "source": "rtsp://"}]}, "no host"),
    ({"cameras": [{"name": "cam1", "source": "${CAM_URL}"}]}, "type the address itself"),
    ({"cameras": [{"name": "cam1", "source": 0, "imgsz": 800}]}, "detector size is one of 640, 960, 1280"),
    ({"cameras": [{"name": "cam 1", "source": 0}]}, "letters, digits"),
    ({"cameras": [{"name": "a", "source": 0}, {"name": "a", "source": 1}]}, "same name"),
    ({"cameras": []}, "cameras"),
    ({"retention_days": -1}, "retention_days"),
    ({"https": "maybe"}, "https"),
    ({"detector_conf": 0.5}, "detector_conf"),           # not a setting of this page
    ({"backup_copy_to": "Z:/not/plugged/in"}, "second-copy folder"),
    ({"backup_copy_to": "relative/folder"}, "second-copy folder"),
    ({"idle_after_s": 5}, "idle_after_s"),
    ({"recording_when": "sometimes"}, "people are in view, or always"),
    ({"recording_keep_days": 0}, "recording_keep_days"),
    ({"fall_after_s": 1}, "fall_after_s"),
    ({"vehicle_kinds": []}, "vehicle_kinds"),
    ({"vehicle_kinds": ["car", "spaceship"]}, "vehicle kinds are some of"),
    ({"vehicle_parked_after_s": 5}, "vehicle_parked_after_s"),
    ({"animal_kinds": []}, "animal_kinds"),
    ({"scene_answers_in_a_row": 0}, "scene_answers_in_a_row"),
    ({"scene_answers_in_a_row": 6}, "scene_answers_in_a_row"),
    ({"animal_kinds": ["dog", "monkey"]}, "animal kinds are some of"),
    ({"fall_after_s": 1000}, "fall_after_s"),
    ({"away_schedule": ["weekdays 08:00-18:00"]}, "unknown day 'weekdays'"),
    ({"away_schedule": ["mon-fri 8-18"]}, "write days and times like mon-fri 08:00-18:00"),
    ({"away_schedule": ["mon 09:00-09:00"]}, "the same time"),
])
def test_bad_values_are_refused(paths, db, change, message):
    app, _ = _app(paths, db)
    r = TestClient(app, client=("127.0.0.1", 5000)).post("/api/settings", json=_form(**change))
    assert r.status_code == 400 and message in r.json()["detail"], r.json()
    assert not paths[1].exists()


def test_daily_backup_settings_are_saved(paths, db, tmp_path):
    app, _ = _app(paths, db)
    usb = tmp_path / "usb"
    usb.mkdir()
    r = TestClient(app, client=("127.0.0.1", 5000)).post(
        "/api/settings", json=_form(backup_enabled=False, backup_copy_to=f"  {usb}  "))
    assert r.status_code == 200, r.json()
    saved = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    assert saved == {"backup": {"enabled": False, "copy_to": str(usb)}}


def test_recording_settings_are_saved(paths, db):
    app, _ = _app(paths, db)
    r = TestClient(app, client=("127.0.0.1", 5000)).post(
        "/api/settings", json=_form(recording_when="always", recording_keep_days=1.5, recording_max_gb=50))
    assert r.status_code == 200, r.json()
    assert yaml.safe_load(paths[1].read_text(encoding="utf-8")) == {
        "recording": {"when": "always", "keep_days": 1.5, "max_gb": 50}}
    r = TestClient(app, client=("127.0.0.1", 5000)).post(   # Phase 42: the minutes kept around alerts
        "/api/settings", json=_form(recording_alert_keep_days=14, recording_alert_before_s=30))
    assert r.status_code == 200, r.json()
    assert yaml.safe_load(paths[1].read_text(encoding="utf-8")) == {
        "recording": {"alert_keep_days": 14, "alert_before_s": 30}}
    bad = TestClient(app, client=("127.0.0.1", 5000)).post("/api/settings", json=_form(recording_alert_after_s=5000))
    assert bad.status_code == 400


def test_idle_mode_settings_are_saved(paths, db):
    app, _ = _app(paths, db)
    r = TestClient(app, client=("127.0.0.1", 5000)).post("/api/settings", json=_form(idle_after_s=300))
    assert r.status_code == 200, r.json()
    assert yaml.safe_load(paths[1].read_text(encoding="utf-8")) == {"idle": {"after_s": 300}}
    assert load_config(paths[0])["idle"] == {"enabled": True, "after_s": 300, "fps": 3}


def test_viewers_cannot_see_or_change_settings(paths, db):
    app, _ = _app(paths, db)
    db.add_user("vic", accounts.hash_password(PW), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": PW})
    assert viewer.get("/api/settings").status_code == 403
    assert viewer.post("/api/settings", json=_form()).status_code == 403
    assert viewer.post("/api/restart").status_code == 403
    assert viewer.get("/settings", follow_redirects=False).status_code == 303


def test_without_a_settings_file_the_page_says_so(db, tmp_path):
    from labeler.config import DEFAULTS
    import copy
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    r = TestClient(create_app(cfg, db, None), client=("127.0.0.1", 5000)).get("/api/settings")
    assert r.status_code == 503


def test_restart_needs_the_supervisor(paths, db, monkeypatch):
    app, _ = _app(paths, db)
    local = TestClient(app, client=("127.0.0.1", 5000))
    calls = []
    app.state.request_restart = lambda: calls.append(1)
    monkeypatch.delenv("VL_SUPERVISED", raising=False)
    assert local.post("/api/restart").status_code == 409 and not calls
    monkeypatch.setenv("VL_SUPERVISED", "1")
    assert local.get("/api/settings").json()["restart_available"] is True
    assert local.post("/api/restart").status_code == 200 and calls == [1]
    assert db.list_audit(1)[0]["action"] == "restart app"


def test_supervisor_restarts_on_request_without_counting_a_crash(monkeypatch, supervisor, tmp_path):
    runs = iter([4] * (supervisor.MAX_CRASHES + 2) + [3, 0])
    calls = []
    monkeypatch.setattr(supervisor, "run_once", lambda argv: calls.append(1) or (next(runs), False))
    assert supervisor.main([]) == 0
    assert len(calls) == supervisor.MAX_CRASHES + 4                # never gave up
    text = (tmp_path / "logs" / "supervisor.log").read_text(encoding="utf-8")
    assert text.count("asked for on the Settings page") == supervisor.MAX_CRASHES + 2
    assert "crash 1 of 5" in text


def test_masked_source_without_a_stored_camera_is_refused():
    with pytest.raises(ValueError):
        settings_mod.check_source("rtsp://a:•••@h/x", None)


def test_a_camera_sent_from_another_device(paths, db):
    from labeler import receiver
    app, _ = _app(paths, db)
    client = TestClient(app, client=("127.0.0.1", 5000))
    form = _form()
    form["cameras"] = form["cameras"] + [{"name": "phone", "source": "PUSH", "mode": "kiosk"}]
    r = client.post("/api/settings", json=form).json()
    password = r["new_passwords"]["phone"]
    assert len(password) == 24 and "camera phone added (push, kiosk)" in r["changes"]
    saved = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    phone = next(c for c in saved["cameras"] if c["name"] == "phone")
    assert phone["source"] == "push" and phone["push_password"] == receiver.hash_password(password)
    assert password not in paths[1].read_text(encoding="utf-8")
    assert all(password not in e["detail"] for e in db.list_audit(20))
    s = client.get("/api/settings").json()
    assert s["push"]["ready"] == {"phone": True} and s["push"]["rtsp_port"] == 8554
    again = client.post("/api/settings", json=form).json()                 # saved again: the same password
    assert again["new_passwords"] == {}
    renewed = client.post("/api/settings/cameras/phone/new-password").json()
    assert renewed["password"] != password
    saved = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    assert next(c for c in saved["cameras"] if c["name"] == "phone")["push_password"] == receiver.hash_password(renewed["password"])
    assert db.list_audit(1)[0]["detail"] == "camera phone: new send password"
    assert client.post("/api/settings/cameras/cam1/new-password").status_code == 404   # not sent from elsewhere
    form["cameras"][-1]["source"] = 1                                       # now a webcam: the password goes
    client.post("/api/settings", json=form)
    saved = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    assert "push_password" not in next(c for c in saved["cameras"] if c["name"] == "phone")


def test_fall_alert_settings_are_saved(paths, db):
    app, _ = _app(paths, db)
    client = TestClient(app, client=("127.0.0.1", 5000))
    form = _form(fall_after_s=20)
    form["cameras"][1]["fall_alert"] = False                   # the door camera never sees the floor
    r = client.post("/api/settings", json=form)
    assert r.status_code == 200, r.json()
    saved = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    assert saved["falls"] == {"after_s": 20}
    cams = {c["name"]: c for c in saved["cameras"]}
    assert cams["door"]["fall_alert"] is False and "fall_alert" not in cams["cam1"]   # on = default, not written
    shown = client.get("/api/settings").json()
    page = shown.get("values", shown)
    assert page["fall_after_s"] == 20 and [c["fall_alert"] for c in page["cameras"]] == [True, False]
    assert "camera door fall_alert: True → False" in db.list_audit(10)[0]["detail"]


def test_vehicle_settings_are_saved(paths, db):
    app, _ = _app(paths, db)
    client = TestClient(app, client=("127.0.0.1", 5000))
    form = _form(vehicle_kinds=["motorbike", "car"], vehicle_parked_after_s=120)
    form["cameras"][0]["vehicles"] = True                       # a kiosk camera that does look for vehicles
    form["cameras"][1]["vehicles"] = True                       # a security camera: that is its default
    r = client.post("/api/settings", json=form)
    assert r.status_code == 200, r.json()
    saved = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    assert saved["vehicles"] == {"kinds": ["car", "motorbike"], "parked_after_s": 120}
    cams = {c["name"]: c for c in saved["cameras"]}
    assert cams["cam1"]["vehicles"] is True and "vehicles" not in cams["door"]   # the default is not written
    shown = client.get("/api/settings").json()
    page = shown.get("values", shown)
    assert [c["vehicles"] for c in page["cameras"]] == [True, True]


def test_every_field_marks_unsaved_changes():
    """Not only the cameras table: one listener on the whole form (checked in the browser on a test copy)."""
    from pathlib import Path
    page = (Path(__file__).resolve().parent.parent / "labeler" / "static" / "settings.html").read_text(encoding="utf-8")
    assert 'document.querySelector("main").addEventListener(kind, () => { if (loaded) changed(); })' in page


def test_safety_checks_each_kind_and_the_main_times(paths, db):
    """Phase 51: each kind on/off and the main times go into their own config sections and are used at the next
    start; night hours are checked and written the same way every time."""
    app, _ = _app(paths, db)
    local = TestClient(app, client=("127.0.0.1", 5000))
    values = local.get("/api/settings").json()["values"]
    assert values["safety_fights_enabled"] is True and values["behaviour_night_hours"] == "22-06"
    assert values["safety_equipment_every_s"] == 600 and values["behaviour_pace_turns"] == 4
    r = local.post("/api/settings", json=_form(safety_fights_enabled=False, behaviour_enabled=False,
                                               behaviour_night_hours="23-5", safety_equipment_every_s=900,
                                               safety_child_alone_s=0))
    assert r.status_code == 200, r.json()
    assert "safety_fights_enabled: True → False" in r.json()["changes"]
    saved = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    assert saved == {"safety": {"fights_enabled": False, "child_alone_s": 0, "equipment_every_s": 900},
                     "behaviour": {"enabled": False, "night_hours": "23-05"}}
    cfg = load_config(paths[0])
    assert cfg["safety"]["fights_enabled"] is False and cfg["behaviour"]["night_hours"] == "23-05"
    assert cfg["safety"]["crowd_enabled"] is True and "_settings_ignored" not in cfg
    for bad in ({"behaviour_night_hours": "25-06"}, {"behaviour_night_hours": "22-22"},
                {"behaviour_night_hours": "night"}, {"safety_equipment_every_s": 60}, {"behaviour_pace_turns": 1}):
        r = local.post("/api/settings", json=_form(**bad))
        assert r.status_code == 400, bad
    assert "night hours look like 22-06" in local.post("/api/settings", json=_form(behaviour_night_hours="x")).json()["detail"]

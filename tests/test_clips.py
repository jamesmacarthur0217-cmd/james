"""Alert clips: a few seconds before and after an alert, played on the Alerts page, deleted with the person."""
import copy
from datetime import datetime, timedelta

import cv2
import numpy as np
from fastapi.testclient import TestClient

from labeler import accounts
from labeler.clips import ClipRecorder, clip_name
from labeler.config import DEFAULTS
from labeler.web import create_app
from test_pipeline_logic import make_pipeline

CFG = {"before_s": 5, "after_s": 5, "fps": 10, "max_width": 960}


def _jpeg(i):
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, (i * 3) % 64] = 255                      # a moving line, so frames differ
    return cv2.imencode(".jpg", img)[1].tobytes()


def _frames_in(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps


def test_clip_has_the_seconds_before_and_after_the_alert(tmp_path):
    rec = ClipRecorder(tmp_path, CFG)
    for i in range(400):                            # 20 s of a 20 fps camera
        t = i * 0.05
        if i == 200:
            rec.request(7, now=t)                   # alert at second 10
        rec.add_frame(_jpeg(i), now=t)
    rec.close()
    n, fps = _frames_in(tmp_path / clip_name(7))
    assert 95 <= n <= 102 and round(fps) == 10      # seconds 5 to 15 at 10 frames per second
    assert not list((tmp_path / "clips").glob("*.part.webm"))


def test_a_15_fps_camera_gives_10_frames_per_second(tmp_path):
    rec = ClipRecorder(tmp_path, CFG)
    for i in range(300):                            # 20 s at 15 fps
        t = i / 15
        if i == 150:
            rec.request(8, now=t)
        rec.add_frame(_jpeg(i), now=t)
    rec.close()
    n, fps = _frames_in(tmp_path / clip_name(8))
    assert 95 <= n <= 102 and round(fps) == 10


def test_stopping_saves_what_the_clip_has_so_far_and_ignores_later_alerts(tmp_path):
    rec = ClipRecorder(tmp_path, CFG)
    for i in range(60):
        rec.add_frame(_jpeg(i), now=i * 0.1)
    rec.request(1, now=6.0)
    rec.close()
    rec.request(2, now=7.0)                          # a late labeling result after the camera stopped
    assert (tmp_path / clip_name(1)).is_file() and not (tmp_path / clip_name(2)).exists()


def test_an_alert_that_fires_starts_a_clip(db, camera):
    p = make_pipeline(db)

    class Alerts:
        def check(self, track_id, camera_name, fallen=False):
            return [{"id": 41}, {"id": 42}]

    class Clips:
        requested = []

        def request(self, event_id, before_s=None):
            self.requested.append(event_id)

    p.alerts, p.clips = Alerts(), Clips()
    p._check_alerts(db.create_track(camera, 1, datetime.now()))
    assert p.clips.requested == [41, 42]


def _alert_with_clip(db, camera, tmp_path):
    rule = db.add_rule("anyone", [])
    t = db.create_track(camera, 1, datetime.now() - timedelta(days=40))
    event = db.record_alert(rule["id"], t, "test-cam", "anyone")
    path = tmp_path / clip_name(event["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"webm")
    return rule, t, event, path


def test_clips_are_deleted_with_the_person_the_rule_and_by_retention(db, camera, tmp_path):
    rule, t, event, path = _alert_with_clip(db, camera, tmp_path)
    assert db.rule_clips(rule["id"]) == [clip_name(event["id"])]
    assert clip_name(event["id"]) in db.delete_tracks([t])[1]
    rule2, t2, event2, _ = _alert_with_clip(db, camera, tmp_path)
    assert clip_name(event2["id"]) in db.cleanup(30)


def test_clips_page_is_admin_only_and_files_go_with_the_person(db, camera, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    app = create_app(cfg, db, None)
    admin = TestClient(app, client=("127.0.0.1", 5000))
    rule, t, event, path = _alert_with_clip(db, camera, tmp_path)
    name = path.name
    assert admin.get("/api/alerts/events").json()["events"][0]["clip"] == name
    r = admin.get(f"/clips/{name}")
    assert r.status_code == 200 and r.headers["content-type"] == "video/webm"
    assert db.list_audit(action="view clip")[0]["detail"] == name
    for bad in ["alert_x.webm", "other.webm", "alert_99999.webm"]:
        assert admin.get(f"/clips/{bad}").status_code == 404
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get(f"/clips/{name}").status_code == 403
    path.unlink()
    assert admin.get("/api/alerts/events").json()["events"][0]["clip"] is None   # not written (yet)
    path.write_bytes(b"webm")
    admin.post("/api/tracks/delete", json={"ids": [t]})
    assert not path.exists()
    rule2, _, _, path2 = _alert_with_clip(db, camera, tmp_path)
    assert admin.delete(f"/api/alerts/rules/{rule2['id']}").status_code == 200
    assert not path2.exists()

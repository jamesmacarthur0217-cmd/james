"""A short video of each person: recorded while they are in view, kept only for real people, deleted with them,
admins only; and the storage line on the Health page."""
import copy
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from labeler import accounts, healthcheck
from labeler.clips import ClipRecorder, person_video_name
from labeler.config import DEFAULTS
from labeler.pipeline import TrackState
from labeler.web import create_app
from labeler.zones import ZoneCounter
from test_clips import CFG, _frames_in, _jpeg
from test_pipeline_logic import make_pipeline

PV = dict(DEFAULTS["person_videos"])          # 20 s at most, 5 fps, 1 s before and after


def _camera(rec, seconds, start=0.0, fps=20, events=()):
    """Feed `seconds` of frames; events: {time: callable(now)} run just before that frame."""
    events = dict(events)
    for i in range(int(seconds * fps)):
        t = start + i / fps
        for at in [a for a in events if a <= t]:
            events.pop(at)(t)
        rec.add_frame(_jpeg(i), now=t)


def test_video_runs_from_a_second_before_to_a_second_after(tmp_path):
    rec = ClipRecorder(tmp_path, CFG, PV)
    name = person_video_name(5, 1)
    _camera(rec, 12, events={3.0: lambda t: rec.start_person("a", now=t),
                             8.0: lambda t: rec.finish_person("a", name, now=t)})
    rec.close()
    n, fps = _frames_in(tmp_path / name)
    assert round(fps) == 5 and 33 <= n <= 37      # seconds 2 to 9 at 5 frames per second


def test_long_stays_are_capped(tmp_path):
    rec = ClipRecorder(tmp_path, CFG, {**PV, "max_s": 4})
    name = person_video_name(6, 1)
    _camera(rec, 15, events={1.0: lambda t: rec.start_person("a", now=t),
                             12.0: lambda t: rec.finish_person("a", name, now=t)})
    rec.close()
    n, _ = _frames_in(tmp_path / name)
    assert 23 <= n <= 27                          # 1 s before + the first 4 s, not 11 s


def test_dropped_people_leave_no_file(tmp_path):
    rec = ClipRecorder(tmp_path, CFG, PV)
    _camera(rec, 6, events={1.0: lambda t: rec.start_person("a", now=t), 3.0: lambda t: rec.drop_person("a")})
    rec.close()
    assert not list(tmp_path.rglob("*.webm"))


def test_off_records_nothing(tmp_path):
    rec = ClipRecorder(tmp_path, CFG, {**PV, "enabled": False})
    rec.start_person("a", now=0.0)
    rec.finish_person("a", person_video_name(1, 1), now=1.0)
    rec.close()
    assert rec.person is None and not list(tmp_path.rglob("*.webm"))


class FakeClips:
    person = PV

    def __init__(self):
        self.calls = []

    def start_person(self, key, now=None):
        self.calls.append(("start", key))

    def finish_person(self, key, name, now=None):
        self.calls.append(("finish", key, name))

    def drop_person(self, key):
        self.calls.append(("drop", key))


def _pipeline(db, camera, tmp_path):
    p = make_pipeline(db)
    p.camera_id, p.owned_ids, p.crop_dir = camera, set(), tmp_path
    p.zone_counter = ZoneCounter([], 3)
    p.clips = FakeClips()
    return p


def test_finished_people_get_numbered_visits_and_noise_is_dropped(db, camera, tmp_path):
    p = _pipeline(db, camera, tmp_path)
    real = TrackState(1, datetime.now(), 0.0, clip_key=11)
    real.n_frames, real.vlm_requested = 30, True
    noise = TrackState(2, datetime.now(), 0.0, clip_key=12)
    noise.n_frames = 3
    p.tracks = {1: real, 2: noise}
    p._finish(1)
    p._finish(2)
    assert ("drop", 12) in p.clips.calls
    name = person_video_name(real.db_id, 1)
    assert ("finish", 11, name) in p.clips.calls
    assert db.get_tracks([real.db_id])[0]["videos"] == [name]
    real.clip_key = 13                            # back for a second visit (re-linked)
    p.tracks[1] = real
    p._finish(1)
    assert db.get_tracks([real.db_id])[0]["videos"] == [name, person_video_name(real.db_id, 2)]


def test_deleting_a_person_removes_their_videos(db, camera, tmp_path):
    t = db.create_track(camera, 1, datetime.now())
    db.update_track(t, video_paths="clips/person_1_1.webm,clips/person_1_2.webm")
    deleted, paths = db.delete_tracks([t])
    assert "clips/person_1_1.webm" in paths and "clips/person_1_2.webm" in paths


def test_retention_removes_their_videos(db, camera):
    t = db.create_track(camera, 1, datetime.now() - timedelta(days=40))
    db.update_track(t, video_paths="clips/person_9_1.webm")
    assert "clips/person_9_1.webm" in db.cleanup(30)


def test_videos_are_for_admins_and_only_when_written(db, camera, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    t = db.create_track(camera, 1, datetime.now())
    written, pending = person_video_name(t, 1), person_video_name(t, 2)
    db.update_track(t, video_paths=f"{written},{pending}")
    (tmp_path / "clips").mkdir()
    (tmp_path / written).write_bytes(b"webm")
    app = create_app(cfg, db, None)
    admin = TestClient(app, client=("127.0.0.1", 5000))
    assert admin.get("/api/tracks").json()[0]["videos"] == [written.split("/")[-1]]   # not the one still recording
    assert admin.get("/clips/" + written.split("/")[-1]).status_code == 200
    assert db.list_audit(1)[0]["action"] == "view clip"
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/tracks").json()[0]["videos"] == []
    assert viewer.get("/clips/" + written.split("/")[-1]).status_code == 403


def test_storage_line(db, camera, tmp_path, monkeypatch):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"].update(crop_dir=str(tmp_path), min_free_gb=0, retention_days=0)
    for i in range(14):                           # 2 people a day over the last week
        db.create_track(camera, i, datetime.now() - timedelta(hours=12 * i))
    use = {"pictures": 14_000_000, "person videos": 14_000_000, "alert clips": 0, "backups": 0, "logs": 0}
    check = healthcheck.check_storage(cfg, db, use=use)
    assert check["level"] == "ok" and "about 2 MB per person, 2 people a day" in check["detail"]

    class Disk:
        free = 100_000_000                        # 100 MB left: 25 days at 4 MB a day
    monkeypatch.setattr(healthcheck.shutil, "disk_usage", lambda p: Disk)
    check = healthcheck.check_storage(cfg, db, use=use)
    assert check["level"] == "warning" and "space for about 25 more days" in check["detail"]
    cfg["storage"]["retention_days"] = 7          # people are deleted after a week: it never fills up
    assert healthcheck.check_storage(cfg, db, use=use)["level"] == "ok"

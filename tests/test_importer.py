"""Importing recorded videos: real start times, fast reading, video time inside the pipeline, the import job, and
the Import page's API."""
import copy
import os
import struct
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeler import accounts, importer
from labeler.config import DEFAULTS
from labeler.db import IMPORTED
from labeler.pipeline import TrackState, VideoClock
from labeler.sources import RecordedVideo
from labeler.web import create_app
from labeler.zones import ZoneCounter

NOW = datetime(2026, 10, 10, 12, 0)


def _video(path: Path, frames=30, fps=20.0) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(frames):
        w.write(np.full((48, 64, 3), i * 8 % 255, np.uint8))
    w.release()
    return path


def _box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


def _mp4(path: Path, created: datetime | None, version=0, moov_last=True) -> Path:
    seconds = 0 if created is None else int((created - datetime(1904, 1, 1, tzinfo=timezone.utc)).total_seconds())
    if version == 1:
        mvhd = _box(b"mvhd", bytes([1, 0, 0, 0]) + struct.pack(">QQ", seconds, seconds) + bytes(80))
    else:
        mvhd = _box(b"mvhd", bytes([0, 0, 0, 0]) + struct.pack(">II", seconds, seconds) + bytes(80))
    parts = [_box(b"ftyp", b"isom" + bytes(4)), _box(b"mdat", bytes(5000)), _box(b"moov", _box(b"trak", b"") + mvhd)]
    if not moov_last:
        parts = [parts[0], parts[2], parts[1]]
    path.write_bytes(b"".join(parts))
    return path


@pytest.mark.parametrize("name,expected", [
    ("VID_20261009_143012", datetime(2026, 10, 9, 14, 30, 12)),
    ("20261009_143012", datetime(2026, 10, 9, 14, 30, 12)),
    ("PXL_20261009_143012345", datetime(2026, 10, 9, 14, 30, 12)),          # Pixel adds milliseconds
    ("2026-10-09 14.30.12", datetime(2026, 10, 9, 14, 30, 12)),
    ("ch01_20261009143012", datetime(2026, 10, 9, 14, 30, 12)),              # recorders
    ("IMG_1234", None),
    ("20261399_250000", None),                                               # not a real date
    ("VID_20991009_143012", None),                                           # in the future
])
def test_time_from_the_file_name(name, expected):
    assert importer.time_from_name(name, now=NOW) == expected


@pytest.mark.parametrize("version,moov_last", [(0, True), (1, True), (0, False)])
def test_time_from_the_mp4_header(tmp_path, version, moov_last):
    utc = datetime(2026, 10, 9, 6, 30, 12, tzinfo=timezone.utc)
    path = _mp4(tmp_path / "clip.mp4", utc, version, moov_last)
    assert importer.mp4_creation_time(path) == utc.astimezone().replace(tzinfo=None)   # shown in local time


def test_unset_or_broken_header_gives_no_time(tmp_path):
    assert importer.mp4_creation_time(_mp4(tmp_path / "zero.mp4", None)) is None
    (tmp_path / "junk.mp4").write_bytes(b"\x00\x00\x00\x01junk")
    assert importer.mp4_creation_time(tmp_path / "junk.mp4") is None


def test_start_time_order(tmp_path):
    utc = datetime(2026, 10, 9, 6, 30, 12, tzinfo=timezone.utc)
    named = _mp4(tmp_path / "VID_20261001_080000.mp4", utc)
    plain = _mp4(tmp_path / "clip.mp4", utc)
    old = _mp4(tmp_path / "noheader.mp4", None)
    os.utime(old, (NOW.timestamp(), NOW.timestamp()))
    typed = datetime(2025, 1, 1, 9, 0)
    assert importer.start_time(named, 60, typed) == (typed, "typed")
    assert importer.start_time(named, 60) == (datetime(2026, 10, 1, 8, 0), "file name")
    assert importer.start_time(plain, 60)[1] == "video header"
    assert importer.start_time(old, 60) == (NOW - timedelta(seconds=60), "file time")   # modified = the end


def test_find_videos(tmp_path):
    (tmp_path / "b").mkdir()
    for name in ("b/2.mp4", "1.MOV", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in importer.find_videos(tmp_path)] == ["1.MOV", "2.mp4"]
    assert importer.find_videos(f'"{tmp_path / "1.MOV"}"')[0].name == "1.MOV"    # pasted with quotes
    with pytest.raises(ValueError, match="not a video"):
        importer.find_videos(tmp_path / "notes.txt")
    with pytest.raises(ValueError, match="no file or folder"):
        importer.find_videos(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no video files"):
        importer.find_videos(tmp_path / "empty")


def test_recorded_video_reads_every_kth_frame_with_its_position(tmp_path):
    path = _video(tmp_path / "v.avi", frames=30, fps=20.0)
    every = [t for _, t in RecordedVideo(path, 1).frames()]
    third = [t for _, t in RecordedVideo(path, 3).frames()]
    assert len(every) == 30 and every[1] == pytest.approx(0.05)
    assert len(third) == 10 and third[1] == pytest.approx(0.15) and third[-1] == pytest.approx(1.35)
    assert importer.stride_for(30, 10) == 3 and importer.stride_for(10, 10) == 1 and importer.stride_for(25, 10) == 2


def test_video_clock():
    c = VideoClock(datetime(2026, 1, 1, 8, 0), base=5000)
    c.t = 90.5
    assert c.mono() == 5090.5 and c.wall() == datetime(2026, 1, 1, 8, 1, 30, 500000)


def test_pipeline_uses_video_time(db, tmp_path):
    from tests.test_pipeline_logic import make_pipeline
    camera = db.get_or_create_camera("phone", IMPORTED, "security")
    p = make_pipeline(db)
    p.camera_id, p.owned_ids, p.crop_dir = camera, set(), tmp_path
    p.zone_counter = ZoneCounter([], 3)
    p.clock = VideoClock(datetime(2025, 5, 1, 9, 0))
    p.clock.t = 40.0
    st = TrackState(7, p.clock.wall(), p.clock.mono())
    st.n_frames, st.vlm_requested = 30, True
    p.tracks[7] = st
    p.clock.t = 45.0                                       # five seconds of video later
    p._finish(7)
    t = db.get_tracks([st.db_id])[0]
    assert t["started_at"].startswith("2025-05-01T09:00:40") and t["ended_at"].startswith("2025-05-01T09:00:45")


class FakeWorker:
    def __init__(self, pending=0):
        self.pending = pending


class FakePipeline:
    """Counts frames per file and records the clock it saw."""

    def __init__(self):
        self.owned_ids, self.tracks, self.recently_lost = set(), {}, {}
        self.frames, self.seen, self.running = 0, [], False

    def process(self, frame):
        self.seen.append((self.clock.mono(), self.clock.wall()))
        if len(self.seen) % 10 == 1:
            self.owned_ids.add(len(self.seen))
            self.tracks[len(self.seen)] = object()

    def _finish(self, tid):
        self.tracks.pop(tid)


def _run(imp, files, **kw):
    imp.start(files, "phone", "security", kw.get("keep", 0), kw.get("fps", 10))
    for _ in range(400):
        if not imp.running:
            break
        time.sleep(0.02)
    return imp.status()["history"][-1]


def test_import_job_runs_files_one_after_another(db, tmp_path):
    a, b = _video(tmp_path / "a.avi", 20, 20.0), _video(tmp_path / "b.avi", 10, 20.0)
    fake, registered = FakePipeline(), []
    db.get_or_create_camera("phone", IMPORTED, "security")
    imp = importer.Importer(DEFAULTS, db, register=registered.append, unregister=registered.remove,
                            pipeline_factory=lambda cfg, cam: fake)
    starts = [datetime(2026, 1, 1, 8, 0), datetime(2026, 1, 1, 9, 0)]
    job = _run(imp, [{"path": a, "start": starts[0]}, {"path": b, "start": starts[1]}], keep=14)
    assert job["state"] == "done" and [f["state"] for f in job["files"]] == ["done", "done"]
    assert len(fake.seen) == 10 + 5                         # 20 fps video, 10 frames per second used
    assert fake.seen[0][1] == starts[0] and fake.seen[10][1] == starts[1]
    monos = [m for m, _ in fake.seen]
    assert monos == sorted(monos) and monos[10] - monos[9] > importer.FILE_GAP_S   # no re-link across files
    assert job["people"] == 2 and registered == []           # registered while running, removed after
    assert db.camera_retention() == {"phone": 14}


def test_import_waits_for_the_vision_model(db, tmp_path):
    worker = FakeWorker(pending=5)
    fake = FakePipeline()
    imp = importer.Importer(DEFAULTS, db, worker=worker, pipeline_factory=lambda cfg, cam: fake)
    imp.start([{"path": _video(tmp_path / "a.avi", 20, 20.0), "start": NOW}], "phone", "security", 0, 10)
    time.sleep(0.3)
    assert fake.seen == []                                   # 5 queued > 2: waiting
    worker.pending = 0
    for _ in range(200):
        if not imp.running:
            break
        time.sleep(0.02)
    assert len(fake.seen) == 10


def test_cancel_keeps_what_was_found(db, tmp_path):
    worker = FakeWorker(pending=5)
    imp = importer.Importer(DEFAULTS, db, worker=worker, pipeline_factory=lambda cfg, cam: FakePipeline())
    imp.start([{"path": _video(tmp_path / "a.avi"), "start": NOW}], "phone", "security", 0, 10)
    with pytest.raises(ValueError, match="already running"):
        imp.start([{"path": tmp_path / "a.avi", "start": NOW}], "phone", "security", 0, 10)
    assert imp.cancel()
    worker.pending = 0
    for _ in range(200):
        if not imp.running:
            break
        time.sleep(0.02)
    job = imp.status()["history"][-1]
    assert job["state"] == "cancelled" and job["done_s"] < job["total_s"]   # counted only as far as it got


# --- the web API -----------------------------------------------------------------------------------

@pytest.fixture
def web(db, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    fake = FakePipeline()
    imp = importer.Importer(cfg, db, pipeline_factory=lambda c, cam: fake)
    app = create_app(cfg, db, None, importer=imp)
    return TestClient(app, client=("127.0.0.1", 5000)), imp


def test_preview_and_start(web, tmp_path, db):
    client, imp = web
    video = _video(tmp_path / "VID_20261009_143012.avi")
    files = client.post("/api/import/preview", json={"path": str(tmp_path)}).json()["files"]
    assert files[0]["start"] == "2026-10-09T14:30:12" and files[0]["start_from"] == "file name"
    assert files[0]["fps"] == 20.0 and files[0]["duration_s"] == 1.5
    r = client.post("/api/import/start", json={"files": [{"path": str(video), "start": "2026-10-09T14:30:12"}],
                                               "camera": "phone", "keep_days": 0})
    assert r.status_code == 200, r.json()
    for _ in range(200):
        if not imp.running:
            break
        time.sleep(0.02)
    assert db.list_audit(5)[0]["action"] == "import videos"
    cams = client.get("/api/cameras").json()
    assert {"name": "phone", "mode": "security", "source": "imported videos", "live": False} in cams


@pytest.mark.parametrize("change,message", [
    ({"camera": "my phone"}, "letters, digits"),
    ({"camera": "cam1"}, "live camera's name"),               # the config's camera
    ({"keep_days": -1}, "keep days"),
    ({"target_fps": 0}, "frames per second"),
    ({"mode": "fast"}, "kiosk or security"),
    ({"imgsz": 800}, "detector size is 640, 960 or 1280"),
    ({"files": [{"path": "C:/nothing.mp4", "start": "2026-10-09T14:30:00"}]}, "not a video file"),
    ({"start": "yesterday"}, "start time must look like"),
])
def test_start_is_checked(web, tmp_path, change, message):
    client, _ = web
    video = _video(tmp_path / "a.avi")
    body = {"files": [{"path": str(video), "start": change.pop("start", "2026-10-09T14:30:00")}],
            "camera": "phone", "mode": "security", "keep_days": 0, "target_fps": 10, **change}
    r = client.post("/api/import/start", json=body)
    assert r.status_code == 400 and message in r.json()["detail"], r.json()


def test_delete_an_import(web, db):
    client, _ = web
    cam = db.get_or_create_camera("phone", IMPORTED, "security")
    for i in range(3):
        db.create_track(cam, i, datetime(2025, 1, 1, 8, i))
    live = db.get_or_create_camera("cam1", "0", "kiosk")
    db.create_track(live, 9, NOW)
    assert client.post("/api/import/delete", json={"camera": "cam1"}).status_code == 404   # not an import
    r = client.post("/api/import/delete", json={"camera": "phone"})
    assert r.json()["deleted"] == 3 and len(db.camera_track_ids("cam1")) == 1
    assert db.list_audit(1)[0]["detail"] == "imported camera phone: 3 people"


def test_imported_footage_keeps_its_own_retention(db, tmp_path):
    from run import _apply_retention
    cfg = copy.deepcopy(DEFAULTS)
    cfg["cameras"] = [{"name": "cam1"}]
    old = datetime.now() - timedelta(days=400)
    for name, keep in (("forever", 0), ("year", 365)):
        cam = db.get_or_create_camera(name, IMPORTED, "security")
        db.set_camera_retention(name, keep)
        db.create_track(cam, 1, old)
    _apply_retention(db, cfg, tmp_path)                     # the 30-day rule does not delete imports
    assert len(db.camera_track_ids("forever")) == 1 and db.camera_track_ids("year") == []


def _imported_person(db, video, offset_s=0.5, in_view=1.0):
    cam = db.get_or_create_camera("phone", IMPORTED, "security")
    start = datetime(2026, 1, 1, 8, 0)
    t = db.create_track(cam, 1, start + timedelta(seconds=offset_s))
    db.update_track(t, in_view_s=in_view)
    db.set_sources([t], str(video), start)
    return t


def test_imported_people_remember_their_file_and_moment(db, tmp_path):
    video = _video(tmp_path / "door cam.avi")
    t = _imported_person(db, video, offset_s=31.4)
    assert db.get_tracks([t])[0]["source"] == {"file": "door cam.avi", "offset_s": 31.4}   # the name, not the folder
    assert db.track_source(t)[0] == str(video)


def test_excerpt_of_the_original(tmp_path):
    video = _video(tmp_path / "v.avi", frames=200, fps=20.0)                # 10 s
    out = importer.make_excerpt(video, offset_s=4.0, in_view_s=0.5, out=tmp_path / "e.webm", fps=10)
    cap = cv2.VideoCapture(str(out))
    n = 0
    while cap.read()[0]:
        n += 1
    assert 43 <= n <= 47                     # 2 s before to 2 s after the 0.5 s in view: 2.0-6.5 s at 10 per second
    with pytest.raises(ValueError, match="no pictures"):
        importer.make_excerpt(video, offset_s=60, in_view_s=1, out=tmp_path / "late.webm")


def test_original_is_played_for_admins_only(web, db, tmp_path):
    client, _ = web
    video = _video(tmp_path / "v.avi", frames=60, fps=20.0)
    t = _imported_person(db, video)
    r = client.get(f"/api/tracks/{t}/original")
    assert r.status_code == 200 and r.headers["content-type"] == "video/webm" and len(r.content) > 0
    assert db.list_audit(1)[0]["detail"] == f"person #{t}: v.avi at 0 s"
    live = db.create_track(db.get_or_create_camera("cam1", "0", "kiosk"), 2, NOW)
    assert client.get(f"/api/tracks/{live}/original").status_code == 404     # not imported
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(client.app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get(f"/api/tracks/{t}/original").status_code == 403
    deleted, paths = db.delete_tracks([t])                                   # the excerpt goes with the person
    assert f"clips/excerpt_{t}.webm" in paths


def test_original_moved_away(web, db, tmp_path):
    client, _ = web
    video = _video(tmp_path / "v.avi")
    t = _imported_person(db, video)
    video.unlink()
    r = client.get(f"/api/tracks/{t}/original")
    assert r.status_code == 404 and "no longer where it was imported from" in r.json()["detail"]


def test_viewers_cannot_import(web, db):
    client, _ = web
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(client.app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/import").status_code == 403
    assert viewer.post("/api/import/preview", json={"path": "C:/"}).status_code == 403
    assert viewer.get("/import", follow_redirects=False).status_code == 303
    assert viewer.get("/api/cameras").status_code == 200

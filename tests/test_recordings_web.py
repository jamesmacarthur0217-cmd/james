"""The Recordings page's API (labeler/recordings_web.py): the day list, finding a moment, who may get what,
the blurred stream for viewers, deleting, and the audit entries."""
import copy
import json
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeler import accounts, privacy
from labeler.config import DEFAULTS
from labeler.recordings_web import frame_index
from labeler.web import create_app

DAY = datetime(2026, 10, 12)
T0 = DAY.replace(hour=14)


def _recording(db, base, start, seconds=3.0, fps=10, open_=False, heads=((10, 10, 40, 40),)):
    """A small real video (mp4v, readable by OpenCV) with its frame list, and its row."""
    rel = f"cam1/{start:%Y-%m-%d}/{start:%H%M%S}.mp4"
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    n = int(seconds * fps)
    lines = [json.dumps({"w": 160, "h": 120, "fps": fps})]
    for i in range(n):
        w.write(np.full((120, 160, 3), 40 + i, np.uint8))
        lines.append(json.dumps([round(i / fps, 2), [list(h) for h in heads]]))
    w.release()
    path.with_suffix(".jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rec_id = db.add_recording("cam1", rel, start, "mp4v")
    if not open_:
        db.update_recording(rec_id, frames=n, size_bytes=path.stat().st_size,
                            ended_at=start + timedelta(seconds=seconds))
    return rec_id


@pytest.fixture
def setup(db, camera, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cfg["recording"]["dir"] = str(tmp_path / "rec")
    cfg["cameras"] = [{"name": "cam1", "source": 0}]
    first = _recording(db, tmp_path / "rec", T0)
    second = _recording(db, tmp_path / "rec", T0 + timedelta(minutes=10), open_=True)
    other_day = _recording(db, tmp_path / "rec", T0 - timedelta(days=1))
    db.create_track(db.get_or_create_camera("cam1", "0", "kiosk"), 1, T0 + timedelta(seconds=1))
    app = create_app(cfg, db, None)
    admin = TestClient(app, client=("127.0.0.1", 5000))
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    assert viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"}).status_code == 200
    return {"admin": admin, "viewer": viewer, "ids": (first, second, other_day), "base": tmp_path / "rec"}


def test_frame_index():
    times = [0.0, 0.1, 0.2, 0.5, 0.6]
    assert frame_index(times, 0) == 0 and frame_index(times, 0.15) == 2 and frame_index(times, 0.3) == 3
    assert frame_index(times, 9) == 4 and frame_index([], 1) == 0


def test_day_list_with_people_and_sizes_for_admins_only(setup):
    first, second, _ = setup["ids"]
    d = setup["admin"].get("/api/recordings?camera=cam1&day=2026-10-12").json()
    assert [r["id"] for r in d["recordings"]] == [first, second]
    assert d["recordings"][1]["open"] is True and "size_mb" in d["recordings"][0]
    assert d["people"][0]["at"] == "2026-10-12T14:00:01" and d["admin"] is True
    v = setup["viewer"].get("/api/recordings?camera=cam1&day=2026-10-12").json()
    assert "size_mb" not in v["recordings"][0] and v["admin"] is False
    assert setup["admin"].get("/api/recordings?day=12-10-2026").status_code == 400


def test_find_a_moment(setup):
    first, second, _ = setup["ids"]
    f = setup["admin"].get("/api/recordings/find?camera=cam1&at=2026-10-12T14:00:01.5").json()
    assert f["id"] == first and f["offset_s"] == 1.5 and f["frame"] == 15 and f["video_s"] == 1.5
    f = setup["admin"].get("/api/recordings/find?camera=cam1&at=2026-10-12T14:09:30").json()   # 30 s before the next
    assert f["id"] == second and f["offset_s"] == 0
    assert setup["admin"].get("/api/recordings/find?camera=cam1&at=2026-10-12T14:05:00").status_code == 404


def test_viewers_may_watch_the_blurred_stream_but_not_get_the_file(setup, monkeypatch):
    first, _, _ = setup["ids"]
    v = setup["viewer"]
    assert v.get("/recordings").status_code == 200
    assert v.get(f"/recordings/file/{first}").status_code == 403
    assert v.get(f"/api/recordings/{first}/download").status_code == 403
    assert v.post("/api/recordings/delete", json={"ids": [first]}).status_code == 403
    blurred = []
    real = privacy.blur_boxes
    monkeypatch.setattr(privacy, "blur_boxes", lambda img, boxes: blurred.append(boxes) or real(img, boxes))
    with v.stream("GET", f"/recordings/stream/{first}?t=1") as r:
        assert r.status_code == 200
        chunk = next(r.iter_bytes())
    assert b"Content-Type: image/jpeg" in chunk and blurred and blurred[0] == [[10, 10, 40, 40]]
    blurred.clear()
    with setup["admin"].stream("GET", f"/recordings/stream/{first}") as r:
        next(r.iter_bytes())
    assert blurred == []                                   # admins see faces


def test_admins_get_the_file_audited_once_per_playback(setup, db):
    first, _, _ = setup["ids"]
    a = setup["admin"]
    before = len(db.list_audit(100))
    r = a.get(f"/recordings/file/{first}", headers={"Range": "bytes=0-"})
    assert r.status_code == 206 and r.headers["content-type"] == "video/mp4"
    a.get(f"/recordings/file/{first}", headers={"Range": "bytes=1000-"})   # the browser seeking: not a new entry
    entries = db.list_audit(100)
    assert len(entries) == before + 1 and entries[0]["action"] == "view recording"
    r = a.get(f"/api/recordings/{first}/download")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert db.list_audit(1)[0]["action"] == "download recording"


def test_delete_removes_files_but_not_the_one_being_written(setup, db):
    first, second, _ = setup["ids"]
    r = setup["admin"].post("/api/recordings/delete", json={"ids": [first, second, 999]}).json()
    assert r == {"deleted": 1}
    assert [x["id"] for x in db.list_recordings(start=DAY, end=DAY + timedelta(days=1))] == [second]
    assert not list((setup["base"] / "cam1" / "2026-10-12").glob("140000.*"))
    assert list((setup["base"] / "cam1" / "2026-10-11").glob("140000.*"))       # another day's is kept
    assert db.list_audit(1)[0]["action"] == "delete recordings"


def test_page_has_faster_playback():
    """Phase 37: speed 1x-8x and -10/+10 s under the video, keys, and the speed kept for the next file."""
    from pathlib import Path
    page = (Path(__file__).resolve().parent.parent / "labeler" / "static" / "recordings.html").read_text(encoding="utf-8")
    for n in (1, 2, 4, 8):
        assert f'data-speed="{n}"' in page
    assert 'data-skip="-10"' in page and 'data-skip="10"' in page
    assert 'addEventListener("keydown"' in page and "video.defaultPlaybackRate = n" in page
    assert page.count('$("speed-bar").hidden = false') == 3   # the admins' file, the time-lapse, the stream
    assert "&speed=${speed}" in page and "function restartStream" in page and "function pauseStream" in page


def test_stream_speed_sends_every_nth_frame_at_the_same_pace(setup):
    """Phase 37 step 2: the viewers' stream at 2x has half the frames and takes half the time; d still ends it."""
    import time
    first, _, _ = setup["ids"]
    v = setup["viewer"]

    def frames(url):
        began = time.monotonic()
        body = v.get(url).content
        return body.count(b"--frame"), time.monotonic() - began

    n1, t1 = frames(f"/recordings/stream/{first}?t=0&d=2")             # 3 s at 10 fps: frames 0.0 ... 2.0 s
    n2, t2 = frames(f"/recordings/stream/{first}?t=0&d=2&speed=2")     # every 2nd of them
    n8, _ = frames(f"/recordings/stream/{first}?t=1&speed=8")          # from 1 s to the end: frames 10, 18, 26
    assert (n1, n2, n8) == (21, 11, 3)
    assert t2 < 0.75 * t1 and t1 > 1.8
    assert v.get(f"/recordings/stream/{first}?speed=3").status_code == 400

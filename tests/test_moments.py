"""Search recordings by words (Phase 16): moments from what a search found (labeler/moments.py) and the API."""
import copy
import json
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeler import accounts
from labeler.config import DEFAULTS
from labeler.moments import moments
from labeler.web import create_app

T0 = datetime(2026, 10, 12, 14, 0, 0)


def _recording(db, base, start, times, fps=10, camera="cam1"):
    """A small real video (mp4v) whose frames were taken at `times` (seconds from its start), and its row."""
    rel = f"{camera}/{start:%Y-%m-%d}/{start:%H%M%S}.mp4"
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    lines = [json.dumps({"w": 160, "h": 120, "fps": fps})]
    for i, t in enumerate(times):
        w.write(np.full((120, 160, 3), 40 + i % 200, np.uint8))
        lines.append(json.dumps([t, [[10, 10, 40, 40]]]))
    w.release()
    path.with_suffix(".jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rec_id = db.add_recording(camera, rel, start, "mp4v")
    db.update_recording(rec_id, frames=len(times), size_bytes=path.stat().st_size,
                        ended_at=start + timedelta(seconds=times[-1]))
    return rec_id


def _even(seconds, fps=10):
    return [round(i / fps, 2) for i in range(int(seconds * fps))]


def _thing(type_, id_, start_s, end_s, camera="cam1", **extra):
    iso = lambda s: (T0 + timedelta(seconds=s)).isoformat(timespec="seconds")   # noqa: E731
    return {"type": type_, "id": id_, "camera": camera, "started_at": iso(start_s),
            "ended_at": iso(end_s) if end_s is not None else None, "crop": f"{type_}_{id_}.jpg", **extra}


def test_people_together_are_one_moment_at_the_right_second(db, tmp_path):
    rec = _recording(db, tmp_path, T0, _even(30))
    man = _thing("person", 1, 10, 14, attributes={"apparent_gender": {"value": "man"},
                                                  "upper_color": {"value": "red"}, "upper_type": {"value": "t-shirt"}})
    woman = _thing("person", 2, 13, 18)
    later = _thing("person", 3, 25, 26)
    got = moments(db, [man, woman, later], tmp_path, 10, pad_s=1)
    assert (got["found"], got["on_video"], got["not_on_video"]) == (3, 3, 0)
    first, second = got["moments"]                                   # newest first
    assert [i["id"] for i in first["items"]] == [3] and [i["id"] for i in second["items"]] == [1, 2]
    assert second["parts"] == [{"recording": rec, "offset_s": 9.0, "end_offset_s": 19.0, "video_s": 9.0,
                                "video_end_s": 19.1, "start": "2026-10-12T14:00:09", "end": "2026-10-12T14:00:19"}]
    assert second["length_s"] == 10.0 and second["picture"] == "person_1.jpg"
    assert second["items"][0]["text"] == "man, red t-shirt"


def test_uneven_frames_place_the_video_by_frame_not_by_clock(db, tmp_path):
    times = [0.0, 0.1, 0.2, 2.0, 2.1, 2.2, 6.0, 6.1]                 # the camera stalled twice
    _recording(db, tmp_path, T0, times)
    part = moments(db, [_thing("person", 1, 2, 2)], tmp_path, 10, pad_s=0)["moments"][0]["parts"][0]
    assert (part["offset_s"], part["end_offset_s"]) == (2.0, 2.0)
    assert (part["video_s"], part["video_end_s"]) == (0.3, 0.4)      # frame 3 of a 10 fps file, not 2 s in


def test_a_visit_over_two_files_has_two_parts(db, tmp_path):
    a = _recording(db, tmp_path, T0, _even(10))
    b = _recording(db, tmp_path, T0 + timedelta(seconds=12), _even(10))
    m = moments(db, [_thing("person", 1, 8, 15)], tmp_path, 10, pad_s=0)["moments"]
    assert len(m) == 1 and [p["recording"] for p in m[0]["parts"]] == [a, b]
    assert m[0]["parts"][0]["offset_s"] == 8.0 and m[0]["parts"][1]["end_offset_s"] == 3.0
    assert (m[0]["start"], m[0]["end"]) == ("2026-10-12T14:00:08", "2026-10-12T14:00:15")


def test_things_without_video_are_counted_not_listed(db, tmp_path):
    _recording(db, tmp_path, T0, _even(10))
    found = [_thing("person", 1, 3, 4), _thing("person", 2, 600, 610),            # 10 minutes later: no file
             _thing("vehicle", 5, 2, 9, camera="gate", kind="car", color="white", make="Toyota"),   # no recording
             _thing("animal", 6, 5, None, kind="dog", color="black")]               # still in view
    got = moments(db, found, tmp_path, 10, pad_s=0, now=T0 + timedelta(seconds=8))
    assert (got["found"], got["on_video"], got["not_on_video"]) == (4, 2, 2)
    assert [[i["text"] for i in m["items"]] for m in got["moments"]] == [["black dog"], ["a person"]]
    assert got["moments"][0]["end"] == "2026-10-12T14:00:08"                       # the dog: until now
    from labeler.moments import describe
    assert describe(found[2]) == "white Toyota car" and describe({**found[2], "make": "unknown"}) == "white car"


# --- the API ---

class Agent:
    def __init__(self, result):
        self.result, self.asked = result, []

    def run(self, question, limit=30):
        self.asked.append((question, limit))
        return self.result


@pytest.fixture
def app_with(db, tmp_path):
    def make(agent=None, embedder=None):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
        cfg["recording"]["dir"] = str(tmp_path / "rec")
        cfg["cameras"] = [{"name": "cam1", "source": 0}]
        app = create_app(cfg, db, None, embedder=embedder, agent=agent)
        admin = TestClient(app, client=("127.0.0.1", 5000))
        if not db.get_user("vic"):
            db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
        viewer = TestClient(app, client=("192.168.1.20", 5000))
        assert viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"}).status_code == 200
        return admin, viewer
    return make


def test_search_returns_moments_for_admins_and_viewers(db, tmp_path, app_with):
    rec = _recording(db, tmp_path / "rec", T0, _even(30))
    agent = Agent({"explanation": "Searched: all cameras", "people": [_thing("person", 1, 10, 12)],
                   "vehicles": [_thing("vehicle", 2, 11, 13, kind="car", color="red")]})
    admin, viewer = app_with(agent=agent)
    for client in (admin, viewer):
        r = client.post("/api/recordings/search", json={"question": "  man   with a backpack ", "limit": 500})
        assert r.status_code == 200, r.text
        d = r.json()
        assert (d["question"], d["how"], d["found"], d["on_video"]) == ("man with a backpack", "question", 2, 2)
        assert [i["type"] for i in d["moments"][0]["items"]] == ["person", "vehicle"]
        assert d["moments"][0]["parts"][0]["recording"] == rec
    assert agent.asked[0] == ("man with a backpack", 100)
    assert db.list_audit(1)[0]["action"] == "search recordings"
    assert admin.post("/api/recordings/search", json={"question": "   "}).status_code == 400


def test_search_without_the_vision_model(db, tmp_path, app_with):
    admin, _ = app_with()
    assert admin.post("/api/recordings/search", json={"question": "red top"}).status_code == 503
    cam = db.get_or_create_camera("cam1", "0", "kiosk")
    tid = db.create_track(cam, 1, T0 + timedelta(seconds=5))
    _recording(db, tmp_path / "rec", T0, _even(20))

    class Embedder:
        model_name = "test"

        def text(self, q):
            return np.ones(4, np.float32)
    db.nearest = lambda vec, model, limit=20, exclude=None: [(tid, 0.9)]
    admin, _ = app_with(embedder=Embedder())
    d = admin.post("/api/recordings/search", json={"question": "red top"}).json()
    assert d["how"] == "appearance" and "vision model is off" in d["explanation"]
    assert d["found"] == 1 and d["moments"][0]["items"][0]["id"] == tid


def test_the_stream_stops_at_the_end_of_the_moment(db, tmp_path, app_with):
    rec = _recording(db, tmp_path / "rec", T0, _even(5))
    _, viewer = app_with()
    with viewer.stream("GET", f"/recordings/stream/{rec}?t=1&d=0.5") as r:
        frames = r.read().count(b"Content-Type: image/jpeg")
    assert frames == 6                                              # 1.0 ... 1.5 s, not to the file's end
    assert db.list_audit(1)[0]["detail"].endswith("from 1 s for 0.5 s")


def test_a_busy_camera_is_not_one_long_moment(db, tmp_path):
    _recording(db, tmp_path, T0, _even(200))
    chain = [_thing("person", i, 25 * i, 25 * i + 30) for i in range(6)]          # each overlaps the next
    got = moments(db, chain, tmp_path, 10, pad_s=0)["moments"]
    assert [[i["id"] for i in m["items"]] for m in reversed(got)] == [[0, 1], [2, 3], [4, 5]]
    assert all(m["length_s"] <= 60 for m in got)
    long = moments(db, [_thing("person", 9, 0, 90)], tmp_path, 10, pad_s=0)["moments"]
    assert len(long) == 1 and long[0]["length_s"] == 90.0                          # one person seen long: whole


def test_recordings_page_has_the_search():
    from pathlib import Path
    page = (Path(__file__).resolve().parent.parent / "labeler" / "static" / "recordings.html").read_text(encoding="utf-8")
    for needle in ('id="find"', 'fetch("/api/recordings/search"', 'id="play-all"', "playMoment(", "&d=${until.seconds}",
                   "?start=${p.video_s}&end=${p.video_end_s}", "none of them is on video"):
        assert needle in page, needle

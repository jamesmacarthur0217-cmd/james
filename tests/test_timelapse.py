"""Time-lapse of a day (Phase 34): labeler/timelapse.py with small real videos and a fake encoder."""
import copy
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from labeler import timelapse
from labeler.config import DEFAULTS
from test_moments import T0, _even, _recording

DAY = T0.strftime("%Y-%m-%d")


class Writer:
    """Collects frames instead of encoding them; writes a small file so the result exists."""
    made = []

    def __init__(self, path, size):
        self.path, self.size, self.frames = path, size, []
        Writer.made.append(self)

    def write(self, frame):
        assert frame.shape[1::-1] == self.size
        self.frames.append(frame)

    def close(self):
        self.path.write_bytes(b"fake mp4")
        return True, ""


@pytest.fixture
def setup(db, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["recording"]["dir"] = str(tmp_path / "rec")
    a = _recording(db, tmp_path / "rec", T0, _even(30))                           # 300 frames
    b = _recording(db, tmp_path / "rec", T0 + timedelta(minutes=5), _even(20))    # 200 frames
    _recording(db, tmp_path / "rec", T0 - timedelta(days=1), _even(5))            # another day
    Writer.made = []
    return cfg, tmp_path / "rec", (a, b)


def test_plan_keeps_every_nth_frame_to_fit_the_length():
    srcs = [{"id": 1, "start": "2026-10-12T14:00:00", "times": _even(30)},
            {"id": 2, "start": "2026-10-12T14:05:00", "times": _even(20)}]
    info = timelapse.plan(srcs, seconds=4)                                      # 4 s x 25 = 100 frames for 500
    assert info["step"] == 5 and info["total"] == 500 and len(info["frames"]) == 100
    assert info["frames"][0] == [1, 0.0, "2026-10-12T14:00:00"] and info["frames"][60] == [2, 0.0, "2026-10-12T14:05:00"]
    assert info["sources"] == [[1, 300], [2, 200]]
    assert timelapse.plan(srcs, seconds=60)["step"] == 1                        # a short day: every frame


def test_fit_keeps_one_size():
    tall = np.full((200, 100, 3), 255, np.uint8)
    out = timelapse.fit(tall, (160, 120))
    assert out.shape == (120, 160, 3) and out[:, :50].max() == 0 and out[60, 80].min() == 255   # black bars beside


def test_make_writes_the_day_and_a_map_that_opens_the_recordings(db, setup):
    cfg, base, (a, b) = setup
    srcs = timelapse.sources(db, base, "cam1", DAY)
    assert [s["id"] for s in srcs] == [a, b]                                     # not the day before
    info = timelapse.make(cfg, base, "cam1", DAY, srcs, seconds=4, writer_factory=Writer)
    w = Writer.made[0]
    assert len(w.frames) == len(info["frames"]) == 100 and w.size == (160, 120)
    assert info["frames"][61] == [b, 0.5, "2026-10-12T14:05:00"]
    folder = base / "cam1" / DAY
    assert (folder / "timelapse.mp4").is_file() and json.loads((folder / "timelapse.json").read_text())["step"] == 5
    assert timelapse.current(base, "cam1", DAY, srcs) is not None
    (folder / "140000.jsonl").write_text((folder / "140000.jsonl").read_text() + "[30.0, []]\n")   # it grew
    assert timelapse.current(base, "cam1", DAY, timelapse.sources(db, base, "cam1", DAY)) is None


def test_today_s_recording_growing_keeps_it_marked_older(db, setup):
    cfg, base, (a, b) = setup
    db.update_recording(b, ended_at=None)                                         # still being written
    srcs = timelapse.sources(db, base, "cam1", DAY)
    timelapse.make(cfg, base, "cam1", DAY, srcs, seconds=4, writer_factory=Writer)
    assert timelapse.current(base, "cam1", DAY, srcs)["older"] is False
    folder = base / "cam1" / DAY
    (folder / "140500.jsonl").write_text((folder / "140500.jsonl").read_text() + "[20.0, []]\n")   # it grew
    grown = timelapse.sources(db, base, "cam1", DAY)
    assert timelapse.current(base, "cam1", DAY, grown)["older"] is True
    maker = timelapse.Maker(cfg, db, writer_factory=Writer)
    assert maker.start("cam1", DAY)["state"] == "done"                            # shown as it is
    assert maker.start("cam1", DAY, again=True)["state"] in ("making", "done")    # made anew on request


def test_a_file_shorter_than_its_frame_list_keeps_the_map_right(db, setup):
    cfg, base, (a, b) = setup
    srcs = timelapse.sources(db, base, "cam1", DAY)
    srcs[0]["times"] = srcs[0]["times"] + [30.0 + i / 10 for i in range(100)]  # 100 frames listed, not in the file
    info = timelapse.make(cfg, base, "cam1", DAY, srcs, seconds=4, writer_factory=Writer)
    assert len(Writer.made[0].frames) == len(info["frames"])
    assert {f[0] for f in info["frames"]} == {a, b}


def test_the_time_lapse_goes_with_the_days_last_recording(db, setup, tmp_path):
    cfg, base, (a, b) = setup
    timelapse.make(cfg, base, "cam1", DAY, timelapse.sources(db, base, "cam1", DAY), 4, writer_factory=Writer)
    folder = base / "cam1" / DAY
    timelapse.drop_if_alone(folder)
    assert (folder / "timelapse.mp4").is_file()                                 # recordings still there
    for p in folder.glob("1*"):
        p.unlink()
    timelapse.drop_if_alone(folder)
    assert not list(folder.glob("timelapse*"))


def test_api_and_page(db, setup):
    from pathlib import Path
    from fastapi.testclient import TestClient
    from labeler import accounts
    from labeler.web import create_app
    cfg, base, _ = setup
    cfg["cameras"] = [{"name": "cam1", "source": 0}]
    app = create_app(cfg, db, None)
    app.state.timelapse.writer_factory = Writer
    admin = TestClient(app, client=("127.0.0.1", 5000))
    q = {"camera": "cam1", "day": DAY}
    assert admin.get("/api/recordings/timelapse", params=q).json()["state"] == "none"
    assert admin.get("/api/recordings/timelapse", params={**q, "camera": "../x"}).status_code == 404
    assert admin.post("/api/recordings/timelapse", json={**q, "day": "2026-10-01"}).status_code == 404
    assert admin.post("/api/recordings/timelapse", json=q).json()["state"] in ("making", "done")
    for _ in range(100):
        d = admin.get("/api/recordings/timelapse", params=q).json()
        if d["state"] == "done":
            break
        time.sleep(0.05)
    assert d["state"] == "done" and len(d["frames"]) == 500 and d["fps"] == 25      # a short day: every frame
    folder = base / "cam1" / DAY
    (folder / "timelapse.mp4").write_bytes(b"plain")
    (folder / "timelapse.blurred.mp4").write_bytes(b"blurred")
    assert admin.get(f"/recordings/timelapse/cam1/{DAY}.mp4").content == b"plain"
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get(f"/recordings/timelapse/cam1/{DAY}.mp4").content == b"blurred"   # faces blurred for viewers
    assert viewer.get("/api/recordings/timelapse", params=q).status_code == 200
    assert [e["action"] for e in db.list_audit(10)].count("view time-lapse") == 2
    page = (Path(__file__).resolve().parent.parent / "labeler" / "static" / "recordings.html").read_text(encoding="utf-8")
    for needle in ('id="lapse"', 'id="lapse-open"', "playAt(ms(f[2]))", "/recordings/timelapse/"):
        assert needle in page, needle


def test_maker_runs_one_at_a_time_and_says_when_it_is_ready(db, setup):
    cfg, base, _ = setup
    maker = timelapse.Maker(cfg, db, writer_factory=Writer)
    with pytest.raises(ValueError):
        maker.start("cam1", "2026-10-01")                                       # nothing recorded
    first = maker.start("cam1", DAY)
    assert first["state"] in ("making", "done")
    for _ in range(100):
        if maker.status()["state"] != "making":
            break
        time.sleep(0.05)
    assert maker.status()["state"] == "done" and maker.status()["frames"] > 0
    again = maker.start("cam1", DAY)                                            # kept: ready at once
    assert again["state"] == "done" and again["frames"] == maker.status()["frames"]


def test_a_blurred_copy_for_viewers(db, setup, monkeypatch):
    cfg, base, _ = setup
    from labeler import privacy
    blurred = []
    real = privacy.blur_boxes
    monkeypatch.setattr(privacy, "blur_boxes", lambda img, boxes: blurred.append(boxes) or real(img, boxes))
    timelapse.make(cfg, base, "cam1", DAY, timelapse.sources(db, base, "cam1", DAY), 4, writer_factory=Writer)
    plain, faces = Writer.made
    assert len(plain.frames) == len(faces.frames) == 100 and blurred[0] == [[10, 10, 40, 40]]
    assert (base / "cam1" / DAY / "timelapse.blurred.mp4").is_file()
    Writer.made = []
    timelapse.make(cfg, base, "cam1", DAY, timelapse.sources(db, base, "cam1", DAY), 4, writer_factory=Writer,
                   blurred=False)
    assert len(Writer.made) == 1

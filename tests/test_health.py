"""Camera health: offline, blank and slow cameras are reported once, after a while, and closed when fixed."""
import copy
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import update

from labeler import accounts
from labeler.config import DEFAULTS
from labeler.db import CameraEvent
from labeler.health import HealthMonitor
from labeler.web import create_app

CFG = DEFAULTS["health"]       # offline 10 s, blank 10 s, slow 30 s below 5 fps, 30 s grace


class FakeCamera:
    def __init__(self, name="cam1"):
        self.name, self.started_mono, self.last_frame_at, self.blank, self.fps = name, 0.0, 0.0, False, 25.0


def _run(monitor, cam, seconds, start, frames=True):
    """Advance time second by second; returns the problems that started."""
    started = []
    for t in range(start, start + seconds):
        if frames:
            cam.last_frame_at = t
        started += monitor.check([cam], now=t)
    return [e["kind"] for e in started]


def test_nothing_is_reported_while_the_camera_warms_up(db):
    m, cam = HealthMonitor(db, CFG), FakeCamera()
    cam.blank, cam.fps = True, 0.5
    assert _run(m, cam, 29, 0) == [] and m.current() == []


def test_blank_picture_after_10_seconds_and_closed_when_back(db):
    m, cam = HealthMonitor(db, CFG), FakeCamera()
    _run(m, cam, 40, 0)
    cam.blank = True
    assert _run(m, cam, 9, 40) == []                       # a short moment: not a problem yet
    assert _run(m, cam, 2, 49) == ["blank"]
    assert _run(m, cam, 20, 51) == []                      # reported once
    cam.blank = False
    _run(m, cam, 1, 71)
    assert m.current() == []
    e = db.list_camera_events()[0]
    assert e["kind"] == "blank" and e["ended_at"] is not None


def test_no_picture_is_offline_and_hides_blank_and_slow(db):
    m, cam = HealthMonitor(db, CFG), FakeCamera()
    _run(m, cam, 40, 0)
    cam.blank, cam.fps = True, 0.0                         # frozen values from the last frame
    assert _run(m, cam, 15, 40, frames=False) == ["offline"]
    assert [p["kind"] for p in m.current()] == ["offline"]
    _run(m, cam, 1, 55)                                    # frames again
    assert all(p["kind"] != "offline" for p in m.current())


def test_slow_camera_after_30_seconds(db):
    m, cam = HealthMonitor(db, CFG), FakeCamera()
    _run(m, cam, 40, 0)
    cam.fps = 3.0
    assert _run(m, cam, 29, 40) == []
    assert _run(m, cam, 2, 69) == ["slow"]
    assert "3.0 fps" in m.current()[0]["message"]


def test_problems_left_open_by_a_stopped_app_are_closed_at_start(db):
    m, cam = HealthMonitor(db, CFG), FakeCamera()
    _run(m, cam, 40, 0)
    _run(m, cam, 15, 40, frames=False)
    assert db.list_camera_events()[0]["ended_at"] is None
    HealthMonitor(db, CFG)                                 # the next start
    assert db.list_camera_events()[0]["ended_at"] is not None


def test_camera_events_retention(db):
    db.open_camera_event("cam1", "blank", "old")
    db.open_camera_event("cam1", "blank", "new")
    with db.Session.begin() as s:
        s.execute(update(CameraEvent).where(CameraEvent.message == "old")
                  .values(started_at=datetime.now() - timedelta(days=400)))
    assert db.cleanup_camera_events(365) == 1
    assert [e["message"] for e in db.list_camera_events()] == ["new"]


def test_live_status_shows_problems_and_history_is_admin_only(db, tmp_path):
    m, cam = HealthMonitor(db, CFG), FakeCamera()
    _run(m, cam, 40, 0)
    _run(m, cam, 15, 40, frames=False)
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    app = create_app(cfg, db, None, health=m)
    admin = TestClient(app, client=("127.0.0.1", 5000))
    assert [p["kind"] for p in admin.get("/api/status").json()["camera_problems"]] == ["offline"]
    assert admin.get("/api/camera-events").json()[0]["camera"] == "cam1"
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/status").json()["camera_problems"][0]["kind"] == "offline"   # on the Live page
    assert viewer.get("/api/camera-events").status_code == 403

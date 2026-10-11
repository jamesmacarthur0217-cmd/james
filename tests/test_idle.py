"""Idle mode: the detector slows down when nobody is in view and wakes on the first person."""
import threading
from types import SimpleNamespace

import numpy as np
import torch

from labeler import pipeline as pipeline_mod
from labeler.health import HealthMonitor
from labeler.pipeline import TrackState

from test_pipeline_logic import make_pipeline


def _idle_pipeline(db, after_s=5, fps=3, enabled=True):
    p = make_pipeline(db)
    p.cfg["idle"].update(enabled=enabled, after_s=after_s, fps=fps)
    p.idle_cfg = p.cfg["idle"]
    p.idle = False
    p.last_person_at = p.last_processed_at = 0.0
    p.wake_conf = 0.5
    p.min_box_area = 0.03
    p.ignore_zones = [(0.0, 0.0, 0.2, 0.2)]
    return p


def test_idle_after_nobody_in_view_then_a_few_frames_a_second(db):
    p = _idle_pipeline(db)
    assert not p.skip_while_idle(4.9) and not p.idle          # still within after_s
    p.last_processed_at = 5.0
    assert p.skip_while_idle(5.1) and p.idle                   # idle: too soon after the last processed frame
    assert not p.skip_while_idle(5.0 + 1 / 3 + 0.01)           # a third of a second later: look again
    p.tracks[1] = TrackState(1, None, 0.0)                     # somebody was found
    assert not p.skip_while_idle(5.4) and not p.idle           # full speed at once
    p.tracks.clear()
    assert not p.skip_while_idle(10.3)                         # they left: full speed for after_s more
    p.last_processed_at = 10.4
    assert p.skip_while_idle(10.5) and p.idle


def test_disabled_never_skips(db):
    p = _idle_pipeline(db, enabled=False)
    assert not p.skip_while_idle(1000) and not p.idle


def _boxes(*dets):
    """Ultralytics-like boxes: (x1, y1, x2, y2, conf) in a 640x480 frame."""
    data = torch.tensor([d[:4] for d in dets], dtype=torch.float32).reshape(-1, 4)
    conf = torch.tensor([d[4] for d in dets], dtype=torch.float32)
    return _B(data, conf)


class _B:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = xyxy, conf

    def __len__(self):
        return len(self.conf)


def test_wakes_on_a_clear_detection_of_a_person_who_would_be_recorded(db):
    shape = (480, 640, 3)
    p = _idle_pipeline(db)
    p._wake_on_detection(_boxes((200, 100, 400, 470, 0.3)), shape, 50.0)   # weak: the tracker would not start it
    p._wake_on_detection(_boxes((300, 300, 340, 360, 0.9)), shape, 50.0)   # too small for a kiosk
    p._wake_on_detection(_boxes((10, 10, 110, 80, 0.9)), shape, 50.0)      # in an ignore zone (a poster)
    p._wake_on_detection(None, shape, 50.0)
    assert p.last_person_at == 0.0
    p._wake_on_detection(_boxes((300, 300, 340, 360, 0.9), (200, 100, 400, 470, 0.7)), shape, 50.0)
    assert p.last_person_at == 50.0


def test_idle_camera_is_not_reported_slow(db):
    monitor = HealthMonitor(db, {"startup_grace_s": 0, "offline_after_s": 10, "min_fps": 5})
    cam = SimpleNamespace(started_mono=0.0, last_frame_at=99.0, blank=False, fps=3.0, idle=True)
    assert monitor.problems_now(cam, 100.0) == set()
    cam.idle = False
    assert monitor.problems_now(cam, 100.0) == {"slow"}


class _Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_run_loop_saves_frames_and_misses_nobody(db, monkeypatch):
    """30 fps for 40 s of fake time; a person stands in view from 20.0 s to 25.0 s."""
    clock = _Clock()

    class Source:
        kind, ended = "video", False

        def __init__(self, *a, **k):
            pass

        def read(self):
            clock.t += 1 / 30
            if clock.t > 40:
                self.ended = True
                return None
            return np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(pipeline_mod, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))
    monkeypatch.setattr(pipeline_mod, "FrameSource", Source)
    p = _idle_pipeline(db, after_s=5, fps=3)
    p.cam = {"source": "fake.mp4", "loop_files": False}
    p.stop_event = threading.Event()
    p.frames, p.fps = 0, 0.0
    processed = []

    def process(frame):
        processed.append(clock.t)
        if 20.0 <= clock.t < 25.0:
            p.tracks[1] = TrackState(1, None, clock.t)
        else:
            p.tracks.clear()
    monkeypatch.setattr(p, "process", process)
    monkeypatch.setattr(p, "_finish", lambda tid: None)
    p.run()

    seen = [t for t in processed if 20.0 <= t < 25.0]
    assert seen[0] - 20.0 <= 1 / 3 + 1 / 30               # found within one idle look of arriving
    assert len(seen) >= 30 * 5 - 12                        # then every frame while they stay
    full = 30 * 40
    assert len(processed) < full * 0.5                     # most empty frames skipped
    idle_before = [t for t in processed if 6.0 <= t < 19.9]
    assert 3 * 13 <= len(idle_before) <= 3 * 14 + 2        # ~3 a second while idle

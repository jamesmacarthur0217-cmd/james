"""The vision model's time (Phase 50): safety checks use its idle time (the per-minute limit holds only while people's
labels wait), and people passing by in under vlm.label_after_s get colours only."""
import copy
from datetime import datetime

import numpy as np

from labeler.config import DEFAULTS
from labeler.incidents import ConfirmQueue
from labeler.vlm import PEOPLE


class Worker:
    """A vision model queue stand-in: labels waiting as set; every check answered at once."""

    def __init__(self, labels_waiting: int):
        self.labels_waiting, self.sent, self.queue = labels_waiting, [], None

    def waiting(self, priority: int) -> int:
        return self.labels_waiting if priority == PEOPLE else 0

    def submit(self, job, pictures, mode=None, priority=0):
        self.sent.append(job)
        self.queue.on_result(job, [{"answer": "no", "what": ""}])
        return True


def checks_sent(labels_waiting: int) -> int:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["safety"]["confirm_per_min"] = 2
    w = Worker(labels_waiting)
    q = ConfirmQueue(cfg, w, {}, clock=lambda: 0.0)
    w.queue = q
    pic = [(datetime(2026, 10, 25, 12), np.zeros((8, 8, 3), np.uint8), [])]
    for n in range(5):   # five warnings about different people within the same second
        q.suspect("cam1", "floor", "something new", people=[n], pictures=pic)
    for _ in range(10):
        q.tick()
    return len(w.sent)


def test_checks_use_idle_time():
    assert checks_sent(labels_waiting=0) == 5   # nothing waits: no limit
    assert checks_sent(labels_waiting=3) == 2   # labels wait: at most confirm_per_min a minute, as before


def test_only_people_who_stay_are_labelled(db):
    from labeler.pipeline import TrackState
    from test_pipeline_logic import make_pipeline
    p = make_pipeline(db)
    p.cfg["vlm"]["label_after_s"] = 2.0
    st = TrackState(1, datetime.now(), 0.0, seen_since=10.0)
    st.n_frames = 25
    assert not p._due_for_label(st, 11.0)      # in view 1 s: a passer-by so far
    assert p._due_for_label(st, 12.0)          # 2 s: labelled
    st.n_frames = 10
    assert not p._due_for_label(st, 20.0)      # too few frames yet
    p.cfg["vlm"]["label_after_s"] = 0
    st.n_frames = 25
    assert p._due_for_label(st, 10.5)          # 0: everyone, as before
    st.vlm_requested = True
    assert not p._due_for_label(st, 30.0)      # asked already


def test_one_picture_only_while_labels_wait(db):
    from test_pipeline_logic import make_pipeline
    p = make_pipeline(db)
    p.vlm = Worker(labels_waiting=0)
    assert p.pictures_per_person() == 2                       # time to spare: two, as before
    p.vlm.labels_waiting = 4
    assert p.pictures_per_person() == 1                       # others wait: one
    p.cfg["vlm"]["one_when_busy"] = False
    assert p.pictures_per_person() == 2

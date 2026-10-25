"""The vision model's time (Phase 50): safety checks use its idle time (the per-minute limit holds only while people's
labels wait), people passing by in under vlm.label_after_s get colours only, and its load of the last 10 minutes is
kept for the Health page."""
import copy
import threading
from datetime import datetime

import numpy as np

from labeler.config import DEFAULTS
from labeler.incidents import ConfirmQueue
from labeler.vlm import PEOPLE, VLMWorker


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


class TwoSecondClient:
    """Each answer takes 2 s on a made-up clock; the first waits until `go` is set."""

    def __init__(self, clock: list):
        self.clock, self.started, self.go = clock, threading.Event(), threading.Event()

    def label(self, person, face=None, mode=None):
        self.started.set()
        assert self.go.wait(5)
        self.clock[0] += 2.0
        return {"answer": "no"}


def test_the_vision_model_keeps_its_load_of_the_last_10_minutes():
    t = [0.0]
    client = TwoSecondClient(t)
    w = VLMWorker(client, lambda job, results: None, max_queue=2, clock=lambda: t[0])
    pic = np.zeros((8, 8, 3), np.uint8)
    assert w.submit(1, [(pic, None)])
    assert client.started.wait(5)              # the first is with the model; two more wait
    assert w.submit(2, [(pic, None)]) and w.submit(3, [(pic, None)])
    assert not w.submit(4, [(pic, None)])      # the queue is full: refused (colour labels only)
    client.go.set()
    assert w.close(5)
    t[0] = 10.0                                # 6 s of answers in the first 10 s
    load = w.load()
    assert load["window_s"] == 10 and abs(load["busy"] - 0.6) < 1e-9 and load["answers_per_min"] == 18
    assert load["labels"] == 3 and load["label_wait_s"] == 2.0 and load["refused"] == 1   # waited 0, 2 and 4 s
    t[0] = 700.0                               # ten minutes on: all of it is older than the window
    assert w.load() == {"window_s": 600, "busy": 0.0, "answers_per_min": 0.0, "labels": 0, "label_wait_s": None,
                        "refused": 0}


def test_dropped_checks_are_counted_by_time():
    clock = [0.0]
    cfg = copy.deepcopy(DEFAULTS)
    q = ConfirmQueue(cfg, Worker(labels_waiting=0), {}, clock=lambda: clock[0])
    pic = [(datetime(2026, 10, 25, 12), np.zeros((8, 8, 3), np.uint8), [])]
    q.sent[99] = object()                      # one being checked: the next waits
    q.suspect("cam1", "floor", "something new", pictures=pic)
    clock[0] = cfg["safety"]["max_wait_s"] + 1
    q.tick()
    assert q.dropped_within(600) == 1
    clock[0] += 601
    assert q.dropped_within(600) == 0

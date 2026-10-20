"""Safety checks (Phase 44 step 2): the picture memory, the confirm queue (critical first, one at a time, a limit per
minute, too old dropped, cooldown), the vision model's shared priority queue, job numbers, the prompt and pictures."""
import copy
import threading
from datetime import datetime, timedelta

import numpy as np
import pytest

from labeler import vlm
from labeler.config import DEFAULTS
from labeler.incidents import CATEGORIES, ConfirmQueue, FrameMemory, job_id, number_of, pictures_for, prompt_for
from labeler.scene import job_id as scene_job

T0 = datetime(2026, 10, 19, 12, 0)


def _memory(n=10, every=1.0):
    m = FrameMemory(keep_s=10, every_s=1.0, side=960)
    for i in range(n):
        m.add(np.full((1080, 1920, 3), i, np.uint8), T0 + timedelta(seconds=i * every), i * every)
    return m


def test_the_memory_keeps_one_picture_a_second_for_the_last_seconds():
    m = FrameMemory(keep_s=10, every_s=1.0, side=960)
    kept = [m.add(np.zeros((1080, 1920, 3), np.uint8), T0, t / 2) for t in range(40)]   # 2 frames a second for 20 s
    assert sum(kept) == 20 and len(m) == 11   # 10 s back from the newest, both ends included
    pic = m.pick(1, 4)[0][1]
    assert pic.shape == (540, 960, 3)          # shrunk to 960 px
    m = _memory()
    got = m.pick(3, 4)
    assert [int(p[0, 0, 0]) for _, p in got] == [5, 7, 9]   # spread over the last 4 s, the newest included
    assert [int(p[0, 0, 0]) for _, p in m.pick(5, 2)] == [7, 8, 9]   # fewer there: all of them
    assert FrameMemory().pick(3, 4) == []


def test_job_numbers_do_not_mix():
    assert number_of(job_id(5)) == 5 and number_of(-5) is None and number_of(17) is None
    assert number_of(scene_job(3)) is None


class Worker:
    def __init__(self, ok=True):
        self.jobs, self.ok = [], ok

    def submit(self, job, frames, mode=None, priority=vlm.PEOPLE):
        self.jobs.append((job, frames, mode, priority))
        return self.ok


def _queue(worker=None, **safety):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["safety"].update(safety)
    clock = [100.0]
    q = ConfirmQueue(cfg, worker or Worker(), {"cam1": _memory()}, clock=lambda: clock[0])
    return q, clock


def test_critical_first_one_at_a_time_and_answers_to_listeners():
    worker = Worker()
    q, clock = _queue(worker)
    heard = []
    q.listeners.append(lambda s, a: heard.append((s.category, a["answer"], a["what"])))
    assert q.suspect("cam1", "floor", "patch on the walkway") == "queued"
    clock[0] += 1
    assert q.suspect("cam1", "person_down", "lying still 20 s", people=[7]) == "queued"
    sent = q.tick()
    assert sent.category == "person_down" and len(worker.jobs) == 1 and q.tick() is None   # one at a time
    job, frames, mode, priority = worker.jobs[0]
    assert priority == vlm.CRITICAL and mode[0] == "incident" and CATEGORIES["person_down"][2] in mode[1]
    assert len(frames) == 1 and len(frames[0][0]) == 3 and "3 pictures" in mode[1] and "2 seconds apart" in mode[1]
    q.on_result(job, [{"answer": "yes", "what": "a man lying on the floor", "seconds": "1.2"}])
    assert heard == [("person_down", "yes", "a man lying on the floor")]
    s = q.tick()
    assert s.category == "floor" and worker.jobs[1][3] == vlm.WARNING
    q.on_result(worker.jobs[1][0], [])   # the vision model failed
    st = q.state()
    assert (st["sent"], st["yes"], st["failed"], st["waiting"], st["checking"]) == (2, 1, 1, 0, 0)
    assert st["recent"][0]["what"] == "a man lying on the floor" and st["wait_median_s"] in (0.0, 1.0)


def test_limit_per_minute_too_old_and_cooldown():
    worker = Worker()
    q, clock = _queue(worker, confirm_per_min=2, max_wait_s=30, cooldown_s=60)
    for cat in ("fire", "fight", "weapon"):
        q.suspect("cam1", cat, "test")
    for _ in range(3):
        s = q.tick()
        if s:
            q.on_result(job_id(s.number), [{"answer": "no", "what": ""}])
    assert len(worker.jobs) == 2 and q.state()["waiting"] == 1   # the limit: 2 a minute
    clock[0] += 31
    assert q.tick() is None and q.state()["dropped_old"] == 1     # waited too long
    assert q.suspect("cam1", "fire", "again") == "cooldown"       # checked 31 s ago
    assert q.suspect("cam1", "fire", "other area", area={"id": 4, "name": "kitchen", "points": [(0, 0), (1, 0), (1, 1)]}) == "queued"
    assert q.suspect("cam1", "fire", "other area", area={"id": 4, "name": "kitchen", "points": []}) == "cooldown"   # already waiting
    assert q.suspect("cam2", "fire", "no memory") == "no pictures"
    with pytest.raises(ValueError):
        q.suspect("cam1", "aliens", "x")


def test_a_full_vision_model_queue_counts_as_failed():
    q, clock = _queue(Worker(ok=False))
    q.suspect("cam1", "crowd", "40 people in the hall")
    q.tick()
    assert q.state()["failed"] == 1 and q.state()["checking"] == 0


def test_the_prompt_and_the_area_cut_out():
    q, clock = _queue()
    area = {"id": 1, "name": "exit", "points": [(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)]}
    q.suspect("cam1", "exit", "boxes", area=area)
    s = q.waiting[0]
    assert 'cut out around the area "exit"' in prompt_for(s, 2) and "never how a person looks" in prompt_for(s, 2)
    assert pictures_for(s)[0].shape[0] < 540   # the area's part of the 960 x 540 picture
    assert vlm.clean_incident({"answer": "YES", "what": "  boxes   in the   doorway "}) == \
        {"answer": "yes", "what": "boxes in the doorway"}
    assert vlm.clean_incident({"answer": "maybe"})["answer"] == "unsure"
    one = vlm.incident_prompt(1, 0, "", "Is there smoke?")
    assert one.startswith("This is a picture from a security camera.") and one.endswith("Question: Is there smoke?")


class SlowClient:
    """Labels one job at a time; the first waits until released, so the others queue up behind it."""

    def __init__(self):
        self.started, self.release, self.order = threading.Event(), threading.Event(), []

    def label(self, person, face, mode=None):
        if not self.started.is_set():
            self.started.set()
            self.release.wait(5)
        self.order.append(mode)
        return {"answer": "no"}


def test_the_vision_model_queue_puts_critical_checks_before_people():
    client, done = SlowClient(), []
    worker = vlm.VLMWorker(client, lambda job, results: done.append(job), max_queue=2)
    pic = np.zeros((8, 8, 3), np.uint8)
    assert worker.submit(1, [(pic, None)], "first")
    assert client.started.wait(5)   # the first is being labeled now; the next job is taken only when it ends
    assert worker.submit(2, [(pic, None)], "person a") and worker.submit(3, [(pic, None)], "person b")
    assert not worker.submit(4, [(pic, None)], "person c")              # people: the queue is full
    assert worker.submit(5, [(pic, None)], "warning", priority=vlm.WARNING) is False   # also behind people
    assert worker.submit(6, [(pic, None)], "critical", priority=vlm.CRITICAL)        # never refused
    assert worker.waiting(vlm.PEOPLE) == 2 and worker.waiting(vlm.CRITICAL) == 1
    client.release.set()
    assert worker.close(5)
    assert client.order == ["first", "critical", "person a", "person b"] and done == [1, 6, 2, 3]

"""Mice and rats (Phase 55, labeler/pests.py): only food areas are looked at, each cut out and enlarged; seen on 2 of
the last 3 looks of an area raises one warning suspect with the pictures of those looks, cut out around the boxes;
that area not again for a minute; the camera's look with the model."""
from datetime import datetime, timedelta

import numpy as np

from labeler.pests import CUT, QUESTION, PestWatch, area_box, food_areas, in_area
from labeler.zones import ZoneCounter
from test_pipeline_logic import make_pipeline

SHAPE = (480, 640, 3)
KITCHEN = {"id": 1, "kind": "area", "role": "food_prep", "name": "kitchen",
           "points": [(0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)]}   # left bottom quarter
TABLE = {"id": 2, "kind": "area", "role": "dining", "name": "table",
         "points": [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]}     # right bottom quarter
WALKWAY = {"id": 3, "kind": "area", "role": "walkway", "name": "hall", "points": [(0, 0), (1, 0), (1, 0.5), (0, 0.5)]}
MOUSE = (100, 400, 130, 415)     # in the kitchen
T0 = datetime(2026, 10, 30, 12)


def looks(w, pattern, found, areas=(KITCHEN,), t0=0.0):
    """One look a second; pattern: "x" = found seen, "." = nothing. Returns (time, *raised) for each raise."""
    out = []
    for i, c in enumerate(pattern):
        frame = np.full(SHAPE, i % 250, np.uint8)
        for got in w.update(t0 + i, T0 + timedelta(seconds=t0 + i), frame, found if c == "x" else {}, list(areas)):
            out.append((t0 + i, *got))
    return out


def test_food_areas_and_boxes():
    assert food_areas([KITCHEN, TABLE, WALKWAY, {**KITCHEN, "kind": "line"}]) == [KITCHEN, TABLE]
    assert area_box(KITCHEN["points"], SHAPE) == (0, 240, 320, 480)
    assert in_area(MOUSE, SHAPE, KITCHEN["points"]) and not in_area(MOUSE, SHAPE, TABLE["points"])


def test_two_of_three_looks_raise_once():
    assert looks(PestWatch({}), "x..x..x", {1: [(0.5, MOUSE)]}) == []     # never 2 within 3 looks
    [(t, area, reason, crop, pictures, cut)] = looks(PestWatch({}), "x.xxx", {1: [(0.6, MOUSE)]})
    assert t == 2 and area is KITCHEN
    assert reason == "a mouse or rat seen 2 of the last 3 looks in kitchen (food preparation; up to 0.60)"
    assert crop == (100 / 640, 400 / 480, 130 / 640, 415 / 480)
    assert [p[0] for p in pictures] == [T0, T0 + timedelta(seconds=2)]       # the looks where it was seen
    assert [int(p[1][0, 0, 0]) for p in pictures] == [0, 2] and pictures[0][2] == []
    assert cut == (CUT[0], CUT[1] * 0.5)   # measured on an area: scaled to the area's share of the picture


def test_each_area_once_a_minute():
    w = PestWatch({"pest_rearm_s": 60})
    both = {1: [(0.5, MOUSE)], 2: [(0.7, (400, 400, 430, 415))]}
    assert [r[1]["name"] for r in looks(w, "xx", both, [KITCHEN, TABLE])] == ["kitchen", "table"]
    assert looks(w, "x" * 50, both, [KITCHEN, TABLE], t0=2) == []
    assert len(looks(w, "xxxxxxxxxx", {1: [(0.5, MOUSE)]}, [KITCHEN, TABLE], t0=52)) == 1   # at 61 s: the kitchen


def test_an_area_gone_is_forgotten():
    w = PestWatch({})
    looks(w, "x", {1: [(0.5, MOUSE)]})
    looks(w, ".", {}, [TABLE], t0=1)
    assert set(w.looks) == {2}


class Boxes:
    def __init__(self, dets):
        self.conf = np.array([d[0] for d in dets], float)
        self.xyxy = np.array([d[1] for d in dets], float).reshape(-1, 4)


class Model:
    """Returns the same boxes (in the cut-out area's pixels) for every area; remembers what it was shown."""
    def __init__(self, dets):
        self.dets, self.shown = dets, []

    def predict(self, frame, **kw):
        self.shown.append((frame.shape, kw["imgsz"], kw["conf"]))
        return [type("R", (), {"boxes": Boxes(self.dets)})()]


class Queue:
    def __init__(self):
        self.raised = []

    def suspect(self, camera, category, reason, **kw):
        self.raised.append((category, reason, kw))
        return "queued"


def camera(db, zones, dets):
    p = make_pipeline(db)
    p.det_cfg = {"device": "cpu"}
    p.zone_counter = ZoneCounter(zones)
    p.pest_watch, p.pest_model, p.incidents = PestWatch({}), Model(dets), Queue()
    return p


def test_the_camera_looks_at_food_areas_and_raises(db):
    # in the kitchen's cut-out (0, 240)-(320, 480): a mouse at (100, 160) -> (100, 400) in the picture; a box whose
    # middle is outside the outline (the area's box is its outline here, so one past its bottom edge) is left out
    p = camera(db, [KITCHEN, WALKWAY], [(0.6, (100, 160, 130, 175)), (0.9, (100, 235, 130, 260))])
    frame = np.zeros(SHAPE, np.uint8)
    for t in range(2):
        p._look_for_pests(frame, float(t))
    assert p.pest_model.shown == [((240, 320, 3), 1280, 0.3)] * 2   # only the kitchen, cut out, enlarged
    [(category, reason, kw)] = p.incidents.raised
    assert category == "pests" and reason.startswith("a mouse or rat seen 2 of the last 2 looks in kitchen")
    assert kw["area"] is KITCHEN and kw["question"] == QUESTION and len(kw["pictures"]) == 2
    assert kw["crop"] == (100 / 640, 400 / 480, 130 / 640, 415 / 480)


def test_refused_by_the_queue_tries_again(db):
    """The queue's own once-a-minute limit counts from when the check was sent, a moment after the area's: refused
    once, the area is tried again at the next sighting, not a minute later (seen in the real run)."""
    p = camera(db, [KITCHEN], [(0.6, (100, 160, 130, 175))])
    answers = iter(["cooldown", "queued"])
    raised = []
    p.incidents.suspect = lambda *a, **kw: raised.append(a) or next(answers)
    for t in range(4):
        p._look_for_pests(np.zeros(SHAPE, np.uint8), float(t))
    assert len(raised) == 2 and 1 in p.pest_watch.raised   # at 1 s (refused), at 3 s (queued)


def test_no_food_area_no_look(db):
    p = camera(db, [WALKWAY], [(0.9, (10, 10, 40, 25))])
    p._look_for_pests(np.zeros(SHAPE, np.uint8), 0.0)
    assert p.pest_model.shown == [] and p.incidents.raised == [] and not p.pest_watch.due(0.5)

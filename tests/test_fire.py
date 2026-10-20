"""Fire and smoke (Phase 46 step 2, labeler/fire.py): seen on 3 of the last 5 looks raises one critical suspect with
the part around the boxes and pictures over 8 s; boxes inside a cooking area are ignored unless they reach past it;
nothing again until nothing was seen for a while; the camera's look with the model."""
import copy
from datetime import datetime, timedelta

import numpy as np

from labeler.config import DEFAULTS
from labeler.fire import FIRE, SMOKE, FireWatch, share_inside
from labeler.incidents import ConfirmQueue, FrameMemory
from labeler.zones import ZoneCounter
from test_pipeline_logic import make_pipeline

SHAPE = (480, 640, 3)
KITCHEN = {"id": 1, "kind": "area", "role": "cooking", "name": "kitchen",
           "points": [(0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)]}
STOVE = (40, 300, 200, 440)      # inside the kitchen (left bottom quarter)
SOFA = (400, 300, 560, 440)      # right bottom: outside


def looks(w, pattern, dets, zones=(), t0=0.0):
    """One look a second; pattern: "x" = the detections seen, "." = nothing. Returns what update() raised."""
    out = []
    for i, c in enumerate(pattern):
        got = w.update(t0 + i, dets if c == "x" else [], SHAPE, list(zones))
        if got:
            out.append((t0 + i, *got))
    return out


def test_share_inside():
    assert share_inside(STOVE, SHAPE, KITCHEN["points"]) == 1.0 and share_inside(SOFA, SHAPE, KITCHEN["points"]) == 0.0
    assert 0.3 < share_inside((200, 300, 400, 440), SHAPE, KITCHEN["points"]) < 0.7


def test_three_of_five_looks_raise_once():
    w = FireWatch({})
    assert looks(w, "x.x..x..", [(SMOKE, 0.6, SOFA)]) == []          # never 3 within 5 looks
    got = looks(FireWatch({}), "xx.x", [(FIRE, 0.7, SOFA), (SMOKE, 0.5, (380, 250, 600, 400))])
    assert len(got) == 1 and got[0][0] == 3
    assert got[0][1] == "flames and smoke seen 3 of the last 4 s"
    assert got[0][2] == (380 / 640, 250 / 480, 600 / 640, 440 / 480)   # around every box seen


def test_again_only_after_a_quiet_while():
    w = FireWatch({"fire_rearm_s": 60})
    assert len(looks(w, "xxx", [(FIRE, 0.7, SOFA)])) == 1
    assert looks(w, "xxxxx" + "." * 30 + "xxx", [(FIRE, 0.7, SOFA)], t0=3) == []   # quiet only 30 s
    assert len(looks(w, "." * 61 + "xxx", [(FIRE, 0.7, SOFA)], t0=50)) == 1


def test_cooking_area():
    assert looks(FireWatch({}), "xxxxx", [(FIRE, 0.8, STOVE)], [KITCHEN]) == []    # a stove: normal
    got = looks(FireWatch({}), "xxx", [(FIRE, 0.8, (100, 200, 400, 440))], [KITCHEN])   # reaching past it
    assert len(got) == 1 and got[0][1].endswith("reaching past the cooking area kitchen")
    dining = {**KITCHEN, "role": "dining"}
    assert len(looks(FireWatch({}), "xxx", [(FIRE, 0.8, STOVE)], [dining])) == 1   # only "cooking" excuses it


def test_pictures_over_eight_seconds():
    mem = FrameMemory(keep_s=10)
    t0 = datetime(2026, 10, 20, 12)
    for i in range(11):
        mem.add(np.full((48, 64, 3), i, np.uint8), t0 + timedelta(seconds=i), float(i))
    q = ConfirmQueue(copy.deepcopy(DEFAULTS), None, {"cam1": mem}, clock=lambda: 10.0)
    q.suspect("cam1", "fire", "smoke", span_s=8)
    assert [int(p[1][0, 0, 0]) for p in q.waiting[0].pictures] == [2, 6, 10]
    q.suspect("cam1", "person_down", "lying")   # the usual 4 s
    assert [int(p[1][0, 0, 0]) for p in q.waiting[1].pictures] == [6, 8, 10]


class Boxes:
    def __init__(self, dets):
        self.cls = np.array([d[0] for d in dets], float)
        self.conf = np.array([d[1] for d in dets], float)
        self.xyxy = np.array([d[2] for d in dets], float)


class Model:
    def __init__(self, dets):
        self.dets = dets

    def predict(self, frame, **kw):
        return [type("R", (), {"boxes": Boxes(self.dets)})()]


class Queue:
    def __init__(self):
        self.raised = []

    def suspect(self, camera, category, reason, **kw):
        self.raised.append((category, reason, kw))
        return "queued"


def test_the_camera_looks_and_raises(db):
    p = make_pipeline(db)
    p.det_cfg = {"device": "cpu"}
    p.zone_counter = ZoneCounter([])
    p.fire_watch, p.fire_model, p.incidents = FireWatch({}), Model([(FIRE, 0.9, SOFA)]), Queue()
    frame = np.zeros(SHAPE, np.uint8)
    for t in range(3):
        p._look_for_fire(frame, float(t))
    [(category, reason, kw)] = p.incidents.raised
    assert category == "fire" and reason == "flames seen 3 of the last 3 s" and kw["span_s"] == 8
    assert kw["crop"] == (400 / 640, 300 / 480, 560 / 640, 440 / 480)

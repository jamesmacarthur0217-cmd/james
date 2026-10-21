"""Unsafe equipment (Phase 47 step 3, labeler/equipment.py): the 10-minute rounds over the areas with a role (two "yes"
a minute apart alert, then "reported" until a "no"), raised by the camera at the lowest priority."""
from types import SimpleNamespace

import numpy as np

from labeler.equipment import EquipmentRounds

SHAPE = (480, 640, 3)
KITCHEN = {"id": 1, "kind": "area", "role": "cooking", "name": "kitchen",
           "points": [(0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)]}
HALL = {"id": 2, "kind": "area", "role": "walkway", "name": "hall",
        "points": [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]}
PLAIN = {"id": 3, "kind": "area", "role": "", "name": "counted", "points": [(0, 0), (1, 0), (1, 1), (0, 1)]}
LINE = {"id": 4, "kind": "line", "name": "door", "points": [(0, 0), (1, 1)]}


def rounds(r, seconds, t0=0, zones=(KITCHEN, HALL, PLAIN, LINE)):
    out = []
    for t in range(t0, t0 + seconds):
        out += [(t, z["name"], alert) for z, alert, _ in r.due(t, list(zones))]
    return out


def test_every_area_with_a_role_in_turn_every_10_min():
    r = EquipmentRounds({})
    got = rounds(r, 100)
    assert got == [(60, "kitchen", False), (80, "hall", False)]   # first after a minute, 20 s apart; plain: not
    r.answered(1, {"answer": "no"})
    r.answered(2, {"answer": "no"})
    got = rounds(r, 700, t0=100)
    assert [(t, n) for t, n, _ in got] == [(700, "kitchen"), (700, "hall")]


def test_two_yes_a_minute_apart_alert_then_reported_until_a_no():
    r = EquipmentRounds({})
    rounds(r, 61)
    r.answered(1, {"answer": "yes", "what": "chair on its side"})   # taken at 61: once more a minute later
    assert rounds(r, 61, t0=61, zones=(KITCHEN,)) == [(121, "kitchen", True)]   # the second look alerts on "yes"
    r.answered(1, {"answer": "yes"})
    assert rounds(r, 601, t0=122, zones=(KITCHEN,)) == [(722, "kitchen", False)]   # still there: no alert again
    assert r.state()["reported"] == 1
    r.answered(1, {"answer": "no"})
    got = rounds(r, 601, t0=723, zones=(KITCHEN,))
    r.answered(1, {"answer": "yes"})
    assert got == [(1323, "kitchen", False)] and rounds(r, 61, t0=1324, zones=(KITCHEN,)) == [(1384, "kitchen", True)]


def test_a_yes_then_no_ends_it_and_no_answer_tries_again_in_a_minute():
    r = EquipmentRounds({})
    rounds(r, 61, zones=(KITCHEN,))
    r.answered(1, {"answer": "yes"})
    rounds(r, 60, t0=61, zones=(KITCHEN,))
    r.answered(1, {"answer": "no"})
    assert rounds(r, 590, t0=121, zones=(KITCHEN,)) == [] and r.state()["checking"] == 0
    r = EquipmentRounds({})
    rounds(r, 61, zones=(KITCHEN,))
    r.answered(1, None)   # dropped: the vision model was busy
    assert rounds(r, 61, t0=61, zones=(KITCHEN,)) == [(121, "kitchen", False)]
    r.answered(1, None)
    rounds(r, 1, t0=200, zones=())   # the area was deleted: forgotten
    assert r.state()["areas"] == 0


class Queue:
    def __init__(self):
        self.raised = []

    def suspect(self, camera, category, reason, **kw):
        self.raised.append((category, reason, kw))
        return "queued"


def test_the_camera_raises_the_rounds(db):
    from datetime import datetime

    from labeler.incidents import FrameMemory
    from labeler.zones import ZoneCounter
    from test_pipeline_logic import make_pipeline
    p = make_pipeline(db)
    p.zone_counter = ZoneCounter([KITCHEN, HALL, PLAIN])
    p.incidents = Queue()
    p.memory, p.equipment = FrameMemory(), EquipmentRounds({"equipment_first_s": 0, "equipment_gap_s": 0})
    p.memory.add(np.zeros(SHAPE, np.uint8), datetime(2026, 10, 21, 12), 0.0)
    p._round_equipment(0.0)
    assert [(c, kw["area"]["name"], kw["level"], kw["alert_if_yes"], len(kw["pictures"]))
            for c, _, kw in p.incidents.raised] == [("equipment", "kitchen", "round", False, 1),
                                                    ("equipment", "hall", "round", False, 1)]
    p.incidents.raised[0][2]["on_done"]({"answer": "yes"})
    p.incidents.raised.clear()
    p._round_equipment(1.0)   # the answer is taken: once more a minute later
    p._round_equipment(61.0)
    assert [(kw["area"]["name"], kw["level"], kw["alert_if_yes"]) for _, _, kw in p.incidents.raised] == \
        [("kitchen", None, True)]   # the second look: a warning, and it alerts


def test_health_line():
    from labeler.healthcheck import check_rounds
    cams = [SimpleNamespace(name="cam1", equipment=EquipmentRounds({}),
                            zone_counter=SimpleNamespace(zones=[KITCHEN, HALL, PLAIN]))]
    assert check_rounds(cams)["detail"] == "cam1: rounds over 2 areas with a role every 10 min"
    assert check_rounds([SimpleNamespace(name="k", equipment=None)]) is None

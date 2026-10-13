"""tools/loiter_check.py: true stays from MOTChallenge markings, and how alerts are scored against them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import loiter_check as lc  # noqa: E402

AREA = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]   # the left half
SIZE = (100, 100)


def _gt(person_frames: dict[int, list[tuple[int, float]]]) -> dict:
    """{person: [(frame, x of their feet)]} -> frame -> [(person, box)], 30 frames a second."""
    gt: dict = {}
    for pid, frames in person_frames.items():
        for frame, x in frames:
            gt.setdefault(frame, []).append((pid, (x - 5, 50, x + 5, 90)))
    return gt


def test_true_stays_and_reached():
    gt = _gt({1: [(f, 20) for f in range(1, 301)],                                   # 10 s on the left
              2: [(f, 20 if f <= 90 or f > 150 else 80) for f in range(1, 301)],     # 3 s, out 2 s, back 5 s
              3: [(f, 80) for f in range(1, 301)]})                                  # always on the right
    stays = lc.true_stays(gt, SIZE, AREA)
    assert stays[1] == [(0.0, 10.0)] and 3 not in stays
    assert stays[2] == [(0.0, 3.0), (5.0, 10.0)]
    assert lc.reached(stays[1], 5) == 5.0
    assert lc.reached(stays[2], 4) == 9.0              # walking out starts the time again
    assert lc.reached(stays[2], 6) is None


def test_score_counts_found_false_and_late():
    gt = _gt({1: [(f, 20) for f in range(1, 301)], 2: [(f, 20) for f in range(1, 61)]})
    box1 = {f: b for f in gt for p, b in gt[f] if p == 1}
    box2 = {f: b for f in gt for p, b in gt[f] if p == 2}
    records = {11: [(f, box1[f]) for f in range(1, 301, 3)], 12: [(f, box2[f]) for f in range(1, 61, 3)]}
    alerts = [{"rule": "all:5", "record": 11, "t": 6.0},        # person 1 reached 5 s at 5.0: 1 s late
              {"rule": "all:5", "record": 12, "t": 5.5}]        # person 2 stayed only 2 s: a false alarm
    lc_areas, lc.AREAS = lc.AREAS, {"all": AREA}
    try:
        rows = lc.score("x", gt, SIZE, records, alerts, [5], {1: 1.0, 2: 1.0}, 0.5)
    finally:
        lc.AREAS = lc_areas
    r = rows[0]
    assert (r["stayed"], r["seen"], r["held"], r["found"], r["alerts"], r["false"]) == (1, 1, 1, 1, 2, 1)
    assert r["late_median_s"] == 1.0

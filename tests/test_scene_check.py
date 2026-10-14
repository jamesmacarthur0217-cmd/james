"""tools/scene_check.py: how answers are scored (made-up answers: the datasets are not in git, so not on GitHub)."""
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import scene_check as sc  # noqa: E402


def _item(question, video, frame, truth, answer):
    return {"question": question, "video": video, "frame": frame, "truth": truth, "answer": answer, "seconds": 0.5}


def test_report_counts_answers_and_wrong_twice_in_a_row():
    items = [_item(sc.CAR, "MOT16-02", f, "no", a) for f, a in [(1, "no"), (16, "yes"), (31, "yes"), (46, "no")]]
    items += [_item(sc.CAR, "MOT16-09", f, "no", a) for f, a in [(1, "yes"), (16, "unsure")]]
    items += [_item(sc.CAR, "MOT16-13", 1, "yes", "yes"), _item(sc.LYING, "adl-01", 5, None, "yes")]
    rows = {(r["question"], r["truth"]): r for r in sc.report(items)}
    no = rows[(sc.CAR, "no")]
    assert (no["pictures"], no["yes"], no["no"], no["unsure"]) == (6, 3, 2, 1)
    assert (no["wrong_twice"], no["pairs"]) == (1, 4)       # frames 16 and 31 of MOT16-02; videos are not joined
    assert rows[(sc.CAR, "yes")]["yes"] == 1
    assert rows[(sc.LYING, "lying on purpose (apart)")]["pictures"] == 1
    assert sc.report([{**items[0], "answer": None} for _ in range(0)]) == []


# --- tools/scene_area_check.py (Phase 27) ---

import scene_area_check as sa  # noqa: E402


def test_area_check_scores_each_way():
    def item(name, truth, part, whole):
        return {"set": name, "truth": truth, "part": {"answer": part, "seconds": 0.5},
                "whole": {"answer": whole, "seconds": 0.7}}
    items = [item("coco-car", "no", "no", "yes"), item("coco-car", "no", "unsure", "yes"),
             item("coco-car", "yes", "yes", "yes"), item("urfd-wall", "no", "no", "yes")]
    rows = {(r["set"], r["truth"]): r for r in sa.report(items, ["whole", "part"])}
    assert rows[("coco-car", "no")]["part"] == {"right": 1, "wrong": 0, "unsure": 1, "seconds": 0.5}
    assert rows[("coco-car", "no")]["whole"]["wrong"] == 2 and rows[("coco-car", "yes")]["whole"]["right"] == 1
    assert list(rows) == [("coco-car", "yes"), ("coco-car", "no"), ("urfd-wall", "no")]


def test_area_check_rectangles_stay_inside_the_picture():
    assert sa.rect(-0.1, 0.2, 0.5, 1.3) == [[0.0, 0.2], [0.5, 0.2], [0.5, 1.0], [0.0, 1.0]]
    assert sa.grow(0.5, 0.5, 0.1, 0.2, 3)[0] == [0.35, pytest.approx(0.2)]
    assert sa.overlaps((0, 0, 0.5, 0.5), (0.4, 0.4, 1, 1)) and not sa.overlaps((0, 0, 0.4, 0.4), (0.4, 0.4, 1, 1))

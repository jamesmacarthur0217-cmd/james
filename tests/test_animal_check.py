"""tools/animal_check.py: marked animals from COCO labels, and how found boxes are matched to them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import animal_check as ac  # noqa: E402


def test_truth_reads_only_animals(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("16 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.1 0.1\n14 0.25 0.25 0.1 0.1\n")   # dog, person, bird
    got = ac.truth(label, 100, 200)
    assert [c for c, _ in got] == [16, 14]
    assert got[0][1] == (40.0, 60.0, 60.0, 140.0)
    assert ac.truth(tmp_path / "missing.txt", 100, 100) == []


def test_match_needs_the_same_kind_and_overlap():
    gts = [(16, (0, 0, 10, 10)), (15, (20, 20, 30, 30))]
    dets = [(16, (1, 1, 10, 10), 0.9), (16, (20, 20, 30, 30), 0.8), (14, (50, 50, 60, 60), 0.6)]
    found, used = ac.match(gts, dets)
    assert found == {0} and used == {0}               # the cat box was called a dog: not found, a wrong box


def test_not_people_drops_only_boxes_that_are_a_person():
    dets = [(1, (0, 0, 10, 20), 16, 0.9), (2, (30, 0, 40, 10), 16, 0.9)]
    assert [d[0] for d in ac.not_people(dets, [(0, 0, 10, 21)])] == [2]
    assert [d[0] for d in ac.not_people(dets, [(0, 0, 10, 21)], min_iou=0.99)] == [1, 2]

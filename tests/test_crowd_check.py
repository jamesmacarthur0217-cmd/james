"""The crowding measuring tool (Phase 45 step 3, tools/crowd_check.py): true counts from MOTChallenge markings
(pedestrians and people standing still, visible enough), stretches over a limit, and the scoring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import crowd_check as cc  # noqa: E402


def test_true_counts(tmp_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("1,1,0,0,10,30,1,1,0.9\n"      # pedestrian
                  "1,2,0,0,10,30,0,7,0.8\n"      # standing still (not considered in MOT's own scoring): counted
                  "1,3,0,0,10,30,1,1,0.1\n"      # hardly visible: not counted
                  "1,4,0,0,10,30,0,3,1.0\n"      # a car: not counted
                  "2,1,0,0,10,30,1,1,1.0\n", encoding="utf-8")
    assert cc.true_counts(gt, 0.25) == {1: 2, 2: 1}


def test_stretches():
    flags = [(t / 10, 3 <= t / 10 < 9) for t in range(120)]
    assert cc.stretches(flags, 5) == [(3.0, 8.9)] and cc.stretches(flags, 7) == []


def test_score():
    true = {f: (12 if 30 <= f < 90 else 4) for f in range(1, 121)}   # over 10 from frame 30 to 89 (10 fps)
    res = {"counts": [(f, 11 if 35 <= f < 95 else 3) for f in range(1, 121)],
           "suspects": [(9.0, "everywhere", "11 people"), (4.0, None, "8 people moving fast the same way")]}
    r = cc.score(true, res, 10, 10, 5)
    assert (r["true_median"], r["counted_median"], r["limit"]) == (8, 7, 10)   # half 12, half 4; half 11, half 3
    assert r["recall"] == round(55 / 60, 2) and r["precision"] == round(55 / 60, 2)
    assert (r["true_stretches"], r["crowd_suspects"], r["rush_suspects"]) == (1, 1, 1)

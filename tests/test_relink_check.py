"""Counting check (tools/relink_check.py) scoring, and the per-mode re-link bar."""
import copy
import sys
from pathlib import Path

import pytest

from labeler.config import DEFAULTS
from labeler.pipeline import relink_settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import relink_check as rc  # noqa: E402

A, B = (10, 10, 40, 100), (300, 10, 330, 100)   # where true people 1 and 2 walk


def _gt(frames=range(1, 21)):
    return {f: [(1, A), (2, B)] for f in frames}


def test_long_gap_bar_per_mode():
    assert relink_settings(DEFAULTS, "kiosk")["long_gap_min_similarity"] == 0.90
    assert relink_settings(DEFAULTS, "security")["long_gap_min_similarity"] == 0.98
    cfg = copy.deepcopy(DEFAULTS)
    cfg["tracking"]["relink"]["long_gap_min_similarity"] = 0.93           # one number for every mode still works
    assert relink_settings(cfg, "security")["long_gap_min_similarity"] == 0.93


def test_read_gt(tmp_path):
    f = tmp_path / "gt.txt"
    f.write_text("1,9,499,158,31.03,75.17,1,-4.1,-7.3,0\n2,9,500,158,31,75,1,0,0,0\n")
    gt = rc.read_gt(f)
    [(pid, box)] = gt[1]
    assert pid == 9 and box == pytest.approx((499.0, 158.0, 530.03, 233.17)) and len(gt[2]) == 1


def test_score_counts_merges_and_splits():
    gt = _gt()
    records = {
        1: [(f, A) for f in range(1, 11)] + [(f, B) for f in range(11, 21)],   # two people joined: wrong
        2: [(f, B) for f in range(1, 9)],                                       # person 2 again: a split
    }
    s = rc.score(records, gt, [])
    assert (s["true"], s["counted"], s["found"], s["merged"], s["split"]) == (2, 2, 2, 1, 1)


def test_each_join_is_judged():
    gt = _gt()
    joins = [{"record": 1, "gap_s": 6.0, "old_seen": [(f, A) for f in range(1, 9)],
              "new_seen": [(f, A) for f in range(12, 18)]},
             {"record": 2, "gap_s": 6.0, "old_seen": [(f, A) for f in range(1, 9)],
              "new_seen": [(f, B) for f in range(12, 18)]}]
    rc.judge(joins, gt)
    assert [j["right"] for j in joins] == [True, False]
    assert joins[1]["speed_heights_s"] > 0.4                                 # 290 px in 6 s, people 90 px tall

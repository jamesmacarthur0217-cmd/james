"""tools/scene_check.py: how answers are scored (made-up answers: the datasets are not in git, so not on GitHub)."""
import sys
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

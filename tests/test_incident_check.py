"""The safety measuring tool (Phase 45 step 0, tools/incident_check.py): the truth file, the scoring (found, false
alarms per hour, late, a little slack before a span), and the pictures it feeds (a half, the last one held)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import incident_check as ic  # noqa: E402


def test_the_truth_file(tmp_path):
    f = tmp_path / "truth.csv"
    f.write_text("video,kind,start_s,end_s,half\n"
                 "a.mp4,fall,3.5,5,right\n"
                 "a.mp4,person_down,5,70,\n"
                 "b.mp4,,,,\n", encoding="utf-8")
    t = ic.read_truth(f)
    assert t == {"a.mp4": {"half": "right", "spans": [("fall", 3.5, 5.0), ("person_down", 5.0, 70.0)]},
                 "b.mp4": {"half": "", "spans": []}}


def test_scoring():
    truth = {"a": {"half": "", "spans": [("fall", 10, 12), ("fall", 100, 102)]}, "b": {"half": "", "spans": []}}
    results = {
        "a": {"seconds": 1800, "suspects": [(9, "fall"), (101, "fall"), (500, "fall")],
              "answers": [(9.5, "fall", "yes"), (101, "fall", "no"), (500, "fall", "yes")]},
        "b": {"seconds": 1800, "suspects": [(50, "fall"), (60, "fire")], "answers": [(50, "fall", "unsure")]},
    }
    fall, fire = ic.score(truth, results, ["fall", "fire"], late_s=30)
    assert (fall["spans"], fall["raised"], fall["found"]) == (2, 2, 1)   # the second: the vision model said no
    assert fall["false_raised"] == 2 and fall["false_found"] == 1 and fall["false_found_per_h"] == 1.0   # one hour
    assert fall["late_median_s"] == 0.0 and fall["answers"] == {"yes": 2, "no": 1, "unsure": 1}   # 0.5 s early: slack
    assert (fire["spans"], fire["false_raised"], fire["false_found"]) == (0, 1, 0)


def test_the_pictures_fed(tmp_path):
    path = tmp_path / "v.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 32))
    for i in range(10):   # 1 s at 10 fps; the left half dark, the right half bright
        frame = np.zeros((32, 64, 3), np.uint8)
        frame[:, 32:] = 200
        out.write(frame)
    out.release()
    got = list(ic.frames(path, 5, "right", hold_s=2))
    assert len(got) == 5 + 10                          # every 2nd frame, then the last held 2 s at 5 a second
    assert got[0][0].shape == (32, 32, 3) and got[0][0].mean() > 150
    assert got[-1][1] > 2.5 and got[-1][0] is got[4][0]

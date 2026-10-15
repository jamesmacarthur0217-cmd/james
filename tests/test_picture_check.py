"""Camera picture check (Phase 30): labeler/picture_check.py and the scoring of tools/picture_check_eval.py
(made-up pictures: the datasets are not in git, so not on GitHub)."""
import sys
from pathlib import Path

import cv2
import numpy as np

from labeler import picture_check as pc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import picture_check_eval as ev  # noqa: E402


BUS = Path(__file__).resolve().parent.parent / "samples" / "bus.jpg"   # a real photo, in git


def _scene():
    """A sharp, normally lit photo (810 x 1080) shrunk to 480 x 640."""
    return cv2.resize(cv2.imread(str(BUS)), (480, 640), interpolation=cv2.INTER_AREA)


def test_a_good_picture_passes_at_any_resolution():
    assert pc.check_frame(_scene()) == []
    assert pc.check_frame(cv2.imread(str(BUS))) == []


def test_each_problem_is_found():
    img = _scene()
    assert pc.check_frame(np.full_like(img, 12)) == ["covered"]
    assert pc.check_frame((img * 0.15).astype(np.uint8)) == ["dark"]
    assert pc.check_frame(np.clip(img.astype(int) * 2 + 120, 0, 255).astype(np.uint8)) == ["bright"]
    assert pc.check_frame(cv2.GaussianBlur(img, (0, 0), 3)) == ["blurry"]
    assert pc.check_frame(cv2.GaussianBlur(img, (0, 0), 0.6)) == []          # slightly soft: fine
    backlit = np.clip(img.astype(int) + 150, 0, 255).astype(np.uint8)
    backlit[150:450, 150:300] = 30                                         # a dark person against a bright window
    assert pc.check_frame(backlit, [(150, 150, 300, 450)]) == ["backlight"]   # not "bright" as well
    assert pc.check_frame(img, [(150, 150, 300, 450)]) == []
    assert set(pc.ADVICE) >= {"covered", "dark", "bright", "blurry", "backlight", "faces", "small_faces"}


def test_faces_from_the_people_recorded():
    def person(view, facing, age="30-39"):
        return {"in_view_s": view, "facing_s": facing, "attributes": {"age_range": {"value": age}}}
    assert pc.faces([person(10, 5)] * 9)[0] == []                          # too few people to tell
    found, numbers = pc.faces([person(10, 1)] * 12)
    assert found == ["faces"] and numbers == {"people": 12, "facing_share": 0.1, "ages_share": 1.0}
    assert pc.faces([person(10, 5, "unknown")] * 12)[0] == ["small_faces"]
    assert pc.faces([person(10, 5)] * 12)[0] == []
    assert pc.faces([{"in_view_s": None}] * 20)[1] == {"people": 0}


def test_eval_makes_pictures_worse_and_scores_them():
    rng = np.random.default_rng(1)
    img = _scene()
    assert ev.worse(img, "dark", [], rng).mean() < img.mean() * 0.3
    lit = ev.worse(img, "backlight", [(0, 0, 100, 100)], rng)
    assert lit[:100, :100].mean() < img[:100, :100].mean() < lit[200:, 200:].mean()
    rows = [{"i": 1, "src": "coco", "kind": "blur", "blank": False, **pc.stats(cv2.GaussianBlur(img, (0, 0), 3))},
            {"i": 1, "src": "coco", "kind": "original", "blank": False, **pc.stats(img)}]
    got = {r["kind"]: r for r in ev.report(rows, pc.LIMITS) if r["pictures"]}
    assert got["blur"]["caught"] == 1 and got["original"]["flagged"] == 0
    tuned = ev.tune(rows * 4)
    assert set(tuned) == set(pc.LIMITS)

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


# --- step 2: over time, the Health line, "Check the picture" ---

import copy  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from labeler.config import DEFAULTS  # noqa: E402
from labeler.healthcheck import check_picture  # noqa: E402
from labeler.web import create_app  # noqa: E402

CFG = {"every_s": 30, "window_s": 600, "lasting_share": 0.8}


class Clock:
    def __init__(self):
        self.mono, self.now = 0.0, datetime(2026, 10, 15, 20, 0)

    def step(self, s):
        self.mono += s
        self.now += timedelta(seconds=s)


def _watch(clock):
    return pc.PictureWatch(CFG, clock=lambda: clock.mono, wall=lambda: clock.now)


def test_a_problem_must_last_most_of_ten_minutes():
    clock = Clock()
    w = _watch(clock)
    assert w.due() and w.lasting() == {}
    for i in range(20):                                    # 10 minutes, a check every 30 s
        w.add(["dark"] if i not in (3, 7) else [])         # 18 of 20 dark: a light on for a moment
        if i == 8:
            assert w.lasting() == {}                       # not enough of the window checked yet
        clock.step(30)
    assert w.lasting() == {"dark": "2026-10-15T20:00:00"}
    assert w.state()["advice"] == {"dark": pc.ADVICE["dark"]} and w.state()["checked_at"] == "2026-10-15T20:09:30"
    for _ in range(5):                                     # 5 of the last 20 fine: no longer 80%
        w.add([])
        clock.step(30)
    assert w.lasting() == {}
    for _ in range(20):
        w.add(["covered"])                                 # the Health "blank picture" problem says it already
        clock.step(30)
    assert w.lasting() == {}
    clock.step(-30)
    assert not w.due()


class Cam:
    def __init__(self, clock, mode="kiosk", frame=None):
        self.name, self.mode = "cam1", mode
        self.picture = _watch(clock)
        self.scene_frame = (frame, datetime.now(), []) if frame is not None else None
        self.scene_people = []


def test_health_line(db):
    cfg = copy.deepcopy(DEFAULTS)
    clock = Clock()
    cam = Cam(clock)
    assert check_picture(cfg, db, [cam])["detail"].startswith("not checked yet")
    for _ in range(20):
        cam.picture.add(["blurry"])
        clock.step(30)
    line = check_picture(cfg, db, [cam])
    assert line["level"] == "warning" and "camera cam1: The picture is blurry" in line["detail"]
    cam.picture = _watch(clock)
    cam.picture.add([])
    assert check_picture(cfg, db, [cam])["detail"].startswith("fine (last checked 20:10")
    cfg["picture_check"]["enabled"] = False
    assert check_picture(cfg, db, [cam]) is None


def test_check_the_picture_now(db, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cam = Cam(Clock(), frame=cv2.GaussianBlur(_scene(), (0, 0), 3))
    cam.status = lambda: {"camera": "cam1"}
    client = TestClient(create_app(cfg, db, [cam]), client=("127.0.0.1", 5000))
    d = client.get("/api/cameras/cam1/picture").json()
    assert d["findings"] == ["blurry"] and d["advice"] == [pc.ADVICE["blurry"]]
    assert d["image"].startswith("data:image/jpeg;base64,") and d["faces"]["people"] == 0
    assert client.get("/api/cameras/nope/picture").status_code == 404
    cam.scene_frame = (cam.scene_frame[0], datetime.now() - timedelta(minutes=5), [])
    assert client.get("/api/cameras/cam1/picture").status_code == 409     # no recent picture


def test_viewers_cannot_check_the_picture(db, tmp_path):
    from labeler import accounts
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cam = Cam(Clock(), frame=_scene())
    cam.status = lambda: {"camera": "cam1"}
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(create_app(cfg, db, [cam]), client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/cameras/cam1/picture").status_code == 403

"""Search by photo: the largest person in the picture is searched (or the whole picture), nothing is stored."""
import copy
from datetime import datetime

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeler import accounts
from labeler.config import DEFAULTS
from labeler.photo import MAX_BYTES, PhotoSearch
from labeler.web import create_app

RED, BLUE = (0, 0, 255), (255, 0, 0)


class FakeEmbedder:
    """Appearance = the crop's average colour, so a red person looks like other red people."""
    model_name = "fake"

    def track_vector(self, crops):
        v = crops[0].reshape(-1, 3).mean(axis=0).astype(np.float32) + 1.0
        return v / np.linalg.norm(v)

    def text(self, q):
        return np.ones(3, np.float32) / 3 ** 0.5


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, image, **kw):
        boxes = np.array(self.boxes, np.float32).reshape(-1, 4)

        class Result:
            pass
        r = Result()
        r.boxes = type("B", (), {"xyxy": type("T", (), {"cpu": lambda s: type("N", (), {"numpy": lambda s2: boxes})()})()})()
        return [r]


def _photo(people):
    """A grey picture with coloured people: [(color, (x1, y1, x2, y2))]."""
    img = np.full((400, 600, 3), 128, np.uint8)
    for color, (x1, y1, x2, y2) in people:
        img[y1:y2, x1:x2] = color
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_the_largest_person_is_searched():
    small, large = (20, 20, 60, 100), (300, 50, 420, 380)
    data = _photo([(BLUE, small), (RED, large)])
    vector, info = PhotoSearch(DEFAULTS, FakeEmbedder(), FakeDetector([small, large])).vector(data)
    assert info["people"] == 2 and info["box"] == list(large)
    assert np.argmax(vector) == 2                               # red (BGR): the large person, not the blue one


def test_the_person_in_the_middle_is_searched_even_if_smaller():
    from labeler.photo import pick
    centre, big_at_edge = (250, 150, 350, 300), (0, 0, 200, 400)
    assert pick([big_at_edge, centre], 600, 400) == centre
    assert pick([(0, 0, 100, 100), (450, 0, 600, 400)], 600, 400) == (450, 0, 600, 400)   # none central: largest


def test_without_a_person_the_whole_picture_is_used():
    vector, info = PhotoSearch(DEFAULTS, FakeEmbedder(), FakeDetector([])).vector(_photo([]))
    assert info["people"] == 0 and info["box"] is None


@pytest.mark.parametrize("data,message", [(b"", "no picture"), (b"not an image", "not a JPEG"),
                                          (b"x" * (MAX_BYTES + 1), "larger than 10 MB")],
                         ids=["empty", "not-a-picture", "too-big"])
def test_bad_pictures(data, message):
    with pytest.raises(ValueError, match=message):
        PhotoSearch(DEFAULTS, FakeEmbedder(), FakeDetector([])).vector(data)


def test_photo_search_api(db, camera, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    emb = FakeEmbedder()
    ids = {}
    for name, color in (("red", RED), ("blue", BLUE)):
        t = ids[name] = db.create_track(camera, 1, datetime.now())
        db.upsert_embedding(t, emb.track_vector([np.full((10, 10, 3), color, np.uint8)]), "fake")
    box = (300, 50, 420, 380)
    app = create_app(cfg, db, None, embedder=emb, photo=PhotoSearch(cfg, emb, FakeDetector([box])))
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    r = viewer.post("/api/search/photo", content=_photo([(RED, box)]), headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200, r.json()                       # viewers may search, as with text search
    assert [p["id"] for p in r.json()["people"]][0] == ids["red"]
    assert r.json()["found"] == "1 person found"
    assert db.list_audit(1)[0]["action"] == "photo search"
    assert not list((tmp_path / "crops").rglob("*.*"))         # the photo is not stored
    assert viewer.post("/api/search/photo", content=b"junk").status_code == 400

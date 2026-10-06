"""Fast classifiers and the web API (delete, review, search guards)."""
import copy
from datetime import datetime

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from labeler import schema
from labeler.config import DEFAULTS
from labeler.student import Student
from labeler.web import create_app


def _student_file(tmp_path, carrying_items):
    """A tiny trained student: 'gender' separable on feature 0; carried items always 'no'."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    gender = np.where(X[:, 0] > 0, "man", "woman")
    attrs = {"apparent_gender": {"model": LogisticRegression().fit(X, gender),
                                 "thresholds": {"man": 0.8, "woman": 0.8}}}
    y_no = np.array(["no"] * 199 + ["yes"])
    for item in carrying_items:
        attrs[f"carrying:{item}"] = {"model": LogisticRegression().fit(X, y_no), "thresholds": {"no": 0.5}}
    path = tmp_path / "attributes.joblib"
    joblib.dump({"embedding_model": "m", "teacher_model": "t", "n_people": 200, "target_agreement": 0.9,
                 "attributes": attrs, "trained_at": "test"}, path)
    return path


def test_student_trusts_only_confident_answers(tmp_path):
    s = Student(_student_file(tmp_path, schema.LIST_FIELDS["carrying"]), "m")
    sure = s.predict(np.array([5, 0, 0, 0]), "kiosk")
    unsure = s.predict(np.array([0.01, 0, 0, 0]), "kiosk")
    assert sure["apparent_gender"][0] == "man" and sure["apparent_gender"][2] is True
    assert unsure["apparent_gender"][2] is False
    assert sure["carrying"][0] == "none" and sure["carrying"][2] is True


def test_student_cannot_rule_out_an_item_it_has_no_classifier_for(tmp_path):
    items = [i for i in schema.LIST_FIELDS["carrying"] if i != "luggage"]
    s = Student(_student_file(tmp_path, items), "m")
    assert s.predict(np.array([5, 0, 0, 0]), "kiosk")["carrying"][2] is False


def test_student_refuses_a_different_embedding_model(tmp_path):
    with pytest.raises(ValueError):
        Student(_student_file(tmp_path, []), "other-model")


@pytest.fixture
def client(db, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, None)), tmp_path / "crops"


def test_delete_api_removes_records_and_pictures(client, db, camera):
    api, crops = client
    ids = []
    for i in range(3):
        t = db.create_track(camera, i, datetime.now())
        db.update_track(t, crop_path=f"{t}.jpg")
        (crops / f"{t}.jpg").write_bytes(b"jpg")
        ids.append(t)
    r = api.post("/api/tracks/delete", json={"ids": ids[:2]}).json()
    assert r == {"deleted": 2, "pictures_removed": 2}
    assert [t["id"] for t in api.get("/api/tracks").json()] == [ids[2]]
    assert sorted(p.name for p in crops.iterdir()) == [f"{ids[2]}.jpg"]
    assert api.post("/api/tracks/delete", json={"everything": True}).json()["deleted"] == 1
    assert api.get("/api/tracks").json() == []


def test_review_api_and_guards(client, db, camera):
    api, _ = client
    t = db.create_track(camera, 1, datetime.now())
    db.add_attributes(t, [("age_range", "20-29", 0.5)], "vlm", "q", threshold=0.6)
    item = api.get("/api/review").json()[0]
    assert item["allowed"] == schema.FIELDS["age_range"]
    assert api.post(f"/api/review/{item['attribute_id']}", json={"value": "30-39"}).status_code == 200
    assert api.get("/api/review").json() == []
    assert api.post("/api/review/999999", json={"value": "x"}).status_code == 404
    assert api.get("/api/search", params={"q": "red shirt"}).status_code == 503   # embeddings off
    assert api.post("/api/shutdown").status_code == 501                           # not running under run.py

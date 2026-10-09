"""Weekly retraining from the user's own cameras: keep a new version only if the answer key says it is not worse."""
import copy
from datetime import datetime, timedelta

import numpy as np
from fastapi.testclient import TestClient

from labeler import accounts, retrain, training
from labeler.config import DEFAULTS
from labeler.retrain import Retrainer
from labeler.web import create_app

rng = np.random.default_rng(0)


def _vec(gender):
    """Appearance vectors where gender is visible on the first number, plus noise."""
    v = rng.normal(0, 0.05, 768).astype(np.float32)
    v[0] = 1.0 if gender == "man" else -1.0
    return v / np.linalg.norm(v)


def _cfg(tmp_path, **retrain_cfg):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["embeddings"]["model"] = "fake"
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cfg["student"]["model_path"] = str(tmp_path / "shipped.joblib")
    cfg["student"]["retrained_path"] = str(tmp_path / "data" / "attributes.joblib")
    cfg["retrain"].update({"coco_dir": str(tmp_path / "no-coco"), "min_own_people": 30, "min_answer_key": 10},
                          **retrain_cfg)
    return cfg


def _shipped_model(cfg, flipped):
    """The classifiers the app starts with; `flipped` = they learned gender the wrong way round."""
    labels, X = [], []
    for i in range(80):
        g = "man" if i % 2 else "woman"
        X.append(_vec(g))
        labels.append({"apparent_gender": ("woman" if g == "man" else "man") if flipped else g})
    models, _ = training.train(np.stack(X), labels, np.arange(80), report=lambda s: None)
    training.save(retrain.resolve(cfg["student"]["model_path"]), models, "fake", "test", 80, 0.9)


def _people(db, camera, n, label_right=True, key=False):
    ids = []
    for i in range(n):
        g = "man" if i % 2 else "woman"
        t = db.create_track(camera, i, datetime.now())
        db.update_track(t, status="finished")
        db.upsert_embedding(t, _vec(g), "fake")
        if key:
            db.save_answer_key(t, {"apparent_gender": g})
        else:
            said = g if label_right else ("woman" if g == "man" else "man")
            db.add_attributes(t, [("apparent_gender", said, 0.9)], "vlm", "q", threshold=0.6)
        ids.append(t)
    return ids


def test_a_better_version_is_kept_and_the_cameras_get_it(db, camera, tmp_path):
    cfg = _cfg(tmp_path)
    _shipped_model(cfg, flipped=True)                     # today's classifiers are bad
    _people(db, camera, 60)
    _people(db, camera, 20, key=True)

    class Cam:
        student = retrain.Student(retrain.student_path(cfg), "fake")

    r = Retrainer(db, cfg, [Cam])
    r.run_now()
    for _ in range(300):
        if not r.running.is_set():
            break
        __import__("time").sleep(0.05)
    entry = db.list_model_history()[0]
    assert entry["kept"] and entry["score_after"] > entry["score_before"]
    assert entry["own_people"] == 60 and entry["key_people"] == 20
    assert retrain.student_path(cfg) == tmp_path / "data" / "attributes.joblib"
    assert Cam.student.predict(_vec("man"), "kiosk")["apparent_gender"][0] == "man"   # swapped in, live


def test_a_worse_version_is_thrown_away(db, camera, tmp_path):
    cfg = _cfg(tmp_path)
    _shipped_model(cfg, flipped=False)                    # today's classifiers are good
    _people(db, camera, 60, label_right=False)            # but the new labels are wrong
    _people(db, camera, 20, key=True)
    entry = retrain.retrain(db, cfg)
    assert not entry["kept"] and entry["score_after"] < entry["score_before"]
    assert retrain.student_path(cfg) == tmp_path / "shipped.joblib"
    assert not (tmp_path / "data" / "attributes.new.joblib").exists()


def test_nothing_changes_without_enough_people(db, camera, tmp_path):
    cfg = _cfg(tmp_path)
    _shipped_model(cfg, flipped=False)
    _people(db, camera, 10)
    entry = retrain.retrain(db, cfg)
    assert not entry["kept"] and "not enough people" in entry["note"]
    _people(db, camera, 30)
    assert "not enough answer-key people" in retrain.retrain(db, cfg)["note"]


def test_training_labels_prefer_the_humans_and_skip_the_test_people(db, camera):
    a, b = (db.create_track(camera, i, datetime.now()) for i in (1, 2))
    for t in (a, b):
        db.update_track(t, status="finished")
        db.upsert_embedding(t, _vec("man"), "fake")
    db.add_attributes(a, [("upper_color", "red", 0.9)], "vlm", "q", threshold=0.6)
    db.add_attributes(a, [("upper_color", "blue", 0.95), ("apparent_gender", "man", 0.9)], "student", "s",
                      threshold=0.6)
    attr = next(x for x in db.review_queue(limit=10) if x["name"] == "upper_color")
    db.apply_review(attr["attribute_id"], "green", "ann")
    db.save_answer_key(b, {"apparent_gender": "man"})
    people = db.training_people("fake")
    assert [(t, labels) for t, _, labels in people] == [(a, {"upper_color": "green"})]   # not the student's


def test_weekly_and_at_night(db, tmp_path):
    r = Retrainer(db, _cfg(tmp_path), [])
    night = datetime(2026, 10, 9, 3, 30)
    assert r.due(night) and not r.due(night.replace(hour=1))
    db.add_model_history(own_people=0, coco_people=0, key_people=0, note="x")
    assert not r.due(datetime.now() + timedelta(days=6))
    assert r.due((datetime.now() + timedelta(days=8)).replace(hour=4))


def test_accuracy_history_and_retrain_api(db, camera, tmp_path):
    t = db.create_track(camera, 1, datetime.now())
    db.add_attributes(t, [("upper_color", "red", 1.0)], "vlm", "q", threshold=0.6)
    db.save_answer_key(t, {"upper_color": "red"})
    db.record_accuracy(datetime.now())
    db.record_accuracy(datetime.now())                     # same day: replaced, not doubled
    cfg = _cfg(tmp_path)
    app = create_app(cfg, db, None)
    admin = TestClient(app, client=("127.0.0.1", 5000))
    hist = admin.get("/api/answer-key/history").json()
    assert len(hist) == 1 and hist[0]["camera"] == "test-cam" and hist[0]["mean_overall"] == 1.0
    assert admin.get("/api/retrain").json()["available"] is False
    assert admin.post("/api/retrain").status_code == 503   # no retrainer outside the app
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.post("/api/retrain").status_code == 403

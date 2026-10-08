"""Database: labels, review rules, deleting, search, retention."""
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import func, select

from labeler.db import Attribute, Embedding, Review, Track


def _person(db, camera, crop=None):
    tid = db.create_track(camera, 1, datetime.now())
    if crop:
        db.update_track(tid, crop_path=crop)
    return tid


def test_disagreement_between_sources_goes_to_review(db, camera):
    t = _person(db, camera)
    db.add_attributes(t, [("upper_color", "red", 0.8)], "color_estimator", "rules", threshold=0.6)
    db.add_attributes(t, [("upper_color", "blue", 1.0)], "vlm", "qwen", threshold=0.6)
    item = db.review_queue()[0]
    assert item["reason"] == "models disagree"
    assert {c["value"] for c in item["candidates"]} == {"red", "blue"}


def test_vlm_answer_beats_a_more_confident_student(db, camera):
    t = _person(db, camera)
    db.add_attributes(t, [("eyewear", "none", 0.9)], "student", "s", threshold=0.6)
    assert db.resolved_values(t)["eyewear"] == "none"                    # student counts until the VLM answers
    db.add_attributes(t, [("eyewear", "glasses", 0.5)], "vlm", "q", threshold=0.6)
    assert db.resolved_values(t)["eyewear"] == "glasses"                 # VLM answered: it wins at lower confidence


def test_low_confidence_goes_to_review_but_unknown_does_not(db, camera):
    t = _person(db, camera)
    db.add_attributes(t, [("age_range", "20-29", 0.5), ("eyewear", "unknown", 1.0)], "vlm", "qwen", threshold=0.6)
    assert [i["name"] for i in db.review_queue()] == ["age_range"]


def test_review_queue_puts_disagreements_first(db, camera):
    low = _person(db, camera)
    db.add_attributes(low, [("age_range", "20-29", 0.5)], "vlm", "qwen", threshold=0.6)
    conflict = _person(db, camera)
    db.add_attributes(conflict, [("upper_color", "red", 0.9)], "color_estimator", "r", threshold=0.6)
    db.add_attributes(conflict, [("upper_color", "blue", 1.0)], "vlm", "q", threshold=0.6)
    assert [i["track_id"] for i in db.review_queue()] == [conflict, low]


def test_human_review_settles_every_source(db, camera):
    t = _person(db, camera)
    db.add_attributes(t, [("upper_color", "red", 0.8)], "color_estimator", "r", threshold=0.6)
    db.add_attributes(t, [("upper_color", "blue", 1.0)], "vlm", "q", threshold=0.6)
    db.apply_review(db.review_queue()[0]["attribute_id"], "red")
    assert db.review_queue() == []
    attrs = db.list_tracks()[0]["attributes"]
    assert attrs["upper_color"] == {"value": "red", "confidence": 1.0, "source": "human", "needs_review": False}
    assert db.list_tracks(name="upper_color", value="red")[0]["id"] == t
    assert db.list_tracks(name="upper_color", value="blue") == []


def test_delete_selected_removes_everything_about_those_people(db, camera):
    keep = _person(db, camera, crop="keep.jpg")
    gone = _person(db, camera, crop="gone.jpg")
    for t in (keep, gone):
        db.add_attributes(t, [("upper_color", "red", 0.8)], "color_estimator", "r", threshold=0.6)
        db.add_attributes(t, [("upper_color", "blue", 1.0)], "vlm", "q", threshold=0.6)
        db.upsert_embedding(t, np.ones(4, np.float32) / 2, "m")
    db.apply_review(next(i for i in db.review_queue() if i["track_id"] == gone)["attribute_id"], "red")

    deleted, paths = db.delete_tracks([gone])
    assert deleted == [gone] and paths == ["gone.jpg"]
    with db.Session() as s:
        assert s.scalar(select(func.count(Track.id))) == 1
        assert s.scalar(select(func.count(Attribute.id)).where(Attribute.track_id == gone)) == 0
        assert s.scalar(select(func.count(Embedding.id)).where(Embedding.track_id == gone)) == 0
        assert s.scalar(select(func.count(Review.id))) == 0
    assert [t for t, _ in db.nearest(np.ones(4, np.float32) / 2, "m")] == [keep]


def test_delete_everything_and_nothing(db, camera):
    for _ in range(3):
        _person(db, camera)
    assert db.delete_tracks([]) == ([], [])
    deleted, _ = db.delete_tracks(everything=True)
    assert len(deleted) == 3 and db.list_tracks() == []


def test_writes_for_a_deleted_person_are_ignored(db, camera):
    t = _person(db, camera)
    db.delete_tracks([t])
    db.update_track(t, status="finished")               # the live pipeline may still be finishing them
    db.add_attributes(t, [("upper_color", "red", 0.9)], "vlm", "q", threshold=0.6)
    db.upsert_embedding(t, np.ones(4, np.float32), "m")
    assert db.list_tracks() == []


def test_nearest_ranks_by_cosine_similarity(db, camera):
    a, b = _person(db, camera), _person(db, camera)
    db.upsert_embedding(a, np.array([1, 0, 0, 0], np.float32), "m")
    db.upsert_embedding(b, np.array([0.6, 0.8, 0, 0], np.float32), "m")
    hits = db.nearest(np.array([0, 1, 0, 0], np.float32), "m")
    assert [t for t, _ in hits] == [b, a]
    assert db.nearest(np.array([1, 0, 0, 0], np.float32), "m", exclude=a)[0][0] == b


def test_retention_deletes_only_old_people(db, camera):
    old = db.create_track(camera, 1, datetime.now() - timedelta(days=40))
    db.update_track(old, crop_path="old.jpg")
    new = _person(db, camera)
    assert db.cleanup(30) == ["old.jpg"]
    assert [t["id"] for t in db.list_tracks()] == [new]
    assert db.cleanup(0) == []


def test_each_camera_can_keep_people_for_its_own_time(db, camera):
    gate = db.get_or_create_camera("gate", "rtsp://x", "security")
    db.create_track(camera, 1, datetime.now() - timedelta(days=10))
    gate_10 = db.create_track(gate, 1, datetime.now() - timedelta(days=10))
    db.create_track(gate, 1, datetime.now() - timedelta(days=40))
    db.cleanup(30, {"test-cam": 7})                                              # kiosk 7 days, others 30
    assert [t["id"] for t in db.list_tracks()] == [gate_10]
    db.cleanup(0, {"gate": 5})                                                   # default keep forever
    assert db.list_tracks() == []


def test_retention_removes_pictures_and_blurred_copies(db, camera, tmp_path):
    from labeler import privacy
    from run import _apply_retention
    old = db.create_track(camera, 1, datetime.now() - timedelta(days=40))
    db.update_track(old, crop_path=f"{old}.jpg")
    (tmp_path / f"{old}.jpg").write_bytes(b"x")
    privacy.blurred_path(tmp_path, f"{old}.jpg").parent.mkdir()
    privacy.blurred_path(tmp_path, f"{old}.jpg").write_bytes(b"x")
    cfg = {"cameras": [{"name": "test-cam"}], "storage": {"retention_days": 30, "audit_retention_days": 365}}
    _apply_retention(db, cfg, tmp_path)
    assert db.list_tracks() == [] and list(tmp_path.rglob("*.jpg")) == []


def test_hourly_retention_stops_with_the_app(db, tmp_path):
    import threading
    from run import _retention_loop
    stop = threading.Event()
    stop.set()
    _retention_loop(db, {}, tmp_path, stop)                                      # returns at once


def test_people_left_active_by_a_crash_are_closed(db, camera):
    t = _person(db, camera)
    assert db.close_stale_tracks() == 1
    row = db.list_tracks()[0]
    assert row["id"] == t and row["status"] == "finished" and row["ended_at"] is not None
    assert db.close_stale_tracks() == 0

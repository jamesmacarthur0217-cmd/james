"""Pipeline decisions without a camera or GPU: re-linking, deleting, label voting, the VLM worker."""
import copy
import threading
import time
from datetime import datetime

import numpy as np

from labeler import schema
from labeler.config import DEFAULTS
from labeler.pipeline import Candidate, LiveClock, Pipeline, TrackState, relink_settings
from labeler.vlm import VLMWorker


class FakeEmbedder:
    model_name = "fake"

    def __init__(self, vector):
        self.vector = np.asarray(vector, np.float32)

    def track_vector(self, crops):
        return self.vector / np.linalg.norm(self.vector)


def make_pipeline(db, embedder=None):
    """A Pipeline with its decision logic but no detector or camera."""
    p = Pipeline.__new__(Pipeline)
    p.cfg = copy.deepcopy(DEFAULTS)
    p.mode = "kiosk"
    p.db = db
    p.vlm = None
    p.embedder = embedder
    p.student = None
    p.vlm_model_name = "fake-vlm"
    p.threshold = 0.6
    p.tracks, p.by_db_id, p.recently_lost = {}, {}, {}
    p.relink_cfg = relink_settings(p.cfg, p.mode)
    p.deleted_ids = set()
    p.alerts = None
    p.name = "test-cam"
    p.falls = p.vehicles = p.animals = None
    p.age = None
    p.age_ids = set()
    p.clips = None
    p.night_ids = set()
    p.clock = LiveClock()
    return p


def _lost_person(p, db, camera, vector, box, lost_at):
    st = TrackState(1, datetime.now(), lost_at, db_id=db.create_track(camera, 1, datetime.now()))
    st.vector = np.asarray(vector, np.float32) / np.linalg.norm(vector)
    st.box = box
    st.candidates = [Candidate(0.5, np.zeros((10, 10, 3), np.uint8), None)]
    p.recently_lost[st.db_id] = (st, lost_at)
    return st


def _new_track(tid, box):
    st = TrackState(tid, datetime.now(), 0.0)
    st.first_box = st.box = box
    st.n_frames = 5
    st.candidates = [Candidate(0.9, np.zeros((10, 10, 3), np.uint8), None)]
    return st


def test_returning_person_continues_their_record(db, camera):
    p = make_pipeline(db, FakeEmbedder([1, 0.05, 0]))
    old = _lost_person(p, db, camera, [1, 0, 0], (100, 100, 200, 300), lost_at=100.0)
    got = p._try_relink(7, _new_track(7, (105, 100, 205, 300)), now=103.0)     # 3 s later, same place
    assert got is old and got.tracker_id == 7 and p.tracks[7] is old and not p.recently_lost


def test_short_gap_needs_a_similar_position(db, camera):
    p = make_pipeline(db, FakeEmbedder([1, 0.3, 0]))                           # similarity ~0.96
    _lost_person(p, db, camera, [1, 0, 0], (100, 100, 200, 300), lost_at=100.0)
    new = _new_track(7, (600, 100, 700, 300))                                  # other side of the frame
    assert p._try_relink(7, new, now=103.0) is new


def test_long_gap_needs_high_similarity(db, camera):
    p = make_pipeline(db, FakeEmbedder([1, 0.6, 0]))                           # similarity ~0.86
    _lost_person(p, db, camera, [1, 0, 0], (100, 100, 200, 300), lost_at=100.0)
    new = _new_track(7, (100, 100, 200, 300))
    assert p._try_relink(7, new, now=115.0) is new                             # 15 s: needs >= 0.90


def test_forget_stops_recording_a_deleted_person(db, camera):
    p = make_pipeline(db)
    st = TrackState(3, datetime.now(), 0.0, db_id=db.create_track(camera, 3, datetime.now()))
    p.tracks[3], p.by_db_id[st.db_id] = st, st
    p.forget([st.db_id])
    assert st.deleted and st.db_id not in p.by_db_id
    p.on_vlm_result(st.db_id, [{"upper_color": "red"}])                       # late result is dropped
    assert db.list_tracks()[0]["attributes"] == {}


def test_labels_are_a_vote_over_recent_answers(db, camera):
    p = make_pipeline(db)
    st = TrackState(3, datetime.now(), 0.0, db_id=db.create_track(camera, 3, datetime.now()))
    p.by_db_id[st.db_id] = st
    base = {f: "unknown" for f in schema.MODE_FIELDS["kiosk"]}
    p.on_vlm_result(st.db_id, [dict(base, upper_color="red"), dict(base, upper_color="red")])
    p.on_vlm_result(st.db_id, [dict(base, upper_color="orange")])
    assert st.labels["upper_color"] == ("red", round(2 / 3, 3))
    assert st.labels["eyewear"] == ("unknown", 1.0)


class FakeClient:
    def __init__(self):
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def label(self, person, face, mode=None):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        if person == "bad":
            raise RuntimeError("model error")
        return {"upper_color": person}


def test_vlm_worker_runs_two_at_once_and_survives_errors():
    got = {}
    client = FakeClient()
    w = VLMWorker(client, lambda tid, res: got.__setitem__(tid, res), workers=2)
    for tid in range(4):
        w.submit(tid, [("red", None), ("bad" if tid == 2 else "blue", None)])
    w.submit(9, [])
    assert w.close(wait_s=10)
    assert client.peak == 2
    assert sorted(got) == [0, 1, 2, 3, 9]
    assert got[2] == [{"upper_color": "red"}] and got[9] == []


def test_each_camera_can_have_its_own_detector_size(db, monkeypatch, tmp_path):
    monkeypatch.setattr("labeler.pipeline.YOLO", lambda path: object())
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    cfg["clips"]["enabled"] = False
    wide = Pipeline(cfg, db, None, camera={"name": "gate", "source": "rtsp://x", "mode": "security", "imgsz": 1280})
    plain = Pipeline(cfg, db, None, camera={"name": "kiosk", "source": 0, "mode": "kiosk"})
    assert wide.imgsz == 1280 and plain.imgsz == cfg["detector"]["imgsz"] == 640


def test_night_frames_store_no_colors_but_everything_else(db, camera):
    p = make_pipeline(db)
    st = TrackState(3, datetime.now(), 0.0, db_id=db.create_track(camera, 3, datetime.now()))
    st.n_frames, st.low_light_frames = 30, 25                    # mostly black and white (infrared)
    p.by_db_id[st.db_id] = st
    p._mark_night(st)
    base = {f: "unknown" for f in schema.MODE_FIELDS["kiosk"]}
    p.on_vlm_result(st.db_id, [dict(base, upper_color="gray", lower_color="black", upper_type="jacket")])
    attrs = db.list_tracks()[0]["attributes"]
    assert attrs["upper_color"]["value"] == "unknown" and attrs["lower_color"]["value"] == "unknown"
    assert attrs["upper_type"]["value"] == "jacket"
    st.low_light_frames = 5                                      # dawn: the camera shows color again
    p._mark_night(st)
    p.on_vlm_result(st.db_id, [dict(base, upper_color="red")])
    assert db.list_tracks()[0]["attributes"]["upper_color"]["value"] == "red"


def test_student_gives_no_colors_at_night(db):
    p = make_pipeline(db)
    preds = {"upper_color": ("black", 0.9, True), "upper_type": ("jacket", 0.9, True)}
    assert set(p._student_trusted(preds, night=True)) == {"upper_type"}
    assert set(p._student_trusted(preds)) == {"upper_color", "upper_type"}


def test_age_model_answer_replaces_the_vlm_age(db, camera):
    p = make_pipeline(db)
    st = TrackState(3, datetime.now(), 0.0, db_id=db.create_track(camera, 3, datetime.now()))
    st.candidates = [Candidate(0.9, np.zeros((10, 10, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))]
    p.by_db_id[st.db_id] = st
    p.age = FakeAge()
    p._store_age(st)
    base = {f: "unknown" for f in schema.MODE_FIELDS["kiosk"]}
    p.on_vlm_result(st.db_id, [dict(base, age_range="50-59", upper_color="red")])
    attrs = db.list_tracks()[0]["attributes"]
    assert attrs["age_range"]["value"] == "30-39" and attrs["age_range"]["source"] == "age_model"
    assert attrs["upper_color"]["value"] == "red"
    assert p._current_labels(st)["age_range"][2] == "age_model"


class FakeAge:
    name = "fake-age"

    def estimate(self, faces):
        return ("30-39", 0.7) if faces else None


class FakeStudent:
    """Confident about every kiosk attribute, including an age that disagrees with the age model."""
    name = "fake-student"

    def predict(self, vector, mode):
        out = {f: ("none" if f == "carrying" else "unknown", 0.95, True) for f in schema.MODE_FIELDS[mode]}
        out["age_range"] = ("50-59", 0.95, True)
        return out


def _person_with_face(db, camera):
    st = TrackState(3, datetime.now(), 0.0, db_id=db.create_track(camera, 3, datetime.now()))
    st.candidates = [Candidate(0.9, np.zeros((10, 10, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))]
    return st


def test_student_gives_no_age_while_the_age_model_is_loaded(db, camera):
    p = make_pipeline(db)
    p.student, p.age = FakeStudent(), FakeAge()
    st = _person_with_face(db, camera)
    p._store_student(st, np.zeros(4, np.float32))
    p._store_age(st)
    attrs = db.list_tracks()[0]["attributes"]
    assert "age_range" not in st.student_labels
    assert attrs["age_range"]["value"] == "30-39" and attrs["age_range"]["source"] == "age_model"
    assert not attrs["age_range"]["needs_review"]


def test_student_skips_the_vlm_only_when_age_is_known(db, camera):
    p = make_pipeline(db)
    p.student, p.age = FakeStudent(), FakeAge()
    st = _person_with_face(db, camera)
    p._store_student(st, np.zeros(4, np.float32))
    assert not p._student_covers_all(st)        # no age yet: the VLM still has to answer it
    p._store_age(st)
    assert p._student_covers_all(st)            # age from the age model, everything else from the student

    p.age = None                                 # without an age model, the student's age is used
    st2 = _person_with_face(db, camera)
    p._store_student(st2, np.zeros(4, np.float32))
    assert st2.student_labels["age_range"][0] == "50-59" and p._student_covers_all(st2)

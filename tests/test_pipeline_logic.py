"""Pipeline decisions without a camera or GPU: re-linking, deleting, label voting, the VLM worker."""
import copy
import threading
import time
from datetime import datetime

import numpy as np

from labeler import schema
from labeler.config import DEFAULTS
from labeler.pipeline import Candidate, Pipeline, TrackState
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
    p.relink_cfg = p.cfg["tracking"]["relink"]
    p.deleted_ids = set()
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

    def label(self, person, face):
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

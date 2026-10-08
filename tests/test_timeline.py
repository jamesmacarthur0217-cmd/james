"""Timeline: look-alikes from all cameras, in time order, without impossible sightings."""
import copy
from datetime import datetime, timedelta

import numpy as np
from fastapi.testclient import TestClient

from labeler import accounts, timeline
from labeler.config import DEFAULTS
from labeler.web import create_app

T0 = datetime(2026, 10, 8, 12, 0)


def _t(id_, camera, start_min, end_min, sim=None):
    t = {"id": id_, "camera": camera, "started_at": (T0 + timedelta(minutes=start_min)).isoformat(),
         "ended_at": (T0 + timedelta(minutes=end_min)).isoformat()}
    if sim is not None:
        t["similarity"] = sim
    return t


def test_sightings_in_time_order_with_gaps():
    anchor = _t(1, "entrance", 0, 2)
    out = timeline.plausible(anchor, [_t(3, "shop", 30, 40, 0.86), _t(2, "lobby", 5, 6, 0.9)], window_h=24)
    assert [t["id"] for t in out] == [1, 2, 3]
    assert [t["gap_s"] for t in out] == [None, 180, 1440]
    assert out[0]["anchor"] and not out[1]["anchor"]


def test_nobody_is_in_two_places_at_once():
    anchor = _t(1, "entrance", 0, 10)
    candidates = [_t(2, "shop", 5, 8, 0.95),          # during the anchor's visit, elsewhere: impossible
                  _t(3, "lobby", 20, 30, 0.90),
                  _t(4, "lobby", 25, 35, 0.85),       # overlaps #3, which is more alike: left out
                  _t(5, "entrance", 2, 4, 0.99)]      # same camera, same time: a second person
    assert [t["id"] for t in timeline.plausible(anchor, candidates, window_h=24)] == [1, 3]


def test_only_within_the_time_window():
    anchor = _t(1, "entrance", 0, 1)
    far = _t(2, "shop", 60 * 30, 60 * 30 + 1, 0.95)    # 30 hours later
    assert [t["id"] for t in timeline.plausible(anchor, [far], window_h=24)] == [1]


class FakeEmbedder:
    model_name = "fake"


def test_timeline_api(db, camera, tmp_path):
    shop = db.get_or_create_camera("shop", "1", "security")
    a = db.create_track(camera, 1, T0)
    b = db.create_track(shop, 1, T0 + timedelta(minutes=20))
    c = db.create_track(shop, 2, T0 + timedelta(minutes=21))       # looks different
    for t in (a, b, c):
        db.update_track(t, status="finished", ended_at=T0 + timedelta(minutes=30 if t == a else 25))
    db.update_track(a, ended_at=T0 + timedelta(minutes=10))
    for t, v in ((a, [1, 0, 0]), (b, [0.95, 0.31, 0]), (c, [0, 1, 0])):
        vec = np.zeros(768, np.float32)
        vec[:3] = v
        db.upsert_embedding(t, vec / np.linalg.norm(vec), "fake")
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    app = create_app(cfg, db, None, embedder=FakeEmbedder())
    admin = TestClient(app, client=("127.0.0.1", 5000))
    out = admin.get(f"/api/tracks/{a}/timeline").json()
    assert [(t["id"], t["camera"]) for t in out] == [(a, "test-cam"), (b, "shop")]
    assert admin.get(f"/api/tracks/{a}/timeline", params={"min_similarity": 0.99}).json()[0]["id"] == a
    assert db.list_audit(action="timeline")[0]["detail"] == f"person #{a}"
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get(f"/api/tracks/{a}/timeline").status_code == 200       # part of searching

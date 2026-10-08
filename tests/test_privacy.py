"""Face blurring: where the head is, saved pictures, what admins and viewers get."""
import copy
from datetime import datetime

import cv2
import numpy as np
from fastapi.testclient import TestClient

from labeler import accounts, geometry, privacy
from labeler.config import DEFAULTS
from labeler.pipeline import Candidate, TrackState
from labeler.web import create_app
from test_pipeline_logic import make_pipeline

PW = "correct horse battery"


def _noise(h=200, w=100, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _kpts(points):
    k = np.zeros((17, 3), np.float32)
    for i, (x, y) in points.items():
        k[i] = (x, y, 0.9)
    return k


def test_head_box_covers_the_face_keypoints():
    k = _kpts({geometry.NOSE: (50, 40), geometry.L_EYE: (45, 35), geometry.R_EYE: (55, 35)})
    x1, y1, x2, y2 = geometry.head_box(k, (0, 0, 100, 300))
    assert x1 < 45 and x2 > 55 and y1 < 35 and y2 > 40


def test_head_box_without_keypoints_is_the_top_of_the_person():
    assert geometry.head_box(None, (10, 20, 110, 320)) == (10, 20, 110, 80)        # standing: top fifth
    assert geometry.head_box(None, (0, 0, 200, 100)) == (0, 0, 200, 35)            # not standing: generous


def test_blur_changes_only_the_box():
    img = _noise()
    out = privacy.blur_boxes(img, [(10, 10, 60, 60), None])
    assert not np.array_equal(out[10:60, 10:60], img[10:60, 10:60])
    assert np.array_equal(out[70:, :], img[70:, :]) and np.array_equal(out[:, 70:], img[:, 70:])
    assert np.array_equal(img, _noise())                                           # original untouched


def _person(db, camera, tmp_path, **storage):
    p = make_pipeline(db)
    p.crop_dir = tmp_path
    p.cfg["storage"].update(storage)
    st = TrackState(1, datetime.now(), 0.0, db_id=db.create_track(camera, 1, datetime.now()))
    st.candidates = [Candidate(0.9, _noise(), _noise(40, 40, 1), head=(20, 10, 80, 60))]
    p._save_crops(st)
    return st.db_id


def test_saved_picture_gets_a_blurred_copy_for_viewers(db, camera, tmp_path):
    tid = _person(db, camera, tmp_path)
    original = cv2.imread(str(tmp_path / f"{tid}.jpg"))
    copy_ = cv2.imread(str(privacy.blurred_path(tmp_path, f"{tid}.jpg")))
    assert copy_ is not None and original is not None
    diff = np.abs(original.astype(int) - copy_.astype(int)).mean(axis=2)
    assert diff[15:55, 25:75].mean() > 20 and diff[100:, :].mean() < 3            # face blurred, rest same


def test_blur_on_disk_keeps_no_face_anywhere(db, camera, tmp_path):
    tid = _person(db, camera, tmp_path, blur_faces=True, save_face_crops=True)
    assert not privacy.blurred_path(tmp_path, f"{tid}.jpg").exists()               # the picture itself is blurred
    assert not (tmp_path / f"{tid}_face.jpg").exists()                             # and no face close-up
    saved = cv2.imread(str(tmp_path / f"{tid}.jpg")).astype(int)
    assert np.abs(saved[15:55, 25:75] - _noise()[15:55, 25:75].astype(int)).mean() > 20


def _clients(db, tmp_path, pipelines=None):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    app = create_app(cfg, db, pipelines)
    db.add_user("vic", accounts.hash_password(PW), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": PW})
    return TestClient(app, client=("127.0.0.1", 5000)), viewer


def test_viewers_get_blurred_pictures_admins_the_original(db, camera, tmp_path):
    tid = _person(db, camera, tmp_path)
    cv2.imwrite(str(tmp_path / "999.jpg"), _noise(seed=5))                        # saved before blurring existed
    admin, viewer = _clients(db, tmp_path)
    name = f"{tid}.jpg"
    assert admin.get(f"/crops/{name}").content == (tmp_path / name).read_bytes()
    assert viewer.get(f"/crops/{name}").content == privacy.blurred_path(tmp_path, name).read_bytes()
    old = viewer.get("/crops/999.jpg")
    assert old.status_code == 200 and old.content != (tmp_path / "999.jpg").read_bytes()   # blurred whole
    assert old.headers["vary"] == "Cookie"
    for bad in ["nothing.jpg", "x.png", "a b.jpg"]:
        assert admin.get(f"/crops/{bad}").status_code == 404


def test_viewers_see_blurred_pictures_only_if_the_option_is_on(db, camera, tmp_path):
    tid = _person(db, camera, tmp_path)
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    cfg["web"]["blur_faces_for_viewers"] = False
    db.add_user("vic", accounts.hash_password(PW), "viewer")
    viewer = TestClient(create_app(cfg, db, None), client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": PW})
    assert viewer.get(f"/crops/{tid}.jpg").content == (tmp_path / f"{tid}.jpg").read_bytes()


def test_deleting_a_person_removes_the_blurred_copy_too(db, camera, tmp_path):
    tid = _person(db, camera, tmp_path)
    admin, _ = _clients(db, tmp_path)
    assert admin.post("/api/tracks/delete", json={"ids": [tid]}).json()["pictures_removed"] == 1
    assert not (tmp_path / f"{tid}.jpg").exists()
    assert not privacy.blurred_path(tmp_path, f"{tid}.jpg").exists()


class FakePipeline:
    name = "cam1"

    def live_thumb(self, tracker_id, blurred=False):
        return b"blurred" if blurred else b"face"


def test_live_thumbnails_are_blurred_for_viewers(db, tmp_path):
    admin, viewer = _clients(db, tmp_path, [FakePipeline()])
    assert admin.get("/api/live/thumb/1").content == b"face"
    assert viewer.get("/api/live/thumb/1").content == b"blurred"


def test_blurred_live_frames_are_made_only_while_a_viewer_watches(db):
    p = make_pipeline(db)
    p._jpeg_lock = __import__("threading").Lock()
    p._jpeg = p._jpeg_blurred = None
    p.viewer_watching_at = -1e9
    frame = _noise(64, 64)
    p._publish(frame, privacy.blur_boxes(frame, [(0, 0, 64, 64)]))
    assert p.latest_jpeg is not None
    assert p.latest_jpeg_blurred() not in (None, p.latest_jpeg)
    assert p.viewer_watching_at > 0                                                # asking keeps them coming

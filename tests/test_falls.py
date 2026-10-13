"""Fall detection from made-up body points: sudden falls, slow lying down, getting up, beds."""
from datetime import datetime

import numpy as np

from labeler.config import DEFAULTS
from labeler.falls import LYING, UNSURE, UPRIGHT, FallWatch, posture
from labeler.geometry import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

CFG = {**DEFAULTS["falls"], "after_s": 10.0, "drop_s": 1.0, "min_drop": 0.4, "getup_s": 1.0}
FPS = 10


def standing(x=300.0, top=100.0, h=300.0):
    """Box and body points of someone standing (shoulders above hips)."""
    k = np.zeros((17, 3), np.float32)
    k[[L_SHOULDER, R_SHOULDER], :] = [[x - 30, top + 0.2 * h, 0.9], [x + 30, top + 0.2 * h, 0.9]]
    k[[L_HIP, R_HIP], :] = [[x - 20, top + 0.5 * h, 0.9], [x + 20, top + 0.5 * h, 0.9]]
    return (x - 50, top, x + 50, top + h), k


def lying(x=300.0, floor=400.0, length=300.0):
    """Box and body points of someone lying on the floor (shoulders level with hips)."""
    k = np.zeros((17, 3), np.float32)
    k[[L_SHOULDER, R_SHOULDER], :] = [[x - 100, floor - 50, 0.9], [x - 100, floor - 30, 0.9]]
    k[[L_HIP, R_HIP], :] = [[x + 10, floor - 50, 0.9], [x + 10, floor - 30, 0.9]]
    return (x - length / 2, floor - 80, x + length / 2, floor), k


def feed(watch, frames, key=1, t0=0.0, lying_ok=False):
    """Feed (box, kpts) frames at FPS; returns [(time, event)] for confirmed falls."""
    out = []
    for i, (box, k) in enumerate(frames):
        t = t0 + i / FPS
        e = watch.update(key, t, box, k, lying_ok)
        if e:
            out.append((t, e))
    return out


def test_posture_from_the_torso_and_from_the_box():
    assert posture(standing()[1], standing()[0]) == UPRIGHT
    assert posture(lying()[1], lying()[0]) == LYING
    assert posture(None, (0, 0, 300, 100)) == LYING        # no body points: a wide box
    assert posture(None, (0, 0, 100, 300)) == UPRIGHT
    assert posture(None, (0, 0, 100, 120)) == UNSURE


def test_sudden_fall_is_confirmed_after_staying_down():
    watch = FallWatch(CFG)
    frames = [standing()] * 30 + [lying()] * 120   # 3 s standing, then on the floor for 12 s
    events = feed(watch, frames)
    assert len(events) == 1
    t, e = events[0]
    assert e["since"] == 3.0 and abs(t - 13.0) < 0.05
    assert watch.state(1) == "fallen"


def test_not_yet_before_after_s():
    watch = FallWatch(CFG)
    assert feed(watch, [standing()] * 30 + [lying()] * 50) == []
    assert watch.state(1) == "down"


def test_lying_down_slowly_is_not_a_fall():
    watch = FallWatch(CFG)
    # Standing, then sitting on a bed a while (no body points), then lying: standing was 4 s before lying.
    sit_box = (250, 180, 350, 400)
    frames = [standing()] * 30 + [(sit_box, None)] * 40 + [lying(floor=300)] * 150
    assert feed(watch, frames) == []
    assert watch.state(1) is None


def test_small_head_drop_is_not_a_fall():
    watch = FallWatch({**CFG, "min_drop": 0.4})
    # Lying on a high bed: the box top comes down only a little.
    frames = [standing(top=100)] * 30 + [lying(floor=200)] * 150
    assert feed(watch, frames) == []


def test_getting_up_cancels_and_a_second_fall_counts_again():
    watch = FallWatch(CFG)
    frames = [standing()] * 30 + [lying()] * 40 + [standing()] * 20
    assert feed(watch, frames) == []
    assert watch.state(1) is None
    events = feed(watch, [standing()] * 10 + [lying()] * 110, t0=9.0)
    assert len(events) == 1


def test_short_upright_flicker_does_not_cancel():
    watch = FallWatch(CFG)
    frames = [standing()] * 30 + [lying()] * 40 + [standing()] * 3 + [lying()] * 80
    assert len(feed(watch, frames)) == 1


def test_one_alert_per_fall():
    watch = FallWatch(CFG)
    assert len(feed(watch, [standing()] * 30 + [lying()] * 300)) == 1


def test_no_fall_where_lying_is_normal():
    watch = FallWatch(CFG)
    assert feed(watch, [standing()] * 30 + [lying()] * 150, lying_ok=True) == []


def test_people_are_watched_separately_and_forgotten():
    watch = FallWatch(CFG)
    assert feed(watch, [standing()] * 30 + [lying()] * 50, key=1) == []
    assert feed(watch, [standing()] * 80, key=2) == []
    assert watch.state(1) == "down" and watch.state(2) is None
    watch.forget(1)
    assert watch.state(1) is None


def test_kneeling_or_crouching_on_the_floor_is_not_a_fall():
    watch = FallWatch(CFG)
    crouch = ((260, 260, 340, 400), None)   # short and narrow: looking under a bed
    assert feed(watch, [standing()] * 30 + [crouch] * 150) == []


def test_fall_out_of_sight_at_the_bottom_edge():
    """Falling at the bottom of the picture, half cut off: the detector loses them, the fall still counts."""
    watch = FallWatch(CFG)
    events = []
    frames = [standing(top=180, h=300)] * 30 + [lying(floor=480)] * 2   # lands at the edge (picture 480 high)
    for i, (box, k) in enumerate(frames):
        watch.update(1, i / FPS, box, k, frame_h=480)
    for i in range(len(frames), len(frames) + 130):   # then nobody is seen
        events += watch.tick(i / FPS)
    assert len(events) == 1 and events[0][0] == 1


def test_walking_out_of_the_bottom_of_the_picture_is_not_a_fall():
    """Upright all the way, cut shorter by the edge (MOT16-04): not down."""
    watch = FallWatch(CFG)
    events = []
    for i in range(30):
        top = 100 + 12 * i
        box = (300, top, 380, min(480, top + 300))
        k = standing(x=340, top=top, h=300)[1]
        watch.update(1, i / FPS, box, k, frame_h=480)
    for i in range(30, 200):
        events += watch.tick(i / FPS)
    assert events == []


def test_new_track_number_during_a_fall_keeps_the_history():
    watch = FallWatch(CFG)
    assert feed(watch, [standing()] * 30, key=1) == []
    events = feed(watch, [lying()] * 110, key=2, t0=3.1)   # the tracker calls the fallen person #2
    assert len(events) == 1


# --- zones where lying is normal, and the camera loop's part ---

from fastapi.testclient import TestClient  # noqa: E402

from labeler import zones  # noqa: E402
from labeler.alerts import AlertEngine  # noqa: E402
from labeler.pipeline import TrackState  # noqa: E402
from labeler.web import create_app  # noqa: E402
from test_pipeline_logic import make_pipeline  # noqa: E402

BED = [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]


def test_zone_marked_lying_is_normal(db):
    bed = db.add_zone("test-cam", "bed", "area", BED, lying_ok=True)
    line = db.add_zone("test-cam", "door", "line", [(0, 0.2), (1, 0.2)], lying_ok=True)
    assert bed["lying_ok"] is True and line["lying_ok"] is False   # only areas can be beds
    zs = db.list_zones("test-cam")
    assert zones.lying_ok(zs, (0.7, 0.8)) and not zones.lying_ok(zs, (0.2, 0.8))


def test_zone_api_takes_the_flag(db, tmp_path):
    import copy
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    client = TestClient(create_app(cfg, db, None, None, AlertEngine(db)), client=("127.0.0.1", 5000))
    r = client.post("/api/zones", json={"camera": "c", "name": "sofa", "kind": "area", "points": BED,
                                        "lying_ok": True})
    assert r.status_code == 200 and r.json()["lying_ok"] is True
    assert client.get("/api/zones").json()[0]["lying_ok"] is True
    assert "lying is normal here" in client.get("/api/audit").json()["entries"][0]["detail"]


def test_pipeline_names_the_person_who_fell(db, camera, caplog):
    p = make_pipeline(db)
    p.falls = FallWatch(CFG)
    p._fall_db = {}
    st = TrackState(7, datetime.now(), 0.0, db_id=db.create_track(camera, 7, datetime.now()))
    p.tracks[7] = st
    p._on_fall(7)
    p._fall_db[8] = 1234                      # a person whose track already ended
    p._on_fall(8)
    text = caplog.text
    assert f"person #{st.db_id} has been down" in text and "person #1234 has been down" in text


def test_one_person_seen_as_two_boxes_gives_one_alert():
    watch = FallWatch(CFG)
    out = []
    for i, (box, k) in enumerate([standing()] * 30 + [lying()] * 120):
        for key, b in ((1, box), (2, (box[0] + 5, box[1], box[2] + 5, box[3]))):   # two boxes on one person
            if watch.update(key, i / FPS, b, k):
                out.append(key)
    assert len(out) == 1


def test_reused_track_number_takes_the_fresher_history():
    """The tracker gives an old number (last seen lying long ago) to the person who just fell."""
    watch = FallWatch(CFG)
    feed(watch, [lying()] * 5, key=3)                          # number 3: someone lying, then gone
    feed(watch, [standing()] * 30, key=5, t0=5.0)              # number 5: standing (8.0 s)
    events = feed(watch, [lying()] * 110, key=3, t0=8.1)       # the fall, seen as number 3 again
    assert len(events) == 1 and abs(events[0][1]["since"] - 8.1) < 0.05


def test_a_second_fall_at_the_same_spot_later_counts():
    watch = FallWatch(CFG)
    assert len(feed(watch, [standing()] * 30 + [lying()] * 120, key=1)) == 1   # first fall, then gone
    events = feed(watch, [standing()] * 30 + [lying()] * 120, key=2, t0=40.0)   # 25 s later, same place
    assert len(events) == 1


def test_number_switch_in_the_same_frame_is_merged_not_out_of_sight():
    """The tracker starts number 2 (lying) in the frame number 1 (falling) is last seen, at the bottom edge."""
    watch = FallWatch(CFG)
    events = []
    for i in range(30):
        watch.update(1, i / FPS, *standing(top=180, h=300), frame_h=480)
    watch.update(1, 3.0, *lying(floor=480), frame_h=480)
    for i in range(30, 150):
        t = i / FPS + 0.001
        events += [(2, e) for e in [watch.update(2, t, *lying(floor=478), frame_h=480)] if e]
        events += watch.tick(t)
    assert len(events) == 1 and events[0][0] == 2   # one alert, for the number still in view
    assert watch.state(2) == "fallen" and 1 not in watch.tracks


def test_out_of_sight_respects_lying_is_normal_here():
    watch = FallWatch(CFG)
    for i in range(30):
        watch.update(1, i / FPS, *standing(top=180, h=300), lying_ok=True, frame_h=480)
    watch.update(1, 3.0, *lying(floor=480), lying_ok=True, frame_h=480)
    assert sum(len(watch.tick(i / FPS)) for i in range(31, 200)) == 0

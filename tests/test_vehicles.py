"""Vehicles (labeler/vehicles.py): one record per visit, parked vehicles once, the vehicles table."""
import copy
from datetime import datetime, timedelta

import numpy as np

from labeler.config import DEFAULTS
from labeler.pipeline import Pipeline
from labeler.vehicles import VehicleStore, VehicleWatch

CFG = {**DEFAULTS["vehicles"], "parked_after_s": 60.0, "parked_memory_s": 3600.0, "lost_after_s": 4.0}
T0 = datetime(2026, 10, 13, 9, 0, 0)
CAR, BICYCLE, LORRY = 2, 1, 7
FRAME = np.zeros((480, 640, 3), np.uint8)


class FakeStore:
    def __init__(self):
        self.rows, self.pictures = {}, {}

    def create(self, kind, started_at, tracker_id):
        i = len(self.rows) + 1
        self.rows[i] = {"kind": kind, "started_at": started_at}
        return i

    def update(self, vehicle_id, **fields):
        self.rows[vehicle_id].update(fields)

    def picture(self, vehicle_id, crop):
        self.pictures[vehicle_id] = crop.shape

    def describe(self, vehicle_id, crop):
        self.rows[vehicle_id]["described"] = self.rows[vehicle_id].get("described", 0) + 1


def run(watch, frames, t0=0.0, fps=5):
    """frames: list of detection lists, one per frame, at fps."""
    for i, dets in enumerate(frames):
        t = t0 + i / fps
        watch.update(t, T0 + timedelta(seconds=t), FRAME, dets)
    return t0 + len(frames) / fps


def passing(key=1, cls=CAR, n=30, conf=0.8):
    """A vehicle driving across the picture."""
    return [[(key, (20 + 15 * i, 200, 140 + 15 * i, 280), cls, conf)] for i in range(n)]


def standing(key=1, cls=CAR, n=10, box=(300, 200, 420, 280), conf=0.8):
    return [[(key, box, cls, conf)] for _ in range(n)]


def test_a_passing_car_is_one_record_with_a_picture():
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    end = run(watch, passing())
    run(watch, [[]] * 30, t0=end)                       # gone for 6 s: finished
    assert list(store.rows) == [1]
    row = store.rows[1]
    assert row["kind"] == "car" and row["ended_at"] is not None and not row.get("parked")
    assert 1 in store.pictures


def test_seen_too_briefly_is_not_stored():
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    end = run(watch, passing(n=3))
    run(watch, [[]] * 30, t0=end)
    assert store.rows == {}


def test_kind_is_the_majority_and_unwanted_kinds_are_skipped():
    store = FakeStore()
    watch = VehicleWatch(CFG, store, kinds=["car", "lorry"])
    frames = [[(1, b, LORRY if i % 3 else CAR, 0.8)] for i, ((_, b, _, _),) in enumerate(passing())]
    end = run(watch, frames + [[]] * 30)
    run(watch, passing(key=2, cls=BICYCLE), t0=end)     # bicycles not wanted here
    assert [r["kind"] for r in store.rows.values()] == ["lorry"]


def test_parked_car_is_one_record_while_hidden_and_found_again():
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    t = run(watch, standing(n=5 * 70))                  # 70 s still: parked after 60 s
    assert store.rows[1]["parked"] is True
    t = run(watch, [[]] * 5 * 20, t0=t)                 # hidden 20 s by someone walking past: lost
    assert store.rows[1]["ended_at"] is not None
    run(watch, standing(key=9, n=50), t0=t)             # found again, with a new tracker number
    assert list(store.rows) == [1] and store.rows[1]["ended_at"] is None


def test_parked_car_driving_off_is_marked_left():
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    t = run(watch, standing(n=5 * 70))
    leave = [[(1, (300 - 20 * i, 200, 420 - 20 * i, 280), CAR, 0.8)] for i in range(1, 12)]
    t = run(watch, leave, t0=t)
    run(watch, [[]] * 30, t0=t)
    row = store.rows[1]
    assert row["left_at"] is not None and list(store.rows) == [1]


def test_a_different_car_parking_elsewhere_is_a_new_record():
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    t = run(watch, standing(n=5 * 70))
    t = run(watch, [[]] * 50, t0=t)
    run(watch, standing(key=5, n=20, box=(20, 300, 140, 380)), t0=t)
    assert list(store.rows) == [1, 2]


def test_moving_and_shown():
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    t = run(watch, passing(n=10))
    assert watch.moving(t - 0.2)
    assert watch.shown(t - 0.2)[0][1].startswith("#1 car")
    t = run(watch, standing(key=2, n=5 * 70), t0=t + 10)
    assert not watch.moving(t - 0.2) and "parked" in watch.shown(t - 0.2)[0][1]


def test_picture_skips_boxes_cut_by_the_border():
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    end = run(watch, standing(n=10, box=(0, 200, 300, 400)))   # touches the left border
    run(watch, [[]] * 30, t0=end)
    assert store.rows and store.pictures == {}


def test_vehicles_table_store_list_delete_and_retention(db, camera, tmp_path):
    store = VehicleStore(db, camera, tmp_path)
    old = store.create("car", datetime.now() - timedelta(days=40), 1)
    new = store.create("motorbike", datetime.now(), 2)
    store.picture(new, np.zeros((40, 60, 3), np.uint8))
    store.update(new, parked=True, color="red")
    rows = db.list_vehicles()
    assert [r["id"] for r in rows] == [new, old] and rows[0]["parked"] and rows[0]["crop"] == f"vehicles/{new}.jpg"
    assert (tmp_path / "vehicles" / f"{new}.jpg").is_file()
    assert db.list_vehicles(kind="car")[0]["id"] == old
    db.cleanup(30)
    assert [r["id"] for r in db.list_vehicles()] == [new]
    assert db.delete_vehicles([new]) == [f"vehicles/{new}.jpg"] and db.list_vehicles() == []


def test_only_security_cameras_look_for_vehicles_by_default(db, monkeypatch, tmp_path):
    monkeypatch.setattr("labeler.pipeline.YOLO", lambda path: object())
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    cfg["clips"]["enabled"] = False
    gate = Pipeline(cfg, db, None, camera={"name": "gate", "source": "rtsp://x", "mode": "security"})
    desk = Pipeline(cfg, db, None, camera={"name": "desk", "source": 0, "mode": "kiosk"})
    yard = Pipeline(cfg, db, None, camera={"name": "yard", "source": 1, "mode": "kiosk", "vehicles": True})
    assert gate.vehicles is not None and desk.vehicles is None and yard.vehicles is not None


def test_standing_vehicle_lost_now_and_then_becomes_parked_as_one_record():
    """A bicycle in a rack, lost for 6 s every 20 s behind passing people (MOT16-04)."""
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    t = 0.0
    for k in range(5):                                   # 5 x (20 s seen + 6 s hidden) = 130 s
        t = run(watch, standing(key=10 + k, cls=BICYCLE, n=5 * 20), t0=t)
        t = run(watch, [[]] * 30, t0=t)
    assert list(store.rows) == [1] and store.rows[1]["parked"] is True


# --- step 2: describing vehicles ---

from labeler.vehicles import apply_description, body_color  # noqa: E402
from labeler.vlm import VEHICLE_TYPES, clean_vehicle, vehicle_schema  # noqa: E402


def test_each_vehicle_is_described_once():
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    end = run(watch, passing(n=40))                     # 8 s in view: described after 5 s
    run(watch, [[]] * 30, t0=end)
    end = run(watch, passing(key=2, n=10), t0=end + 10)  # 2 s in view: described when it leaves
    run(watch, [[]] * 30, t0=end)
    assert [r["described"] for r in store.rows.values()] == [1, 1]


def test_vehicle_answers_are_cleaned():
    assert set(vehicle_schema()["properties"]["type"]["enum"]) == set(VEHICLE_TYPES)
    assert clean_vehicle({"vehicle_in_picture": "no", "type": "car"})["type"] == "not a vehicle"
    out = clean_vehicle({"type": "Car", "color": "silver", "make": "Perodua", "model": "Myvi"})
    assert out == {"type": "car", "color": "silver", "make": "Perodua", "model": "Myvi"}
    out = clean_vehicle({"type": "spaceship", "color": "chrome", "make": "it looks like a small Japanese car",
                         "model": "N/A"})
    assert out == {"type": "unknown", "color": "unknown", "make": "unknown", "model": "unknown"}


def test_description_is_stored_and_not_a_vehicle_is_hidden(db, camera, tmp_path):
    store = VehicleStore(db, camera, tmp_path)
    car = store.create("lorry", datetime.now(), 1)
    sign = store.create("car", datetime.now(), 2)
    apply_description(db, car, {"type": "pickup", "color": "white", "make": "Toyota", "model": "Hilux"})
    apply_description(db, sign, {"type": "not a vehicle", "color": "red", "make": "unknown", "model": "unknown"})
    rows = db.list_vehicles()
    assert [r["id"] for r in rows] == [car]
    assert {k: rows[0][k] for k in ("kind", "color", "make", "model")} == {
        "kind": "pickup", "color": "white", "make": "Toyota", "model": "Hilux"}
    apply_description(db, car, {"type": "unknown", "color": "unknown", "make": "unknown", "model": "unknown"})
    assert db.list_vehicles()[0]["make"] == "Toyota"   # "unknown" never overwrites an answer


def test_first_color_from_the_middle_of_the_body():
    crop = np.zeros((100, 200, 3), np.uint8)
    crop[:, :] = (40, 40, 200)                    # red body (BGR)
    crop[:30, :] = (20, 20, 20)                   # dark windows on top
    assert body_color(crop) == "red"
    assert body_color(np.full((100, 200, 3), 128, np.uint8)) is None   # gray picture: no color (night)


def test_pipeline_sends_vehicles_to_the_vision_model_and_routes_the_answer(db, camera, tmp_path):
    from test_pipeline_logic import make_pipeline
    p = make_pipeline(db)
    p.owned_ids, p.vehicle_cfg = set(), dict(DEFAULTS["vehicles"])
    sent = []

    class VLM:
        def submit(self, job_id, frames, mode):
            sent.append((job_id, mode))
            return True

    p.vlm = VLM()
    vid = VehicleStore(db, camera, tmp_path).create("car", datetime.now(), 1)
    crop = np.zeros((60, 100, 3), np.uint8)
    crop[:, :] = (200, 60, 30)                    # blue
    p._describe_vehicle(vid, crop)
    assert sent == [(-vid, "vehicle")] and -vid in p.owned_ids
    assert db.list_vehicles()[0]["color"] == "blue"
    p.on_vlm_result(-vid, [{"type": "car", "color": "blue", "make": "Proton", "model": "Saga"}])
    assert db.list_vehicles()[0]["make"] == "Proton"

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

    def picture(self, vehicle_id, crop, heads=None):
        self.pictures[vehicle_id] = crop.shape

    def describe(self, vehicle_id, crop):
        self.rows[vehicle_id]["described"] = self.rows[vehicle_id].get("described", 0) + 1

    def changed(self, vehicle_id):
        pass

    def zone_events(self, vehicle_id, events):
        self.rows[vehicle_id].setdefault("zones", []).extend(e["kind"] for e in events)


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


def test_picture_prefers_boxes_not_cut_by_the_border_but_always_has_one():
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    end = run(watch, standing(n=10, box=(0, 200, 300, 400)))   # touches the left border: only an unclear picture
    run(watch, [[]] * 30, t0=end)
    assert store.pictures[1] == (220, 315, 3)
    store = FakeStore()
    watch = VehicleWatch(CFG, store)
    frames = standing(n=10, box=(0, 100, 400, 400)) + standing(n=10, box=(100, 200, 220, 280))   # cut big, then clear
    end = run(watch, frames)
    run(watch, [[]] * 30, t0=end)
    assert store.pictures[1] == (88, 132, 3)                    # the clear one, though smaller


def test_vehicles_table_store_list_delete_and_retention(db, camera, tmp_path):
    store = VehicleStore(db, camera, tmp_path)
    old = store.create("car", datetime.now() - timedelta(days=40), 1)
    new = store.create("motorbike", datetime.now(), 2)
    store.picture(new, np.zeros((40, 60, 3), np.uint8))
    store.update(new, parked=True, color="red")
    rows = db.list_vehicles()
    assert [r["id"] for r in rows] == [new, old] and rows[0]["parked"] and rows[0]["crop"] == f"vehicle_{new}.jpg"
    assert (tmp_path / f"vehicle_{new}.jpg").is_file() and (tmp_path / "blurred" / f"vehicle_{new}.jpg").is_file()
    assert db.list_vehicles(kind="car")[0]["id"] == old
    db.cleanup(30)
    assert [r["id"] for r in db.list_vehicles()] == [new]
    assert db.delete_vehicles([new]) == ([new], [f"vehicle_{new}.jpg"]) and db.list_vehicles() == []


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


# --- step 3: lists, the Ask box, Reports, counting lines ---

from fastapi.testclient import TestClient  # noqa: E402

from labeler import accounts, reports  # noqa: E402
from labeler.agent import QueryAgent, vehicle_query  # noqa: E402
from labeler.alerts import AlertEngine  # noqa: E402
from labeler.web import create_app  # noqa: E402
from labeler.zones import ZoneCounter  # noqa: E402


def _vehicles(db, camera, tmp_path):
    store = VehicleStore(db, camera, tmp_path)
    ids = {}
    for name, kind, color, make, ago in [("suv", "SUV", "silver", "Toyota", 1), ("myvi", "car", "red", "Perodua", 2),
                                          ("bike", "motorbike", "black", None, 3), ("old", "car", "red", None, 50)]:
        i = store.create(kind, datetime(2026, 10, 13, 12) - timedelta(hours=ago), 1)
        db.update_vehicle(i, color=color, make=make)
        ids[name] = i
    return ids


def _app(db, tmp_path, address="127.0.0.1"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, None, None, AlertEngine(db)), client=(address, 5000))


def test_vehicles_api_filters_and_delete(db, camera, tmp_path):
    ids = _vehicles(db, camera, tmp_path)
    client = _app(db, tmp_path)
    got = lambda **q: [v["id"] for v in client.get("/api/vehicles", params=q).json()]   # noqa: E731
    assert got(kind="car") == [ids["suv"], ids["myvi"], ids["old"]]      # "car" also finds the SUV
    assert got(kind="SUV") == [ids["suv"]]
    assert got(color="gray") == [ids["suv"]]                             # silver counts as gray, and back
    assert got(day="2026-10-13") == [ids["suv"], ids["myvi"], ids["bike"]]
    assert client.get("/api/vehicles", params={"day": "13/10"}).status_code == 400
    r = client.post("/api/vehicles/delete", json={"ids": [ids["bike"], 999]})
    assert r.json()["deleted"] == 1 and ids["bike"] not in got()
    assert "1 vehicles" in client.get("/api/audit").json()["entries"][0]["detail"]


def test_viewers_can_list_but_not_delete_vehicles(db, camera, tmp_path):
    _vehicles(db, camera, tmp_path)
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _app(db, tmp_path, address="192.168.1.20")
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/vehicles").status_code == 200
    assert viewer.post("/api/vehicles/delete", json={"ids": [1]}).status_code == 403


def test_ask_box_knows_vehicle_questions():
    q = vehicle_query("any red cars yesterday evening?")
    assert q["kinds"] == ["SUV", "car", "pickup", "van"] and q["colors"] == ["red"]
    assert vehicle_query("a bike at the gate")["kinds"] == ["bicycle", "motorbike"]
    assert vehicle_query("the grey lorry")["colors"] == ["gray", "silver"]
    assert vehicle_query("was the Perodua here?")["make"] == "perodua"
    assert vehicle_query("the Myvi", models={"Myvi"})["model"] == "myvi"
    assert vehicle_query("a man in a red shirt") is None
    assert vehicle_query("who carried a bag") is None


def test_ask_box_answers_with_vehicles(db, camera, tmp_path, monkeypatch):
    ids = _vehicles(db, camera, tmp_path)
    agent = QueryAgent({"vlm": {"host": "http://x", "model": "m"}}, db, None, ["test-cam"])
    monkeypatch.setattr(agent, "_ask_llm", lambda q, now: {"start": "2026-10-13T00:00", "end": "2026-10-13T23:59",
                                                          "camera": "", "conditions": [], "appearance": ""})
    res = agent.run("red cars today", now=datetime(2026, 10, 13, 18))
    assert [v["id"] for v in res["vehicles"]] == [ids["myvi"]] and res["people"] == []
    assert "red" in res["explanation"] and "car" in res["explanation"]
    res = agent.run("any toyota today", now=datetime(2026, 10, 13, 18))
    assert [v["id"] for v in res["vehicles"]] == [ids["suv"]]


def test_reports_count_vehicles_per_hour_and_kind(db, camera, tmp_path):
    _vehicles(db, camera, tmp_path)
    s, e = datetime(2026, 10, 13), datetime(2026, 10, 14)
    out = reports.summarize_vehicles(db.vehicle_rows(s, e), s, e)
    assert out["total"] == 3 and out["unit"] == "hour"
    assert {k["value"]: k["count"] for k in out["kinds"]} == {"SUV": 1, "car": 1, "motorbike": 1}
    assert sum(r["count"] for r in out["timeline"]) == 3
    summary = _app(db, tmp_path).get("/api/reports/summary", params={"start": "2026-10-13", "end": "2026-10-13"}).json()
    assert summary["vehicles"]["total"] == 3


def test_counting_lines_count_vehicles_apart_from_people(db, camera, tmp_path):
    store = VehicleStore(db, camera, tmp_path, camera="test-cam")
    watch = VehicleWatch(CFG, store)
    gate = {"id": 1, "name": "gate", "kind": "line", "points": [(0.5, 0.0), (0.5, 1.0)]}   # a vertical line
    watch.zone_counter = ZoneCounter([gate], 3)
    end = run(watch, passing(n=40))          # drives left to right across the middle
    run(watch, [[]] * 30, t0=end)
    day = T0.replace(hour=0)
    counts = db.zone_counts(day, day + timedelta(days=1))
    assert len(counts) == 1 and counts[0]["vehicles_in"] + counts[0]["vehicles_out"] == 1
    assert counts[0]["in"] + counts[0]["out"] == 0                    # people untouched
    vid = db.list_vehicles()[0]["id"]
    db.delete_vehicles([vid])
    assert db.zone_counts(day, day + timedelta(days=1)) == []


# --- step 4: alert rules on vehicles ---

from labeler.alerts import describe, in_hours, matches  # noqa: E402


def test_vehicle_conditions_match():
    car = {"vehicle": "SUV", "vehicle_color": "silver", "zone": "gate", "hour": 23}
    assert matches([["vehicle", "any"]], car) and matches([["vehicle", "car"]], car)       # "car" includes SUVs
    assert not matches([["vehicle", "motorbike"]], car)
    assert matches([["vehicle", "any"], ["vehicle_color", "gray"]], car)                  # silver = gray here
    assert matches([["vehicle", "any"], ["zone", "gate"], ["hours", "22-06"]], car)
    assert not matches([["vehicle", "any"], ["hours", "06-18"]], car)
    assert not matches([["vehicle", "any"]], {"upper_color": "red"})                     # a person is no vehicle
    assert describe([["vehicle", "any"], ["hours", "22-06"]]) == "any vehicle, between 22:00 and 06:00"
    assert in_hours("22-06", 2) and not in_hours("22-06", 12) and in_hours("18-24", 23) and not in_hours("18-24", 0)


def test_vehicle_rules_fire_once_per_vehicle_and_never_for_people(db, camera, tmp_path):
    engine = AlertEngine(db)
    db.add_rule("night vehicle", [["vehicle", "any"], ["hours", "00-24"]])
    db.add_rule("red motorbike", [["vehicle", "motorbike"], ["vehicle_color", "red"]])
    db.add_rule("anyone", [])                                                            # a person rule
    store = VehicleStore(db, camera, tmp_path)
    bike = store.create("motorbike", datetime.now(), 1)
    assert [e["message"] for e in engine.check_vehicle(bike, "test-cam")] == ["night vehicle: any vehicle, between 00:00 and 00:00"]
    db.update_vehicle(bike, color="red")
    assert [e["message"] for e in engine.check_vehicle(bike, "test-cam")] == ["red motorbike: vehicle = motorbike, vehicle color = red"]
    assert engine.check_vehicle(bike, "test-cam") == []                                  # once per rule and vehicle
    person = db.create_track(camera, 1, datetime.now())
    assert [e["message"] for e in engine.check(person, "test-cam")] == ["anyone: anyone"]  # vehicle rules skip people
    db.update_vehicle(bike, not_vehicle=True)
    other = store.create("car", datetime.now(), 2)
    db.update_vehicle(other, not_vehicle=True)
    assert engine.check_vehicle(other, "test-cam") == []                                 # "not a vehicle": no alert


def test_vehicle_alerts_are_listed_and_deleted_with_the_vehicle(db, camera, tmp_path):
    engine = AlertEngine(db)
    db.add_rule("any vehicle", [["vehicle", "any"]])
    store = VehicleStore(db, camera, tmp_path)
    vid = store.create("car", datetime.now(), 1)
    store.picture(vid, np.zeros((40, 60, 3), np.uint8))
    event = engine.check_vehicle(vid, "test-cam")[0]
    shown = db.list_alerts()[0]
    assert shown["vehicle_id"] == vid and shown["track_id"] is None and shown["crop"] == f"vehicle_{vid}.jpg"
    _, paths = db.delete_vehicles([vid])
    assert f"clips/alert_{event['id']}.webm" in paths and db.list_alerts() == []


def test_rule_api_checks_vehicle_rules(db, tmp_path):
    client = _app(db, tmp_path)
    post = lambda conds: client.post("/api/alerts/rules", json={"name": "r", "conditions": conds})   # noqa: E731
    assert post([["vehicle", "motorbike"], ["vehicle_color", "red"], ["hours", "22-06"]]).status_code == 200
    assert post([["hours", "09-17"]]).status_code == 200                                 # people at office hours
    assert post([["vehicle", "spaceship"]]).status_code == 400
    assert post([["hours", "25-06"]]).status_code == 400
    assert post([["vehicle_color", "red"]]).status_code == 400                           # color without "vehicle"
    assert post([["vehicle", "any"], ["upper_color", "red"]]).status_code == 400         # a person's label


def test_pipeline_checks_vehicle_rules_and_saves_a_clip(db, camera, tmp_path):
    from test_pipeline_logic import make_pipeline
    p = make_pipeline(db)
    asked = []

    class Alerts:
        def check_vehicle(self, vehicle_id, camera_name):
            asked.append(vehicle_id)
            return [{"id": 7}]

    class Clips:
        def request(self, event_id, before_s=None):
            asked.append(("clip", event_id))

    p.alerts, p.clips = Alerts(), Clips()
    p._check_vehicle_alerts(5)
    assert asked == [5, ("clip", 7)]


def test_old_sqlite_database_skips_vehicle_alerts_instead_of_failing(tmp_path):
    from sqlalchemy import text
    from labeler.db import Database
    db = Database("sqlite:///" + (tmp_path / "old.db").as_posix())
    with db.engine.begin() as conn:   # the table as it was before: track_id required
        conn.execute(text("DROP TABLE alert_events"))
        conn.execute(text("CREATE TABLE alert_events (id INTEGER PRIMARY KEY, rule_id INTEGER NOT NULL, "
                          "track_id INTEGER NOT NULL, vehicle_id INTEGER, camera VARCHAR(100), message VARCHAR(300), "
                          "created_at DATETIME)"))
    cam = db.get_or_create_camera("c", "x", "security")
    vid = db.create_vehicle(cam, 1, "car", datetime.now())
    rule = db.add_rule("v", [["vehicle", "any"]])["id"]
    assert db.record_alert(rule, None, "c", "v", vehicle_id=vid) is None

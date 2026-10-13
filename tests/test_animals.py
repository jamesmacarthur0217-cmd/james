"""Animals (Phase 21): dogs, cats and birds from the vehicle detector's pass, followed and stored like vehicles."""
from datetime import datetime, timedelta

import numpy as np

from labeler.animals import ALL_KINDS, AnimalStore, kinds_map
from labeler.config import DEFAULTS
from labeler.vehicles import KINDS as VEHICLE_KINDS, VehicleWatch
from test_pipeline_logic import make_pipeline

T0 = datetime(2026, 10, 14, 12, 0, 0)
DOG, CAT, BIRD, CAR = 16, 15, 14, 2


def _watch(db, camera, tmp_path, kinds=("dog", "cat", "bird")):
    store = AnimalStore(db, camera, tmp_path)
    return VehicleWatch(DEFAULTS["animals"], store, list(kinds), names=kinds_map(list(kinds)), still_word="resting")


def _feed(watch, frames, key, box, cls, start=0.0, fps=10, frame=None):
    for i in range(frames):
        t = start + i / fps
        watch.update(t, T0 + timedelta(seconds=t), frame, [(key, box, cls, 0.9)])
    return start + frames / fps


def test_kinds_map():
    assert kinds_map(["dog", "cat", "bird"]) == {14: "bird", 15: "cat", 16: "dog"}
    assert kinds_map(["cow"]) == {19: "cow"}
    assert not set(ALL_KINDS) & set(VEHICLE_KINDS)          # one detector pass: the classes never overlap


def test_a_dog_is_counted_once_with_its_kind_and_picture(db, camera, tmp_path):
    watch = _watch(db, camera, tmp_path)
    frame = np.full((200, 300, 3), 120, np.uint8)
    _feed(watch, 3, 7, (100, 100, 160, 150), DOG, frame=frame)
    assert db.list_animals() == []                                    # fewer than min_frames: not yet
    t = _feed(watch, 10, 7, (100, 100, 160, 150), DOG, start=0.3, frame=frame)
    animals = db.list_animals()
    assert [(a["kind"], a["camera"], a["crop"]) for a in animals] == [("dog", "test-cam", f"animal_{animals[0]['id']}.jpg")]
    assert (tmp_path / animals[0]["crop"]).is_file() and (tmp_path / "blurred" / animals[0]["crop"]).is_file()
    watch.update(t + 10, T0 + timedelta(seconds=t + 10), frame, [])
    assert db.list_animals()[0]["ended_at"] is not None              # out of view: the visit ends


def test_kinds_not_chosen_and_vehicles_are_ignored(db, camera, tmp_path):
    watch = _watch(db, camera, tmp_path, kinds=("dog",))
    _feed(watch, 10, 1, (10, 10, 50, 40), CAT)
    _feed(watch, 10, 2, (10, 10, 50, 40), CAR)
    assert db.list_animals() == []


def test_an_animal_lying_still_is_resting_and_stays_one_record(db, camera, tmp_path):
    watch = _watch(db, camera, tmp_path)
    box = (100, 100, 180, 140)
    t = _feed(watch, 700, 3, box, DOG)                                 # 70 s without moving
    a = db.list_animals()[0]
    assert a["resting"] and a["resting_at"] is not None
    assert watch.shown(t - 0.05)[0][1].endswith("dog resting")
    watch.update(t + 10, T0 + timedelta(seconds=t + 10), None, [])     # hidden behind someone ...
    _feed(watch, 10, 99, box, DOG, start=t + 20)                      # ... and found again in the same place
    assert len(db.list_animals()) == 1


def test_color_and_kind_votes(db, camera, tmp_path):
    from labeler.vehicles import body_color
    watch = _watch(db, camera, tmp_path)
    frame = np.zeros((200, 300, 3), np.uint8)
    frame[:, :] = (30, 80, 160)                                       # a brownish-orange animal
    _feed(watch, 4, 5, (100, 100, 160, 150), CAT, frame=frame)
    _feed(watch, 60, 5, (100, 100, 160, 150), DOG, start=0.4, frame=frame)
    a = db.list_animals()[0]
    assert a["kind"] == "dog"                                         # the most confident kind so far wins
    assert a["color"] and a["color"] == body_color(frame[100:150, 100:160])
    night = _watch(db, camera, tmp_path)
    _feed(night, 60, 6, (100, 100, 160, 150), DOG, frame=np.zeros((200, 300, 3), np.uint8))
    assert db.list_animals()[0]["color"] is None                      # black and white (night): no color


def test_delete_and_retention(db, camera, tmp_path):
    watch = _watch(db, camera, tmp_path)
    _feed(watch, 10, 1, (10, 10, 50, 40), BIRD, frame=np.zeros((100, 100, 3), np.uint8))
    aid = db.list_animals()[0]["id"]
    db.add_animal_zone_events(aid, "test-cam", [{"zone_id": 1, "zone_name": "yard", "kind": "enter", "at": T0}])
    deleted, paths = db.delete_animals([aid, 999])
    assert deleted == [aid] and paths == [f"animal_{aid}.jpg"] and db.list_animals() == []
    old = db.create_animal(camera, 2, "cat", datetime.now() - timedelta(days=40))
    db.update_animal(old, crop_path=f"animal_{old}.jpg")
    db.create_animal(camera, 3, "cat", datetime.now())
    assert f"animal_{old}.jpg" in db.cleanup(30)
    assert [a["kind"] for a in db.list_animals()] == ["cat"]


def test_one_detector_pass_for_vehicles_and_animals(db, camera, tmp_path):
    """The camera asks the second detector for both kinds of classes once, and each watcher gets its own."""
    p = make_pipeline(db)
    got = {}

    class Watch:
        def __init__(self, name):
            self.name = name

        def update(self, t, wall, frame, dets, heads):
            got[self.name] = [d[2] for d in dets]

    class Boxes:
        import torch
        id = torch.tensor([1, 2, 3])
        xyxy = torch.tensor([[10.0, 10, 50, 40], [60, 10, 90, 40], [10, 50, 40, 90]])
        cls = torch.tensor([CAR, DOG, BIRD])
        conf = torch.tensor([0.9, 0.8, 0.7])

    class Model:
        def track(self, frame, classes, **kw):
            got["classes"] = classes
            return [type("R", (), {"boxes": Boxes()})()]

    from labeler.pipeline import LiveClock
    p.vehicles, p.animals = Watch("vehicles"), Watch("animals")
    p.animal_kinds = kinds_map(["dog", "cat", "bird"])
    p.vehicle_model, p.vehicle_tracker, p.vehicle_cfg = Model(), "bytetrack.yaml", DEFAULTS["vehicles"]
    p.det_cfg, p.ignore_zones, p.idle, p.clock = DEFAULTS["detector"], [], False, LiveClock()
    p._update_vehicles(np.zeros((100, 100, 3), np.uint8), 0.0)
    assert sorted(got["classes"]) == sorted([*VEHICLE_KINDS, DOG, CAT, BIRD])
    assert got["vehicles"] == [CAR] and got["animals"] == [DOG, BIRD]
    p.vehicles = None                                                 # vehicles off on this camera: animals only
    p._update_vehicles(np.zeros((100, 100, 3), np.uint8), 0.0)
    assert sorted(got["classes"]) == [BIRD, CAT, DOG]


# --- step 2: the list, the Ask box, Reports ---

def _app(db, tmp_path, address="127.0.0.1"):
    import copy
    from fastapi.testclient import TestClient
    from labeler.web import create_app
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, None), client=(address, 5000))


def _animals(db, camera):
    ids = {}
    for name, kind, color, ago in [("rex", "dog", "brown", 1), ("tom", "cat", "black", 2), ("crow", "bird", "black", 3),
                                   ("old", "dog", "white", 50)]:
        i = db.create_animal(camera, 1, kind, datetime(2026, 10, 14, 12) - timedelta(hours=ago))
        db.update_animal(i, color=color)
        ids[name] = i
    return ids


def test_animals_api_filters_and_delete(db, camera, tmp_path):
    ids = _animals(db, camera)
    client = _app(db, tmp_path)
    got = lambda **q: [a["id"] for a in client.get("/api/animals", params=q).json()]   # noqa: E731
    assert got(kind="dog") == [ids["rex"], ids["old"]]
    assert got(color="black") == [ids["tom"], ids["crow"]]
    assert got(day="2026-10-14") == [ids["rex"], ids["tom"], ids["crow"]]
    assert client.get("/api/animals", params={"day": "14/10"}).status_code == 400
    r = client.post("/api/animals/delete", json={"ids": [ids["crow"], 999]})
    assert r.json()["deleted"] == 1 and ids["crow"] not in got()
    assert "1 animals" in client.get("/api/audit").json()["entries"][0]["detail"]


def test_viewers_can_list_but_not_delete_animals(db, camera, tmp_path):
    from labeler import accounts
    _animals(db, camera)
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _app(db, tmp_path, address="192.168.1.20")
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert len(viewer.get("/api/animals").json()) == 4
    assert viewer.post("/api/animals/delete", json={"ids": [1]}).status_code == 403


def test_ask_box_words_for_animals():
    from labeler.agent import animal_query, vehicle_query
    assert animal_query("any dogs yesterday?") == {"kinds": ["dog"], "colors": []}
    assert animal_query("black cat at night") == {"kinds": ["cat"], "colors": ["black"]}
    assert animal_query("pets in the garden")["kinds"] == ["cat", "dog"]
    assert animal_query("animals today") == {"kinds": [], "colors": []}
    assert animal_query("man in a red shirt") is None
    assert vehicle_query("any dogs yesterday?") is None


def test_ask_box_answers_with_animals(db, camera):
    from labeler.agent import QueryAgent
    ids = _animals(db, camera)
    agent = QueryAgent.__new__(QueryAgent)
    agent.db, agent.embedder, agent.cameras = db, None, ["test-cam"]
    agent._ask_llm = lambda q, now: {}                                  # the model gives no time or camera
    res = agent.run("black cat", now=datetime(2026, 10, 14, 13))
    assert [a["id"] for a in res["animals"]] == [ids["tom"]] and res["people"] == []
    assert res["explanation"].startswith("Looked for: black cat")


def test_reports_count_animals(db, camera, tmp_path):
    _animals(db, camera)
    r = _app(db, tmp_path).get("/api/reports/summary", params={"start": "2026-10-14", "end": "2026-10-14"}).json()
    assert r["animals"]["total"] == 3
    assert {k["value"]: k["count"] for k in r["animals"]["kinds"]} == {"dog": 1, "cat": 1, "bird": 1}

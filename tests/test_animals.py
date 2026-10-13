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


# --- step 3: alert rules on animals ---

def test_animal_rules_match_and_describe():
    from labeler.alerts import describe, matches, rule_kind
    dog = {"animal": "dog", "zone": "yard"}
    assert matches([["animal", "any"]], dog) and matches([["animal", "dog"], ["zone", "yard"]], dog)
    assert not matches([["animal", "cat"]], dog)
    assert not matches([["animal", "any"]], {"upper_color": "red"})                       # a person is no animal
    assert describe([["animal", "dog"], ["armed", "away"]]) == "animal = dog, only when Away"
    assert describe([["animal", "any"]]) == "any animal"
    assert [rule_kind(c) for c in ([], [["vehicle", "any"]], [["animal", "cat"]])] == ["person", "vehicle", "animal"]


def test_animal_rules_fire_once_per_animal_and_never_for_people_or_vehicles(db, camera):
    from labeler.alerts import AlertEngine
    engine = AlertEngine(db)
    db.add_rule("dog in yard", [["animal", "dog"], ["zone", "yard"]])
    db.add_rule("any animal", [["animal", "any"]])
    db.add_rule("anyone", [])                                                            # a person rule
    db.add_rule("any vehicle", [["vehicle", "any"]])
    dog = db.create_animal(camera, 1, "dog", datetime.now())
    db.update_animal(dog, crop_path=f"animal_{dog}.jpg")
    assert [e["message"] for e in engine.check_animal(dog, "test-cam")] == ["any animal: any animal"]
    db.add_animal_zone_events(dog, "test-cam", [{"zone_id": 1, "zone_name": "yard", "kind": "enter", "at": T0}])
    assert [e["message"] for e in engine.check_animal(dog, "test-cam")] == ["dog in yard: animal = dog, in zone yard"]
    assert engine.check_animal(dog, "test-cam") == []                                     # once per rule and animal
    shown = db.list_alerts()[0]
    assert shown["animal_id"] == dog and shown["track_id"] is None and shown["crop"] == f"animal_{dog}.jpg"
    person = db.create_track(camera, 1, datetime.now())
    assert [e["message"] for e in engine.check(person, "test-cam")] == ["anyone: anyone"]
    _, paths = db.delete_animals([dog])
    assert any(p.startswith("clips/alert_") for p in paths)
    assert all(e["animal_id"] is None for e in db.list_alerts())                         # its alerts went with it


def test_animal_stay_rules(db, camera):
    from labeler.alerts import AlertEngine
    engine = AlertEngine(db)
    db.add_rule("cat in kitchen", [["animal", "cat"], ["stayed", "kitchen:300"]])
    db.add_rule("person in kitchen", [["stayed", "kitchen:30"]])
    assert engine.stay_limits("test-cam", animals=True) == {"kitchen": 300}
    assert engine.stay_limits("test-cam") == {"kitchen": 30}
    assert engine.stay_limits("test-cam", vehicles=True) == {}
    cat = db.create_animal(camera, 1, "cat", datetime.now())
    assert engine.check_animal(cat, "test-cam", stayed={"kitchen": 200}) == []
    assert len(engine.check_animal(cat, "test-cam", stayed={"kitchen": 301})) == 1


def test_pipeline_checks_animal_rules_and_saves_a_clip(db, camera):
    p = make_pipeline(db)
    asked = []

    class Alerts:
        def check_animal(self, animal_id, camera_name, stayed=None):
            asked.append(animal_id)
            return [{"id": 7}]

    class Clips:
        def request(self, event_id, before_s=None):
            asked.append(("clip", event_id))

    p.alerts, p.clips = Alerts(), Clips()
    p._check_animal_alerts(5)
    assert asked == [5, ("clip", 7)]


def test_rule_api_checks_animal_rules(db, tmp_path):
    db.add_zone("test-cam", "yard", "area", [(0, 0), (1, 0), (1, 1)])
    client = _app(db, tmp_path)
    post = lambda conds: client.post("/api/alerts/rules", json={"name": "r", "conditions": conds}).status_code  # noqa: E731
    assert post([["animal", "dog"], ["zone", "yard"], ["hours", "22-06"], ["armed", "away"]]) == 200
    assert post([["animal", "cat"], ["stayed", "yard:300"]]) == 200
    assert post([["animal", "unicorn"]]) == 400
    assert post([["animal", "dog"], ["vehicle", "car"]]) == 400                           # one or the other
    assert post([["animal", "dog"], ["upper_color", "red"]]) == 400                       # a person's label

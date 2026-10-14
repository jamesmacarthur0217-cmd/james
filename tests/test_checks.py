"""Checks ("was this right?", Phase 24): right / wrong / not sure about vehicles, animals, alerts and scene answers."""
import copy
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from labeler import accounts, checks
from labeler.config import DEFAULTS
from labeler.web import create_app


def _things(db, camera):
    """One of each kind, all with pictures, made just now."""
    now = datetime.now()
    car = db.create_vehicle(camera, 1, "car", now - timedelta(minutes=5))
    db.update_vehicle(car, color="white", crop_path=f"vehicle_{car}.jpg")
    dog = db.create_animal(camera, 2, "dog", now - timedelta(minutes=4))
    db.update_animal(dog, crop_path=f"animal_{dog}.jpg")
    fall_rule = db.add_rule("Fall", [["fallen", "yes"]])["id"]
    person = db.create_track(camera, 3, now - timedelta(minutes=3))
    fall = db.record_alert(fall_rule, person, "test-cam", "Fall: fell", repeat=True)
    q = db.add_scene_question("test-cam", "Is the gate open?")
    db.add_scene_answer(q["id"], "no", now - timedelta(minutes=2))
    db.update_scene_question(q["id"], last_answer="no", last_at=now - timedelta(minutes=2), last_picture="scene_1.jpg")
    return {"car": car, "dog": dog, "fall": fall["id"], "question": q["id"]}


def test_each_kind_is_offered_with_its_question(db, camera):
    ids = _things(db, camera)
    got = {k: checks.candidates(db, k) for k in checks.KINDS}
    assert got["vehicle"][0]["question"] == "Is this a white car?" and got["vehicle"][0]["id"] == ids["car"]
    assert got["animal"][0]["question"] == "Is this a dog?" and "not an animal" in got["animal"][0]["corrections"]
    assert "dog" not in got["animal"][0]["corrections"] and "car" not in got["vehicle"][0]["corrections"]
    alert = got["alert"][0]
    assert alert["id"] == ids["fall"] and alert["what"] == "fall" and alert["question"].startswith("Was this fall alert right?")
    scene = got["scene"][0]
    assert scene["question"] == "“Is the gate open?” was answered “no”. Is that right?" and scene["picture"] == "scene_1.jpg"


def test_kinds_take_turns_and_checked_or_skipped_items_go(db, camera):
    ids = _things(db, camera)
    first = checks.next_item(db)
    assert first["kind"] == "vehicle"                                        # nothing checked yet: in KINDS order
    db.save_check("vehicle", first["id"], first["what"], "test-cam", "yes", None, "james")
    assert checks.next_item(db)["kind"] == "animal"                          # the kind with fewest checks next
    assert checks.next_item(db, {("animal", ids["dog"])})["kind"] == "alert"  # skipped on the page
    for kind in ("animal", "alert", "scene"):
        item = checks.next_item(db)
        db.save_check(kind, item["id"], item["what"], "test-cam", "no", None, "james")
    assert checks.next_item(db) is None                                      # everything checked


def test_old_and_unpictured_items_are_not_offered(db, camera):
    old = db.create_vehicle(camera, 1, "car", datetime.now() - timedelta(days=31))
    db.update_vehicle(old, crop_path="vehicle_old.jpg")
    db.create_vehicle(camera, 2, "bus", datetime.now())                     # no picture yet
    assert checks.candidates(db, "vehicle") == []


def test_alert_kinds():
    assert checks.alert_what({"scene_id": 1}, []) == "scene"
    assert checks.alert_what({"fall": True}, [["fallen", "yes"]]) == "fall"
    assert checks.alert_what({"track_id": 1}, [["stayed", "gate:60"]]) == "loitering"
    assert checks.alert_what({"animal_id": 3}, [["animal", "dog"]]) == "animal"
    assert checks.alert_what({"vehicle_id": 3}, [["vehicle", "any"]]) == "vehicle"
    assert checks.alert_what({"track_id": 1}, [["upper_color", "red"]]) == "person"


def _app(db, tmp_path, address="127.0.0.1"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, None), client=(address, 5000))


def test_api_saves_replaces_and_checks_corrections(db, camera, tmp_path):
    ids = _things(db, camera)
    client = _app(db, tmp_path)
    nxt = client.get("/api/check/next").json()
    assert nxt["item"]["kind"] == "vehicle" and nxt["counts"] == {}
    assert client.get("/api/check/next", params={"skip": f"vehicle:{ids['car']},bad,x:y"}).json()["item"]["kind"] == "animal"
    r = client.post("/api/check", json={"kind": "animal", "item_id": ids["dog"], "right": "no", "correction": "cat"})
    assert r.status_code == 200 and r.json()["correction"] == "cat" and r.json()["what"] == "dog"
    assert client.post("/api/check", json={"kind": "animal", "item_id": ids["dog"], "right": "no",
                                           "correction": "unicorn"}).status_code == 400
    r = client.post("/api/check", json={"kind": "animal", "item_id": ids["dog"], "right": "yes", "correction": "cat"})
    assert r.json()["right"] == "yes" and r.json()["correction"] is None     # checked again: replaced; no fix if right
    assert [(c["kind"], c["right"]) for c in db.list_checks()] == [("animal", "yes")]
    assert client.post("/api/check", json={"kind": "vehicle", "item_id": 999, "right": "yes"}).status_code == 404
    assert client.post("/api/check", json={"kind": "person", "item_id": 1, "right": "yes"}).status_code == 400
    assert client.post("/api/check", json={"kind": "vehicle", "item_id": ids["car"], "right": "maybe"}).status_code == 400


def test_only_admins_check(db, camera, tmp_path):
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _app(db, tmp_path, address="192.168.1.20")
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/check/next").status_code == 403
    assert viewer.get("/check", follow_redirects=False).status_code == 303


# --- step 2: results ---

def _c(kind, what, camera, right, correction=None):
    return {"kind": kind, "what": what, "camera": camera, "right": right, "correction": correction}


def test_results_per_thing_and_camera_with_mistakes():
    rows = [_c("vehicle", "car", "gate", "yes")] * 9 + [_c("vehicle", "car", "gate", "no", "SUV"),
                                                         _c("vehicle", "bus", "yard", "yes"), _c("vehicle", "car", "gate", "unsure")]
    rows += [_c("animal", "dog", "gate", "no", "cat"), _c("animal", "dog", "gate", "no", "cat"), _c("animal", "cat", "gate", "yes")]
    rows += [_c("alert", "fall", "room", "yes"), _c("scene", "no", "gate", "no")]
    res = {(r["what"], r["camera"]): r for r in checks.results(rows)}
    v = res[("vehicles", None)]
    assert (v["right"], v["wrong"], v["unsure"], v["share"], v["few"]) == (10, 1, 1, 0.909, False)
    assert v["mistakes"] == [{"what": "car → SUV", "n": 1}]
    assert res[("vehicles", "yard")]["right"] == 1 and res[("vehicles", "yard")]["few"]
    a = res[("animals", None)]
    assert (a["right"], a["wrong"], a["mistakes"]) == (1, 2, [{"what": "dog → cat", "n": 2}])
    assert res[("fall alerts", None)]["share"] == 1.0 and res[("scene answers", None)]["share"] == 0.0
    order = [r["what"] for r in checks.results(rows) if r["camera"] is None]
    assert order == ["vehicles", "animals", "scene answers", "fall alerts"]
    assert checks.summary_line(rows) == ("vehicles 10 of 11 right (91%); animals 1 of 3 right (few checks yet); "
                                         "scene answers 0 of 1 right (few checks yet); fall alerts 1 of 1 right (few checks yet)")
    assert checks.summary_line([_c("animal", "dog", "gate", "unsure")]) is None   # only "not sure": nothing to say
    assert checks.results([]) == []


def test_results_on_the_page_health_and_reports(db, camera, tmp_path):
    from labeler.healthcheck import check_checks
    ids = _things(db, camera)
    client = _app(db, tmp_path)
    assert client.get("/api/check/results").json()["rows"] == [] and check_checks(db) is None
    client.post("/api/check", json={"kind": "vehicle", "item_id": ids["car"], "right": "yes"})
    client.post("/api/check", json={"kind": "animal", "item_id": ids["dog"], "right": "no", "correction": "cat"})
    rows = client.get("/api/check/results").json()["rows"]
    assert [(r["what"], r["camera"], r["right"], r["wrong"]) for r in rows if r["camera"] is None] == \
        [("vehicles", None, 1, 0), ("animals", None, 0, 1)]
    assert check_checks(db)["detail"] == ("vehicles 1 of 1 right (few checks yet); animals 0 of 1 right "
                                          "(few checks yet) (Check page)")
    today = datetime.now().strftime("%Y-%m-%d")
    report = client.get("/api/reports/summary", params={"start": today, "end": today}).json()
    assert [(r["what"], r["right"]) for r in report["checks"]] == [("vehicles", 1), ("animals", 0)]
    other_day = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    assert client.get("/api/reports/summary", params={"start": other_day, "end": other_day}).json()["checks"] == []
    assert client.get("/api/reports/summary", params={"start": today, "end": today, "camera": "other"}).json()["checks"] == []

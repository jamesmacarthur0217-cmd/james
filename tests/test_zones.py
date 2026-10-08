"""Zones and counting lines: geometry, counting without double counts, storage, API."""
import copy
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from labeler import accounts, zones
from labeler.config import DEFAULTS
from labeler.pipeline import TrackState
from labeler.web import create_app
from test_pipeline_logic import make_pipeline

SQUARE = {"id": 1, "name": "counter", "kind": "area", "points": [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]}
DOOR = {"id": 2, "name": "door", "kind": "line", "points": [(0.0, 0.5), (1.0, 0.5)]}   # "in" = downwards
T0 = datetime(2026, 10, 8, 12, 0, 0)


def _walk(counter, person, points, start=T0):
    """Feed one point per second; returns (kind, zone, dwell) of every event."""
    out = []
    for i, pt in enumerate(points):
        out += [(e["kind"], e["zone_name"], e["dwell_s"]) for e in counter.update(person, pt, start + timedelta(seconds=i))]
    return out


def test_geometry():
    assert zones.inside((0.4, 0.4), SQUARE["points"]) and not zones.inside((0.7, 0.4), SQUARE["points"])
    assert zones.side((0.5, 0.6), *DOOR["points"]) == 1 and zones.side((0.5, 0.4), *DOOR["points"]) == -1
    assert zones.side((0.5, 0.6), (0.2, 0.5), (0.4, 0.5)) == 0                   # beside the line's end
    assert zones.anchor((100, 100, 200, 300), (400, 400), "kiosk") == (0.375, 0.5)   # middle of the person
    assert zones.anchor((100, 100, 200, 300), (400, 400), "security") == (0.375, 0.75)  # their feet


def test_area_counts_once_despite_jitter_on_the_border():
    c, p = zones.ZoneCounter([SQUARE], min_frames=3), zones.PersonZones()
    path = [(0.1, 0.4)] * 2 + [(0.3, 0.4), (0.1, 0.4)] * 3 + [(0.3, 0.4)] * 5 + [(0.8, 0.4)] * 3
    events = _walk(c, p, path)
    assert [e[:2] for e in events] == [("enter", "counter"), ("exit", "counter")]
    assert events[1][2] == 5.0                       # in from second 8 to 12: confirmed at 10 and 15


def test_line_counts_crossings_both_ways():
    c, p = zones.ZoneCounter([DOOR], min_frames=2), zones.PersonZones()
    above, below = (0.5, 0.3), (0.5, 0.7)
    events = _walk(c, p, [above] * 3 + [below] * 3 + [(0.5, 0.48), below, (0.5, 0.52)] + [above] * 2)
    assert [e[0] for e in events] == ["in", "out"]   # the first side seen is where they came from, not a crossing


def test_leaving_the_view_closes_open_areas():
    c, p = zones.ZoneCounter([SQUARE], min_frames=1), zones.PersonZones()
    _walk(c, p, [(0.3, 0.3)])
    left = c.leave(p, T0 + timedelta(seconds=30))
    assert [(e["kind"], e["dwell_s"]) for e in left] == [("exit", 30.0)] and not p.inside


def test_zone_points_are_checked():
    with pytest.raises(ValueError):
        zones.check_points("area", [[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        zones.check_points("line", [[0, 0], [1.5, 1]])
    with pytest.raises(ValueError):
        zones.check_points("circle", [[0, 0], [1, 1]])
    assert zones.check_points("line", [[0, 0.5], [1, 0.5]]) == [(0.0, 0.5), (1.0, 0.5)]


def test_zone_events_wait_for_the_persons_record(db, camera):
    p = make_pipeline(db)
    p.zone_counter = zones.ZoneCounter([SQUARE], min_frames=1)
    st = TrackState(3, datetime.now(), 0.0)
    st.zone_log = p.zone_counter.update(st.zones, (0.3, 0.3), T0)
    p._flush_zone_events(st)                                   # no record yet: kept
    assert len(st.zone_log) == 1 and db.zone_counts(T0, T0 + timedelta(hours=1)) == []
    st.db_id = db.create_track(camera, 3, datetime.now())
    p._flush_zone_events(st)
    assert st.zone_log == [] and db.zones_entered(st.db_id) == ["counter"]


def test_zone_storage_counts_and_deleting(db, camera):
    z = db.add_zone("test-cam", "counter", "area", SQUARE["points"])
    with pytest.raises(ValueError):
        db.add_zone("test-cam", "counter", "line", DOOR["points"])          # same name twice
    db.add_zone("test-cam", "door", "line", DOOR["points"])
    assert [x["name"] for x in db.list_zones("test-cam")] == ["counter", "door"]
    a, b = db.create_track(camera, 1, T0), db.create_track(camera, 2, T0)
    for t, dwell in ((a, 10.0), (b, 20.0)):
        db.add_zone_events(t, "test-cam", [
            {"zone_id": z["id"], "zone_name": "counter", "kind": "enter", "at": T0},
            {"zone_id": z["id"], "zone_name": "counter", "kind": "exit", "at": T0, "dwell_s": dwell}])
    db.add_zone_events(a, "test-cam", [{"zone_id": 9, "zone_name": "door", "kind": "in", "at": T0}])
    counts = {c["zone"]: c for c in db.zone_counts(T0, T0 + timedelta(days=1))}
    assert counts["counter"]["enter"] == 2 and counts["counter"]["avg_dwell_s"] == 15.0
    assert counts["door"]["in"] == 1 and counts["door"]["out"] == 0
    assert db.delete_zone(z["id"])["name"] == "counter"
    assert db.zone_counts(T0, T0 + timedelta(days=1))[0]["enter"] == 2   # counts stay after the zone is gone
    db.delete_tracks([a, b])
    assert db.zone_counts(T0, T0 + timedelta(days=1)) == []              # deleting people removes their events


def test_alert_rule_can_ask_for_entering_a_zone(db, camera):
    from labeler.alerts import AlertEngine, describe
    engine = AlertEngine(db)
    db.add_rule("at the door", [["zone", "door"], ["upper_color", "red"]])
    t = db.create_track(camera, 1, T0)
    db.add_attributes(t, [("upper_color", "red", 1.0)], "vlm", "q", threshold=0.6)
    assert engine.check(t, "test-cam") == []                                    # red, but not at the door yet
    db.add_zone_events(t, "test-cam", [{"zone_id": 1, "zone_name": "door", "kind": "in", "at": T0}])
    assert [e["message"] for e in engine.check(t, "test-cam")] == ["at the door: in zone door, upper color = red"]
    assert describe([["zone", "door"]]) == "in zone door"


def test_rules_and_reports_know_the_zones(db, camera, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    admin = TestClient(create_app(cfg, db, None), client=("127.0.0.1", 5000))
    rule = {"name": "door", "conditions": [["zone", "door"]]}
    assert admin.post("/api/alerts/rules", json=rule).status_code == 400          # no zone "door" yet
    db.add_zone("test-cam", "door", "line", DOOR["points"])
    assert admin.post("/api/alerts/rules", json=rule).status_code == 200
    t = db.create_track(camera, 1, datetime.now())
    db.add_zone_events(t, "test-cam", [{"zone_id": 1, "zone_name": "door", "kind": "out", "at": datetime.now()}])
    zones_ = admin.get("/api/reports/summary").json()["zones"]
    assert [(z["zone"], z["in"], z["out"]) for z in zones_] == [("door", 0, 1)]


def test_zone_api_is_admin_only_and_audited(db, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path)
    app = create_app(cfg, db, None)
    admin = TestClient(app, client=("127.0.0.1", 5000))
    r = admin.post("/api/zones", json={"camera": "cam1", "name": "door", "kind": "line",
                                       "points": [[0, 0.5], [1, 0.5]]})
    assert r.status_code == 200 and r.json()["kind"] == "line"
    assert admin.post("/api/zones", json={"camera": "cam1", "name": "x", "kind": "area",
                                          "points": [[0, 0]]}).status_code == 400
    assert [z["name"] for z in admin.get("/api/zones").json()] == ["door"]
    assert admin.get("/api/zones/counts").json() == []
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/zones").status_code == 403
    assert viewer.delete(f"/api/zones/{r.json()['id']}").status_code == 403
    assert admin.delete(f"/api/zones/{r.json()['id']}").status_code == 200
    assert admin.delete(f"/api/zones/{r.json()['id']}").status_code == 404
    assert {"zone added", "zone deleted"} <= {e["action"] for e in db.list_audit(username="this PC")}

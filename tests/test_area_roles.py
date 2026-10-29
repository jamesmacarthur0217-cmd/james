"""Area roles (Phase 44 step 1): an area says what the place is (walkway, exit, cooking, ..., bed); "bed" is the old
"lying is normal here" and stays in step with zones.lying_ok; only areas have a role; the API offers the configured
roles, sets a role when drawing and changes it later (admins only, audited)."""
import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from labeler import accounts, zones
from labeler.config import DEFAULTS
from labeler.web import create_app

SQUARE = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]


def _client(db, tmp_path, roles=None):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cfg["zones"]["roles"] = roles
    return TestClient(create_app(cfg, db, None), client=("127.0.0.1", 5000))


def test_roles_in_the_database(db):
    exit_ = db.add_zone("cam1", "back door", "area", SQUARE, role="exit")
    assert exit_["role"] == "exit" and exit_["lying_ok"] is False
    old_way = db.add_zone("cam1", "sofa", "area", SQUARE, lying_ok=True)   # the page before roles
    assert old_way["role"] == zones.BED and old_way["lying_ok"] is True
    plain = db.add_zone("cam1", "till", "area", SQUARE)
    assert plain["role"] == "" and plain["lying_ok"] is False
    with pytest.raises(ValueError, match="unknown area role"):
        db.add_zone("cam1", "x", "area", SQUARE, role="kitchen sink")
    with pytest.raises(ValueError, match="a role for areas, not counting lines"):
        db.add_zone("cam1", "y", "line", [(0, 0.5), (1, 0.5)], role="exit")
    door = db.add_zone("cam1", "front door", "line", [(0, 0.5), (1, 0.5)], role="door")   # Phase 54: lines too
    assert door["role"] == "door" and door["lying_ok"] is False
    with pytest.raises(ValueError, match="a role for counting lines, not areas"):
        db.add_zone("cam1", "z", "area", SQUARE, role="door")


def test_changing_a_role_keeps_lying_ok_in_step(db):
    z = db.add_zone("cam1", "corner", "area", SQUARE)
    assert db.set_zone_role(z["id"], "bed")["lying_ok"] is True
    assert zones.lying_ok(db.list_zones("cam1"), (0.3, 0.3))   # falls are not checked there now
    again = db.set_zone_role(z["id"], "dining")
    assert again["role"] == "dining" and again["lying_ok"] is False
    assert not zones.lying_ok(db.list_zones("cam1"), (0.3, 0.3))
    assert db.set_zone_role(z["id"], "")["role"] == "" and db.set_zone_role(999, "exit") is None
    line = db.add_zone("cam1", "door", "line", [(0, 0.5), (1, 0.5)])
    with pytest.raises(ValueError):
        db.set_zone_role(line["id"], "exit")


def test_an_area_from_before_roles_reads_as_a_bed(db):
    z = db.add_zone("cam1", "bed", "area", SQUARE, lying_ok=True)
    with db.engine.begin() as conn:   # as stored by the code before Phase 44: lying_ok, no role
        conn.execute(text("UPDATE zones SET role = NULL WHERE id = :i"), {"i": z["id"]})
    assert db.list_zones("cam1")[0]["role"] == zones.BED


def test_api(db, tmp_path):
    client = _client(db, tmp_path)
    roles = client.get("/api/zones/roles").json()
    assert [r["role"] for r in roles] == [*zones.ROLES, *zones.LINE_ROLES]
    assert {r["kind"] for r in roles if r["role"] == "door"} == {"line"}
    narrowed = _client(db, tmp_path, roles=["exit", "walkway", "nonsense"]).get("/api/zones/roles").json()
    assert narrowed == [{"role": "exit", "words": zones.ROLES["exit"], "kind": "area"},
                        {"role": "walkway", "words": zones.ROLES["walkway"], "kind": "area"},
                        {"role": "door", "words": zones.LINE_ROLES["door"], "kind": "line"}]   # lines: always
    made = client.post("/api/zones", json={"camera": "cam1", "name": "kitchen", "kind": "area",
                                           "points": [list(p) for p in SQUARE], "role": "cooking"})
    assert made.status_code == 200 and made.json()["role"] == "cooking"
    bad = client.post("/api/zones", json={"camera": "cam1", "name": "k2", "kind": "area",
                                          "points": [list(p) for p in SQUARE], "role": "garage"})
    assert bad.status_code == 400 and "unknown area role" in bad.json()["detail"]
    zid = made.json()["id"]
    changed = client.patch(f"/api/zones/{zid}", json={"role": "bed"})
    assert changed.status_code == 200 and changed.json()["lying_ok"] is True
    assert client.patch(f"/api/zones/{zid}", json={"role": "garage"}).status_code == 400
    assert client.patch("/api/zones/999", json={"role": "exit"}).status_code == 404
    details = [e["detail"] for e in client.get("/api/audit").json()["entries"]]
    assert "area 'kitchen' on camera cam1: bed or sofa (lying is normal here)" in details
    assert "area 'kitchen' on camera cam1 (cooking area)" in details
    assert not accounts.allowed("viewer", "PATCH", f"/api/zones/{zid}")


def test_the_live_page():
    page = (Path(__file__).resolve().parent.parent / "labeler" / "static" / "index.html").read_text(encoding="utf-8")
    for part in ['id="zone-role"', "zone-role-edit", 'fetch("/api/zones/roles")', 'method: "PATCH"']:
        assert part in page, part
    assert "zone-lying" not in page

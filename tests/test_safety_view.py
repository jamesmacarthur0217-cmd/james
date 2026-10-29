"""The safety overview (Phase 52, labeler/safety_view.py, GET /api/safety "overview"): per safety camera which checks
run and why one does not, the areas' roles and what they mean, the floor check's empty picture, the latest checks from
memory; alerts per kind in the last 7 days with the Check page's answers."""
import copy
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from labeler import safety_view
from labeler.config import DEFAULTS
from labeler.incidents import ConfirmQueue, FrameMemory
from labeler.web import create_app

NOW = datetime(2026, 10, 25, 12, 0)
SQUARE = [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]


class Floor:
    """A floor check stand-in: learned, with its empty picture."""

    def state(self):
        return {"learning": None, "learned_at": "2026-10-25T09:30:00", "saved": "2026-10-25T11:00",
                "relearned": 2, "areas": ["hall"], "ready": 1.0, "checking": 0, "reported": 0}

    def picture(self):
        return np.full((60, 80, 3), 120, np.uint8)


def safety_camera(name="yard", zones=(), floor=None, guns=None, yoloe="yoloe.pt", fire=True, falls=True):
    """A safety camera's pipeline as the overview sees it."""
    return SimpleNamespace(name=name, safety_watch=object(), zone_counter=SimpleNamespace(zones=list(zones)),
                           falls=object() if falls else None, fire_watch=object() if fire else None,
                           _gun_path=guns, _yoloe_path=yoloe, floor_watch=floor)


def _kinds(view, camera=0):
    return {k["key"]: (k["state"], k["why"]) for k in view["cameras"][camera]["kinds"]}


def test_which_checks_run_and_why_not(db):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["safety"]["fights_enabled"] = False
    kiosk = SimpleNamespace(name="desk", safety_watch=None)
    yard = safety_camera(fire=False, falls=False)
    view = safety_view.overview(cfg, db, [kiosk, yard], now=NOW)
    assert [c["name"] for c in view["cameras"]] == ["yard"]          # the kiosk camera has no safety checks
    kinds = _kinds(view)
    assert list(kinds) == [k for k, _, _ in safety_view.KINDS]
    assert kinds["fights"] == ("off", "turned off on the Settings page")
    assert kinds["falls"] == ("off", "turned off for this camera (Settings page, Cameras)")
    assert kinds["fire"][0] == "off" and "no fire model" in kinds["fire"][1]
    assert kinds["weapons"] == ("partly", "guns not looked for (no model on this PC)")
    assert kinds["floor"] == ("partly", "no walkway or exit area yet (draw one on the Live page)")
    assert kinds["crowd"][0] == "partly" and kinds["equipment"][0] == "partly"
    assert kinds["down"] == ("on", "") and kinds["behaviour"] == ("on", "")
    assert view["enabled"] and view["queue"] is None and view["night_hours"] == "22-06"


def test_areas_say_what_their_role_means(db):
    cfg = copy.deepcopy(DEFAULTS)
    zones = [{"id": 1, "kind": "area", "name": "kitchen", "role": "cooking", "points": SQUARE},
             {"id": 2, "kind": "area", "name": "hall", "role": "walkway", "max_people": 4, "points": SQUARE},
             {"id": 3, "kind": "area", "name": "sofa", "role": "bed", "points": SQUARE},
             {"id": 4, "kind": "area", "name": "count", "role": "", "points": SQUARE},
             {"id": 5, "kind": "line", "name": "door line", "points": SQUARE[:2]}]
    view = safety_view.overview(cfg, db, [safety_camera(zones=zones, floor=Floor(), guns="guns.pt")], now=NOW)
    cam = view["cameras"][0]
    areas = {a["name"]: a for a in cam["areas"]}
    assert list(areas) == ["kitchen", "hall", "sofa", "count"]            # areas only, not counting lines
    assert areas["kitchen"]["role_words"] == "cooking area"
    assert "a knife or scissors held here is normal" in areas["kitchen"]["meaning"]
    assert "a child here 3 s is checked" in areas["kitchen"]["meaning"]
    assert areas["hall"]["meaning"].startswith("floor check: something new on the floor 30 s is checked")
    assert areas["hall"]["max_people"] == 4 and areas["sofa"]["meaning"].startswith("lying here is normal")
    assert areas["kitchen"]["meaning"].endswith("looked at for unsafe equipment every 10 min")
    assert areas["count"]["meaning"] == ""                                   # no role: a plain counting area
    kinds = _kinds(view)
    assert kinds["floor"] == kinds["crowd"] == kinds["equipment"] == kinds["weapons"] == ("on", "")
    assert cam["floor"] == {"learned": True, "learning": None, "learned_at": "2026-10-25T09:30:00",
                            "saved": "2026-10-25T11:00", "relearned": 2, "picture": True, "ready": 1.0}
    cfg["safety"]["equipment_enabled"], cfg["safety"]["floor_hold_s"] = False, 45
    assert safety_view.role_meaning("walkway", cfg) == "floor check: something new on the floor 45 s is checked"


def test_latest_checks_from_memory_per_camera(db):
    cfg = copy.deepcopy(DEFAULTS)
    q = ConfirmQueue(cfg, None, {"yard": FrameMemory(), "gate": FrameMemory()})
    for cam, answer in (("yard", "no"), ("gate", "yes"), ("yard", "unsure")):
        q.recent.append({"camera": cam, "category": "fire", "level": "critical", "reason": "seen 3 of 5 looks",
                         "area": None, "at": "2026-10-25T11:00:00", "answer": answer, "what": "", "seconds": 1.4})
    q.counts["sent"] = 3
    view = safety_view.overview(cfg, db, [safety_camera("yard"), safety_camera("gate")], q, now=NOW)
    assert [c["answer"] for c in view["cameras"][0]["checks"]] == ["unsure", "no"]    # newest first
    assert [c["answer"] for c in view["cameras"][1]["checks"]] == ["yes"]
    assert view["queue"]["sent"] == 3 and view["kept"] == 50


def test_alerts_per_kind_with_the_checks(db):
    cfg = copy.deepcopy(DEFAULTS)
    rule = db.add_rule("Fire or smoke", [["incident", "fire"]])
    made = [db.record_incident_alert(rule["id"], cam, "fire or smoke", "fire", "critical", "yes", None)
            for cam in ("yard", "yard", "gate")]
    db.save_check("alert", made[0]["id"], "safety: fire or smoke", "yard", "yes", None, "admin")
    db.save_check("alert", made[1]["id"], "safety: fire or smoke", "yard", "no", None, "admin")
    view = safety_view.overview(cfg, db, [], now=datetime.now())
    assert view["cameras"] == [] and view["days"] == 7
    fire = next(a for a in view["alerts"] if a["category"] == "fire")
    assert fire["alerts"] == 3 and fire["cameras"] == {"yard": 2, "gate": 1} and fire["words"] == "fire or smoke"
    assert fire["checked"] == {"right": 1, "wrong": 1, "unsure": 0, "share": 0.5, "few": True}
    assert [a["category"] for a in view["alerts"]] == ["fire"]          # kinds with no alert and no check: left out


def test_the_api_and_the_empty_picture(db, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    client = TestClient(create_app(cfg, db, None), client=("127.0.0.1", 5000))
    got = client.get("/api/safety").json()
    assert got["on"] is False and got["overview"]["cameras"] == [] and len(got["kinds"]) == 12   # the Alerts page's
    assert client.get("/api/safety/empty/yard").status_code == 404
    from labeler import accounts
    assert not accounts.allowed("viewer", "GET", "/api/safety/empty/yard")    # admins only
    assert not accounts.allowed("viewer", "GET", "/api/safety")


def test_the_page_and_its_links(db, tmp_path):
    from pathlib import Path
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    client = TestClient(create_app(cfg, db, None), client=("127.0.0.1", 5000))
    page = client.get("/safety")
    assert page.status_code == 200 and 'fetch("/api/safety")' in page.text and "/api/safety/empty/" in page.text
    from labeler import accounts
    assert not accounts.allowed("viewer", "GET", "/safety")                  # a viewer is sent to the Live page
    static = Path(__file__).resolve().parent.parent / "labeler" / "static"
    assert 'href="/safety"' in (static / "alerts.html").read_text(encoding="utf-8")
    health = (static / "health.html").read_text(encoding="utf-8")
    assert 'href="/safety"' in health and 'class="admin-only"' in health and '"safety checks"' in health


def test_no_page_declares_a_name_the_browser_keeps():
    """`function top()` stopped the safety page with "Identifier 'top' has already been declared" (window.top cannot
    be redeclared); node --check (tests/test_pages.py) does not see it. The same for location, window, document."""
    import re
    from pathlib import Path
    static = Path(__file__).resolve().parent.parent / "labeler" / "static"
    bad = re.compile(r"^(?:async\s+)?(?:function\s+|(?:let|const|var|class)\s+)(top|location|window|document)\b", re.M)
    for page in [*static.glob("*.html"), *static.glob("*.js")]:
        assert not bad.search(page.read_text(encoding="utf-8")), page.name


def test_a_door_line_is_listed_with_its_meaning(db):
    """Phase 54: counting lines with the role "door" show on the safety page (a plain counting line does not)."""
    zones = [{"id": 1, "kind": "line", "name": "front door", "role": "door", "points": [(0.5, 0), (0.5, 1)]},
             {"id": 2, "kind": "line", "name": "count", "role": "", "points": [(0.2, 0), (0.2, 1)]}]
    view = safety_view.overview(copy.deepcopy(DEFAULTS), db, [safety_camera(zones=zones)], now=NOW)
    [door] = view["cameras"][0]["areas"]
    assert door["name"] == "front door" and door["role_words"].startswith("door")
    assert door["meaning"].startswith("two people through it the same way within 2 s") and "equipment" not in door["meaning"]

"""Audit log: who did what, from where; admin-only page; retention."""
import copy
import sys
from datetime import datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select, update

from labeler import accounts
from labeler.config import DEFAULTS
from labeler.db import Attribute, AuditEntry, Review
from labeler.web import create_app

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import users as users_tool  # noqa: E402

PW = "correct horse battery"


def _client(db, tmp_path, address="192.168.1.20"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, None), client=(address, 5000))


def _logged_in(db, tmp_path, name, role):
    db.add_user(name, accounts.hash_password(PW), role)
    c = _client(db, tmp_path)
    assert c.post("/api/login", json={"username": name, "password": PW}).status_code == 200
    return c


def _actions(db):
    return [(e["username"], e["action"]) for e in reversed(db.list_audit())]


def test_logins_are_recorded(db, tmp_path):
    db.add_user("ann", accounts.hash_password(PW), "admin")
    c = _client(db, tmp_path)
    c.post("/api/login", json={"username": "ann", "password": "wrong password"})
    c.post("/api/login", json={"username": "mallory", "password": "guess guess"})
    c.post("/api/login", json={"username": "ann", "password": PW})
    c.post("/api/logout")
    assert _actions(db) == [("ann", "login failed"), ("mallory", "login failed"), ("ann", "login"),
                            ("ann", "logout")]
    entry = db.list_audit(action="login")[0]
    assert entry["address"] == "192.168.1.20" and entry["role"] == "admin"
    assert db.list_audit(username="mallory")[0]["detail"] == "no such user"


def test_admin_actions_are_recorded_with_the_user(db, camera, tmp_path):
    c = _logged_in(db, tmp_path, "ann", "admin")
    t = db.create_track(camera, 1, datetime.now())
    db.add_attributes(t, [("upper_color", "red", 0.4)], "vlm", "q", threshold=0.6)
    with db.Session() as s:
        attr_id = s.scalar(select(Attribute.id))
    assert c.post(f"/api/review/{attr_id}", json={"value": "blue"}).status_code == 200
    with db.Session() as s:
        assert s.scalar(select(Review.reviewer)) == "ann"         # the logged-in user, not a fixed name
    assert c.post(f"/api/answer-key/{t}", json={"labels": {"upper_color": "blue"}}).status_code == 200
    assert c.post("/api/alerts/rules", json={"name": "red top", "conditions": [["upper_color", "red"]]}).status_code == 200
    assert c.get("/api/reports/export.csv").status_code == 200
    assert c.post("/api/tracks/delete", json={"ids": [t]}).status_code == 200
    got = {e["action"]: e["detail"] for e in db.list_audit(username="ann")}
    assert got["review"] == f"label #{attr_id} = blue"
    assert got["answer key"] == f"person #{t}"
    assert got["alert rule added"] == "red top: upper_color = red"
    assert got["export CSV"].startswith("1 people")
    assert got["delete"] == f"1 people (#{t})"


def test_this_pc_is_recorded_as_this_pc(db, tmp_path):
    local = _client(db, tmp_path, address="127.0.0.1")
    local.post("/api/tracks/delete", json={"everything": True})              # nothing to delete: not logged
    local.post("/api/alerts/rules", json={"name": "x", "conditions": []})
    assert _actions(db) == [("this PC", "alert rule added")]


def test_audit_log_is_admin_only(db, tmp_path):
    viewer = _logged_in(db, tmp_path, "vic", "viewer")
    assert viewer.get("/api/audit").status_code == 403
    assert viewer.get("/audit", follow_redirects=False).headers["location"] == "/"
    admin = _logged_in(db, tmp_path, "ann", "admin")
    data = admin.get("/api/audit").json()
    assert {"vic", "ann"} <= set(data["users"]) and "login" in data["actions"]
    assert admin.get("/audit").status_code == 200


def test_audit_paging_and_filters(db):
    for i in range(5):
        db.add_audit("ann" if i % 2 else "bob", "admin", "x", f"act{i}")
    first = db.list_audit(limit=2)
    assert [e["action"] for e in first] == ["act4", "act3"]
    assert [e["action"] for e in db.list_audit(limit=2, before_id=first[-1]["id"])] == ["act2", "act1"]
    assert [e["action"] for e in db.list_audit(username="ann")] == ["act3", "act1"]


def test_old_audit_entries_are_removed(db):
    db.add_audit("ann", "admin", "x", "old")
    db.add_audit("ann", "admin", "x", "new")
    with db.Session.begin() as s:
        s.execute(update(AuditEntry).where(AuditEntry.action == "old")
                  .values(created_at=datetime.now() - timedelta(days=400)))
    assert db.cleanup_audit(0) == 0                                           # 0 = keep forever
    assert db.cleanup_audit(365) == 1
    assert [e["action"] for e in db.list_audit()] == ["new"]


def test_account_changes_from_the_users_tool_are_recorded(db):
    users_tool.run(db, users_tool.parser().parse_args(["add", "vic"]), ask=lambda: PW)
    users_tool.run(db, users_tool.parser().parse_args(["list"]))
    assert [(e["username"], e["action"], e["detail"]) for e in db.list_audit()] == [
        ("this PC", "account change", "added vic (viewer)")]

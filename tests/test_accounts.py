"""Login accounts: password hashing, logging in, roles, guessing limits, the users tool."""
import argparse
import copy
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from labeler import accounts
from labeler.config import DEFAULTS
from labeler.web import create_app, ensure_first_admin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import users as users_tool  # noqa: E402

PW = "correct horse battery"


def _client(db, tmp_path, address="192.168.1.20"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    return TestClient(create_app(cfg, db, None), client=(address, 5000))


def _login(client, name, pw=PW):
    return client.post("/api/login", json={"username": name, "password": pw})


def test_password_hash_checks_the_password():
    h = accounts.hash_password(PW)
    assert PW not in h and h != accounts.hash_password(PW)              # salted
    assert accounts.verify_password(PW, h)
    assert not accounts.verify_password("wrong password", h)
    assert not accounts.verify_password(PW, "garbage")


def test_other_devices_must_log_in(db, tmp_path):
    db.add_user("ann", accounts.hash_password(PW), "admin")
    remote = _client(db, tmp_path)
    assert remote.get("/api/tracks").status_code == 401
    page = remote.get("/review", follow_redirects=False)
    assert page.status_code == 303 and page.headers["location"] == "/login"
    assert remote.get("/login").status_code == 200
    assert _login(remote, "ann", "wrong password").status_code == 401
    assert _login(remote, "nobody").status_code == 401
    assert _login(remote, "ann").json() == {"username": "ann", "role": "admin"}
    assert remote.get("/api/tracks").status_code == 200
    assert remote.get("/api/me").json() == {"username": "ann", "role": "admin", "local": False}
    remote.post("/api/logout")
    assert remote.get("/api/tracks").status_code == 401


def test_this_pc_needs_no_login(db, tmp_path):
    local = _client(db, tmp_path, address="127.0.0.1")
    assert local.get("/api/tracks").status_code == 200
    assert local.post("/api/tracks/delete", json={"ids": []}).status_code == 200
    assert local.get("/api/me").json()["local"] is True


def test_requests_forwarded_by_a_tool_on_this_pc_must_log_in(db, tmp_path):
    """Cloudflare Tunnel, Tailscale Serve/Funnel or a reverse proxy on this PC: the request comes from
    127.0.0.1 but the person is elsewhere."""
    local = _client(db, tmp_path, address="127.0.0.1")
    for header in [{"X-Forwarded-For": "203.0.113.9"}, {"Forwarded": "for=203.0.113.9"},
                   {"CF-Connecting-IP": "203.0.113.9"}, {"Tailscale-User-Login": "someone@example.com"}]:
        assert local.get("/api/tracks", headers=header).status_code == 401, header
        assert local.post("/api/tracks/delete", json={"everything": True}, headers=header).status_code == 401
    assert local.get("/api/tracks").status_code == 200                 # no forwarding: really this PC


def test_viewer_gets_live_view_and_search_only(db, tmp_path):
    db.add_user("vic", accounts.hash_password(PW), "viewer")
    viewer = _client(db, tmp_path)
    _login(viewer, "vic")
    for path in ["/", "/api/live", "/api/status", "/api/tracks", "/api/cameras", "/api/alerts/events"]:
        assert viewer.get(path).status_code == 200, path
    assert viewer.get("/api/search", params={"q": "red"}).status_code == 503         # allowed; no embedder here
    assert viewer.get("/api/tracks/1/similar").status_code == 503
    assert viewer.post("/api/ask", json={"question": "who?"}).status_code == 503
    assert viewer.post("/api/tracks/delete", json={"everything": True}).status_code == 403
    for path in ["/api/reports/summary", "/api/reports/export.csv", "/api/review", "/api/alerts/rules",
                 "/api/answer-key/score"]:
        assert viewer.get(path).status_code == 403, path
    assert viewer.post("/api/shutdown").status_code == 403
    assert viewer.post("/api/alerts/rules", json={"name": "x"}).status_code == 403
    page = viewer.get("/reports", follow_redirects=False)
    assert page.status_code == 303 and page.headers["location"] == "/"


def test_viewer_rules_are_an_allowlist():
    assert accounts.allowed("viewer", "GET", "/api/tracks/5/similar")
    assert not accounts.allowed("viewer", "GET", "/api/tracks/5")            # a future per-person page
    assert not accounts.allowed("viewer", "GET", "/api/something-new")
    assert not accounts.allowed("viewer", "DELETE", "/api/tracks")
    assert not accounts.allowed("nobody", "GET", "/")
    assert accounts.allowed("admin", "POST", "/api/anything")


def test_repeated_wrong_passwords_lock_out_that_device(db, tmp_path):
    db.add_user("ann", accounts.hash_password(PW), "admin")
    remote = _client(db, tmp_path)
    for _ in range(5):
        assert _login(remote, "ann", "wrong password").status_code == 401
    assert _login(remote, "ann").status_code == 429                  # even the right password, for a while
    assert _login(_client(db, tmp_path, address="192.168.1.21"), "ann").status_code == 200   # other device


def test_limiter_forgets_old_failures():
    lim = accounts.LoginLimiter(max_failures=2, window_s=60)
    lim.failed("a", now=0)
    lim.failed("a", now=10)
    assert lim.blocked_for("a", now=20) == pytest.approx(40)
    assert lim.blocked_for("a", now=61) == 0


def test_password_or_role_change_logs_the_user_out(db, tmp_path):
    db.add_user("ann", accounts.hash_password(PW), "admin")
    remote = _client(db, tmp_path)
    _login(remote, "ann")
    db.update_user("ann", role="viewer")
    assert remote.get("/api/tracks").status_code == 401
    _login(remote, "ann")
    assert remote.get("/api/me").json()["role"] == "viewer"


def test_expired_session_is_refused(db):
    uid = db.add_user("ann", accounts.hash_password(PW), "admin")
    assert db.session_user(db.create_session(uid, hours=1))["username"] == "ann"
    assert db.session_user(db.create_session(uid, hours=-1)) is None
    assert db.session_user("made-up-token") is None


def test_old_shared_password_becomes_the_first_admin(db):
    ensure_first_admin(db, {"username": "admin", "password": ""})
    assert db.count_users() == 0
    ensure_first_admin(db, {"username": "admin", "password": PW})
    ensure_first_admin(db, {"username": "admin", "password": PW})      # only once
    assert db.list_users()[0]["role"] == "admin" and db.count_users() == 1
    assert accounts.verify_password(PW, db.get_user("admin")["password_hash"])


def test_users_tool(db):
    def cmd(*argv):
        return users_tool.run(db, users_tool.parser().parse_args(list(argv)), ask=lambda: PW)

    assert cmd("add", "ann", "--role", "admin") == "added ann (admin)"
    assert cmd("add", "vic") == "added vic (viewer)"
    with pytest.raises(ValueError):
        cmd("add", "vic")
    assert "vic" in cmd("list") and "viewer" in cmd("list")
    cmd("role", "vic", "admin")
    assert db.get_user("vic")["role"] == "admin"
    cmd("remove", "vic")
    assert db.get_user("vic") is None
    with pytest.raises(ValueError):
        users_tool.run(db, argparse.Namespace(command="password", name="nobody"), ask=lambda: PW)


def test_short_passwords_are_refused():
    with pytest.raises(ValueError):
        accounts.check_new_password("short")

"""Update button (Phase 35): labeler/updater.py on scratch git repositories (a "GitHub", this copy, a developer)."""
import subprocess

import pytest

from labeler.updater import Updater, find_git

GIT = find_git()
pytestmark = pytest.mark.skipif(GIT is None, reason="git is not installed")


def git(cwd, *args):
    r = subprocess.run([GIT, *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def commit(repo, name, text, message):
    (repo / name).write_text(text, encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repos(tmp_path):
    hub = tmp_path / "hub.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(hub))
    dev = tmp_path / "dev"
    git(tmp_path, "clone", "-q", str(hub), str(dev))
    for repo in (dev,):
        git(repo, "config", "user.email", "dev@example.com")
        git(repo, "config", "user.name", "Dev")
    git(dev, "checkout", "-q", "-b", "main")
    commit(dev, "app.py", "v1\n", "First version")
    commit(dev, "requirements.txt", "numpy\n", "Requirements")
    git(dev, "push", "-q", "-u", "origin", "main")
    app = tmp_path / "app"
    git(tmp_path, "clone", "-q", str(hub), str(app))
    git(app, "config", "user.email", "app@example.com")
    git(app, "config", "user.name", "App")
    return hub, dev, app


def test_up_to_date_then_new_changes_listed(repos):
    _, dev, app = repos
    up = Updater({}, root=app, git=GIT)
    s = up.check()
    assert s["available"] and s["behind"] == 0 and s["error"] is None and s["version"].endswith("Requirements")
    assert up.problem(s) == "already up to date"
    commit(dev, "app.py", "v2\n", "Help sign alert")
    commit(dev, "app.py", "v3\n", "Time-lapse of a day")
    git(dev, "push", "-q")
    s = up.check()
    assert s["behind"] == 2 and s["changes"] == ["Time-lapse of a day", "Help sign alert"] and up.problem(s) is None


def test_update_fast_forwards_and_installs_packages_only_when_they_changed(repos):
    _, dev, app = repos
    pip_runs, backups = [], []
    up = Updater({}, root=app, git=GIT, pip=lambda: pip_runs.append(1) or (True, ""),
                 backup=lambda: backups.append(1) or True)
    commit(dev, "app.py", "v2\n", "New feature")
    git(dev, "push", "-q")
    r = up.update()
    assert r["ok"] and r["changes"] == ["New feature"] and r["requirements"] is False
    assert (app / "app.py").read_text() == "v2\n" and pip_runs == [] and backups == [1]
    commit(dev, "requirements.txt", "numpy\nhttpx2\n", "New package")
    git(dev, "push", "-q")
    r = up.update()
    assert r["ok"] and r["requirements"] is True and pip_runs == [1]
    assert up.state["behind"] == 0


def test_a_failed_package_install_puts_the_old_version_back(repos):
    _, dev, app = repos
    up = Updater({}, root=app, git=GIT, pip=lambda: (False, "no such package"))
    old = git(app, "rev-parse", "HEAD")
    commit(dev, "requirements.txt", "numpy\nnot-a-package\n", "Bad package")
    git(dev, "push", "-q")
    r = up.update()
    assert not r["ok"] and "old version was put back" in r["message"]
    assert git(app, "rev-parse", "HEAD") == old and (app / "requirements.txt").read_text() == "numpy\n"


def test_refuses_when_changed_by_hand_or_with_own_commits_or_backup_failed(repos):
    _, dev, app = repos
    commit(dev, "app.py", "v2\n", "New feature")
    git(dev, "push", "-q")
    (app / "app.py").write_text("my own edit\n")
    up = Updater({}, root=app, git=GIT)
    r = up.update()
    assert not r["ok"] and "changed by hand (app.py)" in r["message"]
    assert (app / "app.py").read_text() == "my own edit\n"                      # nothing touched
    git(app, "checkout", "-q", "--", "app.py")
    commit(app, "notes.txt", "x\n", "A change made here")
    r = up.update()
    assert not r["ok"] and "changes of its own" in r["message"]
    git(app, "reset", "-q", "--hard", "HEAD~1")
    r = Updater({}, root=app, git=GIT, backup=lambda: False).update()
    assert not r["ok"] and "backup failed" in r["message"] and (app / "app.py").read_text() == "v1\n"


def test_not_a_git_copy(tmp_path):
    s = Updater({}, root=tmp_path, git=GIT).check()
    assert not s["available"] and s["error"]
    no_git = Updater({}, root=tmp_path, git="")
    no_git.git_exe = None
    assert no_git.check()["error"] == "Git is not installed on this PC"


def test_api_and_health_line(repos, db, tmp_path, monkeypatch):
    import copy
    from fastapi.testclient import TestClient
    from labeler import accounts, settings as settings_mod
    from labeler.config import DEFAULTS
    from labeler.healthcheck import check_updates
    from labeler.web import create_app
    _, dev, app_repo = repos
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    app = create_app(cfg, db, None)
    app.state.updater = up = Updater(cfg, root=app_repo, git=GIT)
    restarts = []
    app.state.request_restart = lambda: restarts.append(1)
    monkeypatch.setattr(settings_mod, "restart_available", lambda: True)
    admin = TestClient(app, client=("127.0.0.1", 5000))
    assert check_updates(up)["detail"].startswith("not checked yet")
    s = admin.get("/api/update").json()
    assert s["behind"] == 0 and s["problem"] == "already up to date" and s["can_restart"] is True
    assert check_updates(up)["detail"].startswith("up to date: ")
    assert admin.post("/api/update").status_code == 409
    commit(dev, "app.py", "v2\n", "New feature")
    git(dev, "push", "-q")
    s = admin.get("/api/update?check=true").json()
    assert s["behind"] == 1 and s["changes"] == ["New feature"] and s["problem"] is None
    assert check_updates(up)["detail"].startswith("1 new change on GitHub")
    r = admin.post("/api/update").json()
    assert r["ok"] and r["restarting"] is True
    import time
    time.sleep(1.5)
    assert restarts == [1] and (app_repo / "app.py").read_text() == "v2\n"
    assert db.list_audit(1)[0]["action"] == "update app" and "New feature" in db.list_audit(1)[0]["detail"]
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = TestClient(app, client=("192.168.1.20", 5000))
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/update").status_code == 403 and viewer.post("/api/update").status_code == 403
    (app_repo / "app.py").write_text("edited\n")
    up.check(fetch=False)
    assert check_updates(up)["level"] == "warning"


def test_settings_page_has_the_updates_box():
    from pathlib import Path
    page = (Path(__file__).resolve().parent.parent / "labeler" / "static" / "settings.html").read_text(encoding="utf-8")
    for needle in ('id="upd-now"', 'id="upd-check"', 'fetch("/api/update", { method: "POST" })', "waitForRestart("):
        assert needle in page, needle

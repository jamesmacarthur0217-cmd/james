"""Daily backups while running: when they are due, the backup tool (with a fake docker), and the health line."""
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from labeler import backups, healthcheck

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import backup_db  # noqa: E402

NIGHT = datetime(2026, 10, 10, 3, 0)


class FakeDocker:
    """Answers `docker exec ...` like the database container would; records the commands."""

    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def __call__(self, cmd, **kw):
        tool = cmd[cmd.index(backup_db.CONTAINER) + 1]   # docker exec [-i] CONTAINER <tool> ...
        self.calls.append(tool)
        code = 1 if tool in self.fail else 0
        if tool == "pg_dump" and not code:
            kw["stdout"].write(b"PGDMP fake dump")
        out = "27\n" if tool == "psql" else ""
        if kw.get("text"):
            return subprocess.CompletedProcess(cmd, code, out, "boom" if code else "")
        return subprocess.CompletedProcess(cmd, code, out.encode(), b"boom" if code else b"")


@pytest.fixture(autouse=True)
def fake_docker_path(monkeypatch):
    monkeypatch.setattr(backup_db, "docker", lambda: "docker")


def _dump(folder: Path, name: str, age_h: float) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(b"x")
    t = time.time() - age_h * 3600
    os.utime(p, (t, t))
    return p


def test_due_from_the_hour_on_when_the_newest_is_old_enough():
    old = (NIGHT - timedelta(hours=21)).timestamp()
    assert not backups.due(NIGHT.replace(hour=1), 2, old)                    # before 2:00
    assert backups.due(NIGHT, 2, None)                                       # no backup yet
    assert backups.due(NIGHT, 2, old)
    assert not backups.due(NIGHT, 2, (NIGHT - timedelta(hours=5)).timestamp())   # e.g. stop.bat made one
    assert backups.due(NIGHT.replace(hour=15), 2, old)                       # PC was off at night: catch up


def test_restore_check_weekly():
    assert backups.verify_due(NIGHT, None)
    assert not backups.verify_due(NIGHT, {"at": (NIGHT - timedelta(days=3)).isoformat()})
    assert backups.verify_due(NIGHT, {"at": (NIGHT - timedelta(days=8)).isoformat()})


def test_tick_runs_the_tool_only_for_postgres_and_when_due(tmp_path, monkeypatch, db):
    monkeypatch.setattr(backups, "folder", lambda: tmp_path)
    cfg = {"backup": {"enabled": True, "hour": 2, "keep": 5, "copy_to": "E:/copies"}}
    commands = []

    def runner(cmd, **kw):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "Backup saved: x.dump (1.0 MB)\n", "")

    assert backups.tick(cfg, db, NIGHT, runner) is None                      # SQLite: no backups
    monkeypatch.setattr(db, "is_postgres", True)
    assert backups.tick(cfg, db, NIGHT, runner) is True
    assert commands[0][-5:] == ["--keep", "5", "--copy-to", "E:/copies", "--verify"]
    _dump(tmp_path, "visionlabeler-20261010-020000.dump", 1)
    assert backups.tick(cfg, db, NIGHT, runner) is None                      # made an hour ago
    cfg["backup"]["enabled"] = False
    assert backups.tick(cfg, db, NIGHT + timedelta(days=2), runner) is None


def test_a_failed_backup_is_logged_as_an_error(caplog):
    cfg = {"backup": {"enabled": True, "hour": 2, "keep": 7, "copy_to": ""}}
    fail = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "Backup FAILED: no container")  # noqa: E731
    with caplog.at_level(logging.INFO):
        assert backups.run_backup(cfg, False, fail) is False
    assert any(r.levelno == logging.ERROR and "no container" in r.getMessage() for r in caplog.records)


def test_backup_tool_saves_prunes_copies_and_verifies(tmp_path):
    folder, copies = tmp_path / "backups", tmp_path / "usb"
    copies.mkdir()
    for i in range(3):
        _dump(folder, f"visionlabeler-2026100{i + 1}-020000.dump", 72 - i * 24)
        _dump(copies, f"visionlabeler-2026100{i + 1}-020000.dump", 72 - i * 24)
    run = FakeDocker()
    assert backup_db.main(["--keep", "2", "--copy-to", str(copies), "--verify"], folder, run) == 0
    assert len(backup_db.backups(folder)) == 2 and len(backup_db.backups(copies)) == 2
    newest = backup_db.backups(folder)[-1]
    assert newest.read_bytes() == b"PGDMP fake dump" and (copies / newest.name).is_file()
    verify = json.loads((folder / backup_db.VERIFY_FILE).read_text())
    assert verify["ok"] and verify["people"] == 27 and verify["backup"] == newest.name
    assert run.calls.count("dropdb") == 2                                   # before and after: never left behind


def test_failed_restore_check_is_reported_and_cleaned_up(tmp_path):
    run = FakeDocker(fail={"pg_restore"})
    dump = _dump(tmp_path, "visionlabeler-20261010-020000.dump", 1)
    result = backup_db.verify(dump, run)
    assert not result["ok"] and "pg_restore reported errors" in result["message"]
    assert run.calls[-1] == "dropdb"


def test_missing_second_copy_folder_fails_but_keeps_the_backup(tmp_path):
    folder = tmp_path / "backups"
    assert backup_db.main(["--copy-to", str(tmp_path / "unplugged")], folder, FakeDocker()) == 1
    assert len(backup_db.backups(folder)) == 1


def test_failed_dump_leaves_no_partial_file(tmp_path):
    folder = tmp_path / "backups"
    assert backup_db.main(["--copy-to", ""], folder, FakeDocker(fail={"pg_dump"})) == 1
    assert list(folder.iterdir()) == []


def test_health_line_follows_the_daily_backup(db, tmp_path, monkeypatch):
    monkeypatch.setattr(db, "is_postgres", True)
    now = time.time()
    _dump(tmp_path, "visionlabeler-20261007-020000.dump", 3 * 24)
    on = {"enabled": True, "hour": 2, "copy_to": ""}
    assert healthcheck.check_backup(db, now, tmp_path, on)["level"] == "warning"       # 3 days: daily missed
    assert healthcheck.check_backup(db, now, tmp_path, {**on, "enabled": False})["level"] == "ok"
    fresh = _dump(tmp_path, "visionlabeler-20261010-020000.dump", 1)
    (tmp_path / backup_db.VERIFY_FILE).write_text(json.dumps({"at": "2026-10-10T02:01:00", "ok": True}))
    check = healthcheck.check_backup(db, now, tmp_path, on)
    assert check["level"] == "ok" and "restore check 2026-10-10 OK" in check["detail"]
    usb = tmp_path / "usb"
    usb.mkdir()
    check = healthcheck.check_backup(db, now, tmp_path, {**on, "copy_to": str(usb)})
    assert check["level"] == "warning" and "no second copy" in check["detail"]
    (usb / fresh.name).write_bytes(b"x")
    assert healthcheck.check_backup(db, now, tmp_path, {**on, "copy_to": str(usb)})["level"] == "ok"
    (tmp_path / backup_db.VERIFY_FILE).write_text(json.dumps({"at": "2026-10-10T02:01:00", "ok": False,
                                                              "message": "restoring x failed"}))
    assert "FAILED" in healthcheck.check_backup(db, now, tmp_path, on)["detail"]

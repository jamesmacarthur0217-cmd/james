"""Safe-update test (Phase 41 step 1): the parts that need no database server (comparing, the words, the copy's
address, Docker off). The whole tool is run by hand on PostgreSQL (README "Safe-update test")."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import upgrade_check  # noqa: E402

BEFORE = {"rows": {"tracks": 27, "alert_events": 3, "zones": 1}, "unique": ["alert_events|alert_events_rule_id_track_id_key"],
          "notnull": ["alert_events.track_id", "tracks.id"]}
AFTER = {"rows": {"tracks": 27, "alert_events": 3, "zones": 1, "vehicles": 0}, "unique": [], "notnull": ["tracks.id"]}
OPENED = {"seconds": 1.2, "created": ["recordings", "vehicles"], "added": ["zones.lying_ok"]}


def test_compare_and_describe():
    r = upgrade_check.compare(BEFORE, AFTER, OPENED)
    assert r["unique_removed"] == ["alert_events|alert_events_rule_id_track_id_key"]
    assert r["may_be_empty_now"] == ["alert_events.track_id"] and r["lost"] == {} and r["people"] == 27
    assert r["whats_new"][0] == "Recording" and len(r["whats_new"]) == r["entries"]   # from Phase 13 on: all of it
    words = upgrade_check.describe(r)
    assert words[0] == "OK: the update took 1.2 s on a copy of your database (27 people)."
    assert "2 tables (recordings, vehicles), 1 columns (zones.lying_ok)" in words[1]
    assert "nothing lost" in " ".join(words)
    lost = upgrade_check.compare(BEFORE, {**AFTER, "rows": {**AFTER["rows"], "tracks": 20}}, OPENED)
    assert lost["lost"] == {"tracks": {"before": 27, "after": 20}} and "ROWS LOST" in " ".join(upgrade_check.describe(lost))
    failed = upgrade_check.describe({"error": "the update did not finish within 300 s",
                                     "waiting": ["idle in transaction | Client | ALTER TABLE alert_events ..."]})
    assert failed[0].startswith("FAILED") and "waiting: idle in transaction" in failed[1]


def test_copy_url_keeps_the_server_and_changes_only_the_database():
    url = upgrade_check.copy_url("postgresql+psycopg://visionlabeler:pw@127.0.0.1:5432/visionlabeler")
    assert url == "postgresql+psycopg://visionlabeler:pw@127.0.0.1:5432/visionlabeler_upgrade"


def test_docker_off_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(upgrade_check.backup_db, "docker", lambda: "docker")
    calls = []

    def run(args, **kw):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", "No such container")

    assert upgrade_check.main(["--report", str(tmp_path / "r.json")], run=run) == 2
    assert len(calls) == 1 and calls[0][1] == "inspect"   # nothing else was tried
    assert not (tmp_path / "r.json").exists()


def test_a_failed_copy_is_reported_and_cleaned_up(tmp_path, monkeypatch):
    monkeypatch.setattr(upgrade_check.backup_db, "docker", lambda: "docker")
    calls = []

    def run(args, **kw):
        calls.append(" ".join(args))
        if args[1] == "inspect":
            return subprocess.CompletedProcess(args, 0, "true\n", "")
        if "pg_dump" in " ".join(args):
            return subprocess.CompletedProcess(args, 1, "", "pg_dump: error: connection failed")
        return subprocess.CompletedProcess(args, 0, "", "")

    assert upgrade_check.main(["--report", str(tmp_path / "r.json")], run=run) == 1
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["ok"] is False and "copying failed: pg_dump: error: connection failed" in report["error"]
    assert calls[-1].endswith("DROP DATABASE IF EXISTS visionlabeler_upgrade")   # the copy is dropped again


def test_health_line(tmp_path):
    import copy
    from labeler.config import DEFAULTS
    from labeler.healthcheck import check_upgrade
    cfg = copy.deepcopy(DEFAULTS)
    cfg["updates"]["check_report"] = str(tmp_path / "upgrade_check.json")
    cfg["database"]["url"] = "sqlite:///x.db"
    assert check_upgrade(cfg) is None   # SQLite (tests, test copies): no such test
    cfg["database"]["url"] = "postgresql+psycopg://u:p@127.0.0.1/visionlabeler"
    assert check_upgrade(cfg)["level"] == "off" and "not run yet" in check_upgrade(cfg)["detail"]
    (tmp_path / "upgrade_check.json").write_text(json.dumps({"at": "2026-10-19T01:05:00", "ok": True, "seconds": 0.9,
                                                             "created": ["a"] * 10, "added": ["b"] * 5}))
    assert check_upgrade(cfg) == {"name": "safe-update test", "level": "ok",
                                  "detail": "OK on 2026-10-19 01:05: 0.9 s, 10 tables and 5 columns added, nothing lost"}
    (tmp_path / "upgrade_check.json").write_text(json.dumps({"at": "2026-10-19T01:05:00", "ok": False,
                                                             "error": "the update did not finish within 300 s"}))
    assert check_upgrade(cfg)["level"] == "problem"


def test_start_up_says_what_the_database_update_did(tmp_path, caplog, monkeypatch):
    import threading
    import run
    url = "sqlite:///" + (tmp_path / "t.db").as_posix()
    with caplog.at_level("INFO", logger="run"):
        run._open_database(url)
    words = [r.message for r in caplog.records]
    assert words[0].startswith("opening the database") and any("a new database was made" in w for w in words)
    from sqlalchemy import create_engine, text
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE checks"))
    engine.dispose()
    caplog.clear()
    with caplog.at_level("INFO", logger="run"):
        run._open_database(url)
    assert any("database updated for this version" in r.message and "1 new tables" in r.message
               for r in caplog.records)
    caplog.clear()
    slow = threading.Event()

    class Slow:   # a start-up update that takes a while: it says so while waiting
        def __init__(self, u):
            slow.wait(0.25)
            self.created_tables, self.added_columns = set(), set()

    monkeypatch.setattr(run, "Database", Slow)
    with caplog.at_level("INFO", logger="run"):
        run._open_database(url, every_s=0.05)
    assert any("still opening / updating the database" in r.message for r in caplog.records)
    assert not any("database updated for this version" in r.message for r in caplog.records)   # nothing added


def test_settings_api_shows_the_last_test_on_postgresql(db, tmp_path):
    import copy
    from fastapi.testclient import TestClient
    from labeler.config import DEFAULTS
    from labeler.web import create_app
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cfg["updates"]["check_report"] = str(tmp_path / "upgrade_check.json")
    app = create_app(cfg, db, None)

    class Up:   # the update state without git
        state, busy = {"checked_at": "2026-10-19T01:00:00", "available": True, "behind": 0}, False

        def problem(self, s):
            return None

    app.state.updater = Up()
    client = TestClient(app, client=("127.0.0.1", 5000))
    assert client.get("/api/update").json()["safe_test"] is None   # SQLite: none
    db.is_postgres = True
    (tmp_path / "upgrade_check.json").write_text(json.dumps({"at": "2026-10-19T01:05:00", "ok": True, "seconds": 0.9,
                                                             "created": [], "added": []}))
    assert client.get("/api/update").json()["safe_test"] == {"ok": True,
                                                             "words": "OK on 2026-10-19 01:05: 0.9 s, nothing to add, nothing lost"}

"""Restore drill (tools/restore_drill.py): its config stays away from the user's data; the checks it makes."""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import restore_drill as rd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_drill_config_uses_only_the_drill_database_and_work_folder(tmp_path):
    project = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = rd.drill_config(project, tmp_path, "pw")
    assert cfg["database"]["url"] == "postgresql+psycopg://visionlabeler:pw@127.0.0.1:5433/visionlabeler"
    for path in (cfg["storage"]["crop_dir"], cfg["logs"]["dir"], cfg["settings_file"], cfg["backup"]["dir"],
                 cfg["student"]["retrained_path"]):
        assert Path(path).is_relative_to(tmp_path), path
    assert cfg["storage"]["retention_days"] == 0                   # restored people are not deleted
    assert cfg["backup"]["enabled"] is False and cfg["backup"]["copy_to"] == ""
    assert cfg["retrain"]["enabled"] is False
    assert cfg["web"]["host"] == "127.0.0.1" and cfg["web"]["https"] is False
    assert cfg["cameras"] == [{"name": "drill", "source": "samples/bus.jpg"}]
    assert project["database"]["url"] == "${DATABASE_URL}"         # the project's config is not changed


def test_wait_for():
    t = [0.0]
    calls = []
    assert rd.wait_for(lambda: calls.append(1) or len(calls) == 3, 10, sleep=lambda s: t.__setitem__(0, t[0] + s),
                       clock=lambda: t[0])
    assert not rd.wait_for(lambda: False, 5, sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])


def _fake_app(stored=27, health=200, picture=200):
    def fetch(path):
        if path == "/api/status":
            return 200, json.dumps({"tracks": stored}).encode()
        if path.startswith("/api/tracks?"):
            return 200, json.dumps([{"id": 9, "crop": "track_9.jpg"}, {"id": 3, "crop": "track_3.jpg"}]).encode()
        if path.startswith("/crops/"):
            return picture, b""
        if path == "/api/health":
            return health, json.dumps({"problems": [] if health == 200 else ["database: down"]}).encode()
        return 200, b"{}"
    return fetch


def test_check_app_all_good():
    results = rd.check_app(27, _fake_app())
    assert all(ok for _, ok, _ in results), results
    assert {w for w, _, _ in results} >= {"page /", "page /settings", "people in the app", "look-alikes (vector search)",
                                         "a restored person's picture", "health check"}


def test_check_app_reports_what_failed():
    results = {w: (ok, d) for w, ok, d in rd.check_app(27, _fake_app(stored=0, health=503, picture=404))}
    assert results["people in the app"] == (False, "0 (restored 27)")
    assert results["health check"] == (False, "database: down")
    assert results["a restored person's picture"][0] is False

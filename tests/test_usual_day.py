"""The usual day as a chart (Phase 43 step 1): GET /api/routine gives each kiosk camera's share of days with someone
seen per hour (weekday / weekend), the days behind each hour, the usual hours, and today's quiet stretch as the
"Unusual quiet" check sees it; viewers may read it; nothing without the usual day."""
import copy
from datetime import timedelta

from fastapi.testclient import TestClient

from labeler import accounts
from labeler.config import DEFAULTS
from labeler.web import create_app
from test_routine import _people, _regular
from test_unusual import Arming, _run, _setup, at


def _client(db, tmp_path, watch=None, address="127.0.0.1", enabled=True):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cfg["unusual"]["enabled"] = enabled
    app = create_app(cfg, db, None)
    if watch is not None:
        app.state.routine, app.state.unusual = watch.routine, watch
    return TestClient(app, client=(address, 5000))


def test_a_regular_household_mid_morning(db, camera, tmp_path):
    watch, cam, clock = _setup(db, camera)
    _people(db, camera, [(at(7, 10), at(8, 40))])   # up as usual today
    _run(watch, clock, at(10, 0), step=timedelta(minutes=10))
    got = _client(db, tmp_path, watch).get("/api/routine").json()
    assert got["date"] == "2026-10-19" and got["kind"] == "weekday"
    assert (got["usual_share"], got["quiet_share"], got["min_quiet_h"]) == (0.8, 0.97, 3)
    [c] = got["cameras"]
    assert c["camera"] == "test-cam" and not c["learning"] and c["days"] == {"weekday": 40, "weekend": 16}
    assert c["usual_hours"]["weekday"] == [7, 8, 12, 18, 19, 20, 21]
    assert c["usual_hours"]["weekend"][:2] == [9, 10]
    assert len(c["share"]["weekday"]) == len(c["asked"]["weekend"]) == 24
    assert c["share"]["weekday"][7] == 1.0 and c["share"]["weekday"][9] == 0.0 and c["share"]["weekend"][7] == 0.0
    assert c["asked"]["weekday"][3] == 40 and c["need"] == {"weekday": 20, "weekend": 8}
    t = c["today"]   # quiet since 08:40: one whole hour (09:00-10:00), which nobody is usually seen in
    assert t["since"] == "2026-10-19T08:40" and (t["from_h"], t["to_h"], t["hours"]) == (9, 10, 1)
    assert (t["seen_days"], t["days"], t["share"]) == (0, 40, 0.0)
    assert not t["unusual"] and not t["fired"] and t["watched"]


def test_nobody_up_shows_the_alert(db, camera, tmp_path):
    watch, cam, clock = _setup(db, camera)
    assert len(_run(watch, clock, at(8, 0), step=timedelta(minutes=5))) == 1
    t = _client(db, tmp_path, watch).get("/api/routine").json()["cameras"][0]["today"]
    assert t["since"] == "2026-10-19T00:00" and t["hours"] == 8 and (t["seen_days"], t["days"]) == (40, 40)
    assert t["share"] == 1.0 and t["unusual"] and t["fired"]


def test_away_is_not_watched_and_a_fresh_stretch_has_no_hours(db, camera, tmp_path):
    away = Arming("away", at(0))
    watch, cam, clock = _setup(db, camera, arming=away)
    _run(watch, clock, at(7, 30), step=timedelta(minutes=5))
    t = _client(db, tmp_path, watch).get("/api/routine").json()["cameras"][0]["today"]
    assert not t["watched"] and t["since"] == "2026-10-19T00:00"   # Away since midnight: the rule waits for Home
    away.mode, away.since = "home", at(7, 30)
    t = _client(db, tmp_path, watch).get("/api/routine").json()["cameras"][0]["today"]
    assert t["watched"] and t["since"] == "2026-10-19T07:30" and t["hours"] == 0
    assert t["seen_days"] is None and t["share"] is None and not t["unusual"]


def test_still_learning(db, camera, tmp_path):
    _people(db, camera, _regular(days=10))
    watch, cam, clock = _setup(db, camera, history=False)
    c = _client(db, tmp_path, watch).get("/api/routine").json()["cameras"][0]
    assert c["learning"] and c["total_days"] == 10 and c["min_days"] == 14
    assert c["usual_hours"] == {"weekday": [], "weekend": []}


def test_nothing_without_the_usual_day(db, camera, tmp_path):
    assert _client(db, tmp_path).get("/api/routine").json() == {"cameras": []}   # made without run.py
    watch, cam, clock = _setup(db, camera)
    assert _client(db, tmp_path, watch, enabled=False).get("/api/routine").json() == {"cameras": []}
    watch.routine.cameras = []   # no kiosk camera
    assert _client(db, tmp_path, watch).get("/api/routine").json()["cameras"] == []


def test_the_pages(db, tmp_path):
    from pathlib import Path
    static = Path(__file__).resolve().parent.parent / "labeler" / "static"
    health = (static / "health.html").read_text(encoding="utf-8")
    for part in ['id="usual"', 'id="usual-cams"', 'id="usual-key"', 'fetch("/api/routine")', "ResizeObserver",
                 "Show as a table", "--series-1: #2a78d6", "--series-1: #3987e5", "usual-line", "hollow", "quiet today"]:
        assert part in health, part
    assert 'href="/health#usual"' in (static / "alerts.html").read_text(encoding="utf-8")
    assert _client(db, tmp_path).get("/health").status_code == 200


def test_viewers_may_read_it(db, camera, tmp_path):
    watch, cam, clock = _setup(db, camera)
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _client(db, tmp_path, watch, address="192.168.1.20")
    assert viewer.get("/api/routine").status_code == 401
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    assert viewer.get("/api/routine").json()["cameras"][0]["camera"] == "test-cam"

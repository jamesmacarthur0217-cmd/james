"""What's new (Phase 36 step 1): which entries are new after an update, the database's guess, the page and API."""
import copy
import json
import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from labeler import accounts, whatsnew
from labeler.config import DEFAULTS
from labeler.db import NEW_COLUMNS, Base, Database
from labeler.web import create_app
from labeler.whatsnew import ENTRIES, Entry, WhatsNew

E = [Entry("a", 1, "aaaaaaa", "A", "a text", "/"), Entry("b", 2, "bbbbbbb", "B", "b text", "/", "admin", "recordings"),
     Entry("c", 3, "ccccccc", "C", "c text", "/", marker="zones.lying_ok")]


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(whatsnew, "version", lambda: "1234abc 2026-10-17")


def _new(w, admin=True):
    return [i["key"] for i in w.new_items(admin)]


def test_the_list_is_sound():
    keys = [e.key for e in ENTRIES]
    assert len(keys) == len(set(keys))
    for e in ENTRIES:
        assert e.who in ("all", "admin") and e.title and e.text
        # A commit cannot name itself: the newest entry may wait for the next one.
        assert re.fullmatch(r"[0-9a-f]{7}", e.commit) or (e.commit == "" and e is ENTRIES[-1]), e.key
        assert e.page in ("", "/", "/recordings", "/summary", "/alerts", "/check", "/settings", "/health", "/first-run",
                          "/safety")
        if e.marker:   # a table, or a column added at start
            table, _, column = e.marker.partition(".")
            assert table in Base.metadata.tables and (not column or column in NEW_COLUMNS[table]), e.marker


def test_first_start_on_a_new_database_shows_nothing(tmp_path):
    path = tmp_path / "last_version.json"
    w = WhatsNew.start(path, {"tracks", "recordings", "vehicles"}, set(), E)
    assert _new(w) == []
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["last"] == "c" and state["before"] == "c" and state["version"] == "1234abc 2026-10-17"


def test_first_start_on_an_old_database_guesses_from_what_was_added(tmp_path):
    # Tables made at this start: the phase that added them never ran here, nor anything after it.
    assert _new(WhatsNew.start(tmp_path / "1.json", {"recordings"}, set(), E)) == ["c", "b"]
    assert _new(WhatsNew.start(tmp_path / "2.json", set(), {"zones.lying_ok"}, E)) == ["c"]
    assert _new(WhatsNew.start(tmp_path / "3.json", set(), set(), E)) == []   # up to date already
    assert whatsnew.guess_before({"vehicles"}, set()) == "p17"   # the real list: Phase 18 first new


def test_an_update_shows_what_it_added_until_the_next_one(tmp_path):
    path = tmp_path / "last_version.json"
    path.write_text(json.dumps({"last": "a"}), encoding="utf-8")
    w = WhatsNew.start(path, entries=E)
    assert _new(w) == ["c", "b"] and w.ran
    assert _new(WhatsNew.start(path, entries=E)) == ["c", "b"]   # a restart of the same version keeps it
    more = E + [Entry("d", 4, "ddddddd", "D", "d text", "")]
    assert _new(WhatsNew.start(path, entries=more)) == ["d"]
    assert json.loads(path.read_text(encoding="utf-8"))["before"] == "c"


def test_a_broken_or_unknown_state_shows_nothing_wrong(tmp_path):
    path = tmp_path / "last_version.json"
    path.write_text("not json", encoding="utf-8")
    assert _new(WhatsNew.start(path, entries=E)) == []
    assert _new(WhatsNew("gone", entries=E)) == []   # an entry taken out of the list
    assert _new(WhatsNew("", entries=E)) == ["c", "b", "a"]   # the guess said: before the first entry


def test_viewers_see_only_their_part():
    w = WhatsNew("a", entries=E)
    assert _new(w, admin=False) == ["c"]
    s = w.summary(False)
    assert s["new"] == 1 and [i["key"] for i in s["items"]] == ["c", "a"] and "marker" not in s["items"][0]


def test_database_says_what_it_had_to_add(tmp_path):
    url = "sqlite:///" + (tmp_path / "t.db").as_posix()
    first = Database(url)
    assert "tracks" in first.created_tables and first.added_columns == set()
    with first.engine.begin() as conn:
        conn.execute(text("DROP TABLE recordings"))
        conn.execute(text("ALTER TABLE zones DROP COLUMN lying_ok"))
    first.engine.dispose()
    again = Database(url)
    assert again.created_tables == {"recordings"} and again.added_columns == {"zones.lying_ok"}


def _client(db, tmp_path, address="127.0.0.1", w=None):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    app = create_app(cfg, db, None)
    if w is not None:
        app.state.whats_new = w
    return TestClient(app, client=(address, 5000))


def test_page_and_api_for_admins_and_viewers(db, tmp_path):
    w = WhatsNew("p25")
    admin = _client(db, tmp_path, w=w)
    got = admin.get("/api/whats-new").json()
    assert got["newest"] == ENTRIES[-1].key and got["version"] is None
    assert [i["key"] for i in got["items"] if i["new"]][-1] == "p27" and got["new"] == len(ENTRIES) - 1 - [e.key for e in ENTRIES].index("p25")
    assert admin.get("/whats-new").status_code == 200
    db.add_user("vic", accounts.hash_password("correct horse battery"), "viewer")
    viewer = _client(db, tmp_path, address="192.168.1.20", w=w)
    viewer.post("/api/login", json={"username": "vic", "password": "correct horse battery"})
    seen = viewer.get("/api/whats-new").json()
    assert all(i["who"] == "all" for i in seen["items"]) and seen["new"] < got["new"]
    assert viewer.get("/whats-new", follow_redirects=False).status_code == 200
    assert _client(db, tmp_path).get("/api/whats-new").json()["new"] == 0   # made without run.py: nothing new

"""PostgreSQL only (skipped unless VL_TEST_POSTGRES_URL names a server and a user that may create databases, e.g. the
.env URL): an old database is updated at start without hanging. Found 2026-10-18: a database from before Phase 17
hung forever, because the start-up check read alert_events' columns through a second connection while the first had
just changed that table. Run:  set VL_TEST_POSTGRES_URL=<url> & .venv\\Scripts\\python.exe -m pytest tests/test_postgres.py
"""
import os
import threading

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from labeler.db import Database

URL = os.environ.get("VL_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="needs VL_TEST_POSTGRES_URL (a PostgreSQL server)")
NAME = "vl_test_old_layout"


@pytest.fixture
def scratch_url():
    base = make_url(URL)
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f"DROP DATABASE IF EXISTS {NAME}"))
        c.execute(text(f"CREATE DATABASE {NAME}"))
    yield base.set(database=NAME).render_as_string(hide_password=False)
    with admin.connect() as c:
        c.execute(text(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{NAME}'"))
        c.execute(text(f"DROP DATABASE IF EXISTS {NAME}"))
    admin.dispose()


def test_an_old_database_is_updated_at_start(scratch_url):
    first = Database(scratch_url)
    with first.engine.begin() as conn:   # the layout before Phase 17-18
        conn.execute(text("ALTER TABLE alert_events ADD CONSTRAINT alert_events_rule_id_track_id_key "
                          "UNIQUE (rule_id, track_id)"))
        conn.execute(text("ALTER TABLE alert_events ALTER COLUMN track_id SET NOT NULL"))
        conn.execute(text("ALTER TABLE zones DROP COLUMN lying_ok"))
        conn.execute(text("ALTER TABLE recordings DROP COLUMN kept"))   # Phase 42
        conn.execute(text("DROP TABLE checks"))
    first.engine.dispose()
    opened = []
    t = threading.Thread(target=lambda: opened.append(Database(scratch_url)), daemon=True)
    t.start()
    t.join(60)
    assert opened, "opening the old database hung"
    db = opened[0]
    assert db.created_tables == {"checks"} and db.added_columns == {"zones.lying_ok", "recordings.kept"}
    insp = inspect(db.engine)
    assert insp.get_unique_constraints("alert_events") == []
    assert next(c for c in insp.get_columns("alert_events") if c["name"] == "track_id")["nullable"]
    db.engine.dispose()

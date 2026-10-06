import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labeler.db import Database  # noqa: E402


@pytest.fixture
def db(tmp_path):
    """A fresh SQLite database per test (same models as Postgres, minus pgvector)."""
    return Database("sqlite:///" + (tmp_path / "test.db").as_posix())


@pytest.fixture
def camera(db):
    return db.get_or_create_camera("test-cam", "test", "kiosk")

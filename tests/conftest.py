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
def supervisor(tmp_path, monkeypatch):
    """labeler.supervisor writing its log into tmp_path/logs, never into the user's data/logs."""
    from labeler import supervisor as module
    cfg = tmp_path / "test_config.yaml"
    cfg.write_text(f"logs:\n  dir: {(tmp_path / 'logs').as_posix()}\n", encoding="utf-8")
    monkeypatch.setattr(module, "_config_path", lambda argv: str(cfg))
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    yield module
    for h in list(module.log.handlers):   # release the file so tmp_path can be removed
        module.log.removeHandler(h)
        h.close()


@pytest.fixture
def camera(db):
    return db.get_or_create_camera("test-cam", "test", "kiosk")

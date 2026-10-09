"""Log files: the app and the supervisor write rotating files, so messages survive a closed window."""
import logging

import pytest

from labeler import logs


@pytest.fixture
def app_logs(tmp_path):
    """logs.setup() into tmp_path; the root logger and exception hooks are put back afterwards."""
    import sys
    import threading
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    hooks = sys.excepthook, threading.excepthook
    yield {"dir": str(tmp_path / "logs"), "max_mb": 10, "keep": 5}
    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)
    sys.excepthook, threading.excepthook = hooks


def test_app_log_file_gets_messages_with_the_date(app_logs, tmp_path):
    path = logs.setup(app_logs)
    assert path == tmp_path / "logs" / logs.APP_LOG
    logging.getLogger("run").info("hello from the test")
    text = path.read_text(encoding="utf-8")
    assert "INFO run: hello from the test" in text
    assert text[:4].isdigit() and text[4] == "-"          # 2026-10-09 12:00:00 ...


def test_setup_twice_does_not_write_every_line_twice(app_logs):
    logs.setup(app_logs)
    path = logs.setup(app_logs)
    logging.getLogger("run").warning("once")
    assert path.read_text(encoding="utf-8").count("once") == 1


def test_rotation_keeps_the_configured_number_of_files(app_logs, tmp_path):
    handler = logs.file_handler({**app_logs, "max_mb": 0.001, "keep": 2}, "rotate.log")   # 1,000 bytes
    assert handler.maxBytes == 1000 and handler.backupCount == 2
    logger = logging.getLogger("rotate-test")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        for i in range(100):
            logger.warning("line %03d with some padding to fill the file faster", i)
    finally:
        logger.removeHandler(handler)
        handler.close()
    names = sorted(p.name for p in (tmp_path / "logs").iterdir())
    assert names == ["rotate.log", "rotate.log.1", "rotate.log.2"]
    assert "line 099" in (tmp_path / "logs" / "rotate.log").read_text(encoding="utf-8")


def test_recent_errors_forget_after_a_day():
    errors = logs.RecentErrors(keep_s=100)
    for created, msg in ((1000.0, "old"), (1950.0, "new")):
        record = logging.LogRecord("cam", logging.ERROR, __file__, 1, msg, None, None)
        record.created = created
        errors.handle(record)
    errors.handle(logging.LogRecord("cam", logging.WARNING, __file__, 1, "warning only", None, None))
    assert errors.recent(now=2000.0) == [(1950.0, "cam: new")]


def test_supervisor_logs_restarts_and_giving_up(monkeypatch, supervisor, tmp_path):
    monkeypatch.setattr(supervisor, "run_once", lambda argv: (3, False))
    supervisor.main(["--config", "ignored-by-the-fixture.yaml"])
    text = (tmp_path / "logs" / logs.SUPERVISOR_LOG).read_text(encoding="utf-8")
    assert "starting the app" in text
    assert text.count("stopped unexpectedly (exit code 3)") == supervisor.MAX_CRASHES - 1
    assert "ERROR supervisor: The app crashed 5 times" in text


def test_supervisor_finds_the_config_in_the_app_arguments():
    from labeler.supervisor import _config_path
    assert _config_path(["--config", "a.yaml", "--port", "8001"]) == "a.yaml"
    assert _config_path(["--config=b.yaml"]) == "b.yaml"
    assert _config_path(["--port", "8001"]) is None

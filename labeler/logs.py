"""Log files: what the console window shows also goes to data/logs/, so a crash or a warning can still be read
after the window is closed (or when the app was restarted by the supervisor).

    data/logs/visionlabeler.log     the app (run.py), 10 MB x 5 files by default (logs: in config.yaml)
    data/logs/supervisor.log        restarts and give-ups (labeler/supervisor.py)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

from labeler.config import resolve

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
APP_LOG = "visionlabeler.log"
SUPERVISOR_LOG = "supervisor.log"


def file_handler(logs: dict, filename: str) -> RotatingFileHandler:
    """A log file in logs["dir"] that rolls over at logs["max_mb"], keeping logs["keep"] old files.
    The file has the full date (the console shows only the time: it is read the same day)."""
    folder = resolve(logs["dir"])
    folder.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(folder / filename, maxBytes=int(float(logs["max_mb"]) * 1_000_000),
                                  backupCount=int(logs["keep"]), encoding="utf-8", delay=True)
    handler.setFormatter(logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    return handler


class RecentErrors(logging.Handler):
    """Remembers the errors logged in the last day (time + message), for the health check, without reading the
    log files back."""

    def __init__(self, keep_s: float = 86400, max_items: int = 200):
        super().__init__(level=logging.ERROR)
        self.keep_s = keep_s
        self._items: deque[tuple[float, str]] = deque(maxlen=max_items)
        self._lock_items = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < self.level:
            return
        with self._lock_items:
            self._items.append((record.created, f"{record.name}: {record.getMessage()}"))

    def recent(self, now: float | None = None) -> list[tuple[float, str]]:
        cutoff = (now or time.time()) - self.keep_s
        with self._lock_items:
            return [item for item in self._items if item[0] >= cutoff]


recent_errors = RecentErrors()


class DropClosedConnections(logging.Filter):
    """On Windows, Python's asyncio logs an ERROR with a traceback when a browser drops a connection the server is
    closing (WinError 10054): a video player seeking in a recording does it all the time. Harmless, but it filled the
    log and the Health page's "errors in the log" line, so it is left out."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not (isinstance(exc, ConnectionResetError) and "connection_lost" in record.getMessage())


def setup(logs: dict) -> Path:
    """Console (time only) + rotating app log file + the recent-errors memory. Returns the log file's path."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):   # a second call (tests) must not log every line twice
        if getattr(h, "_visionlabeler", False):
            root.removeHandler(h)
            h.close()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
    file = file_handler(logs, APP_LOG)
    for h in (console, file, recent_errors):
        h._visionlabeler = True
        root.addHandler(h)
    # The model downloader logs every web request it makes at startup ("HTTP Request: GET https://..."): noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    aio = logging.getLogger("asyncio")
    if not any(isinstance(f, DropClosedConnections) for f in aio.filters):
        aio.addFilter(DropClosedConnections())
    _log_uncaught()
    return Path(file.baseFilename)


def _log_uncaught() -> None:
    """Crashes outside the camera loops (which log their own) go into the log file too, not only the console."""
    import sys

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
        else:   # the console handler prints the traceback, so not sys.__excepthook__ as well
            logging.getLogger("run").critical("the app crashed", exc_info=(exc_type, exc, tb))

    def thread_hook(args):
        if args.exc_type is not SystemExit:
            logging.getLogger("run").error("thread %s crashed", args.thread.name if args.thread else "?",
                                           exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = hook
    threading.excepthook = thread_hook

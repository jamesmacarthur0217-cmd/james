"""Run the app and restart it if it crashes (used by start.bat).

    python -m labeler.supervisor [run.py arguments]

A normal stop (Stop button, stop.bat, Ctrl+C) ends it. A crash restarts the app after 5 seconds; after
5 crashes within 10 minutes it gives up, since something needs fixing (the log above says what).
Its own messages also go to data/logs/supervisor.log (the app's to visionlabeler.log), so a crash at night
can be read in the morning even after the window was closed.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_CRASHES = 5
WINDOW_S = 600
RESTART_DELAY_S = 5

log = logging.getLogger("supervisor")


def run_once(argv: list[str]) -> tuple[int, bool]:
    """Run the app until it exits. Returns (exit code, whether the user pressed Ctrl+C)."""
    proc = subprocess.Popen([sys.executable, str(ROOT / "run.py"), *argv], cwd=ROOT)
    interrupted = False
    while True:
        try:
            return proc.wait(), interrupted
        except KeyboardInterrupt:
            # The app received Ctrl+C too and is saving; keep waiting rather than killing it.
            interrupted = True


def say(message: str, level: int = logging.INFO) -> None:
    """The console shows the message as before; the log file gets it with the date."""
    print(message, flush=True)
    log.log(level, message.strip())


def _config_path(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def setup_log(argv: list[str]) -> None:
    """Log to the folder the app's config names (logs.dir), so a test config keeps its logs in its own folder."""
    from labeler.config import DEFAULTS, load_config
    from labeler.logs import SUPERVISOR_LOG, file_handler
    try:
        logs = load_config(_config_path(argv))["logs"]
    except Exception:   # the app itself reports a broken config; the supervisor still needs a log
        logs = DEFAULTS["logs"]
    log.setLevel(logging.INFO)
    log.propagate = False   # the console already gets print()
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    log.addHandler(file_handler(logs, SUPERVISOR_LOG))


def main(argv: list[str]) -> int:
    setup_log(argv)
    crashes: list[float] = []
    log.info("starting the app")
    while True:
        code, interrupted = run_once(argv)
        if code == 0 or interrupted:
            log.info("the app stopped normally (exit code %s)", code)
            return 0
        now = time.monotonic()
        crashes = [t for t in crashes if now - t < WINDOW_S] + [now]
        if len(crashes) >= MAX_CRASHES:
            say(f"\nThe app crashed {len(crashes)} times within {WINDOW_S // 60} minutes; not restarting again.\n"
                "Check the messages above (or data\\logs\\visionlabeler.log), fix the cause, then run start.bat "
                "again.", logging.ERROR)
            return code
        say(f"\nThe app stopped unexpectedly (exit code {code}). Restarting in {RESTART_DELAY_S} seconds "
            f"(crash {len(crashes)} of {MAX_CRASHES} allowed in {WINDOW_S // 60} minutes)...\n", logging.WARNING)
        try:
            time.sleep(RESTART_DELAY_S)
        except KeyboardInterrupt:
            return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

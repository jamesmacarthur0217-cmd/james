"""Run the app and restart it if it crashes (used by start.bat).

    python -m labeler.supervisor [run.py arguments]

A normal stop (Stop button, stop.bat, Ctrl+C) ends it. A crash restarts the app after 5 seconds; after
5 crashes within 10 minutes it gives up, since something needs fixing (the log above says what).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_CRASHES = 5
WINDOW_S = 600
RESTART_DELAY_S = 5


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


def main(argv: list[str]) -> int:
    crashes: list[float] = []
    while True:
        code, interrupted = run_once(argv)
        if code == 0 or interrupted:
            return 0
        now = time.monotonic()
        crashes = [t for t in crashes if now - t < WINDOW_S] + [now]
        if len(crashes) >= MAX_CRASHES:
            print(f"\nThe app crashed {len(crashes)} times within {WINDOW_S // 60} minutes; not restarting again.\n"
                  "Check the messages above, fix the cause, then run start.bat again.", flush=True)
            return code
        print(f"\nThe app stopped unexpectedly (exit code {code}). Restarting in {RESTART_DELAY_S} seconds "
              f"(crash {len(crashes)} of {MAX_CRASHES} allowed in {WINDOW_S // 60} minutes)...\n", flush=True)
        try:
            time.sleep(RESTART_DELAY_S)
        except KeyboardInterrupt:
            return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

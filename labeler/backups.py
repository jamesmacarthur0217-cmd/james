"""Daily database backups while the app runs (stop.bat also makes one when it stops the app).

Once a day, from backup.hour on (night, default 2), if the newest backup is older than 20 hours, the app runs
tools/backup_db.py: a pg_dump of the PostgreSQL database into data/backups, keeping the newest backup.keep (7).
With backup.copy_to (e.g. a USB drive or another disk) each backup is copied there as well, since a backup on
the same disk is lost with the disk. Once a week the newest backup is also restored into a temporary database to
check that it really works (a backup nobody ever restored may be broken); the result is shown on the Health
page. A PC that is off at night catches up when the app next runs after backup.hour.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from .config import ROOT, resolve

log = logging.getLogger(__name__)

PATTERN = "visionlabeler-*.dump"
VERIFY_FILE = "last_verify.json"   # written by tools/backup_db.py --verify
MIN_AGE_H = 20                     # same as stop.bat: at most one backup a day
VERIFY_EVERY_D = 7


def folder(cfg: dict) -> Path:
    """backup.dir (data/backups; a test config keeps its own)."""
    return resolve(cfg["backup"].get("dir") or "data/backups")


def newest(where: Path) -> Path | None:
    return max(where.glob(PATTERN), key=lambda p: p.stat().st_mtime, default=None) if where.is_dir() else None


def last_verify(where: Path) -> dict | None:
    """{"at": iso time, "ok": bool, "message": str} of the last restore check, or None."""
    path = where / VERIFY_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def due(now: datetime, hour: int, newest_mtime: float | None) -> bool:
    """Backup time: from `hour` on, when the newest backup is at least MIN_AGE_H hours old (or there is none)."""
    if now.hour < hour:
        return False
    return newest_mtime is None or now - datetime.fromtimestamp(newest_mtime) >= timedelta(hours=MIN_AGE_H)


def verify_due(now: datetime, verify: dict | None) -> bool:
    if not verify:
        return True
    try:
        return now - datetime.fromisoformat(verify["at"]) >= timedelta(days=VERIFY_EVERY_D)
    except (KeyError, ValueError):
        return True


def command(cfg: dict, verify: bool) -> list[str]:
    bcfg = cfg["backup"]
    cmd = [sys.executable, str(ROOT / "tools" / "backup_db.py"), "--keep", str(int(bcfg["keep"])),
           "--dir", str(folder(cfg))]
    if bcfg.get("copy_to"):
        cmd += ["--copy-to", str(bcfg["copy_to"])]
    if verify:
        cmd.append("--verify")
    return cmd


def run_backup(cfg: dict, verify: bool, runner=subprocess.run) -> bool:
    """One backup (and maybe a restore check) in a separate process; its output goes to the log."""
    t = time.monotonic()
    r = runner(command(cfg, verify), cwd=ROOT, capture_output=True, text=True, timeout=3600)
    for line in (r.stdout or "").splitlines():
        if line.strip():
            log.info("backup: %s", line.strip())
    if r.returncode != 0:
        log.error("database backup failed (exit code %s): %s", r.returncode,
                  (r.stderr or r.stdout or "").strip()[-500:] or "no message")
        return False
    log.info("database backup finished in %.0f s", time.monotonic() - t)
    return True


def tick(cfg: dict, db, now: datetime | None = None, runner=subprocess.run) -> bool | None:
    """Make a backup if one is due. Returns None when nothing was due, else whether it worked."""
    bcfg = cfg["backup"]
    if not bcfg["enabled"] or not db.is_postgres:
        return None
    now = now or datetime.now()
    latest = newest(folder(cfg))
    if not due(now, int(bcfg["hour"]), latest.stat().st_mtime if latest else None):
        return None
    return run_backup(cfg, verify_due(now, last_verify(folder(cfg))), runner)


def loop(cfg: dict, db, stop: threading.Event, every_s: float = 600, retry_s: float = 3600) -> None:
    """Checks every 10 minutes; after a failure, tries again in an hour (not every 10 minutes)."""
    wait = every_s
    while not stop.wait(wait):
        try:
            ok = tick(cfg, db)
        except Exception:
            log.exception("database backup check failed")
            ok = False
        wait = retry_s if ok is False else every_s

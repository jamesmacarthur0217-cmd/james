"""Back up the database to data/backups, keeping the newest few. Called by stop.bat and, once a day, by the
running app (labeler/backups.py).

    python tools/backup_db.py                         # back up now
    python tools/backup_db.py --if-older-than-hours 20 --keep 7
    python tools/backup_db.py --copy-to E:\\VisionLabelerBackups   # also copy it there (default: backup.copy_to)
    python tools/backup_db.py --verify                # also restore it into a test database to check it works
    python tools/backup_db.py --verify-only           # only check the newest backup

Restore a backup (replaces the current data; stop the app first):
    docker exec -i visionlabeler-db pg_restore -U visionlabeler -d visionlabeler --clean --if-exists < data/backups/<file>.dump
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUPS = ROOT / "data" / "backups"
CONTAINER = "visionlabeler-db"
USER = DB = "visionlabeler"
VERIFY_DB = "visionlabeler_verify"
PATTERN = "visionlabeler-*.dump"
VERIFY_FILE = "last_verify.json"


def docker() -> str:
    found = shutil.which("docker")
    if found:
        return found
    for candidate in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/DockerDesktop/resources/bin/docker.exe",
                      Path(os.environ.get("ProgramFiles", "")) / "Docker/Docker/resources/bin/docker.exe"):
        if candidate.exists():
            return str(candidate)
    raise SystemExit("docker not found")


def in_container(args: list[str], run=subprocess.run, **kw) -> subprocess.CompletedProcess:
    """A command inside the database container (docker exec)."""
    interactive = ["-i"] if "stdin" in kw else []
    return run([docker(), "exec", *interactive, CONTAINER, *args], **kw)


def backups(folder: Path) -> list[Path]:
    return sorted(folder.glob(PATTERN))   # names hold the date and time, so this is oldest first


def make_backup(folder: Path, run=subprocess.run) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"visionlabeler-{time.strftime('%Y%m%d-%H%M%S')}.dump"
    tmp = out.with_suffix(".partial")
    with open(tmp, "wb") as f:
        r = in_container(["pg_dump", "-U", USER, "-d", DB, "-Fc"], run, stdout=f, stderr=subprocess.PIPE)
    if r.returncode != 0 or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        message = r.stderr.decode(errors="replace").strip() if r.stderr else ""
        raise RuntimeError(message or "empty output")
    tmp.replace(out)
    return out


def prune(folder: Path, keep: int) -> list[str]:
    """Delete all but the newest `keep` backups; returns the removed names."""
    removed = []
    for old in backups(folder)[:-keep] if keep > 0 else []:
        old.unlink()
        removed.append(old.name)
    return removed


def copy_to(dump: Path, target: Path, keep: int) -> Path:
    """A second copy elsewhere (another disk, a USB drive): a backup on the same disk is lost with the disk."""
    if not target.is_dir():
        raise RuntimeError(f"the folder for the second copy does not exist: {target} (drive not plugged in?)")
    out = target / dump.name
    tmp = out.with_suffix(".partial")
    shutil.copy2(dump, tmp)
    tmp.replace(out)
    prune(target, keep)
    return out


def _count(db: str, run) -> int:
    r = in_container(["psql", "-U", USER, "-d", db, "-tAc", "SELECT count(*) FROM tracks"], run,
                     capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "psql failed")
    return int(r.stdout.strip())


def verify(dump: Path, run=subprocess.run) -> dict:
    """Restore the backup into a separate test database, count its people, and delete the test database.
    The real database is not touched. The result is saved for the Health page."""
    result = {"at": datetime.now().isoformat(timespec="seconds"), "backup": dump.name, "ok": False}
    try:
        in_container(["dropdb", "-U", USER, "--if-exists", VERIFY_DB], run, capture_output=True)
        r = in_container(["createdb", "-U", USER, VERIFY_DB], run, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"could not make the test database: {r.stderr.strip()}")
        with open(dump, "rb") as f:
            r = in_container(["pg_restore", "-U", USER, "-d", VERIFY_DB, "--no-owner"], run, stdin=f,
                             capture_output=True)
        if r.returncode != 0:
            raise RuntimeError("pg_restore reported errors: " + r.stderr.decode(errors="replace").strip()[-300:])
        restored, live = _count(VERIFY_DB, run), _count(DB, run)
        result.update(ok=True, people=restored,
                      message=f"restored {dump.name}: {restored} people (the database has {live} now)")
    except (RuntimeError, ValueError, OSError) as e:
        result["message"] = f"restoring {dump.name} failed: {e}"
    finally:
        in_container(["dropdb", "-U", USER, "--if-exists", VERIFY_DB], run, capture_output=True)
    (dump.parent / VERIFY_FILE).write_text(json.dumps(result), encoding="utf-8")
    return result


def configured_copy_to() -> str:
    """backup.copy_to from the app's settings (config.yaml, or the Settings page)."""
    try:
        sys.path.insert(0, str(ROOT))
        from labeler.config import load_config
        return str(load_config()["backup"].get("copy_to") or "")
    except Exception:
        return ""


def configured_dir() -> Path:
    """backup.dir from the app's settings (data/backups unless changed)."""
    try:
        sys.path.insert(0, str(ROOT))
        from labeler.backups import folder
        from labeler.config import load_config
        return folder(load_config())
    except Exception:
        return BACKUPS


def main(argv: list[str] | None = None, folder: Path | None = None, run=subprocess.run) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=7, help="how many backups to keep")
    ap.add_argument("--if-older-than-hours", type=float, default=0,
                    help="skip if the newest backup is younger than this")
    ap.add_argument("--copy-to", help="also copy each backup into this folder (default: backup.copy_to)")
    ap.add_argument("--verify", action="store_true", help="restore the new backup into a test database")
    ap.add_argument("--verify-only", action="store_true", help="only check the newest backup")
    ap.add_argument("--dir", type=Path, help="the backups folder (default: backup.dir, data/backups)")
    args = ap.parse_args(argv)
    folder = folder or args.dir or configured_dir()

    if args.verify_only:
        existing = backups(folder)
        if not existing:
            print("No backup to check.", file=sys.stderr)
            return 1
        result = verify(existing[-1], run)
        print(result["message"], file=sys.stdout if result["ok"] else sys.stderr)
        return 0 if result["ok"] else 1

    existing = backups(folder)
    if existing and args.if_older_than_hours:
        age_h = (time.time() - existing[-1].stat().st_mtime) / 3600
        if age_h < args.if_older_than_hours:
            print(f"Backup skipped: the newest is {age_h:.1f} h old ({existing[-1].name}).")
            return 0
    try:
        out = make_backup(folder, run)
    except RuntimeError as e:
        print("Backup FAILED:", e, file=sys.stderr)
        return 1
    print(f"Backup saved: {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    for name in prune(folder, args.keep):
        print(f"Old backup removed: {name}")

    failed = False
    target = args.copy_to if args.copy_to is not None else configured_copy_to()
    if target:
        try:
            print(f"Second copy saved: {copy_to(out, Path(target), args.keep)}")
        except (RuntimeError, OSError) as e:
            print(f"Second copy FAILED: {e}", file=sys.stderr)
            failed = True
    if args.verify:
        result = verify(out, run)
        print(result["message"], file=sys.stdout if result["ok"] else sys.stderr)
        failed = failed or not result["ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

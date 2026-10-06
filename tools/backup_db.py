"""Back up the database to data/backups, keeping the newest few. Called by stop.bat.

    python tools/backup_db.py                         # back up now
    python tools/backup_db.py --if-older-than-hours 20 --keep 7

Restore a backup (replaces the current data; stop the app first):
    docker exec -i visionlabeler-db pg_restore -U visionlabeler -d visionlabeler --clean --if-exists < data/backups/<file>.dump
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUPS = ROOT / "data" / "backups"
CONTAINER = "visionlabeler-db"


def docker() -> str:
    found = shutil.which("docker")
    if found:
        return found
    for candidate in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/DockerDesktop/resources/bin/docker.exe",
                      Path(os.environ.get("ProgramFiles", "")) / "Docker/Docker/resources/bin/docker.exe"):
        if candidate.exists():
            return str(candidate)
    raise SystemExit("docker not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=7, help="how many backups to keep")
    ap.add_argument("--if-older-than-hours", type=float, default=0,
                    help="skip if the newest backup is younger than this")
    args = ap.parse_args()

    BACKUPS.mkdir(parents=True, exist_ok=True)
    existing = sorted(BACKUPS.glob("visionlabeler-*.dump"))
    if existing and args.if_older_than_hours:
        age_h = (time.time() - existing[-1].stat().st_mtime) / 3600
        if age_h < args.if_older_than_hours:
            print(f"Backup skipped: the newest is {age_h:.1f} h old ({existing[-1].name}).")
            return 0

    out = BACKUPS / f"visionlabeler-{time.strftime('%Y%m%d-%H%M%S')}.dump"
    tmp = out.with_suffix(".partial")
    with open(tmp, "wb") as f:
        r = subprocess.run([docker(), "exec", CONTAINER, "pg_dump", "-U", "visionlabeler", "-d", "visionlabeler",
                            "-Fc"], stdout=f, stderr=subprocess.PIPE)
    if r.returncode != 0 or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        print("Backup FAILED:", r.stderr.decode(errors="replace").strip() or "empty output", file=sys.stderr)
        return 1
    tmp.rename(out)
    print(f"Backup saved: {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")

    for old in sorted(BACKUPS.glob("visionlabeler-*.dump"))[:-args.keep]:
        old.unlink()
        print(f"Old backup removed: {old.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

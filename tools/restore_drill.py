"""Restore drill: prove that a backup brings the system back, not only that the dump file can be read.

    python tools/restore_drill.py [--backup <file>.dump] [--no-pictures] [--keep]

What it does, without touching the real database or the user's files:
  1. a new, empty database container (visionlabeler-drill, 127.0.0.1:5433, from the same image; nothing downloaded)
  2. restores the newest backup into it (or --backup) and counts the people
  3. starts a test copy of the app on it (port 8002, demo picture, no VLM) with a copy of the pictures folder
  4. opens every page and a few API calls on restored people (their picture, look-alikes, the answer-key score)
  5. stops the copy, removes the container and the work folder, and prints the result with the times taken

The weekly check of the daily backup (tools/backup_db.py --verify) only restores and counts; this drill is the
whole way back, as after a broken disk: run it now and then (README "Backups").
"""
from __future__ import annotations

import argparse
import copy
import json
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import backup_db  # noqa: E402
from labeler.backups import folder as backup_folder  # noqa: E402
from labeler.config import load_config, resolve  # noqa: E402

CONTAINER = "visionlabeler-drill"
IMAGE = "pgvector/pgvector:pg17"
DB_PORT = 5433
APP_PORT = 8002
PAGES = ["/", "/review", "/reports", "/alerts", "/answer-key", "/import", "/health", "/settings", "/audit"]


def docker(args: list[str], run=subprocess.run, **kw) -> subprocess.CompletedProcess:
    return run([backup_db.docker(), *args], **kw)


def drill_config(project_cfg: dict, work: Path, password: str) -> dict:
    """The project's config.yaml pointed at the drill database, with everything else inside `work`: the
    restored people are kept (retention off, so the count can be compared), and nothing runs on a schedule."""
    cfg = copy.deepcopy(project_cfg)
    cfg["database"] = {"url": f"postgresql+psycopg://{backup_db.USER}:{password}@127.0.0.1:{DB_PORT}/{backup_db.DB}"}
    cfg["cameras"] = [{"name": "drill", "source": "samples/bus.jpg"}]
    cfg.setdefault("storage", {}).update(crop_dir=(work / "crops").as_posix(), retention_days=0)
    cfg.setdefault("web", {}).update(host="127.0.0.1", https=False)
    cfg.setdefault("logs", {})["dir"] = (work / "logs").as_posix()
    cfg["settings_file"] = (work / "settings.yaml").as_posix()
    cfg.setdefault("backup", {}).update(enabled=False, dir=(work / "backups").as_posix(), copy_to="")
    cfg.setdefault("retrain", {})["enabled"] = False
    cfg.setdefault("recording", {}).update(enabled=False, dir=(work / "recordings").as_posix())
    cfg.setdefault("student", {})["retrained_path"] = (work / "models" / "attributes.joblib").as_posix()
    return cfg


def wait_for(check, timeout_s: float, every_s: float = 1.0, sleep=time.sleep, clock=time.monotonic) -> bool:
    end = clock() + timeout_s
    while True:
        if check():
            return True
        if clock() >= end:
            return False
        sleep(every_s)


def get(path: str, method: str = "GET", timeout: float = 60) -> tuple[int | None, bytes]:
    req = urllib.request.Request(f"http://127.0.0.1:{APP_PORT}{path}", method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except OSError:
        return None, b""


def check_app(restored: int, fetch=get) -> list[tuple[str, bool, str]]:
    """(what, ok, detail) for each check on the running copy."""
    out = []
    for page in PAGES:
        code, _ = fetch(page)
        out.append((f"page {page}", code == 200, str(code)))
    code, body = fetch("/api/status")
    stored = json.loads(body).get("tracks", -1) if code == 200 else -1
    out.append(("people in the app", stored >= restored, f"{stored} (restored {restored})"))
    code, body = fetch("/api/tracks?limit=5")
    people = json.loads(body) if code == 200 else []
    out.append(("list of people", code == 200 and bool(people) == bool(restored), f"{len(people)} shown"))
    if people:
        first = people[-1]
        picture = first.get("crop") or ""
        if picture:
            code, _ = fetch("/crops/" + Path(picture).name)
            out.append(("a restored person's picture", code == 200, str(code)))
        code, _ = fetch(f"/api/tracks/{first['id']}/similar")
        out.append(("look-alikes (vector search)", code == 200, str(code)))
    code, _ = fetch("/api/answer-key/score")
    out.append(("answer-key score", code == 200, str(code)))
    code, _ = fetch("/api/reports/summary")
    out.append(("reports", code == 200, str(code)))
    code, body = fetch("/api/health")
    if code in (200, 503):
        problems = json.loads(body).get("problems", [])
        out.append(("health check", code == 200, "ok" if code == 200 else "; ".join(problems)))
    else:
        out.append(("health check", False, str(code)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Restore the newest backup into a new database and run the app on it.")
    ap.add_argument("--backup", type=Path, help="this backup instead of the newest in backup.dir")
    ap.add_argument("--no-pictures", action="store_true", help="do not copy the pictures folder (people show none)")
    ap.add_argument("--keep", action="store_true", help="keep the work folder (logs) afterwards")
    args = ap.parse_args()

    project = load_config()
    dumps = backup_db.backups(backup_folder(project))
    dump = args.backup or (dumps[-1] if dumps else None)
    if dump is None or not dump.exists():
        print("No backup found.", file=sys.stderr)
        return 1
    if docker(["image", "inspect", IMAGE], capture_output=True).returncode != 0:
        print(f"The database image {IMAGE} is not on this PC.", file=sys.stderr)
        return 1
    work = Path(tempfile.mkdtemp(prefix="vl-drill-"))
    password = secrets.token_urlsafe(16)
    app = None
    results: list[tuple[str, bool, str]] = []
    times: dict[str, float] = {}
    t0 = time.monotonic()
    try:
        print(f"Backup: {dump.name} ({dump.stat().st_size / 1e6:.1f} MB)")
        docker(["rm", "-f", CONTAINER], capture_output=True)
        r = docker(["run", "-d", "--rm", "--pull", "never", "--name", CONTAINER, "-e", f"POSTGRES_USER={backup_db.USER}",
                    "-e", f"POSTGRES_DB={backup_db.DB}", "-e", f"POSTGRES_PASSWORD={password}",
                    "-p", f"127.0.0.1:{DB_PORT}:5432", IMAGE], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("could not start the drill database: " + r.stderr.strip())
        # Over TCP inside the container: during its first start the image runs a temporary server on the
        # socket only, then restarts; the TCP port answers only on the real one.
        ready = wait_for(lambda: docker(["exec", CONTAINER, "pg_isready", "-h", "127.0.0.1", "-U", backup_db.USER,
                                         "-d", backup_db.DB], capture_output=True).returncode == 0, 90)
        if not ready:
            raise RuntimeError("the drill database did not start within 90 s")
        times["new database"] = time.monotonic() - t0

        t = time.monotonic()
        with open(dump, "rb") as f:
            r = docker(["exec", "-i", CONTAINER, "pg_restore", "-U", backup_db.USER, "-d", backup_db.DB, "--no-owner"],
                       stdin=f, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError("pg_restore reported errors: " + r.stderr.decode(errors="replace").strip()[-300:])
        r = docker(["exec", CONTAINER, "psql", "-U", backup_db.USER, "-d", backup_db.DB, "-tAc",
                    "SELECT count(*) FROM tracks"], capture_output=True, text=True)
        restored = int(r.stdout.strip())
        times["restore"] = time.monotonic() - t
        results.append(("restore", True, f"{restored} people"))

        t = time.monotonic()
        cfg = drill_config(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")), work, password)
        if not args.no_pictures:
            shutil.copytree(resolve(project["storage"]["crop_dir"]), work / "crops")
        (work / "drill_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        with open(work / "console.txt", "wb") as console:
            app = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--config", str(work / "drill_config.yaml"),
                                    "--port", str(APP_PORT), "--no-vlm"], cwd=ROOT, stdout=console,
                                   stderr=subprocess.STDOUT)
        if not wait_for(lambda: app.poll() is not None or get("/api/status", timeout=5)[0] == 200, 240):
            raise RuntimeError("the app did not answer within 4 minutes")
        if app.poll() is not None:
            raise RuntimeError(f"the app stopped (exit {app.returncode}); see {work / 'console.txt'}")
        times["app start"] = time.monotonic() - t
        results += check_app(restored)
    except (RuntimeError, OSError, ValueError) as e:
        results.append(("drill", False, str(e)))
    finally:
        if app is not None and app.poll() is None:
            get("/api/shutdown", method="POST")
            try:
                app.wait(90)
            except subprocess.TimeoutExpired:
                app.kill()
        docker(["rm", "-f", CONTAINER], capture_output=True)
        if args.keep:
            print(f"Work folder kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)   # holds a copy of the pictures: removed

    for what, ok, detail in results:
        print(f"  {'ok     ' if ok else 'FAILED '} {what}: {detail}")
    print("Times: " + ", ".join(f"{k} {v:.0f} s" for k, v in times.items())
          + f"; total {time.monotonic() - t0:.0f} s")
    failed = [w for w, ok, _ in results if not ok]
    print("Result: " + ("the backup brings the system back" if not failed else "FAILED: " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

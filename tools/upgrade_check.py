"""Safe-update test (Phase 41): open a copy of the real database with today's code before the app is restarted.

    python tools/upgrade_check.py [--source visionlabeler] [--timeout 300] [--keep]

The app changes its database by itself at start (new tables and columns, old rules removed). On 2026-10-18 a database
built by old code showed that one such change hung for ever (pitfall 30): made-up layouts did not show it. This tool
runs the same start-up update on a copy of the user's own database, where only real data can surprise:

  1. a throwaway database (visionlabeler_upgrade) on the same PostgreSQL server, filled inside the container from
     `pg_dump` of the real one (`--source`); the real database is only read, also while the old app is running, and no
     copy of the data is written to a file;
  2. counts the rows of every table and notes the old layout (unique rules, columns that may not be empty);
  3. opens it with today's code (`Database(url)`) in a child process, with a time limit; if it does not finish, what
     the database is waiting for is shown (as found in Phase 36);
  4. counts again: nothing may be lost; lists the tables and columns added, the old rules removed, how long it took,
     and what the What's new page would list;
  5. drops the copy (unless --keep) and writes the result to data/upgrade_check.json (the Update button and the
     Health page show it: Phase 41 step 2).

Exit code 0 when the update worked, 1 when it failed or took too long, 2 when it could not run (Docker off, ...).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import backup_db  # noqa: E402

COPY = "visionlabeler_upgrade"
REPORT = ROOT / "data" / "upgrade_check.json"
OPEN_CODE = """
import json, sys, time
sys.path.insert(0, sys.argv[2])
t = time.monotonic()
from labeler.db import Database
d = Database(sys.argv[1])
print(json.dumps({"seconds": round(time.monotonic() - t, 2), "created": sorted(d.created_tables),
                  "added": sorted(d.added_columns)}))
"""
LAYOUT_SQL = {
    # table -> rows, one line per table (the copy only)
    "rows": "SELECT format('%s|%s', relname, n) FROM (SELECT c.relname, (xpath('/row/n/text()', query_to_xml("
            "format('SELECT count(*) AS n FROM %I', c.relname), false, true, '')))[1]::text::bigint AS n FROM pg_class c "
            "JOIN pg_namespace s ON s.oid = c.relnamespace WHERE c.relkind = 'r' AND s.nspname = 'public') t "
            "ORDER BY relname",
    "unique": "SELECT format('%s|%s', conrelid::regclass, conname) FROM pg_constraint WHERE contype = 'u' "
              "AND connamespace = 'public'::regnamespace ORDER BY 1",
    "notnull": "SELECT format('%s.%s', table_name, column_name) FROM information_schema.columns "
               "WHERE table_schema = 'public' AND is_nullable = 'NO' ORDER BY 1",
}


def psql(sql: str, db: str, run=subprocess.run, timeout: float = 120) -> list[str]:
    """Lines of a query's answer, run inside the database container."""
    r = backup_db.in_container(["psql", "-U", backup_db.USER, "-d", db, "-v", "ON_ERROR_STOP=1", "-tAc", sql], run,
                               capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-300:])
    return [line for line in r.stdout.splitlines() if line.strip()]


def layout(db: str, run=subprocess.run) -> dict:
    rows = dict(line.split("|", 1) for line in psql(LAYOUT_SQL["rows"], db, run))
    return {"rows": {k: int(v) for k, v in rows.items()}, "unique": psql(LAYOUT_SQL["unique"], db, run),
            "notnull": psql(LAYOUT_SQL["notnull"], db, run)}


def compare(before: dict, after: dict, opened: dict) -> dict:
    """What the update did, and whether anything was lost."""
    lost = {t: (n, after["rows"].get(t)) for t, n in before["rows"].items() if after["rows"].get(t, -1) < n}
    from labeler.whatsnew import ENTRIES, WhatsNew, guess_before
    first_new = WhatsNew(guess_before(set(opened["created"]), set(opened["added"])))
    return {"seconds": opened["seconds"], "created": opened["created"], "added": opened["added"],
            "unique_removed": sorted(set(before["unique"]) - set(after["unique"])),
            "may_be_empty_now": sorted(set(before["notnull"]) - set(after["notnull"])),
            "lost": {t: {"before": a, "after": b} for t, (a, b) in lost.items()},
            "people": before["rows"].get("tracks", 0),
            "whats_new": [e["title"] for e in first_new.new_items(True)][::-1], "entries": len(ENTRIES)}


def waiting_for(db: str, run=subprocess.run) -> list[str]:
    """What the copy's connections are doing now (for a start-up update that does not finish)."""
    try:
        return psql("SELECT format('%s | %s | %s', state, coalesce(wait_event_type, '-'), left(query, 160)) "
                    f"FROM pg_stat_activity WHERE datname = '{COPY}'", "postgres", run, timeout=30)
    except (RuntimeError, subprocess.SubprocessError, OSError):
        return []


def drop_copy(run=subprocess.run) -> None:
    psql(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{COPY}'", "postgres", run)
    psql(f"DROP DATABASE IF EXISTS {COPY}", "postgres", run)


def copy_url(url: str) -> str:
    from sqlalchemy.engine import make_url
    return make_url(url).set(database=COPY).render_as_string(hide_password=False)


def describe(result: dict) -> list[str]:
    """The result in plain words, one line each."""
    if result.get("error"):
        out = [f"FAILED: {result['error']}"]
        out += [f"  waiting: {w}" for w in result.get("waiting", [])]
        return out
    out = [f"OK: the update took {result['seconds']:.1f} s on a copy of your database ({result['people']} people)."]
    if result["created"] or result["added"]:
        out.append(f"  added: {len(result['created'])} tables ({', '.join(result['created']) or '-'}), "
                   f"{len(result['added'])} columns ({', '.join(result['added']) or '-'})")
    else:
        out.append("  nothing to add: the database is up to date already")
    if result["unique_removed"] or result["may_be_empty_now"]:
        out.append(f"  old rules removed: {', '.join(result['unique_removed'] + result['may_be_empty_now'])}")
    out.append("  nothing lost: every table has as many rows as before" if not result["lost"]
               else f"  ROWS LOST: {result['lost']}")
    if result["whats_new"]:
        out.append(f"  What's new will list {len(result['whats_new'])}: {', '.join(result['whats_new'])}")
    return out


def main(argv=None, run=subprocess.run) -> int:
    ap = argparse.ArgumentParser(description="Open a copy of the real database with today's code (read only).")
    ap.add_argument("--source", default=backup_db.DB, help="the database to copy (default: the app's)")
    ap.add_argument("--timeout", type=float, default=300, help="seconds the update may take (default 300)")
    ap.add_argument("--keep", action="store_true", help=f"keep the copy ({COPY}) afterwards")
    ap.add_argument("--report", type=Path, default=REPORT, help="where to write the result (JSON)")
    args = ap.parse_args(argv)
    from labeler.config import load_config
    result = {"at": datetime.now().isoformat(timespec="seconds"), "source": args.source}
    t0 = time.monotonic()
    touched = False   # the server was reached and the copy may exist: drop it at the end
    try:
        try:
            docker = backup_db.docker()
        except SystemExit:
            print("Docker is not installed on this PC.")
            return 2
        if run([docker, "inspect", "-f", "{{.State.Running}}", backup_db.CONTAINER],
               capture_output=True, text=True).stdout.strip() != "true":
            print("The database container is not running: start Docker Desktop (start.bat does) and try again.")
            return 2
        touched = True
        drop_copy(run)   # one left by an earlier run
        psql(f"CREATE DATABASE {COPY}", "postgres", run)
        print(f"Copying {args.source} into {COPY} (inside the database container)...")
        r = backup_db.in_container(["sh", "-c", f"pg_dump -U {backup_db.USER} -d {args.source} -Fc | pg_restore -U "
                                    f"{backup_db.USER} -d {COPY} --no-owner"], run, capture_output=True, text=True,
                                   timeout=1800)
        if r.returncode != 0 and "error" in (r.stderr or "").lower():
            raise RuntimeError("copying failed: " + r.stderr.strip()[-300:])
        result["copy_s"] = round(time.monotonic() - t0, 1)
        before = layout(COPY, run)
        url = copy_url(load_config()["database"]["url"])
        print("Opening the copy with today's code...")
        child = subprocess.Popen([sys.executable, "-c", OPEN_CODE, url, str(ROOT)], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        try:
            out, err = child.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            result["waiting"] = waiting_for(run)
            child.kill()
            child.communicate()
            raise RuntimeError(f"the update did not finish within {args.timeout:.0f} s (it would hang at start)")
        if child.returncode != 0:
            raise RuntimeError("the update failed: " + (err.strip().splitlines() or ["?"])[-1][:300])
        opened = json.loads(out.strip().splitlines()[-1])
        result.update(compare(before, layout(COPY, run), opened))
        result["ok"] = not result["lost"]
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as e:
        result.update(ok=False, error=str(e))
    finally:
        if touched and not args.keep:
            try:
                drop_copy(run)
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:
                print(f"(could not drop {COPY}: {e})")
    result["total_s"] = round(time.monotonic() - t0, 1)
    for line in describe(result):
        print(line)
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=1), encoding="utf-8")
    except OSError as e:
        print(f"(could not write {args.report}: {e})")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

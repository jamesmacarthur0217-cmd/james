"""First-run check (Phase 36 step 2): after a big update, one page says whether everything works and what to do.

The page (static/first-run.html, admins) reads the Health check and each camera's picture check itself; this module
adds what only the server knows: the database has every table and column of this version (and what this start
added), the built-in alert rules exist, a recording was made today, and the Home / Away state. Each item is
{"key", "title", "ok": True / False / None (for information), "detail", "todo", "link"}.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import inspect

from . import alerts as alerts_mod
from .db import NEW_COLUMNS, Base


FRESH = "tracks"   # made at this start: a new database


def _item(key, title, ok, detail, todo="", link="") -> dict:
    return {"key": key, "title": title, "ok": ok, "detail": detail, "todo": todo, "link": link}


def check_database(db) -> dict:
    insp = inspect(db.engine)
    tables = set(insp.get_table_names())
    missing = sorted(set(Base.metadata.tables) - tables)
    for table, columns in NEW_COLUMNS.items():
        if table in tables:
            have = {c["name"] for c in insp.get_columns(table)}
            missing += [f"{table}.{c}" for c in columns if c not in have]
    created, added = getattr(db, "created_tables", set()), getattr(db, "added_columns", set())
    if missing:
        return _item("database", "Database updated", False, "missing: " + ", ".join(missing[:8]),
                     "Restart the app (stop.bat, then start.bat); it updates the database at start. If this stays, "
                     "look at the log file (data/logs) for a database error.", "/health")
    if FRESH in created:
        detail = "a new database, made at this start"
    elif created or added:
        what = []
        if created:
            what.append(f"{len(created)} new {'table' if len(created) == 1 else 'tables'}")
        if added:
            what.append(f"{len(added)} new {'column' if len(added) == 1 else 'columns'}")
        detail = "updated at this start: " + " and ".join(what)
    else:
        detail = "up to date (nothing had to be added at this start)"
    return _item("database", "Database updated", True, detail)


BUILT_IN = [   # (title, config section that turns it on, test on a rule's conditions)
    ("Fall", "falls", alerts_mod.is_fall_rule),
    ("Help sign", "help_sign", alerts_mod.is_help_rule),
    ("No movement", "quiet", alerts_mod.is_quiet_rule),
]


def check_rules(cfg: dict, db) -> list[dict]:
    rules = db.list_rules()
    out = []
    for title, section, test in BUILT_IN:
        key = "rule_" + section
        name = f'Alert rule "{title}"'
        if not (cfg.get(section) or {}).get("enabled"):
            out.append(_item(key, name, None, "turned off on the Settings page", link="/settings"))
            continue
        found = [r for r in rules if test(r["conditions"])]
        if not found:
            out.append(_item(key, name, False, "missing", "Restart the app: it adds the rule at start.", "/alerts"))
        elif not any(r["enabled"] for r in found):
            out.append(_item(key, name, None, "there, but turned off on the Alerts page", link="/alerts"))
        else:
            out.append(_item(key, name, True, "on"))
    return out


def check_recording(cfg: dict, db, pipelines, now: datetime | None = None) -> dict:
    rcfg = cfg["recording"]
    if not rcfg.get("enabled"):
        return _item("recording", "Recording today", None, "recording is turned off on the Settings page",
                     link="/settings")
    now = now or datetime.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    files, latest, now_on = 0, "", False
    for p in pipelines:
        for r in db.list_recordings(p.name, start, start + timedelta(days=1)):
            files += 1
            now_on = now_on or r["ended_at"] is None
            latest = max(latest, r["ended_at"] or "")
    if files:
        detail = f"{files} {'file' if files == 1 else 'files'} today, " + (
            "recording now" if now_on else f"the newest until {latest[11:16]}")
        return _item("recording", "Recording today", True, detail, link="/recordings")
    todo = ("Stand in front of a camera for a few seconds (recording starts when someone is in view), then press "
            "Check again." if rcfg.get("when") == "people" else
            "Wait a minute, then press Check again. If it stays, look at the Health page's recording line.")
    return _item("recording", "Recording today", False, "no recording yet today", todo, "/recordings")


def check_arming(arming) -> dict:
    if arming is None:
        return _item("arming", "Home / Away", None, "not available in this mode")
    s = arming.state()
    detail = ("Away" if s["mode"] == "away" else "Home") + (f" since {s['since'][:16].replace('T', ' ')}"
                                                            if s["since"] else "")
    if s["schedule"]:
        detail += "; weekly Away schedule: " + ", ".join(s["schedule"])
    return _item("arming", "Home / Away", None, detail + ". Alert rules marked \"only when Away\" work only then.",
                 link="/settings")


def server_checks(cfg: dict, db, pipelines, arming, now: datetime | None = None) -> dict:
    items = [check_database(db), *check_rules(cfg, db), check_recording(cfg, db, pipelines, now),
             check_arming(arming)]
    cameras = [{"name": p.name, "mode": p.mode, "help_sign": getattr(p, "gestures", None) is not None}
               for p in pipelines]
    return {"items": items, "cameras": cameras}

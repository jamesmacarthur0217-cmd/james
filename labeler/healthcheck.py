"""The health check: is everything the app needs working right now? (GET /api/health and the Health page.)

Each check is "ok", "warning" (worth a look, nothing is broken: an old backup, errors in the log), "problem"
(something does not work: the database, Ollama, a camera, a full disk) or "off" (turned off in config.yaml).
The answer is HTTP 200 with no problems and 503 with any, so a monitoring tool (Uptime Kuma, a phone app, a
script) can alert on the status code alone.

Camera addresses are never included (they can hold a password), and the error messages from the log only for
admins: a viewer or a monitoring tool sees what is wrong, not the details.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

from . import logs
from .config import ROOT, mask_password, resolve

BACKUP_WARN_DAYS = 7          # stop.bat backs up at most daily; a week without one is worth a look
BACKUP_DIR = "data/backups"


def _version() -> str | None:
    """The git commit the app runs (None on a copy without git)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=3)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


VERSION = _version()


def _check(name: str, level: str, detail: str) -> dict:
    return {"name": name, "level": level, "detail": detail}


def check_database(db) -> dict:
    try:
        ms = db.ping()
        return _check("database", "ok", f"answers in {ms:.1f} ms" if ms < 10 else f"answers in {ms:.0f} ms")
    except Exception as e:
        return _check("database", "problem", f"the database does not answer ({e.__class__.__name__}); "
                                              "is Docker Desktop running? start.bat starts it")


def check_ollama(cfg: dict, pipelines: list) -> dict:
    """Ollama reachable, the model pulled, and the app actually using it (it is only connected at start)."""
    vcfg = cfg["vlm"]
    if not vcfg["enabled"]:
        return _check("vision model (Ollama)", "off", "turned off (vlm.enabled); labels come from colors and "
                                                      "the fast classifiers")
    from .vlm import VLMClient
    ok, reason = VLMClient(vcfg, cfg["mode"]).available()
    if not ok:
        return _check("vision model (Ollama)", "problem", reason)
    try:
        version = requests.get(f"{vcfg['host'].rstrip('/')}/api/version", timeout=3).json().get("version", "?")
    except (requests.RequestException, ValueError):
        version = "?"
    if pipelines and not any(p.vlm is not None for p in pipelines):
        return _check("vision model (Ollama)", "problem",
                      f"Ollama {version} is running now, but it was not when the app started, so people get color "
                      "labels only; restart the app")
    return _check("vision model (Ollama)", "ok", f"Ollama {version}, model {vcfg['model']}")


def check_cameras(cfg: dict, pipelines: list, camera_health, now: float) -> list[dict]:
    current = camera_health.current() if camera_health is not None else []
    out = []
    for p in pipelines:
        name = f"camera {p.name}"
        since = max(0.0, now - p.last_frame_at)
        problems = [e["message"] for e in current if e["camera"] == p.name]
        if not p.running:
            out.append(_check(name, "problem", "not running" + (" (it crashed; see the log)" if p.crashed else "")))
        elif problems:
            out.append(_check(name, "problem", "; ".join(problems)))
        elif since > cfg["health"]["offline_after_s"]:
            out.append(_check(name, "problem", f"no picture for {since:.0f} s"))
        else:
            out.append(_check(name, "ok", f"{p.fps:.1f} fps, last picture {since:.1f} s ago, "
                                          f"{len(p.tracks)} in view"))
    return out


def check_disk(cfg: dict) -> dict:
    path = resolve(cfg["storage"]["crop_dir"])
    free_gb = shutil.disk_usage(path).free / 1e9
    minimum = cfg["storage"]["min_free_gb"]
    if free_gb < minimum:
        return _check("disk", "problem", f"only {free_gb:.1f} GB free (below storage.min_free_gb {minimum} GB): "
                                         "delete old people or lower storage.retention_days")
    return _check("disk", "ok", f"{free_gb:.0f} GB free")


def check_backup(db, now: float, folder: Path | None = None) -> dict:
    if not db.is_postgres:
        return _check("database backup", "off", "only for the PostgreSQL database (stop.bat makes them)")
    folder = folder or resolve(BACKUP_DIR)
    newest = max(folder.glob("visionlabeler-*.dump"), key=lambda p: p.stat().st_mtime, default=None)
    if newest is None:
        return _check("database backup", "warning", "no backup yet; stop.bat makes one when it stops the app")
    age_d = (now - newest.stat().st_mtime) / 86400
    level = "warning" if age_d > BACKUP_WARN_DAYS else "ok"
    detail = f"newest is {age_d:.1f} days old ({newest.name})"
    if level == "warning":
        detail += "; stop.bat makes a new one"
    return _check("database backup", level, detail)


def check_retraining(cfg: dict, db) -> dict:
    if not cfg["retrain"]["enabled"]:
        return _check("weekly retraining", "off", "turned off (retrain.enabled)")
    history = db.list_model_history(limit=1)
    if not history:
        return _check("weekly retraining", "ok", "has not run yet")
    last = history[0]
    return _check("weekly retraining", "ok", f"last run {last['created_at'].replace('T', ' ')}: "
                                             + ("new version kept" if last["kept"] else last["note"] or "not kept"))


def check_errors(now: float) -> tuple[dict, list[dict]]:
    errors = logs.recent_errors.recent(now)
    latest = [{"at": datetime.fromtimestamp(t).isoformat(timespec="seconds"), "message": mask_password(m)[:300]}
              for t, m in errors[-5:]][::-1]
    if not errors:
        return _check("errors in the log", "ok", "none in the last 24 hours"), latest
    return _check("errors in the log", "warning", f"{len(errors)} in the last 24 hours (newest first below; "
                                                  "all in data\\logs\\visionlabeler.log)"), latest


def run_checks(cfg: dict, db, pipelines: list, camera_health=None, started_at: float | None = None,
               detail: str = "full") -> dict:
    """detail: "full" (admins), "viewer" (no log messages) or "status" (monitoring token: problems only)."""
    now_mono, now = time.monotonic(), time.time()
    errors_check, latest_errors = check_errors(now)
    checks = [check_database(db), check_ollama(cfg, pipelines), *check_cameras(cfg, pipelines, camera_health, now_mono),
              check_disk(cfg), check_backup(db, now), check_retraining(cfg, db), errors_check]
    if cfg.get("_settings_error"):   # the app started with config.yaml only
        checks.append(_check("Settings page file", "warning", cfg["_settings_error"]))
    problems = [f"{c['name']}: {c['detail']}" for c in checks if c["level"] == "problem"]
    warnings = [f"{c['name']}: {c['detail']}" for c in checks if c["level"] == "warning"]
    out = {"ok": not problems, "checked_at": datetime.now().isoformat(timespec="seconds"),
           "problems": problems, "warnings": warnings}
    if detail == "status":
        return out
    out["app"] = {"version": VERSION, "uptime_s": round(now - started_at) if started_at else None}
    out["checks"] = checks
    if detail == "full":
        out["latest_errors"] = latest_errors
    return out

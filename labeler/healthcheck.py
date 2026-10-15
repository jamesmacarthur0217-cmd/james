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
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from . import backups, logs
from .config import ROOT, mask_password, resolve

# The newest backup is "worth a look" after this many days: with the daily backup the app makes one every night;
# without it only stop.bat makes them.
BACKUP_WARN_DAYS = {True: 2, False: 7}


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


STORAGE_CACHE_S = 600     # adding up thousands of files is not needed every 30 s
STORAGE_WARN_DAYS = 30
_storage_cache: dict = {}


def _size(paths) -> int:
    total = 0
    for f in paths:
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def _mb(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


def storage_use(cfg: dict) -> dict:
    """Bytes used by what the app stores, by kind."""
    crops = resolve(cfg["storage"]["crop_dir"])
    clips = crops / "clips"
    return {"pictures": _size(list(crops.glob("*.jpg")) + list((crops / "blurred").glob("*.jpg"))),
            "person videos": _size(clips.glob("person_*.webm")),
            "alert clips": _size(clips.glob("alert_*.webm")),
            "backups": _size(backups.folder(cfg).glob("*.dump")) if backups.folder(cfg).is_dir() else 0,
            "logs": _size(resolve(cfg["logs"]["dir"]).glob("*.log*"))}


def check_storage(cfg: dict, db, now_wall: float | None = None, use: dict | None = None) -> dict:
    """What the app's files use, and how many days of space are left at the last week's rate."""
    from datetime import timedelta
    now = datetime.fromtimestamp(now_wall) if now_wall else datetime.now()
    if use is None:
        cached = _storage_cache.get("value")
        if cached and time.monotonic() - _storage_cache["at"] < STORAGE_CACHE_S:
            use = cached
        else:
            use = storage_use(cfg)
            _storage_cache.update(value=use, at=time.monotonic())
    parts = ", ".join(f"{k} {_mb(v)}" for k, v in use.items() if v)
    people = db.count_tracks()
    week = db.count_tracks(now - timedelta(days=7))
    per_person = (use["pictures"] + use["person videos"]) / people if people else 0
    per_day = per_person * week / 7
    free = shutil.disk_usage(resolve(cfg["storage"]["crop_dir"])).free - cfg["storage"]["min_free_gb"] * 1e9
    if per_day <= 0:
        return _check("storage", "ok", (parts or "nothing stored yet") + "; no new people in the last 7 days")
    days = max(0, free) / per_day
    detail = (f"{parts}; about {_mb(int(per_person))} per person, {round(week / 7, 1):g} people a day (last 7 days): "
              f"space for about {days:,.0f} more days")
    retention = cfg["storage"]["retention_days"]
    if retention and days > retention:
        detail += f" (people are kept {retention} days, so it does not fill up)"
    elif days < STORAGE_WARN_DAYS:
        return _check("storage", "warning", detail + ": free space, lower storage.retention_days, or turn off "
                                                     "person videos (Settings)")
    return _check("storage", "ok", detail)


def check_backup(db, now: float, folder: Path | None = None, bcfg: dict | None = None) -> dict:
    """folder: backup.dir (backups.folder(cfg)); bcfg: the backup settings."""
    """The newest backup's age, the second copy (backup.copy_to) and the last restore check."""
    if not db.is_postgres:
        return _check("database backup", "off", "only for the PostgreSQL database")
    bcfg = bcfg or {"enabled": True, "hour": 2, "copy_to": ""}
    daily = bool(bcfg.get("enabled"))
    maker = f"the app makes one every day from {int(bcfg.get('hour', 2))}:00" if daily else \
        "stop.bat makes one when it stops the app (the daily backup is off)"
    folder = folder or backups.folder({"backup": bcfg})
    newest = backups.newest(folder)
    if newest is None:
        return _check("database backup", "warning", f"no backup yet; {maker}")
    age_d = (now - newest.stat().st_mtime) / 86400
    level = "warning" if age_d > BACKUP_WARN_DAYS[daily] else "ok"
    parts = [f"newest is {age_d:.1f} days old ({newest.name})"]
    if level == "warning":
        parts.append(maker + "; see the log if it failed")
    copy_to = bcfg.get("copy_to")
    if copy_to:
        if not (Path(copy_to) / newest.name).is_file():
            level = "warning"
            parts.append(f"no second copy of it in {copy_to} (drive not plugged in?)")
        else:
            parts.append(f"second copy in {copy_to}")
    verify = backups.last_verify(folder)
    if verify is None:
        parts.append("not restore-checked yet (weekly)")
    elif verify.get("ok"):
        parts.append(f"restore check {verify['at'][:10]} OK")
    else:
        level = "warning"
        parts.append(f"restore check {verify.get('at', '?')[:10]} FAILED: {verify.get('message', '')}")
    return _check("database backup", level, "; ".join(parts))


def check_retraining(cfg: dict, db) -> dict:
    if not cfg["retrain"]["enabled"]:
        return _check("weekly retraining", "off", "turned off (retrain.enabled)")
    history = db.list_model_history(limit=1)
    if not history:
        return _check("weekly retraining", "ok", "has not run yet")
    last = history[0]
    return _check("weekly retraining", "ok", f"last run {last['created_at'].replace('T', ' ')}: "
                                             + ("new version kept" if last["kept"] else last["note"] or "not kept"))


def check_recording(cfg: dict, db, pipelines: list) -> dict:
    """Recording (labeler/recorder.py): FFmpeg found, each camera's recorder running, not paused, space used."""
    rc = cfg["recording"]
    if not rc["enabled"]:
        return _check("recording", "off", "turned off (Settings page)")
    from . import recorder
    ffmpeg = recorder.ffmpeg_path(cfg)
    if not ffmpeg.exists():
        return _check("recording", "warning", f"on, but FFmpeg is not found at {ffmpeg}: nothing is recorded "
                                              "(install.bat sets it up, or set recording.ffmpeg)")
    recorders = [(p.name, getattr(p, "recorder", None)) for p in pipelines]
    missing = [name for name, r in recorders if r is None]
    if recorders and len(missing) == len(recorders):
        return _check("recording", "warning", "on, but not recording: no working H.264 encoder was found "
                                              "(see the log at start)")
    stats = db.recording_stats()
    gb, oldest = stats["bytes"] / 1e9, stats["oldest"]
    detail = (f"{stats['files']} files, {gb:.1f} of {float(rc['max_gb']):g} GB, kept {float(rc['keep_days']):g} days"
              + (f", oldest {oldest.replace('T', ' ')[:16]}" if oldest else "")
              + f"; {'while people are in view' if rc['when'] == 'people' else 'always'}")
    encoders = sorted({r.encoder for _, r in recorders if r is not None})
    if encoders:
        detail += f", {', '.join(encoders)}"
    paused = [f"{name}: {r.paused}" for name, r in recorders if r is not None and r.paused]
    dropped = [f"{name}: {r.dropped} frames dropped" for name, r in recorders if r is not None and r.dropped]
    if paused:
        return _check("recording", "problem", "paused (" + "; ".join(paused) + "); " + detail)
    if missing or dropped:
        return _check("recording", "warning", "; ".join([*(f"{m}: not recording" for m in missing), *dropped])
                      + "; " + detail)
    return _check("recording", "ok", detail)


def check_receiver(cfg: dict, pipelines: list, rec=None, firewall: str | None = None, now: float | None = None) -> dict | None:
    """Cameras sent from other devices (labeler/receiver.py): MediaMTX running, who is sending now, firewall."""
    from . import receiver, remote
    pushed = [p for p in pipelines if receiver.is_push(getattr(p, "cam", {}))]
    if not pushed:
        return None
    rec = rec if rec is not None else receiver.CURRENT
    if rec is None or not rec.running():
        why = (rec.problem if rec is not None and rec.problem else "not running")
        return _check("cameras from other devices", "problem", f"the receiver is {why}")
    now = time.monotonic() if now is None else now
    # A frame really arrived lately (last_frame_at alone is also the loop's start time).
    sending = [p.name for p in pushed if getattr(p, "frames", 0) and now - getattr(p, "last_frame_at", -1e9) < 15]
    waiting = [p.name for p in pushed if p.name not in sending]
    detail = (("sending now: " + ", ".join(sending)) if sending else "none sending now") + \
             ("; waiting for: " + ", ".join(waiting) if waiting else "")
    ports = [int(cfg["receiver"]["rtsp_port"]), int(cfg["receiver"]["rtmp_port"])]
    if firewall is not None:
        missing = [port for port in ports if not remote.firewall_rules(port, firewall)]
        if missing:
            return _check("cameras from other devices", "warning", detail + "; the Windows firewall does not let "
                          "Tailscale devices reach port " + " and ".join(map(str, missing)) + " (README: 'A camera at "
                          "another site')")
    if rec.restarts:
        detail += f"; the receiver was started again {rec.restarts} time(s)"
    return _check("cameras from other devices", "ok", detail)


def check_vehicles(cfg: dict, db, pipelines: list) -> dict | None:
    """Vehicles (labeler/vehicles.py): which cameras look for them, the detector file, how many today."""
    vcfg = cfg.get("vehicles") or {}
    if not vcfg.get("enabled"):
        return _check("vehicles", "off", "turned off (Settings page)")
    missing = [p for p in pipelines if getattr(p, "vehicles_missing", None)]
    if missing:
        return _check("vehicles", "warning", f"the vehicle detector {missing[0].vehicles_missing} is missing, so "
                      f"{', '.join(p.name for p in missing)} sees no vehicles: run install.bat again or "
                      "python tools/models.py download")
    on = [p.name for p in pipelines if getattr(p, "vehicles", None) is not None]
    if not on:
        return _check("vehicles", "off", "no camera looks for vehicles (on by default for security cameras; "
                                         "per camera on the Settings page)")
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    n = len(db.vehicle_rows(today, datetime.now() + timedelta(minutes=1)))
    return _check("vehicles", "ok", f"{', '.join(on)}: {n} {'vehicle' if n == 1 else 'vehicles'} today"
                                    + ("" if vcfg.get("describe", True) else "; not described (vehicles.describe off)"))


def check_animals(cfg: dict, db, pipelines: list) -> dict | None:
    """Animals (labeler/animals.py): which cameras look for them, the detector file (the vehicles'), how many today."""
    acfg = cfg.get("animals") or {}
    if not acfg.get("enabled"):
        return _check("animals", "off", "turned off (Settings page)")
    missing = [p for p in pipelines if getattr(p, "vehicles_missing", None)]
    if missing:
        return _check("animals", "warning", f"the vehicle detector {missing[0].vehicles_missing} is missing, so "
                      f"{', '.join(p.name for p in missing)} sees no animals: run install.bat again or "
                      "python tools/models.py download")
    on = [p.name for p in pipelines if getattr(p, "animals", None) is not None]
    if not on:
        return _check("animals", "off", "no camera looks for animals (on by default for security cameras; "
                                        "per camera on the Settings page)")
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    n = len(db.animal_rows(today, datetime.now() + timedelta(minutes=1)))
    return _check("animals", "ok", f"{', '.join(on)}: {n} {'animal' if n == 1 else 'animals'} today "
                                   f"({', '.join(acfg.get('kinds') or [])})")


def check_scene(cfg: dict, db, now: datetime | None = None, arming=None) -> dict | None:
    """Scene questions (labeler/scene.py): how many, their last answers, and any that are overdue (not answered in
    twice their interval: the vision model busy all the time, the camera off, or the vision model down). A question
    outside its limits (only when Away, hours) is never late: it is not asked then."""
    scfg = cfg.get("scene") or {}
    questions = [q for q in db.list_scene_questions() if q["enabled"]]
    if not scfg.get("enabled"):
        return _check("scene questions", "off", "turned off (Settings page)") if questions else None
    if not questions:
        return None   # nothing asked: no line
    if not cfg["vlm"].get("enabled"):
        return _check("scene questions", "warning", f"{len(questions)} questions, but the vision model is turned off "
                                                    "(vlm.enabled): none are asked")
    from .arming import AWAY, HOME
    from .scene import asked_now, limits_since
    now = now or datetime.now()
    state = arming.state() if arming is not None else {"mode": HOME, "since": None}
    away_since = datetime.fromisoformat(state["since"]) if state["mode"] == AWAY and state["since"] else None

    def ago(q) -> float:
        return (now - datetime.fromisoformat(q["last_at"] or q["created_at"])).total_seconds()

    def is_late(q) -> bool:
        if not asked_now(q, now, state["mode"]):
            return False
        start = max(filter(None, [datetime.fromisoformat(q["last_at"] or q["created_at"]),
                                  limits_since(q, now, away_since)]))
        return (now - start).total_seconds() > 2 * q["every_s"] + 60

    late = [q for q in questions if is_late(q)]
    answers = "; ".join(f"{q['question']} {q['last_answer'] or 'not asked yet'}"
                        + (f" ({int(ago(q) // 60)} min ago)" if q["last_at"] else "") for q in questions[:3])
    detail = f"{len(questions)} {'question' if len(questions) == 1 else 'questions'}: {answers}"
    if late:
        return _check("scene questions", "warning", f"{len(late)} not answered for a while ({', '.join(q['question'] for q in late[:3])}): "
                      "the vision model may be busy all the time, or the camera or the vision model is down. " + detail)
    return _check("scene questions", "ok", detail)


def check_picture(cfg: dict, db, pipelines: list, now: datetime | None = None) -> dict | None:
    """Phase 30: what is wrong with each camera's picture (labeler/picture_check.py), when it lasts; on kiosk cameras
    also whether faces are seen (people of the last 7 days)."""
    if not (cfg.get("picture_check") or {}).get("enabled", True):
        return None
    from .picture_check import ADVICE, faces
    now = now or datetime.now()
    notes, checked = [], []
    for p in pipelines:
        watch = getattr(p, "picture", None)
        if watch is None:
            continue
        state = watch.state()
        mine = [f"{ADVICE[c]} (since {since[11:16]})" for c, since in state["lasting"].items()]
        if getattr(p, "mode", "") == "kiosk":
            found, n = faces(db.people_between(now - timedelta(days=7), now, p.name, limit=200, newest_first=True))
            mine += [f"{ADVICE[c]} ({n['facing_share']:.0%} of the time facing the camera, ages for "
                     f"{n['ages_share']:.0%} of the last {n['people']} people)" for c in found]
        if mine:
            notes.append(f"camera {p.name}: " + " ".join(mine))
        if state["checked_at"]:
            checked.append(state["checked_at"][11:16])
    if notes:
        return _check("camera picture", "warning", "; ".join(notes))
    if not checked:
        return _check("camera picture", "ok", "not checked yet (every 30 s once the cameras run)")
    return _check("camera picture", "ok", f"fine (last checked {max(checked)}; Settings page: Check the picture)")


def check_quiet(quiet) -> dict | None:
    """Phase 31: the "no movement" watch (labeler/quiet.py): how long each watched camera has been quiet."""
    if quiet is None:
        return None
    items = quiet.state()
    if not items:
        return _check("no movement", "off", "no rule asks for it, or no kiosk camera to watch (Alerts page)")
    from .alerts import nice_seconds

    several = len({i["rule"] for i in items}) > 1

    def one(i):
        who = f"{i['camera']} ({i['rule']})" if several else i["camera"]
        if "resting" in i:
            return f"{who}: not counted now ({i['resting']})"
        m = i["quiet_s"] // 60
        took = f"{m // 60} h {m % 60} min" if m >= 60 else f"{m} min"
        return (f"{who}: nobody seen for {took} (alert after {nice_seconds(i['after_s'])})"
                + (", alerted" if i["alerted"] else ""))
    level = "warning" if any(i["alerted"] for i in items) else "ok"
    return _check("no movement", level, "; ".join(one(i) for i in items))


def check_checks(db) -> dict | None:
    """Phase 24: what the admins' checks on the Check page say (no line before the first check)."""
    from . import checks
    line = checks.summary_line(db.list_checks())
    return _check("checked by you", "ok", line + " (Check page)") if line else None


def check_arming(arming) -> dict:
    """Home / Away (labeler/arming.py): never a problem, but good to see from another device."""
    s = arming.state()
    nice = {"home": "Home", "away": "Away"}
    detail = nice[s["mode"]]
    if s["since"]:
        detail += f" since {s['since'].replace('T', ' ')[:16]} ({'schedule' if s['why'] == 'schedule' else s['by']})"
    detail += (f"; schedule: {', '.join(s['schedule'])}, next {nice[s['next']['mode']]} at "
               f"{s['next']['at'].replace('T', ' ')[:16]}" if s["next"] else
               f"; schedule: {', '.join(s['schedule'])}" if s["schedule"] else "; no weekly schedule (Settings page)")
    return _check("Home / Away", "ok", detail)


def check_errors(now: float) -> tuple[dict, list[dict]]:
    errors = logs.recent_errors.recent(now)
    latest = [{"at": datetime.fromtimestamp(t).isoformat(timespec="seconds"), "message": mask_password(m)[:300]}
              for t, m in errors[-5:]][::-1]
    if not errors:
        return _check("errors in the log", "ok", "none in the last 24 hours"), latest
    return _check("errors in the log", "warning", f"{len(errors)} in the last 24 hours (newest first below; "
                                                  "all in data\\logs\\visionlabeler.log)"), latest


def run_checks(cfg: dict, db, pipelines: list, camera_health=None, started_at: float | None = None,
               detail: str = "full", arming=None, quiet=None) -> dict:
    """detail: "full" (admins), "viewer" (no log messages) or "status" (monitoring token: problems only)."""
    now_mono, now = time.monotonic(), time.time()
    errors_check, latest_errors = check_errors(now)
    checks = [check_database(db), check_ollama(cfg, pipelines), *check_cameras(cfg, pipelines, camera_health, now_mono),
              check_disk(cfg), check_backup(db, now, backups.folder(cfg), cfg["backup"]), check_retraining(cfg, db), errors_check]
    try:
        checks.insert(-1, check_storage(cfg, db))
    except Exception as e:   # never break the health check
        checks.insert(-1, _check("storage", "warning", f"could not be measured ({e.__class__.__name__})"))
    try:
        from . import remote
        line = check_receiver(cfg, pipelines, firewall=remote._system(subprocess.run)[1] if sys.platform == "win32" else None)
        if line is not None:
            checks.insert(-1, line)
    except Exception as e:
        checks.insert(-1, _check("cameras from other devices", "warning", f"could not be checked ({e.__class__.__name__})"))
    try:
        line = check_vehicles(cfg, db, pipelines)
        if line is not None:
            checks.insert(-1, line)
    except Exception as e:
        checks.insert(-1, _check("vehicles", "warning", f"could not be checked ({e.__class__.__name__})"))
    try:
        line = check_checks(db)
        if line is not None:
            checks.insert(-1, line)
    except Exception as e:
        checks.insert(-1, _check("checked by you", "warning", f"could not be read ({e.__class__.__name__})"))
    try:
        line = check_quiet(quiet)
        if line is not None:
            checks.insert(-1, line)
    except Exception as e:
        checks.insert(-1, _check("no movement", "warning", f"could not be checked ({e.__class__.__name__})"))
    try:
        line = check_picture(cfg, db, pipelines)
        if line is not None:
            checks.insert(-1, line)
    except Exception as e:
        checks.insert(-1, _check("camera picture", "warning", f"could not be checked ({e.__class__.__name__})"))
    try:
        line = check_scene(cfg, db, arming=arming)
        if line is not None:
            checks.insert(-1, line)
    except Exception as e:
        checks.insert(-1, _check("scene questions", "warning", f"could not be checked ({e.__class__.__name__})"))
    try:
        line = check_animals(cfg, db, pipelines)
        if line is not None:
            checks.insert(-1, line)
    except Exception as e:
        checks.insert(-1, _check("animals", "warning", f"could not be checked ({e.__class__.__name__})"))
    try:
        checks.insert(-1, check_recording(cfg, db, pipelines))
    except Exception as e:
        checks.insert(-1, _check("recording", "warning", f"could not be checked ({e.__class__.__name__})"))
    if arming is not None:
        try:
            checks.insert(-1, check_arming(arming))
        except Exception as e:
            checks.insert(-1, _check("Home / Away", "warning", f"could not be checked ({e.__class__.__name__})"))
    try:   # access from other devices and Tailscale (labeler/remote.py); never breaks the health check
        from . import remote
        checks[-1:-1] = remote.checks(cfg, db)   # before "errors in the log"
    except Exception as e:
        checks.insert(-1, _check("other devices", "warning", f"could not be checked ({e.__class__.__name__})"))
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

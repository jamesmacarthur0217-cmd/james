"""Local web UI: live view, people list with search, and the human review queue."""
from __future__ import annotations

import asyncio
import logging
import re
import secrets
import threading
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool

from . import accounts, healthcheck, privacy, reports, schema
from . import settings as settings_mod
from . import timeline as timeline_mod
from .clips import clip_name
from .config import ROOT, load_config, mask_password, resolve
from .db import Database
from .pipeline import Pipeline

STATIC = ROOT / "labeler" / "static"
log = logging.getLogger(__name__)
CLIP_NAME = re.compile(r"(alert_\d+|person_\d+_\d+)\.webm")   # alert clips and person videos (admins)


class ReviewIn(BaseModel):
    value: str


class DeleteIn(BaseModel):
    ids: list[int] = []
    everything: bool = False


class RuleIn(BaseModel):
    name: str
    camera: str | None = None
    conditions: list[list[str]] = []


class RuleUpdate(BaseModel):
    enabled: bool


class ArmingIn(BaseModel):
    mode: str


class HelpTestIn(BaseModel):
    camera: str


class SceneIn(BaseModel):
    camera: str
    question: str
    every_min: int = 5
    alert_on: str = "yes"
    zone_id: int | None = None    # only this area of the picture (an area drawn on the Live page)
    only_away: bool = False       # asked only when the house is set to Away
    hours: str | None = None      # asked only then, e.g. "22-06"


class CheckIn(BaseModel):
    kind: str
    item_id: int
    right: str
    correction: str | None = None


class SceneUpdate(BaseModel):
    enabled: bool | None = None
    every_min: int | None = None
    zone_id: int | None = None    # 0 = the whole picture again
    only_away: bool | None = None
    hours: str | None = None      # "" = all day again


class ZoneIn(BaseModel):
    camera: str
    name: str
    kind: str                     # area | line
    points: list[list[float]]
    lying_ok: bool = False        # an area where lying is normal (a bed, a sofa): no fall alerts there
    role: str = ""                # what the area is (labeler/zones.py ROLES); "bed" = lying_ok
    max_people: int | None = None  # "at most N people" (crowding, labeler/safety.py)


class ZoneUpdate(BaseModel):
    role: str | None = None       # "" = no role; left out = unchanged
    max_people: int | None = None  # 0 = no limit; left out = unchanged


class AnswerIn(BaseModel):
    labels: dict[str, str]


class AskIn(BaseModel):
    question: str
    limit: int = 30


class ImportPath(BaseModel):
    path: str


class ImportFile(BaseModel):
    path: str
    start: str                    # 2026-10-09T14:30:12


class ImportStart(BaseModel):
    files: list[ImportFile]
    camera: str
    mode: str = "security"
    keep_days: int = 0            # 0 = until deleted by hand
    target_fps: float = 10
    imgsz: int | None = None      # detector picture size: 640, 960 or 1280 (None: detector.imgsz)


class CameraName(BaseModel):
    camera: str


def _day_range(start: str | None, end: str | None) -> tuple[datetime, datetime]:
    """Dates as YYYY-MM-DD (end inclusive); default today."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        s = datetime.fromisoformat(start) if start else today
        e = datetime.fromisoformat(end) if end else s
    except ValueError:
        raise HTTPException(400, "dates must look like 2026-10-07")
    if e < s:
        raise HTTPException(400, "end is before start")
    return s, e + timedelta(days=1)


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
LOCAL_USER = {"username": "this PC", "role": "admin", "local": True}
# A monitoring tool calling /api/health?token=... (web.health_token): that one request, status only.
MONITOR_USER = {"username": "health monitor", "role": "monitor"}
# Headers that forwarding tools on this PC add (reverse proxies, Cloudflare Tunnel, Tailscale Serve/Funnel).
# Such a request only looks local; the person is somewhere else, so they must log in.
PROXY_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip", "cf-connecting-ip", "true-client-ip",
                 "tailscale-user-login", "x-forwarded-host")


def is_this_pc(request: Request) -> bool:
    client = request.client.host if request.client else ""
    return client in LOCAL_HOSTS and not any(h in request.headers for h in PROXY_HEADERS)
COOKIE = "vl_session"


class LoginIn(BaseModel):
    username: str
    password: str


def check_access(web: dict, n_users: int) -> None:
    """While nobody can log in, the app stays on this PC only, even if web.host would open it to the network
    (changes `web` in place). The camera keeps working; other devices simply can't connect yet."""
    if web["host"] not in LOCAL_HOSTS and n_users == 0:
        log.warning("web.host is %s, but there is no login account yet, so the app opens on this PC only. "
                    "Create one with: .venv\\Scripts\\python.exe tools\\users.py add <name> --role viewer "
                    "(then restart the app)", web["host"])
        web["host"] = "127.0.0.1"


def ensure_first_admin(db: Database, web: dict) -> None:
    """Older setups had one shared password (VL_WEB_PASSWORD): it becomes the first admin account."""
    if web.get("password") and db.count_users() == 0:
        db.add_user(web.get("username") or "admin", accounts.hash_password(web["password"]), "admin")


def create_app(cfg: dict, db: Database, pipelines: Pipeline | list[Pipeline] | None, embedder=None,
               alerts=None, agent=None, health=None, retrainer=None, importer=None, photo=None,
               arming=None) -> FastAPI:
    app = FastAPI(title="VisionLabeler")
    if arming is None:   # one shared with the alert checks, so a switch here counts there at once
        from .arming import Arming
        arming = getattr(alerts, "arming", None) or Arming(db, cfg)
    app.state.stopping = threading.Event()
    app.state.started_at = time.time()
    health_token = str(cfg["web"].get("health_token") or "")
    if health_token and len(health_token) < 16:
        log.warning("web.health_token is short (%d characters); use a long random one, e.g. 32 characters",
                    len(health_token))
    session_hours = cfg["web"].get("session_hours", 12)
    limiter = accounts.LoginLimiter()

    def audit(request: Request, action: str, detail: str = "", username: str | None = None) -> None:
        """Record who did what (shown on the admin-only Audit log page)."""
        user = getattr(request.state, "user", None) or {}
        db.add_audit(username or user.get("username", "?"), user.get("role", ""),
                     request.client.host if request.client else "", action, detail)

    def who(request: Request) -> str:
        return (getattr(request.state, "user", None) or {}).get("username", "?")

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        """This PC itself (stop.bat, the local browser) is always admin. Other devices must log in, and a
        viewer may only do what accounts.VIEWER lists. A request passed on by a forwarding tool on this PC
        looks local but is not (is_this_pc), so it must log in too."""
        method, path = request.method, request.url.path
        if path == "/api/health" and "token" in request.query_params:
            given = request.query_params["token"].encode()
            if not health_token or not secrets.compare_digest(given, health_token.encode()):
                return JSONResponse({"detail": "wrong health token"}, status_code=401)
            request.state.user = MONITOR_USER
            return await call_next(request)
        page = method == "GET" and not path.startswith(("/api/", "/stream", "/crops/", "/clips/", "/static/",
                                                         "/recordings/"))
        if is_this_pc(request):
            request.state.user = LOCAL_USER
        else:
            user = await run_in_threadpool(db.session_user, request.cookies.get(COOKIE, ""))
            request.state.user = user
            if user is None:
                if not accounts.is_public(method, path):
                    if page:
                        return RedirectResponse("/login", status_code=303)
                    return JSONResponse({"detail": "login required"}, status_code=401)
            elif not accounts.allowed(user["role"], method, path):
                if page:
                    return RedirectResponse("/", status_code=303)   # a viewer opening an admin page
                return JSONResponse({"detail": "your account may not do this"}, status_code=403)
        response = await call_next(request)
        # Pages, scripts and styles: always ask whether there is a newer version (an unchanged file is a
        # quick "not modified"), so a browser never keeps showing a page from before an update.
        if (page or path.startswith("/static/")) and "cache-control" not in response.headers:
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/login")
    def login_page():
        return FileResponse(STATIC / "login.html")

    @app.post("/api/login")
    def login(body: LoginIn, request: Request, response: Response):
        address = request.client.host if request.client else ""
        name = body.username.strip()
        wait = limiter.blocked_for(address)
        if wait:
            audit(request, "login blocked", "too many wrong passwords", username=name)
            raise HTTPException(429, f"Too many wrong passwords. Try again in {int(wait // 60) + 1} min.")
        user = db.get_user(name)
        # A wrong name is checked against a dummy hash, so it takes as long as a wrong password.
        ok = accounts.verify_password(body.password, user["password_hash"] if user else accounts.DUMMY_HASH)
        if not (ok and user):
            limiter.failed(address)
            audit(request, "login failed", "" if user else "no such user", username=name)
            raise HTTPException(401, "Wrong user name or password.")
        limiter.succeeded(address)
        request.state.user = {"username": user["username"], "role": user["role"]}
        audit(request, "login")
        token = db.create_session(user["id"], session_hours)
        response.set_cookie(COOKIE, token, max_age=int(session_hours * 3600), httponly=True, samesite="strict",
                            secure=request.url.scheme == "https")
        return {"username": user["username"], "role": user["role"]}

    @app.post("/api/logout")
    def logout(request: Request, response: Response):
        token = request.cookies.get(COOKIE)
        if token:
            db.end_session(token)
            audit(request, "logout")
        response.delete_cookie(COOKIE)
        return {"ok": True}

    @app.get("/api/me")
    def me(request: Request):
        user = request.state.user
        return {**user, "local": user.get("local", False)} if user else None

    if pipelines is None:
        pipelines = []
    elif isinstance(pipelines, Pipeline):
        pipelines = [pipelines]
    by_name = {p.name: p for p in pipelines}

    def pick(camera: str | None) -> Pipeline | None:
        if camera:
            if camera not in by_name:
                raise HTTPException(404, f"no camera named {camera!r}")
            return by_name[camera]
        return pipelines[0] if pipelines else None
    crop_dir = resolve(cfg["storage"]["crop_dir"])
    crop_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def blur_for(request: Request) -> bool:
        """Viewers see blurred faces (web.blur_faces_for_viewers); admins see faces."""
        user = getattr(request.state, "user", None) or {}
        return cfg["web"].get("blur_faces_for_viewers", True) and user.get("role") != "admin"

    from . import recordings_web
    recordings_web.add_routes(app, cfg, db, audit, blur_for, agent, embedder, photo)   # the Recordings page (labeler/recorder.py)

    # Same address, different picture for admins and viewers: never let a shared browser cache mix them up.
    PICTURE_HEADERS = {"Cache-Control": "private, max-age=60", "Vary": "Cookie"}

    @app.get("/crops/{name}")
    def picture(name: str, request: Request):
        path = crop_dir / name
        if not privacy.PICTURE_NAME.fullmatch(name) or not path.is_file():
            raise HTTPException(404, "no such picture")
        if blur_for(request):
            blurred = privacy.blurred_path(crop_dir, name)
            if blurred.is_file():
                return FileResponse(blurred, headers=PICTURE_HEADERS)
            # Saved before blurring existed (or a face close-up): the face position is unknown, blur it all.
            image = cv2.imread(str(path))
            if image is None:
                raise HTTPException(404, "no such picture")
            return Response(privacy.encode_jpeg(privacy.blur_whole(image)), media_type="image/jpeg",
                            headers=PICTURE_HEADERS)
        return FileResponse(path, headers=PICTURE_HEADERS)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/review")
    def review_page():
        return FileResponse(STATIC / "review.html")

    @app.get("/reports")
    def reports_page():
        return FileResponse(STATIC / "reports.html")

    @app.get("/alerts")
    def alerts_page():
        return FileResponse(STATIC / "alerts.html")

    @app.get("/answer-key")
    def answer_key_page():
        return FileResponse(STATIC / "answer-key.html")

    @app.get("/api/answer-key/next")
    def answer_key_next(skip: str = ""):
        exclude = [int(x) for x in skip.split(",") if x.strip().isdigit()]
        return {"person": db.next_for_answer_key(exclude)}

    @app.post("/api/answer-key/{track_id}")
    def answer_key_save(track_id: int, body: AnswerIn, request: Request):
        allowed = {**schema.FIELDS, **{n: v + ["none"] for n, v in schema.LIST_FIELDS.items()}}
        labels = {}
        for name, value in body.labels.items():
            if name not in allowed:
                raise HTTPException(400, f"unknown attribute {name}")
            if name in schema.LIST_FIELDS:
                items = sorted({v.strip() for v in value.split(",") if v.strip()} - {"none"})
                if any(i not in schema.LIST_FIELDS[name] for i in items):
                    raise HTTPException(400, f"unknown item in {value}")
                labels[name] = ",".join(items) or "none"
            elif value in allowed[name] and value != "unknown":
                labels[name] = value
            else:
                raise HTTPException(400, f"{value!r} is not a value of {name}")
        if not db.save_answer_key(track_id, labels, labeled_by=who(request)):
            raise HTTPException(404, "person not found")
        audit(request, "answer key", f"person #{track_id}")
        return {"ok": True}

    @app.get("/api/answer-key/score")
    def answer_key_score(camera: str | None = None):
        return db.answer_key_score(camera)

    @app.get("/api/answer-key/history")
    def accuracy_history(days: int = 365):
        return db.accuracy_history(min(days, 3650))

    # --- learning from the user's cameras (labeler/retrain.py) -----------------------------

    @app.get("/api/retrain")
    def retrain_status():
        return {"running": bool(retrainer and retrainer.running.is_set()), "available": retrainer is not None,
                "history": db.list_model_history(), "settings": {
                    k: cfg["retrain"][k] for k in ("enabled", "every_days", "min_own_people", "min_answer_key")}}

    @app.post("/api/retrain")
    def retrain_now(request: Request):
        if retrainer is None:
            raise HTTPException(503, "retraining runs only inside the app")
        if not retrainer.run_now():
            raise HTTPException(409, "a retraining is already running")
        audit(request, "retrain")
        return {"ok": True}

    # --- reports -------------------------------------------------------------------------

    def _checked(s, e, camera):
        """Checks made on these days (the Check page): right N of M per thing; one camera's, or all cameras'."""
        from . import checks
        rows = [c for c in db.list_checks(s, e) if not camera or c["camera"] == camera]
        return [r for r in checks.results(rows) if r["camera"] is None]

    @app.get("/api/reports/summary")
    def report_summary(request: Request, start: str | None = None, end: str | None = None,
                       camera: str | None = None):
        s, e = _day_range(start, end)
        audit(request, "view report", f"{s:%Y-%m-%d} to {(e - timedelta(days=1)):%Y-%m-%d}"
                                      + (f", camera {camera}" if camera else ""))
        return {**reports.summarize(db.people_between(s, e, camera), s, e, cfg["engagement"]["looked_s"]),
                "zones": db.zone_counts(s, e, camera),
                "vehicles": reports.summarize_vehicles(db.vehicle_rows(s, e, camera), s, e),
                "animals": reports.summarize_vehicles(db.animal_rows(s, e, camera), s, e),
                "checks": _checked(s, e, camera)}

    @app.get("/api/reports/export.csv")
    def report_csv(request: Request, start: str | None = None, end: str | None = None, camera: str | None = None):
        s, e = _day_range(start, end)
        people = db.people_between(s, e, camera)
        audit(request, "export CSV", f"{len(people)} people, {s:%Y-%m-%d} to {(e - timedelta(days=1)):%Y-%m-%d}"
                                     + (f", camera {camera}" if camera else ""))
        body = reports.to_csv(people)
        name = f"visionlabeler-{s:%Y%m%d}-{(e - timedelta(days=1)):%Y%m%d}.csv"
        return Response("\ufeff" + body, media_type="text/csv",   # BOM so Excel reads UTF-8
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    # --- alerts --------------------------------------------------------------------------

    def _rules_changed():
        if alerts is not None:
            alerts.invalidate()

    @app.get("/api/alerts/rules")
    def list_rules():
        return db.list_rules()

    @app.post("/api/alerts/rules")
    def add_rule(body: RuleIn, request: Request):
        from .alerts import parse_stayed, valid_hours
        from .incidents import CATEGORIES
        from .vlm import VEHICLE_COLORS
        allowed = {**schema.FIELDS, **{n: v + ["none"] for n, v in schema.LIST_FIELDS.items()},
                   "zone": sorted({z["name"] for z in db.list_zones()}), "fallen": ["yes"], "help": ["yes"],
                   "vehicle": ["any", "car", "SUV", "van", "pickup", "lorry", "bus", "motorbike", "bicycle"],
                   "vehicle_color": [c for c in VEHICLE_COLORS if c != "unknown"], "armed": ["away", "home"], "unusual": ["yes"],
                   "animal": ["any", "dog", "cat", "bird", "horse", "sheep", "cow"],
                   "incident": list(CATEGORIES)}
        areas = {z["name"] for z in db.list_zones() if z["kind"] == "area"}
        for cond in body.conditions:
            if len(cond) == 2 and cond[0] == "hours":
                if not valid_hours(cond[1]):
                    raise HTTPException(400, f"hours must look like 22-06, not {cond[1]!r}")
                continue
            if len(cond) == 2 and cond[0] == "quiet":   # no movement: minutes with nobody seen (labeler/quiet.py)
                if not (str(cond[1]).isdigit() and 10 <= int(cond[1]) <= 24 * 60):
                    raise HTTPException(400, "no movement needs 10 to 1440 minutes")
                continue
            if len(cond) == 2 and cond[0] == "stayed":
                try:
                    zone, _ = parse_stayed(cond[1])
                except ValueError as e:
                    raise HTTPException(400, str(e))
                if zone not in areas:
                    raise HTTPException(400, f"no area named {zone!r} (draw it on the Live page; lines have no inside)")
                continue
            if len(cond) != 2 or cond[0] not in allowed or cond[1] not in allowed[cond[0]]:
                raise HTTPException(400, f"unknown condition {cond}")
        names = [c[0] for c in body.conditions]
        if "vehicle_color" in names and "vehicle" not in names:
            raise HTTPException(400, 'a vehicle color needs a "vehicle" condition too (e.g. vehicle = any)')
        if "animal" in names and "vehicle" in names:
            raise HTTPException(400, "a rule is about a vehicle or an animal, not both")
        if "animal" in names and set(names) - {"animal", "zone", "hours", "armed", "stayed"}:
            raise HTTPException(400, "an animal rule can ask only for the animal, a zone, how long it stayed, the hours "
                                     "and Home / Away")
        if "quiet" in names and set(names) - {"quiet", "hours", "armed"}:
            raise HTTPException(400, "a no-movement rule can ask only for the time, the hours and Home / Away")
        if "incident" in names and (set(names) - {"incident", "zone", "hours", "armed"} or names.count("incident") > 1):
            raise HTTPException(400, "a safety rule can ask only for one kind of incident, an area, the hours and "
                                     "Home / Away")
        if "unusual" in names and set(names) - {"unusual", "armed"}:
            raise HTTPException(400, "an unusual-quiet rule can ask only for Home / Away (it learns its own hours)")
        if "fallen" in names and "help" in names:
            raise HTTPException(400, "a rule is about a fall or a help sign, not both")
        if "fallen" in names and "stayed" in names:
            raise HTTPException(400, "a fall rule cannot also ask how long someone stayed")
        if "vehicle" in names and set(names) - {"vehicle", "vehicle_color", "zone", "hours", "armed", "stayed"}:
            raise HTTPException(400, "a vehicle rule can ask only for the vehicle, its color, a zone, how long it "
                                     "stayed, the hours and Home / Away")
        if not body.name.strip():
            raise HTTPException(400, "the rule needs a name")
        rule = db.add_rule(body.name.strip()[:100], body.conditions, body.camera)
        _rules_changed()
        audit(request, "alert rule added", f"{body.name.strip()[:100]}: "
              + (", ".join(f"{n} = {v}" for n, v in body.conditions) or "anyone")
              + (f" (camera {body.camera})" if body.camera else ""))
        return rule

    @app.patch("/api/alerts/rules/{rule_id}")
    def update_rule(rule_id: int, body: RuleUpdate, request: Request):
        if not db.update_rule(rule_id, body.enabled):
            raise HTTPException(404, "rule not found")
        _rules_changed()
        audit(request, "alert rule " + ("enabled" if body.enabled else "disabled"), f"rule #{rule_id}")
        return {"ok": True}

    @app.delete("/api/alerts/rules/{rule_id}")
    def delete_rule(rule_id: int, request: Request):
        clips = db.rule_clips(rule_id)
        if not db.delete_rule(rule_id):
            raise HTTPException(404, "rule not found")
        for name in clips:
            privacy.remove_picture(crop_dir, name)
        _rules_changed()
        audit(request, "alert rule deleted", f"rule #{rule_id}")
        return {"ok": True}

    # --- checks: "was this right?" (labeler/checks.py): admins ----------------------------

    @app.get("/check")
    def check_page():
        return FileResponse(STATIC / "check.html")

    def _clip_for(event_id: int) -> str | None:
        name = clip_name(event_id)
        return name.split("/")[-1] if (crop_dir / name).is_file() else None

    @app.get("/api/check/next")
    def check_next(skip: str = ""):
        from . import checks
        skipped = set()
        for part in filter(None, skip.split(",")):
            kind, _, n = part.partition(":")
            if n.isdigit():
                skipped.add((kind, int(n)))
        return {"item": checks.next_item(db, skipped, clip_for=_clip_for), "counts": db.check_counts()}

    @app.get("/api/check/results")
    def check_results():
        from . import checks
        return {"rows": checks.results(db.list_checks()), "few": checks.FEW, "flags": checks.flags(db)}

    @app.get("/api/check/flags")
    def check_flags():
        """Rules, scene questions and vehicle/animal kinds often wrong by the checks (notes; nothing changes)."""
        from . import checks
        return checks.flags(db)

    @app.post("/api/check")
    def check_save(body: CheckIn, request: Request):
        from . import checks
        if body.kind not in checks.KINDS or body.right not in checks.RIGHT:
            raise HTTPException(400, "kind or answer not known")
        item = next((i for i in checks.candidates(db, body.kind) if i["id"] == body.item_id), None)
        if item is None and body.item_id not in db.checked_ids(body.kind):
            raise HTTPException(404, "nothing to check with that number (deleted, or older than 30 days)")
        if item is None:   # checked before: the new answer replaces the old one
            old = next(c for c in db.list_checks() if c["kind"] == body.kind and c["item_id"] == body.item_id)
            item = {"what": old["what"], "camera": old["camera"], "corrections": checks.CORRECTIONS.get(body.kind, [])}
        correction = body.correction if body.right == "no" else None
        if correction and correction not in item["corrections"]:
            raise HTTPException(400, f"correction must be one of {', '.join(item['corrections'])}")
        return db.save_check(body.kind, body.item_id, item["what"], item["camera"], body.right, correction, who(request))

    # --- scene questions (labeler/scene.py): admins --------------------------------------

    def _scene_every(minutes: int) -> int:
        if not 1 <= minutes <= 60:
            raise HTTPException(400, "ask every 1 to 60 minutes")
        return minutes * 60

    def _scene_area(camera: str, zone_id: int) -> dict:
        z = next((z for z in db.list_zones(camera) if z["id"] == zone_id), None)
        if z is None or z["kind"] != "area":
            raise HTTPException(400, f"camera {camera} has no area #{zone_id}")
        return z

    def _scene_hours(value: str | None) -> str | None:
        from .alerts import valid_hours
        value = (value or "").replace(" ", "")
        if value and not valid_hours(value):
            raise HTTPException(400, "hours look like 22-06 (from 22:00 until 06:00)")
        return value or None

    def _scene_limits(q: dict) -> str:
        limits = (["only when Away"] if q["only_away"] else []) + ([f"hours {q['hours']}"] if q["hours"] else [])
        return f", {', '.join(limits)}" if limits else ""

    @app.get("/api/scene")
    def scene_list():
        return db.list_scene_questions()

    @app.post("/api/scene")
    def scene_add(body: SceneIn, request: Request):
        cams = [c["name"] for c in cfg["cameras"]]
        text = " ".join(body.question.split())
        if body.camera not in cams:
            raise HTTPException(400, f"no camera {body.camera!r}")
        if not 5 <= len(text) <= 200:
            raise HTTPException(400, "the question needs 5 to 200 characters, e.g. Is the gate open?")
        if body.alert_on not in ("yes", "no"):
            raise HTTPException(400, "alert when the answer is yes or no")
        area = _scene_area(body.camera, body.zone_id) if body.zone_id else None
        q = db.add_scene_question(body.camera, text, _scene_every(body.every_min), body.alert_on,
                                  area["id"] if area else None, body.only_away, _scene_hours(body.hours))
        audit(request, "scene question added", f"{body.camera}: {text} (every {body.every_min} min, "
                                               f"alert on {body.alert_on}"
                                               + (f", area {area['name']}" if area else "") + _scene_limits(q) + ")")
        return q

    @app.patch("/api/scene/{question_id}")
    def scene_update(question_id: int, body: SceneUpdate, request: Request):
        fields = {}
        if body.enabled is not None:
            fields["enabled"] = body.enabled
        if body.every_min is not None:
            fields["every_s"] = _scene_every(body.every_min)
        if body.zone_id is not None:
            old = next((q for q in db.list_scene_questions() if q["id"] == question_id), None)
            if old is None:
                raise HTTPException(404, "question not found")
            fields["zone_id"] = _scene_area(old["camera"], body.zone_id)["id"] if body.zone_id else None
        if body.only_away is not None:
            fields["only_away"] = body.only_away
        if body.hours is not None:
            fields["hours"] = _scene_hours(body.hours)
        q = db.update_scene_question(question_id, **fields)
        if q is None:
            raise HTTPException(404, "question not found")
        audit(request, "scene question changed", f"#{question_id}: " + ", ".join(f"{k} = {v}" for k, v in fields.items()))
        return q

    @app.delete("/api/scene/{question_id}")
    def scene_delete(question_id: int, request: Request):
        q = db.delete_scene_question(question_id)
        if q is None:
            raise HTTPException(404, "question not found")
        for name in q["files"]:
            privacy.remove_picture(crop_dir, name)   # pictures (and blurred copies) and alert clips
        audit(request, "scene question deleted", f"#{question_id} {q['camera']}: {q['question']}")
        return {"ok": True}

    # --- Home / Away (labeler/arming.py): admins and viewers ------------------------------

    @app.get("/api/arming")
    def arming_state():
        return arming.state()

    @app.post("/api/arming")
    def arming_set(body: ArmingIn, request: Request):
        try:
            changed = arming.set(body.mode, who(request))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if changed:
            audit(request, "home/away", f"set to {body.mode.capitalize()}")
        return arming.state()

    @app.get("/safety")
    def safety_page():
        """The safety overview (Phase 52, admins: not in accounts.VIEWER)."""
        return FileResponse(STATIC / "safety.html")

    @app.get("/api/safety")
    def safety_state():
        """Safety incidents (labeler/incidents.py, admins): the kinds with their level and words, and the confirm
        queue's counts and latest answers (yes, no, unsure) since the app started; "on": False without a safety
        camera or the vision model. "overview": the safety page's view (Phase 52, labeler/safety_view.py)."""
        from . import safety_view
        from .incidents import CATEGORIES
        queue = getattr(app.state, "incidents", None)
        return {"on": queue is not None,
                "kinds": [{"category": k, "level": lv, "words": w} for k, (lv, w, _) in CATEGORIES.items()],
                "state": queue.state() if queue is not None else None,
                "overview": safety_view.overview(cfg, db, pipelines, queue)}

    @app.get("/api/safety/empty/{name}")
    def safety_empty(name: str):
        """The floor check's empty picture of a safety camera now (admins; the safety page, Phase 52)."""
        p = by_name.get(name)
        watch = getattr(p, "floor_watch", None)
        pic = watch.picture() if watch is not None else None
        if pic is None:
            raise HTTPException(404, "no empty picture (yet) for this camera")
        ok, jpeg = cv2.imencode(".jpg", pic, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(500, "could not make the picture")
        return Response(jpeg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/alerts/events")
    def alert_events(since_id: int = 0, limit: int = 50):
        events = db.list_alerts(since_id, min(limit, 200))
        for e in events:   # a clip is there once it has been written (a few seconds after the alert)
            name = clip_name(e["id"])
            e["clip"] = name.split("/")[-1] if (crop_dir / name).is_file() else None
        return {"latest_id": db.latest_alert_id(), "events": events}

    @app.get("/clips/{name}")
    def clip(name: str, request: Request):
        """An alert's video clip (admins; viewers are not on the allowlist)."""
        path = crop_dir / "clips" / name
        if not CLIP_NAME.fullmatch(name) or not path.is_file():
            raise HTTPException(404, "no such clip")
        audit(request, "view clip", name)
        return FileResponse(path, media_type="video/webm", headers={"Cache-Control": "private, max-age=3600"})

    # --- zones and counting lines (admin) ------------------------------------------------

    @app.get("/api/zones")
    def list_zones(camera: str | None = None):
        """The zones; areas on a safety camera also say how many people are in them now ("people_now", the camera's
        once-a-second count, labeler/pipeline.py _look_objects), to set a people limit by."""
        zones = db.list_zones(camera)
        for z in zones:
            counts = getattr(by_name.get(z["camera"]), "_crowd_counts", None)
            if counts is not None and z["id"] in counts:
                z["people_now"] = counts[z["id"]]
        return zones

    @app.get("/api/zones/roles")
    def zone_roles():
        """The roles the Live page offers, with their words and kind: area roles (config zones.roles, else all), then
        the counting lines' ("door", Phase 54)."""
        from .zones import LINE_ROLES, ROLES
        offered = cfg["zones"].get("roles") or list(ROLES)
        return ([{"role": r, "words": ROLES[r], "kind": "area"} for r in offered if r in ROLES]
                + [{"role": r, "words": w, "kind": "line"} for r, w in LINE_ROLES.items()])

    @app.patch("/api/zones/{zone_id}")
    def update_zone(zone_id: int, body: ZoneUpdate, request: Request):
        """A zone's role and/or an area's people limit (only the fields sent change)."""
        from .zones import ALL_ROLES as ROLES
        fields = {k: getattr(body, k) for k in body.model_fields_set}
        try:
            zone = db.update_zone(zone_id, **fields)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if zone is None:
            raise HTTPException(404, "zone not found")
        if zone["camera"] in by_name:
            by_name[zone["camera"]].reload_zones()
        what = []
        if "role" in fields:
            what.append(ROLES[zone["role"]] if zone["role"] else "no role")
        if "max_people" in fields:
            what.append(f"at most {zone['max_people']} people" if zone["max_people"] else "no people limit")
        audit(request, "zone changed", f"{'area' if zone['kind'] == 'area' else 'line'} {zone['name']!r} on camera "
                                        f"{zone['camera']}: " + ", ".join(what))
        return zone

    @app.post("/api/zones")
    def add_zone(body: ZoneIn, request: Request):
        if pipelines and body.camera not in by_name:
            raise HTTPException(404, f"no camera named {body.camera!r}")
        try:
            zone = db.add_zone(body.camera, body.name, body.kind, body.points, body.lying_ok, body.role, body.max_people)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if body.camera in by_name:
            by_name[body.camera].reload_zones()
        from .zones import ALL_ROLES as ROLES
        audit(request, "zone added", f"{zone['kind']} {zone['name']!r} on camera {zone['camera']}"
                                     + (f" ({ROLES[zone['role']]})" if zone["role"] else "")
                                     + (f", at most {zone['max_people']} people" if zone["max_people"] else ""))
        return zone

    @app.delete("/api/zones/{zone_id}")
    def delete_zone(zone_id: int, request: Request):
        zone = db.delete_zone(zone_id)
        if zone is None:
            raise HTTPException(404, "zone not found")
        if zone["camera"] in by_name:
            by_name[zone["camera"]].reload_zones()
        audit(request, "zone deleted", f"{zone['kind']} {zone['name']!r} on camera {zone['camera']}"
                                       + (f" (scene questions now about the whole picture: {'; '.join(zone['questions'])})"
                                          if zone["questions"] else ""))
        return {"ok": True}

    @app.get("/api/zones/counts")
    def zone_counts(start: str | None = None, end: str | None = None, camera: str | None = None):
        s, e = _day_range(start, end)
        return db.zone_counts(s, e, camera)

    @app.get("/api/camera-events")
    def camera_events(limit: int = 50):
        """Camera problems, newest first (ongoing ones have no end)."""
        return db.list_camera_events(min(limit, 200))

    @app.get("/api/cameras")
    def cameras():
        """Live cameras, then cameras with recorded people only (imported videos, removed cameras):
        "live" false, so the Live page shows no video tab for them but lists their people."""
        # Masked: viewers may call this, and an IP camera address can hold its password.
        out = [{"name": p.name, "mode": p.mode, "source": mask_password(p.cam["source"]), "live": True,
                "vehicles": getattr(p, "vehicles", None) is not None,
                "animals": getattr(p, "animals", None) is not None} for p in pipelines]
        out += [{"name": c["name"], "mode": c["mode"], "source": "imported videos" if c["imported"] else "recorded",
                 "live": False} for c in db.list_cameras()
                if c["name"] not in by_name and (c["people"] or c["imported"])]
        return out

    # --- importing recorded videos (admin; labeler/importer.py) -------------------------------

    @app.get("/import")
    def import_page():
        return FileResponse(STATIC / "import.html")

    def need_importer():
        if importer is None:
            raise HTTPException(503, "importing runs only inside the app")
        return importer

    @app.post("/api/import/preview")
    def import_preview(body: ImportPath):
        from .importer import describe, find_videos
        try:
            files = find_videos(body.path)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"files": describe(files)}

    @app.post("/api/import/start")
    def import_start(body: ImportStart, request: Request):
        from .importer import MAX_FILES
        from .settings import CAMERA_NAME
        from .sources import VIDEO_EXTS
        imp = need_importer()
        camera = body.camera.strip()
        if not CAMERA_NAME.fullmatch(camera):
            raise HTTPException(400, "the camera name is 1-32 letters, digits, - or _")
        known = {c["name"]: c for c in db.list_cameras()}
        configured = {c.get("name") for c in cfg.get("cameras") or [cfg.get("camera") or {}]}
        if camera in by_name or camera in configured or (
                camera in known and not known[camera]["imported"]):
            raise HTTPException(400, f"{camera!r} is a live camera's name; choose another name for the videos")
        if body.mode not in ("kiosk", "security"):
            raise HTTPException(400, "mode is kiosk or security")
        if not 0 <= body.keep_days <= 3650:
            raise HTTPException(400, "keep days is 0-3650 (0 = until deleted by hand)")
        if not 1 <= body.target_fps <= 30:
            raise HTTPException(400, "frames per second is 1-30")
        if body.imgsz is not None and body.imgsz not in (640, 960, 1280):
            raise HTTPException(400, "detector size is 640, 960 or 1280")
        if not 1 <= len(body.files) <= MAX_FILES:
            raise HTTPException(400, f"choose 1-{MAX_FILES} videos")
        files = []
        for f in body.files:
            path = Path(f.path)
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                raise HTTPException(400, f"not a video file: {f.path}")
            try:
                start = datetime.fromisoformat(f.start)
            except ValueError:
                raise HTTPException(400, f"{path.name}: the start time must look like 2026-10-09T14:30:00")
            if start.tzinfo is not None:
                start = start.astimezone().replace(tzinfo=None)
            files.append({"path": path, "start": start})
        try:
            job = imp.start(files, camera, body.mode, body.keep_days, body.target_fps, who(request), body.imgsz)
        except ValueError as e:
            raise HTTPException(409, str(e))
        audit(request, "import videos", f"{len(files)} video(s) into camera {camera} from "
                                        f"{files[0]['path'].parent}"[:500])
        return job

    @app.get("/api/import")
    def import_status():
        imp = need_importer()
        return {**imp.status(), "recorded": [c for c in db.list_cameras() if c["imported"]],
                "vlm_queue": imp.worker.pending if imp.worker is not None else None}

    @app.post("/api/import/cancel")
    def import_cancel(request: Request):
        if not need_importer().cancel():
            raise HTTPException(409, "no import is running")
        audit(request, "import cancelled")
        return {"ok": True}

    @app.post("/api/import/delete")
    def import_delete(body: CameraName, request: Request):
        """Delete every person of an imported camera (with pictures, labels and history)."""
        imp = need_importer()
        cams = {c["name"]: c for c in db.list_cameras()}
        if body.camera not in cams or not cams[body.camera]["imported"]:
            raise HTTPException(404, "no imported videos under that name")
        if imp.running and imp.job["camera"] == body.camera:
            raise HTTPException(409, "an import into this camera is running; cancel it first")
        deleted, paths = db.delete_tracks(db.camera_track_ids(body.camera))
        removed = sum(privacy.remove_picture(crop_dir, name) for name in paths)
        audit(request, "delete", f"imported camera {body.camera}: {len(deleted)} people")
        return {"deleted": len(deleted), "pictures_removed": removed}

    @app.get("/health")
    def health_page():
        return FileResponse(STATIC / "health.html")

    @app.get("/api/health")
    def health_check(request: Request):
        """Is everything working? 200 if so, 503 if not (labeler/healthcheck.py). Admins get the log's error
        messages too; viewers and a monitoring tool (?token=, web.health_token) only what is wrong."""
        role = (getattr(request.state, "user", None) or {}).get("role")
        detail = {"admin": "full", "monitor": "status"}.get(role, "viewer")
        result = healthcheck.run_checks(cfg, db, pipelines, health, app.state.started_at, detail, arming,
                                        getattr(app.state, "quiet", None), getattr(app.state, "updater", None),
                                        getattr(app.state, "pcwatch", None), getattr(app.state, "routine", None),
                                        getattr(app.state, "incidents", None))
        return JSONResponse(result, status_code=200 if result["ok"] else 503, headers={"Cache-Control": "no-store"})

    @app.get("/stream")
    async def stream(request: Request, camera: str | None = None):
        pipeline = pick(camera)
        await run_in_threadpool(audit, request, "view live", f"camera {pipeline.name}" if pipeline else "")
        blurred = blur_for(request)

        async def frames():
            last = None
            # End when the app is stopping: an endless stream would otherwise keep the server from exiting.
            while not app.state.stopping.is_set():
                if pipeline is None:
                    jpeg = None
                else:
                    jpeg = pipeline.latest_jpeg_blurred() if blurred else pipeline.latest_jpeg
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(1 / 30)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/shutdown")
    def shutdown(request: Request):
        """Stop the app cleanly: finish current people, save their labels, then exit."""
        stop = getattr(app.state, "request_shutdown", None)
        if stop is None:
            raise HTTPException(501, "shutdown is not available in this mode")
        audit(request, "stop app")
        stop()
        return {"ok": True}

    @app.get("/summary")
    def summary_page():
        return FileResponse(STATIC / "summary.html")

    @app.get("/first-run")
    def first_run_page():
        return FileResponse(STATIC / "first-run.html")

    @app.get("/api/first-run")
    def first_run():
        """After a big update: database, built-in rules, today's recording, Home / Away (labeler/firstrun.py); the
        page adds the Health check and each camera's picture check."""
        from . import firstrun
        return firstrun.server_checks(cfg, db, pipelines, arming)

    @app.post("/api/first-run/help-test")
    def help_test_start(body: HelpTestIn, request: Request):
        """"Try the help sign": for 30 s a confirmed sign on this camera is shown on the page, with no alert."""
        p = by_name.get(body.camera)
        if p is None:
            raise HTTPException(404, f"camera {body.camera!r} is not running")
        if getattr(p, "gestures", None) is None:
            raise HTTPException(409, "the help sign is turned off for this camera (Settings page)")
        p.start_help_test(30.0)
        audit(request, "try help sign", f"camera {p.name}")
        return p.help_test_state()

    @app.get("/api/first-run/help-test")
    def help_test_state(camera: str):
        p = by_name.get(camera)
        if p is None:
            raise HTTPException(404, f"camera {camera!r} is not running")
        return p.help_test_state()

    @app.get("/whats-new")
    def whats_new_page():
        return FileResponse(STATIC / "whats-new.html")

    @app.get("/api/whats-new")
    def whats_new_api(request: Request):
        """What this version added since the one that ran before (labeler/whatsnew.py); viewers: their part."""
        w = getattr(app.state, "whats_new", None)
        if w is None:   # tests and tools that make the app without run.py
            from .whatsnew import ENTRIES, WhatsNew
            w = WhatsNew(ENTRIES[-1].key)   # nothing new
        return w.summary((getattr(request.state, "user", None) or {}).get("role") == "admin")

    @app.get("/api/summary")
    def summary_api(request: Request, period: str | None = None, start: str | None = None, end: str | None = None,
                    week: str | None = None):
        """What happened: period away / night / today / week (the last 7 days; with week=2026-10-12 the Monday-Sunday
        week of that day), or start and end (2026-10-14T18:00). Viewers: no clips and no camera problems
        (labeler/summary.py)."""
        from . import summary
        now = datetime.now()
        admin = (getattr(request.state, "user", None) or {}).get("role") == "admin"
        if period == "week":
            try:
                day = datetime.fromisoformat(week).date() if week else None
            except ValueError:
                raise HTTPException(400, "week looks like 2026-10-12 (any day of that week)")
            s, e = summary.week_period(now, day)
            if e <= s:
                raise HTTPException(400, "that week has not begun yet")
            return {"period": "week", **summary.week(db, s, e, admin=admin, cfg=cfg, now=now,
                                                    clip_for=lambda i: _clip_for(i) if admin else None)}
        if start or end:
            try:
                s = datetime.fromisoformat(start) if start else now - timedelta(days=1)
                e = datetime.fromisoformat(end) if end else now
            except ValueError:
                raise HTTPException(400, "start and end look like 2026-10-14T18:00")
            if e <= s:
                raise HTTPException(400, "the end must be after the start")
            if e - s > timedelta(days=31):
                raise HTTPException(400, "at most 31 days at once")
            name = "custom"
        else:
            name = period or "away"
            try:
                got = summary.period(db, name, now)
            except ValueError as err:
                raise HTTPException(400, str(err))
            if got is None:
                return {"period": name, "none": "never switched to Away yet"}
            s, e = got
        admin = (getattr(request.state, "user", None) or {}).get("role") == "admin"
        return {"period": name, **summary.build(db, s, e, admin=admin,
                                               clip_for=lambda i: _clip_for(i) if admin else None)}

    @app.get("/api/routine")
    def routine_api():
        """The usual day of each kiosk camera (Phase 43, the Health page chart): per kind of day and hour the share of
        days someone was seen (labeler/routine.py), today's quiet stretch and whether "Unusual quiet" fired
        (labeler/unusual.py). No cameras when unusual quiet is off or there is no kiosk camera. Viewers too."""
        from .routine import kind_of
        routine, unusual = getattr(app.state, "routine", None), getattr(app.state, "unusual", None)
        if routine is None or not cfg["unusual"]["enabled"]:
            return {"cameras": []}
        today = routine.clock().date()
        r = cfg["routine"]
        cameras = []
        for cam in routine.cameras:
            item = routine.chart(cam)
            item["today"] = unusual.today(cam) if unusual is not None else None
            cameras.append(item)
        return {"date": today.isoformat(), "kind": kind_of(today), "weeks": int(r["weeks"]),
                "usual_share": float(r["usual_share"]),
                "quiet_share": float(r["quiet_share"]), "min_quiet_h": int(r["min_quiet_h"]), "cameras": cameras}

    @app.get("/api/today")
    def today_summary(request: Request):
        """Today at a glance (labeler/today.py); viewers get the people numbers only."""
        from . import today
        admin = (getattr(request.state, "user", None) or {}).get("role") == "admin"
        return today.summary(db, admin=admin, cfg=cfg)

    @app.get("/api/live")
    def live(camera: str | None = None):
        """People in view right now and their current labels."""
        pipeline = pick(camera)
        return {"camera": pipeline.name if pipeline else None,
                "people": pipeline.live() if pipeline else [],
                "fps": round(pipeline.fps, 1) if pipeline else 0}

    @app.get("/api/live/thumb/{tracker_id}")
    def live_thumb(tracker_id: int, request: Request, camera: str | None = None):
        pipeline = pick(camera)
        jpeg = pipeline.live_thumb(tracker_id, blurred=blur_for(request)) if pipeline else None
        if jpeg is None:
            raise HTTPException(404, "not in view")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{name}/picture")
    def picture_now(name: str):
        """Check a camera's picture now (Settings page, "Check the picture"; admins): what is wrong, what to do,
        the picture checked, and what lasted over the last minutes (labeler/picture_check.py)."""
        import base64
        from . import picture_check as pc
        from .sources import is_blank
        p = by_name.get(name)
        if p is None:
            raise HTTPException(404, f"camera {name!r} is not running")
        got = getattr(p, "scene_frame", None)
        if got is None or (datetime.now() - got[1]).total_seconds() > 30:
            raise HTTPException(409, "no recent picture from this camera (starting, or no picture)")
        frame = got[0]
        st = pc.stats(frame, getattr(p, "scene_people", []) or [])
        found = pc.findings(st, bool(is_blank(frame)))
        out = {"camera": name, "at": got[1].isoformat(timespec="seconds"), "findings": found,
               "advice": [pc.ADVICE[c] for c in found], "stats": {k: round(v, 3) for k, v in st.items()},
               "lasting": p.picture.state()["lasting"] if getattr(p, "picture", None) else {}}
        if p.mode == "kiosk":
            now = datetime.now()
            faces, numbers = pc.faces(db.people_between(now - timedelta(days=7), now, name, limit=200, newest_first=True))
            out["faces"] = {"findings": faces, "advice": [pc.ADVICE[c] for c in faces], **numbers}
        small = cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out["image"] = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode() if ok else None
        return out

    @app.get("/api/status")
    def status(camera: str | None = None):
        pipeline = pick(camera)
        free_gb = shutil.disk_usage(crop_dir).free / 1e9
        return {**(pipeline.status() if pipeline else {"mode": cfg["mode"]}),
                "cameras": [p.status() for p in pipelines],
                "disk_free_gb": round(free_gb, 1),
                "disk_low": free_gb < cfg["storage"]["min_free_gb"],
                "camera_problems": (health.current() if health is not None else [])
                + (pcwatch.current() if (pcwatch := getattr(app.state, "pcwatch", None)) is not None else []),
                **db.stats()}

    @app.get("/api/schema")
    def get_schema():
        fields = schema.mode_fields(cfg["mode"], cfg.get("labels"))
        # Fields turned off in Settings are never stored: not offered for filters or alert rules either.
        off = set(schema.OPTIONAL_FIELDS) - set(schema.mode_fields("kiosk", cfg.get("labels")))
        return {
            "mode": cfg["mode"],
            "fields": {n: schema.allowed_values(n) or schema.LIST_FIELDS[n] + ["none"] for n in fields},
            "list_fields": list(schema.LIST_FIELDS),
            # Every attribute of every mode, for alert rules (cameras can use different modes).
            "all_fields": {n: v for n, v in {**schema.FIELDS, **schema.LIST_FIELDS}.items() if n not in off},
        }

    def videos(request: Request, people: list[dict]) -> list[dict]:
        """Person videos for admins, only those already written (a video is saved a second after the person
        leaves); viewers get none (like alert clips)."""
        admin = (getattr(request.state, "user", None) or {}).get("role") == "admin"
        for person in people:
            person["videos"] = [v.split("/")[-1] for v in person.get("videos") or []
                                if admin and (crop_dir / v).is_file()]
        return people

    @app.get("/api/tracks")
    def tracks(request: Request, limit: int = 50, name: str | None = None, value: str | None = None,
               camera: str | None = None):
        return videos(request, db.list_tracks(limit=min(limit, 500), name=name, value=value, camera=camera))

    @app.get("/api/vehicles")
    def vehicles_list(limit: int = 50, camera: str | None = None, kind: str | None = None, color: str | None = None,
                      day: str | None = None):
        """Vehicles seen (labeler/vehicles.py), newest first. kind "car" also finds SUVs, vans and pickups."""
        from .agent import VEHICLE_COLOR_WORDS, VEHICLE_WORDS
        start = end = None
        if day:
            try:
                start = datetime.fromisoformat(day).replace(hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                raise HTTPException(400, "day must look like 2026-10-13")
            end = start + timedelta(days=1)
        kinds = sorted(VEHICLE_WORDS.get(kind, {kind})) if kind else None
        colors = sorted(VEHICLE_COLOR_WORDS.get(color, {color})) if color else None
        return db.list_vehicles(min(limit, 500), camera=camera, kinds=kinds, colors=colors, start=start, end=end)

    @app.get("/api/animals")
    def animals_list(limit: int = 50, camera: str | None = None, kind: str | None = None, color: str | None = None,
                     day: str | None = None):
        """Animals seen (labeler/animals.py), newest first."""
        from .agent import VEHICLE_COLOR_WORDS
        start = end = None
        if day:
            try:
                start = datetime.fromisoformat(day).replace(hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                raise HTTPException(400, "day must look like 2026-10-13")
            end = start + timedelta(days=1)
        colors = sorted(VEHICLE_COLOR_WORDS.get(color, {color})) if color else None
        return db.list_animals(min(limit, 500), camera=camera, kinds=[kind] if kind else None, colors=colors,
                               start=start, end=end)

    @app.post("/api/animals/delete")
    def delete_animals(body: DeleteIn, request: Request):
        deleted, paths = db.delete_animals(body.ids)
        if deleted:
            audit(request, "delete", f"{len(deleted)} animals (" + ", ".join(f"#{i}" for i in sorted(deleted)[:50]) + ")")
        removed = sum(privacy.remove_picture(crop_dir, name) for name in paths)
        return {"deleted": len(deleted), "pictures_removed": removed}

    @app.post("/api/vehicles/delete")
    def delete_vehicles(body: DeleteIn, request: Request):
        deleted, paths = db.delete_vehicles(body.ids)
        if deleted:
            audit(request, "delete", f"{len(deleted)} vehicles (" + ", ".join(f"#{i}" for i in sorted(deleted)[:50]) + ")")
        removed = sum(privacy.remove_picture(crop_dir, name) for name in paths)
        return {"deleted": len(deleted), "pictures_removed": removed}

    @app.post("/api/tracks/delete")
    def delete_tracks(body: DeleteIn, request: Request):
        """Delete selected people (or everyone) with their pictures, labels and history."""
        deleted, paths = db.delete_tracks(body.ids, everything=body.everything)
        if deleted:
            shown = ", ".join(f"#{i}" for i in sorted(deleted)[:50]) + (" …" if len(deleted) > 50 else "")
            audit(request, "delete", ("everyone: " if body.everything else "") + f"{len(deleted)} people ({shown})")
        for p in pipelines:
            p.forget(deleted)
        removed = sum(privacy.remove_picture(crop_dir, name) for name in paths)
        return {"deleted": len(deleted), "pictures_removed": removed}

    @app.get("/api/search")
    def search(q: str, request: Request, limit: int = 20):
        """Find people by a free-text description, e.g. "man in a beige coat"."""
        if embedder is None:
            raise HTTPException(503, "embeddings are disabled in config.yaml")
        audit(request, "search", q[:300])
        hits = db.nearest(embedder.text(q), embedder.model_name, limit=min(limit, 100))
        tracks = db.get_tracks([tid for tid, _ in hits])
        sims = dict(hits)
        for t in tracks:
            t["similarity"] = round(sims[t["id"]], 4)
            t["match"] = round(embedder.match_probability(sims[t["id"]]), 4)
        return videos(request, tracks)

    @app.post("/api/search/photo")
    async def search_photo(request: Request, limit: int = 20):
        """People who look like the person in a picture. The picture is the request body (JPEG or PNG, up to
        10 MB), sent by the page as is; it is not stored."""
        if embedder is None:
            raise HTTPException(503, "embeddings are disabled in config.yaml")
        from .photo import PhotoSearch
        searcher = photo or getattr(app.state, "photo", None)
        if searcher is None:
            searcher = app.state.photo = PhotoSearch(cfg, embedder)
        data = await request.body()
        try:
            vector, info = await run_in_threadpool(searcher.vector, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        n = info["people"]
        found = ("1 person found" if n == 1 else f"{n} people found, the one in the middle (or the largest) searched"
                 if n else "nobody found, the whole picture searched")
        await run_in_threadpool(audit, request, "photo search", f"picture {info['width']}x{info['height']}, {found}")

        def rank():
            hits = db.nearest(vector, embedder.model_name, limit=min(limit, 100))
            people = db.get_tracks([tid for tid, _ in hits])
            sims = dict(hits)
            for t in people:
                t["similarity"] = round(sims[t["id"]], 4)
            return videos(request, people)
        return {"people": await run_in_threadpool(rank), "info": info, "found": found}

    @app.post("/api/ask")
    def ask(body: AskIn, request: Request):
        """Answer a question like "who wore a red top after 3pm?" (see labeler/agent.py)."""
        if agent is None:
            raise HTTPException(503, "questions need the VLM; it is disabled or not running")
        if not body.question.strip():
            raise HTTPException(400, "empty question")
        audit(request, "question", body.question.strip()[:300])
        result = agent.run(body.question.strip()[:300], limit=min(body.limit, 100))
        videos(request, result.get("people") or [])
        return result

    @app.get("/api/tracks/{track_id}/similar")
    def similar(track_id: int, request: Request, limit: int = 10):
        """People who look most like this one (same clothes; across cameras too)."""
        if embedder is None:
            raise HTTPException(503, "embeddings are disabled in config.yaml")
        audit(request, "look-alike search", f"person #{track_id}")
        vector = db.get_embedding(track_id, embedder.model_name)
        if vector is None:
            raise HTTPException(404, "no embedding for this person yet")
        hits = db.nearest(vector, embedder.model_name, limit=min(limit, 100), exclude=track_id)
        tracks = db.get_tracks([tid for tid, _ in hits])
        sims = dict(hits)
        for t in tracks:
            t["similarity"] = round(sims[t["id"]], 4)
        return videos(request, tracks)

    @app.get("/api/tracks/{track_id}/original")
    def original(track_id: int, request: Request):
        """An imported person's moment in their original video, as a short excerpt that plays in a browser
        (admins; viewers may only open look-alikes and the timeline below a person). Made on first view."""
        from .db import excerpt_name
        from .importer import make_excerpt
        source = db.track_source(track_id)
        if source is None:
            raise HTTPException(404, "this person was not imported from a video")
        file, offset, in_view = source
        path = Path(file)
        if not path.is_file():
            raise HTTPException(404, f"the original video {path.name} is no longer where it was imported from")
        out = crop_dir / excerpt_name(track_id)
        if not out.is_file():
            try:
                make_excerpt(path, offset, in_view, out)
            except (ValueError, RuntimeError) as e:
                raise HTTPException(500, str(e))
        audit(request, "view original", f"person #{track_id}: {path.name} at {offset:.0f} s")
        return FileResponse(out, media_type="video/webm", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/tracks/{track_id}/timeline")
    def timeline(track_id: int, request: Request, min_similarity: float | None = None):
        """Where this person was seen: look-alikes from all cameras in time order (labeler/timeline.py)."""
        if embedder is None:
            raise HTTPException(503, "embeddings are disabled in config.yaml")
        vector = db.get_embedding(track_id, embedder.model_name)
        if vector is None:
            raise HTTPException(404, "no embedding for this person yet")
        tcfg = cfg["timeline"]
        floor = tcfg["min_similarity"] if min_similarity is None else min_similarity
        hits = [(tid, sim) for tid, sim in db.nearest(vector, embedder.model_name, limit=200, exclude=track_id)
                if sim >= floor]
        sims = dict(hits)
        found = db.get_tracks([track_id] + [tid for tid, _ in hits])
        anchor = next(t for t in found if t["id"] == track_id)
        candidates = [dict(t, similarity=round(sims[t["id"]], 4)) for t in found if t["id"] != track_id]
        audit(request, "timeline", f"person #{track_id}")
        return videos(request, timeline_mod.plausible(anchor, candidates, tcfg["window_h"]))

    @app.get("/api/review")
    def review_queue(request: Request, limit: int = 50):
        items = db.review_queue(limit=min(limit, 200), spot_every=cfg["review"].get("spot_check_every", 0))
        # The person's video helps to answer ("is that a backpack?"): attached from their record.
        people = {t["id"]: t for t in videos(request, db.get_tracks(list({i["track_id"] for i in items})))}
        for item in items:
            item["allowed"] = schema.allowed_values(item["name"])
            item["videos"] = people.get(item["track_id"], {}).get("videos", [])
        return items

    @app.post("/api/review/{attribute_id}")
    def review(attribute_id: int, body: ReviewIn, request: Request):
        value = body.value.strip().lower()
        try:
            db.apply_review(attribute_id, value, who(request))   # the logged-in user, not what the page says
        except KeyError:
            raise HTTPException(404, "attribute not found")
        audit(request, "review", f"label #{attribute_id} = {value}")
        return {"ok": True}

    # --- settings (admin; labeler/settings.py) ------------------------------------------------

    @app.get("/settings")
    def settings_page():
        return FileResponse(STATIC / "settings.html")

    def settings_configs():
        """(settings file, config.yaml alone, config.yaml + settings file): read fresh, as the next start will."""
        path = settings_mod.settings_path(cfg)
        if path is None or not cfg.get("_config_path"):
            raise HTTPException(503, "this app was started with a config without a settings file (settings_file:)")
        base = load_config(cfg["_config_path"], use_settings=False)
        return path, base, load_config(cfg["_config_path"])

    def restart_needed(path) -> bool:
        try:
            saved = settings_mod.read(path)
        except (OSError, ValueError):
            return True
        return saved != (cfg.get("_settings_loaded") or {})

    @app.get("/api/settings")
    def get_settings():
        path, _, effective = settings_configs()
        return {"values": settings_mod.current(effective),    # what the next start uses
                "running": settings_mod.current(cfg),          # what runs now
                "restart_needed": restart_needed(path),
                "restart_available": settings_mod.restart_available()
                and getattr(app.state, "request_restart", None) is not None,
                "cli_source": bool(cfg.get("_cli_source")),
                "file": str(path), "error": effective.get("_settings_error"),
                "ignored": effective.get("_settings_ignored", []),
                "push": settings_mod.push_info(effective, _tailscale_ip())}

    @app.post("/api/settings")
    def save_settings(body: dict, request: Request):
        try:
            parsed = settings_mod.SettingsIn.model_validate(body)
        except ValidationError as e:
            raise HTTPException(400, settings_mod.validation_message(e))
        path, base, effective = settings_configs()
        new_passwords: dict = {}
        try:
            overlay, changes = settings_mod.build(parsed, base, effective, new_passwords)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if changes:
            settings_mod.write(path, overlay)
            audit(request, "settings changed", mask_password("; ".join(changes))[:2000])
        # Shown once on the page (only their hashes are kept): never in the audit log or the app's log.
        return {"ok": True, "changes": changes, "restart_needed": restart_needed(path), "new_passwords": new_passwords}

    @app.post("/api/settings/cameras/{name}/new-password")
    def new_camera_password(name: str, request: Request):
        """A new send password for a camera sent from another device (the old one stops working after a restart)."""
        path, _, effective = settings_configs()
        try:
            password = settings_mod.renew_password(path, effective, name)
        except ValueError as e:
            raise HTTPException(404, str(e))
        audit(request, "settings changed", f"camera {name}: new send password")
        return {"name": name, "password": password, "restart_needed": restart_needed(path)}

    def _tailscale_ip() -> str | None:
        """This PC's Tailscale address for the send addresses (labeler/remote.py, cached), or None."""
        try:
            from . import remote
            ts, _ = remote._system(subprocess.run)
            return ts["ip"] if ts and ts.get("state") == "Running" else None
        except Exception:
            return None

    # --- updates (labeler/updater.py): admins -------------------------------------------------------------------

    def _updater():
        if getattr(app.state, "updater", None) is None:
            from .updater import Updater
            app.state.updater = Updater(cfg)
        return app.state.updater

    @app.get("/api/update")
    def update_state(check: bool = False):
        """What is new on GitHub (the state of the last check; check=true fetches now)."""
        up = _updater()
        state = up.check(fetch=True) if check or not up.state.get("checked_at") else dict(up.state)
        from .updater import check_words, last_check
        tested = last_check(resolve(cfg["updates"]["check_report"])) if db.is_postgres else None
        return {**state, "problem": up.problem(state), "busy": up.busy,
                "safe_test": {"ok": bool(tested.get("ok")), "words": check_words(tested)} if tested else None,
                "can_restart": getattr(app.state, "request_restart", None) is not None
                and settings_mod.restart_available()}

    @app.post("/api/update")
    def update_now(request: Request):
        """Update to the newest version, then restart (Settings page, "Update now"; asks first on the page)."""
        up = _updater()
        result = up.update()
        if not result["ok"]:
            raise HTTPException(409, result["message"])
        audit(request, "update app", f"{result['from']} -> {result['to']}: " + "; ".join(result["changes"])[:900])
        do = getattr(app.state, "request_restart", None)
        result["restarting"] = do is not None and settings_mod.restart_available()
        if result["restarting"]:
            threading.Timer(1.0, do).start()   # after the page got its answer
        return result

    @app.post("/api/restart")
    def restart(request: Request):
        """Stop cleanly and start again (needs the supervisor, i.e. start.bat), e.g. to apply new settings."""
        do = getattr(app.state, "request_restart", None)
        if do is None or not settings_mod.restart_available():
            raise HTTPException(409, "this copy was not started with start.bat, so it cannot start itself again: "
                                     "stop it, then start it again by hand")
        audit(request, "restart app")
        do()
        return {"ok": True}

    @app.get("/audit")
    def audit_page():
        return FileResponse(STATIC / "audit.html")

    @app.get("/api/audit")
    def audit_log(limit: int = 100, user: str | None = None, action: str | None = None,
                  before_id: int | None = None):
        return {"entries": db.list_audit(min(limit, 500), user or None, action or None, before_id),
                **db.audit_choices()}

    return app

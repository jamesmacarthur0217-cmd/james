"""Local web UI: live view, people list with search, and the human review queue."""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import shutil
from datetime import datetime, timedelta

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from . import accounts, privacy, reports, schema
from . import timeline as timeline_mod
from .clips import clip_name
from .config import ROOT, resolve
from .db import Database
from .pipeline import Pipeline

STATIC = ROOT / "labeler" / "static"
log = logging.getLogger(__name__)
CLIP_NAME = re.compile(r"alert_\d+\.webm")


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


class ZoneIn(BaseModel):
    camera: str
    name: str
    kind: str                     # area | line
    points: list[list[float]]


class AnswerIn(BaseModel):
    labels: dict[str, str]


class AskIn(BaseModel):
    question: str
    limit: int = 30


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
               alerts=None, agent=None, health=None) -> FastAPI:
    app = FastAPI(title="VisionLabeler")
    app.state.stopping = threading.Event()
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
        page = method == "GET" and not path.startswith(("/api/", "/stream", "/crops/", "/clips/", "/static/"))
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

    # --- reports -------------------------------------------------------------------------

    @app.get("/api/reports/summary")
    def report_summary(request: Request, start: str | None = None, end: str | None = None,
                       camera: str | None = None):
        s, e = _day_range(start, end)
        audit(request, "view report", f"{s:%Y-%m-%d} to {(e - timedelta(days=1)):%Y-%m-%d}"
                                      + (f", camera {camera}" if camera else ""))
        return {**reports.summarize(db.people_between(s, e, camera), s, e, cfg["engagement"]["looked_s"]),
                "zones": db.zone_counts(s, e, camera)}

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
        allowed = {**schema.FIELDS, **{n: v + ["none"] for n, v in schema.LIST_FIELDS.items()},
                   "zone": sorted({z["name"] for z in db.list_zones()})}
        for cond in body.conditions:
            if len(cond) != 2 or cond[0] not in allowed or cond[1] not in allowed[cond[0]]:
                raise HTTPException(400, f"unknown condition {cond}")
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
        return db.list_zones(camera)

    @app.post("/api/zones")
    def add_zone(body: ZoneIn, request: Request):
        if pipelines and body.camera not in by_name:
            raise HTTPException(404, f"no camera named {body.camera!r}")
        try:
            zone = db.add_zone(body.camera, body.name, body.kind, body.points)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if body.camera in by_name:
            by_name[body.camera].reload_zones()
        audit(request, "zone added", f"{zone['kind']} {zone['name']!r} on camera {zone['camera']}")
        return zone

    @app.delete("/api/zones/{zone_id}")
    def delete_zone(zone_id: int, request: Request):
        zone = db.delete_zone(zone_id)
        if zone is None:
            raise HTTPException(404, "zone not found")
        if zone["camera"] in by_name:
            by_name[zone["camera"]].reload_zones()
        audit(request, "zone deleted", f"{zone['kind']} {zone['name']!r} on camera {zone['camera']}")
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
        return [{"name": p.name, "mode": p.mode, "source": str(p.cam["source"])} for p in pipelines]

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

    @app.get("/api/status")
    def status(camera: str | None = None):
        pipeline = pick(camera)
        free_gb = shutil.disk_usage(crop_dir).free / 1e9
        return {**(pipeline.status() if pipeline else {"mode": cfg["mode"]}),
                "cameras": [p.status() for p in pipelines],
                "disk_free_gb": round(free_gb, 1),
                "disk_low": free_gb < cfg["storage"]["min_free_gb"],
                "camera_problems": health.current() if health is not None else [],
                **db.stats()}

    @app.get("/api/schema")
    def get_schema():
        fields = schema.MODE_FIELDS[cfg["mode"]]
        return {
            "mode": cfg["mode"],
            "fields": {n: schema.allowed_values(n) or schema.LIST_FIELDS[n] + ["none"] for n in fields},
            "list_fields": list(schema.LIST_FIELDS),
            # Every attribute of every mode, for alert rules (cameras can use different modes).
            "all_fields": {**{n: v for n, v in schema.FIELDS.items()}, **schema.LIST_FIELDS},
        }

    @app.get("/api/tracks")
    def tracks(limit: int = 50, name: str | None = None, value: str | None = None, camera: str | None = None):
        return db.list_tracks(limit=min(limit, 500), name=name, value=value, camera=camera)

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
        return tracks

    @app.post("/api/ask")
    def ask(body: AskIn, request: Request):
        """Answer a question like "who wore a red top after 3pm?" (see labeler/agent.py)."""
        if agent is None:
            raise HTTPException(503, "questions need the VLM; it is disabled or not running")
        if not body.question.strip():
            raise HTTPException(400, "empty question")
        audit(request, "question", body.question.strip()[:300])
        return agent.run(body.question.strip()[:300], limit=min(body.limit, 100))

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
        return tracks

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
        return timeline_mod.plausible(anchor, candidates, tcfg["window_h"])

    @app.get("/api/review")
    def review_queue(limit: int = 50):
        items = db.review_queue(limit=min(limit, 200))
        for item in items:
            item["allowed"] = schema.allowed_values(item["name"])
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

    @app.get("/audit")
    def audit_page():
        return FileResponse(STATIC / "audit.html")

    @app.get("/api/audit")
    def audit_log(limit: int = 100, user: str | None = None, action: str | None = None,
                  before_id: int | None = None):
        return {"entries": db.list_audit(min(limit, 500), user or None, action or None, before_id),
                **db.audit_choices()}

    return app

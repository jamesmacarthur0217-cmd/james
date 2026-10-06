"""Local web UI: live view, people list with search, and the human review queue."""
from __future__ import annotations

import asyncio
import base64
import secrets
import shutil

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import schema
from .config import ROOT, resolve
from .db import Database
from .pipeline import Pipeline

STATIC = ROOT / "labeler" / "static"


class ReviewIn(BaseModel):
    value: str
    reviewer: str = "reviewer"


class DeleteIn(BaseModel):
    ids: list[int] = []
    everything: bool = False


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def check_access(web: dict) -> None:
    """Refuse to open the UI to the network without a password."""
    if web["host"] not in LOCAL_HOSTS and not web.get("password"):
        raise SystemExit(
            f"web.host is {web['host']!r}, so other devices could open the app, but no password is set. "
            "Add a line VL_WEB_PASSWORD=<a long password> to .env, or set web.host back to 127.0.0.1.")


def create_app(cfg: dict, db: Database, pipelines: Pipeline | list[Pipeline] | None, embedder=None) -> FastAPI:
    app = FastAPI(title="VisionLabeler")

    password = cfg["web"].get("password") or ""
    if password:
        username = cfg["web"].get("username") or "admin"

        @app.middleware("http")
        async def require_login(request: Request, call_next):
            """Other devices must log in; this PC itself (stop.bat, the local browser) does not."""
            client = request.client.host if request.client else ""
            if client not in LOCAL_HOSTS:
                ok = False
                header = request.headers.get("authorization", "")
                if header.startswith("Basic "):
                    try:
                        user, _, pw = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                        ok = secrets.compare_digest(user, username) and secrets.compare_digest(pw, password)
                    except Exception:
                        ok = False
                if not ok:
                    return Response("Login required", status_code=401,
                                    headers={"WWW-Authenticate": 'Basic realm="VisionLabeler"'})
            return await call_next(request)
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
    app.mount("/crops", StaticFiles(directory=crop_dir), name="crops")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/review")
    def review_page():
        return FileResponse(STATIC / "review.html")

    @app.get("/api/cameras")
    def cameras():
        return [{"name": p.name, "mode": p.mode, "source": str(p.cam["source"])} for p in pipelines]

    @app.get("/stream")
    async def stream(camera: str | None = None):
        pipeline = pick(camera)

        async def frames():
            last = None
            while True:
                jpeg = pipeline.latest_jpeg if pipeline else None
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(1 / 30)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/shutdown")
    def shutdown():
        """Stop the app cleanly: finish current people, save their labels, then exit."""
        request = getattr(app.state, "request_shutdown", None)
        if request is None:
            raise HTTPException(501, "shutdown is not available in this mode")
        request()
        return {"ok": True}

    @app.get("/api/live")
    def live(camera: str | None = None):
        """People in view right now and their current labels."""
        pipeline = pick(camera)
        return {"camera": pipeline.name if pipeline else None,
                "people": pipeline.live() if pipeline else [],
                "fps": round(pipeline.fps, 1) if pipeline else 0}

    @app.get("/api/live/thumb/{tracker_id}")
    def live_thumb(tracker_id: int, camera: str | None = None):
        pipeline = pick(camera)
        jpeg = pipeline.live_thumb(tracker_id) if pipeline else None
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
                **db.stats()}

    @app.get("/api/schema")
    def get_schema():
        fields = schema.MODE_FIELDS[cfg["mode"]]
        return {
            "mode": cfg["mode"],
            "fields": {n: schema.allowed_values(n) or schema.LIST_FIELDS[n] + ["none"] for n in fields},
            "list_fields": list(schema.LIST_FIELDS),
        }

    @app.get("/api/tracks")
    def tracks(limit: int = 50, name: str | None = None, value: str | None = None, camera: str | None = None):
        return db.list_tracks(limit=min(limit, 500), name=name, value=value, camera=camera)

    @app.post("/api/tracks/delete")
    def delete_tracks(body: DeleteIn):
        """Delete selected people (or everyone) with their pictures, labels and history."""
        deleted, paths = db.delete_tracks(body.ids, everything=body.everything)
        for p in pipelines:
            p.forget(deleted)
        removed = 0
        for name in paths:
            path = crop_dir / name
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
        return {"deleted": len(deleted), "pictures_removed": removed}

    @app.get("/api/search")
    def search(q: str, limit: int = 20):
        """Find people by a free-text description, e.g. "man in a beige coat"."""
        if embedder is None:
            raise HTTPException(503, "embeddings are disabled in config.yaml")
        hits = db.nearest(embedder.text(q), embedder.model_name, limit=min(limit, 100))
        tracks = db.get_tracks([tid for tid, _ in hits])
        sims = dict(hits)
        for t in tracks:
            t["similarity"] = round(sims[t["id"]], 4)
            t["match"] = round(embedder.match_probability(sims[t["id"]]), 4)
        return tracks

    @app.get("/api/tracks/{track_id}/similar")
    def similar(track_id: int, limit: int = 10):
        """People who look most like this one (same clothes; across cameras too)."""
        if embedder is None:
            raise HTTPException(503, "embeddings are disabled in config.yaml")
        vector = db.get_embedding(track_id, embedder.model_name)
        if vector is None:
            raise HTTPException(404, "no embedding for this person yet")
        hits = db.nearest(vector, embedder.model_name, limit=min(limit, 100), exclude=track_id)
        tracks = db.get_tracks([tid for tid, _ in hits])
        sims = dict(hits)
        for t in tracks:
            t["similarity"] = round(sims[t["id"]], 4)
        return tracks

    @app.get("/api/review")
    def review_queue(limit: int = 50):
        items = db.review_queue(limit=min(limit, 200))
        for item in items:
            item["allowed"] = schema.allowed_values(item["name"])
        return items

    @app.post("/api/review/{attribute_id}")
    def review(attribute_id: int, body: ReviewIn):
        try:
            db.apply_review(attribute_id, body.value.strip().lower(), body.reviewer)
        except KeyError:
            raise HTTPException(404, "attribute not found")
        return {"ok": True}

    return app

"""Local web UI: live view, people list with search, and the human review queue."""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
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


def create_app(cfg: dict, db: Database, pipeline: Pipeline | None, embedder=None) -> FastAPI:
    app = FastAPI(title="VisionLabeler")
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

    @app.get("/stream")
    async def stream():
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
    def live():
        """People in view right now and their current labels."""
        return {"people": pipeline.live() if pipeline else [],
                "fps": round(pipeline.fps, 1) if pipeline else 0}

    @app.get("/api/live/thumb/{tracker_id}")
    def live_thumb(tracker_id: int):
        jpeg = pipeline.live_thumb(tracker_id) if pipeline else None
        if jpeg is None:
            raise HTTPException(404, "not in view")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status():
        return {**(pipeline.status() if pipeline else {"mode": cfg["mode"]}), **db.stats()}

    @app.get("/api/schema")
    def get_schema():
        fields = schema.MODE_FIELDS[cfg["mode"]]
        return {
            "mode": cfg["mode"],
            "fields": {n: schema.allowed_values(n) or schema.LIST_FIELDS[n] + ["none"] for n in fields},
            "list_fields": list(schema.LIST_FIELDS),
        }

    @app.get("/api/tracks")
    def tracks(limit: int = 50, name: str | None = None, value: str | None = None):
        return db.list_tracks(limit=min(limit, 500), name=name, value=value)

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

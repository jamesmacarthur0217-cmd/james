"""The Recordings page and its API (labeler/recorder.py writes the files).

Admins play the MP4 file itself (the browser seeks with HTTP range requests), download a stretch (cut by FFmpeg
without re-encoding) and delete recordings. Viewers get a stream made here, frame by frame, with faces blurred from
the head boxes saved next to each video, so they never receive the file. Every viewing, download and delete is
in the Audit log (one entry per playback, not one per range request).

Search by words (Phase 16, labeler/moments.py): a description ("man with a backpack yesterday evening") finds
people, vehicles or animals as the Ask box does, and returns their moments in the recordings, for admins and viewers.
"""
from __future__ import annotations

import asyncio
import bisect
import json
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from . import privacy, recorder

STATIC = Path(__file__).resolve().parent / "static"
FIND_AHEAD_S = 120   # "play from 14:05" finds the next recording that starts within this time


class DeleteIn(BaseModel):
    ids: list[int]


class SearchIn(BaseModel):
    question: str
    limit: int = 100


class TimelapseIn(BaseModel):
    camera: str
    day: str
    again: bool = False   # make it anew (it is older than today's recording being written)


def frame_list(path: Path) -> tuple[dict, list[float], list[list]]:
    """A recording's header, each frame's time in the file (seconds from its start) and head boxes."""
    header, times, boxes = {}, [], []
    side = path.with_suffix(".jsonl")
    if not side.exists():
        return header, times, boxes
    with open(side, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                break   # a line cut short by a crash, or still being written
            if isinstance(row, dict):
                header = row
            else:
                times.append(row[0])
                boxes.append(row[1])
    return header, times, boxes


def frame_index(times: list[float], offset_s: float) -> int:
    """The first frame at or after `offset_s` seconds into the recording (the last one if none)."""
    return min(bisect.bisect_left(times, offset_s - 1e-6), max(0, len(times) - 1))


def add_routes(app, cfg: dict, db, audit, blur_for, agent=None, embedder=None, photo=None) -> None:
    base = recorder.folder(cfg)

    def is_admin(request: Request) -> bool:
        return (getattr(request.state, "user", None) or {}).get("role") == "admin"

    def get(rec_id: int) -> tuple[dict, Path]:
        rec = db.get_recording(rec_id)
        path = base / rec["path"] if rec else None
        if rec is None or not path.is_file():
            raise HTTPException(404, "no such recording")
        return rec, path

    @app.get("/recordings")
    def recordings_page():
        return FileResponse(STATIC / "recordings.html")

    @app.get("/api/recordings")
    def recordings_of_day(request: Request, camera: str | None = None, day: str | None = None):
        """One camera's recordings on one day, and when people appeared (for the day bar)."""
        try:
            start = datetime.fromisoformat(day) if day else datetime.now()
        except ValueError:
            raise HTTPException(400, "day must look like 2026-10-12")
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        cameras = sorted(set(db.recording_cameras()) | {c["name"] for c in cfg["cameras"]})
        camera = camera or (cameras[0] if cameras else None)
        admin = is_admin(request)
        recs = []
        for r in db.list_recordings(camera, start, end):
            item = {"id": r["id"], "start": r["started_at"], "end": r["ended_at"], "open": r["ended_at"] is None,
                    "frames": r["frames"]}
            if admin:
                item["size_mb"] = round(r["size_bytes"] / 1e6, 1)
            recs.append(item)
        people = [{"id": p["id"], "at": p["started_at"]}
                  for p in db.people_between(start, end, camera, limit=3000)] if camera else []
        return {"camera": camera, "day": start.date().isoformat(), "cameras": cameras, "recordings": recs,
                "people": people, "admin": admin, "fps": cfg["recording"]["fps"],
                "keep_days": cfg["recording"]["keep_days"], "enabled": cfg["recording"]["enabled"]}

    @app.get("/api/recordings/find")
    def find_recording(camera: str, at: str):
        """The recording showing `at` (or the next one starting soon after) and where in it: for "▶ Recording"
        buttons on people, alerts and the timeline."""
        try:
            when = datetime.fromisoformat(at)
        except ValueError:
            raise HTTPException(400, "at must look like 2026-10-12T14:05:00")
        if when.tzinfo is not None:
            when = when.astimezone().replace(tzinfo=None)
        for r in db.list_recordings(camera, when - timedelta(seconds=1), when + timedelta(seconds=FIND_AHEAD_S)):
            start = datetime.fromisoformat(r["started_at"])
            offset = max(0.0, (when - start).total_seconds())
            _, times, _ = frame_list(base / r["path"])
            if not times or (r["ended_at"] and offset > times[-1] + 1):
                continue
            i = frame_index(times, offset)
            return {"id": r["id"], "start": r["started_at"], "offset_s": times[i], "frame": i,
                    "video_s": round(i / float(cfg["recording"]["fps"]), 2)}
        raise HTTPException(404, "nothing was recorded then")

    @app.post("/api/recordings/search")
    def search_recordings(body: SearchIn, request: Request):
        """Moments in the recordings for a description (labeler/moments.py). With the vision model, the Ask box's
        search (time, camera, labels, vehicles, animals); without it, people ranked by how they look only."""
        from . import moments
        question = " ".join(body.question.split())[:300]
        if not question:
            raise HTTPException(400, "type what to look for, e.g. woman in a red top this morning")
        limit = max(1, min(body.limit, 100))
        if agent is not None:
            result = agent.run(question, limit=limit)
            items = ([{**p, "type": "person"} for p in result.get("people") or []]
                     + [{**v, "type": "vehicle"} for v in result.get("vehicles") or []]
                     + [{**a, "type": "animal"} for a in result.get("animals") or []])
            explanation, how = result["explanation"], "question"
        elif embedder is not None:
            hits = db.nearest(embedder.text(question), embedder.model_name, limit=min(limit, 30))
            items = [{**t, "type": "person"} for t in db.get_tracks([tid for tid, _ in hits])]
            explanation = (f"Looked for people who look like “{question}” (the vision model is off, so times, "
                           "cameras and vehicles are not understood; the closest look-alikes come first)")
            how = "appearance"
        else:
            raise HTTPException(503, "searching needs the vision model or the embeddings; both are off")
        audit(request, "search recordings", question)
        out = moments.moments(db, items, base, float(cfg["recording"]["fps"]))
        return {"question": question, "explanation": explanation, "how": how, **out,
                "recording": bool(cfg["recording"]["enabled"]), "keep_days": cfg["recording"]["keep_days"]}

    # --- the day's time-lapse (labeler/timelapse.py) ---------------------------------------------------------------
    from . import timelapse
    maker = app.state.timelapse = timelapse.Maker(cfg, db)

    def _day(day: str) -> str:
        try:
            return datetime.fromisoformat(day).date().isoformat()
        except ValueError:
            raise HTTPException(400, "day must look like 2026-10-12")

    def _camera(camera: str) -> str:
        if camera not in set(db.recording_cameras()) | {c["name"] for c in cfg["cameras"]}:
            raise HTTPException(404, f"no camera {camera!r}")
        return camera

    @app.get("/api/recordings/timelapse")
    def timelapse_status(camera: str, day: str):
        """Whether the day's time-lapse is ready (with its frame map), being made (with progress), or not made."""
        camera, day = _camera(camera), _day(day)
        srcs = timelapse.sources(db, base, camera, day)
        info = timelapse.current(base, camera, day, srcs) if srcs else None
        if info is not None:
            return {"state": "done", "camera": camera, "day": day, "fps": info["fps"], "frames": info["frames"],
                    "seconds": round(len(info["frames"]) / info["fps"], 1), "older": info["older"],
                    "made_at": info.get("made_at"), "version": info.get("made_at") or len(info["frames"])}
        job = maker.status()
        if job and (job["camera"], job["day"]) == (camera, day) and job["state"] in ("making", "failed"):
            return job
        return {"state": "none", "camera": camera, "day": day, "recorded": bool(srcs)}

    @app.post("/api/recordings/timelapse")
    def timelapse_start(body: TimelapseIn, request: Request):
        """Make the day's time-lapse in the background (one at a time); the page asks for its progress."""
        day = _day(body.day)
        try:
            job = maker.start(_camera(body.camera), day, again=body.again)
        except ValueError as e:
            raise HTTPException(404, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        if job["state"] == "making":
            audit(request, "make time-lapse", f"{body.camera} {day}")
        return job

    @app.get("/recordings/timelapse/{camera}/{day}.mp4")
    def timelapse_file(camera: str, day: str, request: Request):
        """The time-lapse: admins the picture as recorded, viewers the copy with faces blurred."""
        camera, day = _camera(camera), _day(day)
        path = timelapse.day_folder(base, camera, day) / (timelapse.BLURRED if blur_for(request) else timelapse.NAME)
        if not path.is_file():
            raise HTTPException(404, "no time-lapse of that day yet")
        if request.headers.get("range", "bytes=0-").startswith("bytes=0-"):   # a new playback, not a seek
            audit(request, "view time-lapse", f"{camera} {day}")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, no-cache"})

    @app.post("/api/recordings/search/photo")
    async def search_recordings_photo(request: Request):
        """Moments of the people who look like the person in a photo (the request body, a JPEG or PNG; not stored;
        labeler/photo.py), the closest first; only people at least photo_search.min_similarity alike."""
        from . import moments
        from .photo import PhotoSearch
        if embedder is None:
            raise HTTPException(503, "embeddings are disabled in config.yaml")
        searcher = photo or getattr(app.state, "photo", None)
        if searcher is None:
            searcher = app.state.photo = PhotoSearch(cfg, embedder)
        data = await request.body()
        try:
            vector, info = await run_in_threadpool(searcher.vector, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        pcfg = cfg.get("photo_search") or {}
        least, limit = float(pcfg.get("min_similarity", 0.85)), int(pcfg.get("limit", 20))
        hits = [(tid, s) for tid, s in await run_in_threadpool(db.nearest, vector, embedder.model_name, limit)
                if s >= least]
        sims = dict(hits)
        items = [{**t, "type": "person", "similarity": round(sims[t["id"]], 3)}
                 for t in await run_in_threadpool(db.get_tracks, [tid for tid, _ in hits])]
        n = info["people"]
        searched = ("the person in the photo" if n == 1 else "the person in the middle of the photo" if n
                    else "the whole photo (nobody found in it)")
        await run_in_threadpool(audit, request, "search recordings by photo",
                                f"picture {info['width']}x{info['height']}, {len(items)} people alike")
        out = await run_in_threadpool(lambda: moments.moments(db, items, base, float(cfg["recording"]["fps"]),
                                                              closest_first=True))
        return {"explanation": f"Looked for people who look like {searched}, the closest first (at least "
                               f"{least:.2f} alike; a look-alike, not a certain match)",
                "how": "photo", **out, "recording": bool(cfg["recording"]["enabled"]),
                "keep_days": cfg["recording"]["keep_days"]}

    @app.get("/recordings/file/{rec_id}")
    def recording_file(rec_id: int, request: Request):
        """The MP4 itself, for admins (not on the viewers' allowlist)."""
        rec, path = get(rec_id)
        if request.headers.get("range", "bytes=0-").startswith("bytes=0-"):   # a new playback, not a seek
            audit(request, "view recording", f"{rec['camera']} {rec['started_at']}")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=600"})

    @app.get("/recordings/stream/{rec_id}")
    async def recording_stream(rec_id: int, request: Request, t: float = 0.0, d: float | None = None):
        """The recording from `t` seconds on (for `d` seconds, a search's moment; else to the file's end), as a
        picture stream like the live view, faces blurred for viewers (the time is written on each frame)."""
        rec, path = get(rec_id)
        header, times, boxes = await run_in_threadpool(frame_list, path)
        if not times:
            raise HTTPException(404, "this recording has no frames yet")
        blur = blur_for(request)
        fps = float(header.get("fps") or cfg["recording"]["fps"])
        await run_in_threadpool(audit, request, "view recording", f"{rec['camera']} {rec['started_at']} from {t:.0f} s"
                                + (f" for {d:g} s" if d is not None else ""))

        def open_at(i: int):
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            return cap

        def next_jpeg(cap, i: int) -> bytes | None:
            ok, frame = cap.read()
            if not ok:
                return None
            if blur and i < len(boxes) and boxes[i]:
                frame = privacy.blur_boxes(frame, boxes[i])
            return privacy.encode_jpeg(frame)

        async def frames():
            i = frame_index(times, t)
            cap = await run_in_threadpool(open_at, i)
            try:
                while not app.state.stopping.is_set():
                    if d is not None and i < len(times) and times[i] > t + d:
                        break   # the moment is over
                    jpeg = await run_in_threadpool(next_jpeg, cap, i)
                    if jpeg is None:
                        break
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    i += 1
                    await asyncio.sleep(1 / fps)
            finally:
                cap.release()
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/recordings/{rec_id}/download")
    async def download_recording(rec_id: int, request: Request, start: float | None = None, end: float | None = None):
        """The whole file, or the stretch from `start` to `end` seconds (cut at key frames, no re-encoding)."""
        rec, path = get(rec_id)
        name = f"{rec['camera']}_{rec['started_at'].replace(':', '').replace('-', '')}"
        if start is None and end is None:
            await run_in_threadpool(audit, request, "download recording", f"{rec['camera']} {rec['started_at']}")
            return FileResponse(path, media_type="video/mp4", filename=f"{name}.mp4")
        ffmpeg = recorder.ffmpeg_path(cfg)
        if not ffmpeg.exists():
            raise HTTPException(503, "FFmpeg is not installed")
        out = Path(tempfile.mkstemp(suffix=".mp4", prefix="vl_cut_")[1])
        cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(max(0.0, start or 0.0))]
        if end is not None:
            cmd += ["-to", str(end)]
        cmd += ["-i", str(path), "-c", "copy", "-movflags", "+faststart", str(out)]
        result = await run_in_threadpool(lambda: subprocess.run(cmd, capture_output=True, timeout=300))
        if result.returncode != 0 or not out.stat().st_size:
            out.unlink(missing_ok=True)
            raise HTTPException(500, "the video could not be cut")
        await run_in_threadpool(audit, request, "download recording",
                                f"{rec['camera']} {rec['started_at']} {start or 0:.0f}-{end if end is not None else 'end'} s")
        return FileResponse(out, media_type="video/mp4", filename=f"{name}_{int(start or 0)}s.mp4",
                            background=BackgroundTask(out.unlink, missing_ok=True))

    @app.post("/api/recordings/delete")
    def delete_recordings(body: DeleteIn, request: Request):
        """Delete whole recordings (admins). One being written is left alone."""
        gone = []
        for rec_id in body.ids[:500]:
            rec = db.get_recording(rec_id)
            if rec is None or rec["ended_at"] is None:
                continue
            path = base / rec["path"]
            path.unlink(missing_ok=True)
            path.with_suffix(".jsonl").unlink(missing_ok=True)
            from .timelapse import drop_if_alone
            drop_if_alone(path.parent)
            gone.append(rec)
        db.delete_recordings([r["id"] for r in gone])
        if gone:
            audit(request, "delete recordings", ", ".join(f"{r['camera']} {r['started_at']}" for r in gone)[:1000])
        return {"deleted": len(gone)}

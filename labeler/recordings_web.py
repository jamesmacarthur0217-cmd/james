"""The Recordings page and its API (labeler/recorder.py writes the files).

Admins play the MP4 file itself (the browser seeks with HTTP range requests), download a stretch (cut by FFmpeg
without re-encoding) and delete recordings. Viewers get a stream made here, frame by frame, with faces blurred from
the head boxes saved next to each video, so they never receive the file. Every viewing, download and delete is
in the Audit log (one entry per playback, not one per range request).
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


def add_routes(app, cfg: dict, db, audit, blur_for) -> None:
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

    @app.get("/recordings/file/{rec_id}")
    def recording_file(rec_id: int, request: Request):
        """The MP4 itself, for admins (not on the viewers' allowlist)."""
        rec, path = get(rec_id)
        if request.headers.get("range", "bytes=0-").startswith("bytes=0-"):   # a new playback, not a seek
            audit(request, "view recording", f"{rec['camera']} {rec['started_at']}")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=600"})

    @app.get("/recordings/stream/{rec_id}")
    async def recording_stream(rec_id: int, request: Request, t: float = 0.0):
        """The recording from `t` seconds on, as a picture stream like the live view, faces blurred for viewers
        (the time is already written on each frame by the recorder)."""
        rec, path = get(rec_id)
        header, times, boxes = await run_in_threadpool(frame_list, path)
        if not times:
            raise HTTPException(404, "this recording has no frames yet")
        blur = blur_for(request)
        fps = float(header.get("fps") or cfg["recording"]["fps"])
        await run_in_threadpool(audit, request, "view recording", f"{rec['camera']} {rec['started_at']} from {t:.0f} s")

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
            gone.append(rec)
        db.delete_recordings([r["id"] for r in gone])
        if gone:
            audit(request, "delete recordings", ", ".join(f"{r['camera']} {r['started_at']}" for r in gone)[:1000])
        return {"deleted": len(gone)}

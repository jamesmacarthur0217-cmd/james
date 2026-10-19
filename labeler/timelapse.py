"""Time-lapse of a day (Phase 34): about a minute of video showing everything one camera recorded that day.

From the day's recording files (labeler/recorder.py, 10 frames a second, the camera name and time written on every
frame), every `step`-th frame is kept so the day fits `seconds` (60 by default) at 25 frames a second, and encoded
with the recorder's encoder (the graphics card when there is one). Decoding every recorded frame is the work (see
README "Time-lapse" for the time it takes). The files are read one by one rather than joined by FFmpeg: a day whose
camera changed size (FFmpeg's joining then dropped half the frames) still comes out whole, with black bars.

Made on request and kept next to the recordings (<camera>/<day>/timelapse.mp4) with a frame map (timelapse.json:
for each time-lapse frame, the recording and the second in it), so a click in the time-lapse opens the real recording.
It is made again when the day's recordings change (a new one, one deleted, the one being written grew).
"""
from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from . import recorder
from .recordings_web import frame_list

OUT_FPS = 25
DEFAULT_SECONDS = 60
NAME, MAP, BLURRED = "timelapse.mp4", "timelapse.json", "timelapse.blurred.mp4"


def day_folder(base: Path, camera: str, day: str) -> Path:
    return base / camera / day


def sources(db, base: Path, camera: str, day: str) -> list[dict]:
    """The day's recordings in order, with their frame times (only the ones with frames)."""
    start = datetime.fromisoformat(day)
    out = []
    for r in db.list_recordings(camera, start, start + timedelta(days=1)):
        if not r["started_at"].startswith(day):
            continue   # began the day before: it belongs to that day's time-lapse
        path = base / r["path"]
        _, times, boxes = frame_list(path)
        if times and path.is_file():
            out.append({"id": r["id"], "path": path, "start": r["started_at"], "times": times, "boxes": boxes,
                        "open": r["ended_at"] is None})
    return out


def plan(srcs: list[dict], seconds: float = DEFAULT_SECONDS) -> dict:
    """Which frames to keep: every `step`-th of the day's frames, and for each kept frame where it is from."""
    total = sum(len(s["times"]) for s in srcs)
    step = max(1, math.ceil(total / (seconds * OUT_FPS)))
    frames, n = [], 0
    for s in srcs:
        for i in range(len(s["times"])):
            if n % step == 0:
                at = datetime.fromisoformat(s["start"]) + timedelta(seconds=s["times"][i])
                frames.append([s["id"], round(s["times"][i], 2), at.isoformat(timespec="seconds")])
            n += 1
    return {"step": step, "total": total, "frames": frames, "fps": OUT_FPS,
            "sources": [[s["id"], len(s["times"])] for s in srcs]}


def current(base: Path, camera: str, day: str, srcs: list[dict]) -> dict | None:
    """The kept time-lapse's frame map, if it is made from these recordings. When only the recording still being
    written has grown since (today's, while someone is in view), it is still shown, marked "older": the page offers
    to make it again."""
    folder = day_folder(base, camera, day)
    try:
        info = json.loads((folder / MAP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (folder / NAME).is_file():
        return None
    saved, now = info.get("sources") or [], [[s["id"], len(s["times"])] for s in srcs]
    if saved == now:
        return {**info, "older": False}
    grown = (len(saved) == len(now) and all(a[0] == b[0] for a, b in zip(saved, now))
             and all(a[1] == b[1] or (s["open"] and b[1] >= a[1]) for a, b, s in zip(saved, now, srcs)))
    return {**info, "older": True} if grown else None


def drop_if_alone(folder: Path) -> None:
    """A day's time-lapse goes with the day's last recording (deleted by hand or by keep_days / max_gb); the parts kept
    around alerts do not hold it (its frame map points to the recordings themselves)."""
    if folder.is_dir() and not [p for p in folder.iterdir() if not p.name.startswith("timelapse")
                                and "-alert" not in p.name]:
        for p in folder.glob("timelapse*"):
            p.unlink(missing_ok=True)


class Maker:
    """One time-lapse at a time (FFmpeg uses the processor); the pages ask for its progress."""

    def __init__(self, cfg: dict, db, writer_factory=None):
        self.cfg, self.db, self.writer_factory = cfg, db, writer_factory
        self.base = recorder.folder(cfg)
        self._lock = threading.Lock()
        self.job: dict | None = None      # {"camera", "day", "state": "making" / "done" / "failed", ...}

    def status(self) -> dict | None:
        with self._lock:
            return dict(self.job) if self.job else None

    def start(self, camera: str, day: str, seconds: float = DEFAULT_SECONDS, again: bool = False) -> dict:
        """Start making a day's time-lapse in the background (or say it is ready, or busy with another).
        again: make it anew although one is kept (it is older than today's recording being written)."""
        srcs = sources(self.db, self.base, camera, day)
        if not srcs:
            raise ValueError("nothing was recorded that day")
        info = current(self.base, camera, day, srcs)
        if info is not None and not (again and info["older"]):
            return {"state": "done", "camera": camera, "day": day, **_summary(info)}
        with self._lock:
            if self.job and self.job["state"] == "making":
                if (self.job["camera"], self.job["day"]) == (camera, day):
                    return dict(self.job)
                raise RuntimeError(f"busy making the time-lapse of {self.job['camera']} {self.job['day']}")
            self.job = {"state": "making", "camera": camera, "day": day, "done_frames": 0,
                        "total": sum(len(s["times"]) for s in srcs), "started": time.time()}
        threading.Thread(target=self._make, args=(camera, day, srcs, seconds), name="timelapse", daemon=True).start()
        return self.status()

    def _make(self, camera: str, day: str, srcs: list[dict], seconds: float) -> None:
        try:
            info = make(self.cfg, self.base, camera, day, srcs, seconds, self._progress, self.writer_factory)
            with self._lock:
                self.job.update(state="done", took_s=round(time.time() - self.job["started"], 1), **_summary(info))
        except Exception as e:
            with self._lock:
                self.job.update(state="failed", error=str(e)[:300])

    def _progress(self, frames: int) -> None:
        with self._lock:
            if self.job:
                self.job["done_frames"] = frames


def _summary(info: dict) -> dict:
    return {"frames": len(info["frames"]), "step": info["step"], "seconds": round(len(info["frames"]) / OUT_FPS, 1),
            "size_mb": info.get("size_mb")}


def fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The frame in a picture of `size` (width, height), shrunk to fit with black bars (a camera's size can change)."""
    w, h = size
    fh, fw = frame.shape[:2]
    if (fw, fh) == (w, h):
        return frame
    scale = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    out = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def make(cfg: dict, base: Path, camera: str, day: str, srcs: list[dict], seconds: float = DEFAULT_SECONDS,
         progress=None, writer_factory=None, blurred: bool = True) -> dict:
    """Make <camera>/<day>/timelapse.mp4 and its frame map from the day's recordings; returns the map.

    The files are read one after the other (OpenCV: frames not kept are only decoded, not converted) and every
    `step`-th frame goes to the recorder's FFmpeg writer, fitted to the first frame's size; the map lists the frames
    actually written, so it stays right when a file has fewer frames than its frame list (the one being written).
    blurred: also a copy with faces blurred (from the head boxes kept with each recording), for viewers."""
    from . import privacy
    ffmpeg = recorder.ffmpeg_path(cfg)
    if writer_factory is None:
        if not ffmpeg.exists():
            raise RuntimeError("FFmpeg is not installed (tools/get_ffmpeg.py)")
        encoder = recorder.pick_encoder(ffmpeg, cfg["recording"]["encoder"])
        if encoder is None:
            raise RuntimeError("no H.264 encoder works on this PC")
        quality = int(cfg["recording"]["quality"])
        writer_factory = lambda path, size: recorder.FFmpegWriter(ffmpeg, path, size, OUT_FPS, encoder, quality)  # noqa: E731
    info = plan(srcs, seconds)
    step = info["step"]
    folder = day_folder(base, camera, day)
    out, tmp = folder / NAME, folder / "timelapse.part.mp4"
    bout, btmp = folder / BLURRED, folder / "timelapse.blurred.part.mp4"
    writer, bwriter, size, frames, n = None, None, None, [], 0
    try:
        for s in srcs:
            cap = cv2.VideoCapture(str(s["path"]))
            i = 0
            while i < len(s["times"]) and cap.grab():
                if n % step == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        if writer is None:
                            size = (frame.shape[1] // 2 * 2, frame.shape[0] // 2 * 2)
                            writer = writer_factory(tmp, size)
                            bwriter = writer_factory(btmp, size) if blurred else None
                        writer.write(fit(frame, size))
                        if bwriter is not None:
                            heads = s["boxes"][i] if i < len(s.get("boxes") or []) else []
                            bwriter.write(fit(privacy.blur_boxes(frame, heads) if heads else frame, size))
                        at = datetime.fromisoformat(s["start"]) + timedelta(seconds=s["times"][i])
                        frames.append([s["id"], round(s["times"][i], 2), at.isoformat(timespec="seconds")])
                        if progress is not None and len(frames) % 50 == 0:
                            progress(n)
                i += 1
                n += 1
            cap.release()
    finally:
        closed = writer.close() if writer is not None else (False, "no frames could be read")
        bclosed = bwriter.close() if bwriter is not None else (True, "")
    if not closed[0] or not tmp.is_file() or not bclosed[0]:
        tmp.unlink(missing_ok=True)
        btmp.unlink(missing_ok=True)
        raise RuntimeError(f"the time-lapse could not be made: {closed[1] or bclosed[1]}")
    tmp.replace(out)
    if blurred:
        btmp.replace(bout)
    info["frames"] = frames
    info["size_mb"] = round(out.stat().st_size / 1e6, 1)
    info["made_at"] = datetime.now().isoformat(timespec="seconds")
    (folder / MAP).write_text(json.dumps(info), encoding="utf-8")
    return info

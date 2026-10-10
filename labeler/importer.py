"""Import recorded videos (the Import page): people in old footage are found, labeled and stored like people seen
live, with the real time they were filmed, so they can be searched, reported and reviewed.

Fast: frames are read as quickly as the GPU can process them (no waiting for the video's own speed), and only
about `target_fps` frames per video second are used (tracking works on fewer frames than a camera sends; README
has the measurement). The vision model still labels every person: the import waits while more than
MAX_VLM_QUEUE jobs are queued, so live cameras are delayed by at most that many jobs.

Real times: each file's start time is, first that works: typed on the page > the file name (phones and
recorders put date and time in it: VID_20261009_143012.mp4, 20261009_143012.mp4, ch01_20261009143012.avi) >
the MP4/MOV header (creation time, UTC) > the file's modified time minus its length. The page shows which one
was used, so it can be corrected before starting.

Old footage is not live: no alerts, no alert clips, no camera health checks. It is kept until deleted by hand,
unless the import says otherwise (cameras.retention_days), since the 30-day rule would otherwise delete
footage older than 30 days within the hour.
"""
from __future__ import annotations

import copy
import logging
import re
import struct
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .db import IMPORTED
from .sources import VIDEO_EXTS, RecordedVideo, probe_video

log = logging.getLogger(__name__)

MAX_VLM_QUEUE = 2
MAX_FILES = 500
FILE_GAP_S = 3600          # between files of one import, so nobody is re-linked from one file to the next
DRAIN_WAIT_S = 600         # at the end: wait for the last labels to come back

# YYYY MM DD [sep] hh mm ss, with optional - _ . : separators (VID_20261009_143012, 2026-10-09 14.30.12, ...).
NAME_TIME = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ _T.-]?(\d{2})[-_.:h]?(\d{2})[-_.:m]?(\d{2})")


# --- start time of a recording ---------------------------------------------------------------------

def time_from_name(name: str, now: datetime | None = None) -> datetime | None:
    now = now or datetime.now()
    for m in NAME_TIME.finditer(name):
        try:
            t = datetime(*(int(g) for g in m.groups()))
        except ValueError:
            continue
        if datetime(1990, 1, 1) <= t <= now + timedelta(days=1):
            return t
    return None


MP4_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)


def mp4_creation_time(path: Path) -> datetime | None:
    """The creation time in an MP4/MOV file's movie header (moov/mvhd), as local time. Reads only box headers,
    so it is quick for big files (the header is often at the end)."""
    try:
        with open(path, "rb") as f:
            size = path.stat().st_size

            def boxes(start: int, end: int):
                pos = start
                while pos + 8 <= end:
                    f.seek(pos)
                    box_size, kind = struct.unpack(">I4s", f.read(8))
                    header = 8
                    if box_size == 1:
                        box_size, header = struct.unpack(">Q", f.read(8))[0], 16
                    elif box_size == 0:
                        box_size = end - pos
                    if box_size < header:
                        return
                    yield kind, pos + header, pos + box_size
                    pos += box_size

            for kind, body, end in boxes(0, size):
                if kind != b"moov":
                    continue
                for child, cbody, _ in boxes(body, end):
                    if child == b"mvhd":
                        f.seek(cbody)
                        version = f.read(4)[0]
                        seconds = struct.unpack(">Q" if version == 1 else ">I", f.read(8 if version == 1 else 4))[0]
                        if seconds < 86400 * 365 * 86:   # 0 (unset) or before 1990
                            return None
                        return (MP4_EPOCH + timedelta(seconds=seconds)).astimezone().replace(tzinfo=None)
    except (OSError, struct.error, IndexError):
        return None
    return None


def start_time(path: Path, duration_s: float, typed: datetime | None = None) -> tuple[datetime, str]:
    """(start, how it was found)."""
    if typed is not None:
        return typed, "typed"
    t = time_from_name(path.stem)
    if t is not None:
        return t, "file name"
    t = mp4_creation_time(path) if path.suffix.lower() in (".mp4", ".mov", ".m4v", ".3gp") else None
    if t is not None:
        return t, "video header"
    return datetime.fromtimestamp(path.stat().st_mtime) - timedelta(seconds=duration_s), "file time"


def find_videos(path: str | Path) -> list[Path]:
    """A video file, or the videos in a folder and its subfolders, in name order."""
    p = Path(str(path).strip().strip('"'))
    if p.is_file():
        if p.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"{p.name} is not a video file ({', '.join(sorted(VIDEO_EXTS))})")
        return [p]
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
        if not files:
            raise ValueError(f"no video files in {p}")
        if len(files) > MAX_FILES:
            raise ValueError(f"{len(files)} videos in {p}: import at most {MAX_FILES} at once")
        return files
    raise ValueError(f"no file or folder {p}")


def describe(files: list[Path]) -> list[dict]:
    """What the page shows before an import: each file's length and proposed start time."""
    out = []
    for f in files:
        try:
            info = probe_video(f)
            start, how = start_time(f, info["duration_s"])
            out.append({"path": str(f), "name": f.name, **info, "start": start.isoformat(timespec="seconds"),
                        "start_from": how})
        except ValueError as e:
            out.append({"path": str(f), "name": f.name, "error": str(e)})
    return out


EXCERPT_BEFORE_S = 2
EXCERPT_MAX_S = 30


def make_excerpt(src: Path, offset_s: float, in_view_s: float, out: Path, fps: float = 10, width: int = 640) -> Path:
    """A short web-playable video (VP8 WebM) of the original file around one person's moment: from 2 s before
    they appear, as long as they were in view (+2 s), at most 30 s. Works for any format OpenCV reads (an AVI
    from a recorder plays in no browser). Written under a temporary name, then renamed."""
    import cv2
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"cannot open {src.name}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        start = max(0.0, offset_s - EXCERPT_BEFORE_S)
        end = start + min(EXCERPT_MAX_S, in_view_s + 2 * EXCERPT_BEFORE_S)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        fps = min(fps, src_fps)
        step, due, writer, part = 1 / fps, start, None, out.with_name(out.stem + ".part.webm")
        out.parent.mkdir(parents=True, exist_ok=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if t > end:
                break
            if t + 1e-6 < due:
                continue
            due += step
            h, w = frame.shape[:2]
            if writer is None:
                size = (min(width, w) // 2 * 2, int(h * min(width, w) / w) // 2 * 2)
                writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
                if not writer.isOpened():
                    raise RuntimeError("the VP8 video encoder is not available")
            writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        if writer is None:
            raise ValueError(f"no pictures at {offset_s:.0f} s in {src.name}")
        writer.release()
    finally:
        cap.release()
    part.replace(out)
    return out


def stride_for(fps: float, target_fps: float) -> int:
    return max(1, round(fps / max(target_fps, 0.1)))


# --- the import job --------------------------------------------------------------------------------

class Importer:
    """Runs one import at a time inside the app, sharing its models (detector weights, embeddings, age model,
    classifiers, the vision model worker). `register(p)` / `unregister(p)` add an import pipeline to the
    routing of vision model answers (run.py)."""

    def __init__(self, cfg: dict, db, worker=None, embedder=None, student=None, age=None,
                 register=None, unregister=None, pipeline_factory=None):
        self.cfg, self.db, self.worker = cfg, db, worker
        self.embedder, self.student, self.age = embedder, student, age
        self.register, self.unregister = register or (lambda p: None), unregister or (lambda p: None)
        self._factory = pipeline_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.job: dict | None = None
        self.history: list[dict] = []

    @property
    def running(self) -> bool:
        return self.job is not None and self.job["state"] in ("running", "finishing")

    def status(self) -> dict:
        with self._lock:
            return {"job": copy.deepcopy(self.job), "history": copy.deepcopy(self.history[-10:])}

    def start(self, files: list[dict], camera: str, mode: str, keep_days: int, target_fps: float,
              user: str = "?", imgsz: int | None = None) -> dict:
        """files: [{"path", "start" (datetime)}]. Raises ValueError when an import is already running."""
        with self._lock:
            if self.running:
                raise ValueError("an import is already running; wait for it or cancel it")
            self._stop.clear()
            self.job = {"id": len(self.history) + 1, "camera": camera, "mode": mode, "keep_days": keep_days,
                        "target_fps": target_fps, "imgsz": imgsz, "user": user, "state": "running", "error": None,
                        "started_at": datetime.now().isoformat(timespec="seconds"), "finished_at": None,
                        "files": [{"path": str(f["path"]), "name": Path(f["path"]).name,
                                   "start": f["start"].isoformat(timespec="seconds"), "state": "waiting",
                                   "people": 0} for f in files],
                        "current": None, "done_s": 0.0, "total_s": 0.0, "people": 0, "speed": None}
            job = self.job
        threading.Thread(target=self._run, args=(job, files), name="import", daemon=True).start()
        return copy.deepcopy(job)

    def cancel(self) -> bool:
        if not self.running:
            return False
        self._stop.set()
        return True

    def _pipeline(self, camera: str, mode: str, imgsz: int | None = None):
        cfg = copy.deepcopy(self.cfg)
        cfg["clips"]["enabled"] = False      # no alert clips for old footage
        cfg.setdefault("person_videos", {})["enabled"] = False   # the original video is there already
        cam = {"name": camera, "source": IMPORTED, "mode": mode, "loop_files": False, "ignore_zones": []}
        if imgsz:
            cam["imgsz"] = imgsz   # people far away or seen from above need a bigger detector picture
        if self._factory is not None:
            return self._factory(cfg, cam)
        from .pipeline import Pipeline
        p = Pipeline(cfg, self.db, self.worker, self.embedder, self.student, camera=cam)
        p.alerts = None                     # no alerts for old footage
        p.age = self.age
        return p

    def _set(self, job: dict, **fields) -> None:
        with self._lock:
            job.update(fields)

    def _run(self, job: dict, files: list[dict]) -> None:
        from .pipeline import VideoClock
        t0 = time.monotonic()
        p = None
        try:
            infos = [probe_video(f["path"]) for f in files]
            self._set(job, total_s=round(sum(i["duration_s"] for i in infos), 1))
            self.db.get_or_create_camera(job["camera"], IMPORTED, job["mode"])
            self.db.set_camera_retention(job["camera"], int(job["keep_days"]))
            p = self._pipeline(job["camera"], job["mode"], job.get("imgsz"))
            self.register(p)
            p.running = True
            base, done_before = 0.0, 0.0
            log.info("import into camera %s: %d files, %.0f s of video", job["camera"], len(files), job["total_s"])
            for i, (f, info) in enumerate(zip(files, infos)):
                if self._stop.is_set():
                    break
                with self._lock:
                    job["current"] = i
                    job["files"][i]["state"] = "running"
                before = set(p.owned_ids)
                self._new_tracker(p)
                p.clock = VideoClock(f["start"], base)
                video = RecordedVideo(f["path"], stride_for(info["fps"], job["target_fps"]))
                for frame, t in video.frames(self._stop):
                    p.clock.t = t
                    while self.worker is not None and self.worker.pending > MAX_VLM_QUEUE and not self._stop.is_set():
                        time.sleep(0.05)
                    p.process(frame)
                    p.frames += 1
                    if p.frames % 10 == 0:
                        elapsed = time.monotonic() - t0
                        self._set(job, done_s=round(done_before + t, 1), people=len(p.owned_ids),
                                  speed=round((done_before + t) / elapsed, 1) if elapsed > 0 else None)
                for tid in list(p.tracks):   # the end of a file: everyone in view leaves
                    p._finish(tid)
                # Each person remembers the original file and their moment in it ("seen in door.mp4 at 0:31").
                self.db.set_sources(set(p.owned_ids) - before, str(f["path"]), f["start"])
                p.recently_lost.clear()
                base += info["duration_s"] + FILE_GAP_S
                # A cancelled file counts only as far as it got.
                done_before += p.clock.t if self._stop.is_set() else info["duration_s"]
                with self._lock:
                    job["files"][i].update(state="cancelled" if self._stop.is_set() else "done",
                                           people=len(set(p.owned_ids) - before))
                    job["people"] = len(p.owned_ids)
                    job["done_s"] = round(done_before, 1)
            self._set(job, state="finishing")
            self._drain()
            elapsed = time.monotonic() - t0
            self._set(job, state="cancelled" if self._stop.is_set() else "done", people=len(p.owned_ids),
                      speed=round(job["done_s"] / elapsed, 1) if elapsed > 0 else None)
            log.info("import into camera %s %s: %d people from %.0f s of video in %.0f s", job["camera"],
                     job["state"], job["people"], job["done_s"], elapsed)
        except Exception as e:
            log.exception("import into camera %s failed", job["camera"])
            self._set(job, state="failed", error=str(e)[:300])
        finally:
            if p is not None:
                p.running = False
                self.unregister(p)
            with self._lock:
                job["finished_at"] = datetime.now().isoformat(timespec="seconds")
                job["current"] = None
                self.history.append(copy.deepcopy(job))

    def _new_tracker(self, p) -> None:
        """A fresh tracker per file: track numbers and motion from the previous file must not carry over."""
        if hasattr(p, "det_cfg"):
            from ultralytics import YOLO
            p.model = YOLO(p.det_cfg["model"])

    def _drain(self) -> None:
        """Wait for the last labeling jobs, so their answers still reach this import's people."""
        deadline = time.monotonic() + DRAIN_WAIT_S
        while self.worker is not None and self.worker.pending and time.monotonic() < deadline:
            time.sleep(0.2)

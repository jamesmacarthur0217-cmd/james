"""Recording: H.264 videos of each live camera, written by FFmpeg (PHASE13-PLAN.md).

With `recording.when: people` a camera records from `before_s` seconds before the first person appears (kept in
memory as JPEGs) until `after_s` seconds after the last one leaves; a person coming back within that time
continues the same recording. `when: always` records all the time. Long recordings are split into files of
`max_file_min` minutes: <recording.dir>/<camera>/<YYYY-MM-DD>/<HHMMSS>.mp4, fragmented MP4, so a file cut short by
a crash still plays. Next to each video, <HHMMSS>.jsonl has one line per frame: its time in the file (seconds from
`started_at`, from the real clock, so a camera that stalled is still found at the right moment) and the head boxes
of the people in it (from the pipeline's keypoints), used to blur faces when a viewer watches. With
storage.blur_faces the faces are blurred before writing. The camera's name and the time are written on each frame.

The camera loop only resizes the frame and hands it over; encoding runs in FFmpeg (on the graphics card with
h264_nvenc), fed by a writer thread through a small queue. If the queue is full (a very slow disk), frames are
dropped and counted rather than slowing the camera. Recordings are listed in the `recordings` table; `prune()`
deletes the oldest by `keep_days` and `max_gb`, and recording pauses below storage.min_free_gb.

Before a recording is deleted, the minutes around each alert in it (`alert_before_s` before to `alert_after_s` after;
alerts close together make one part) are copied into a part of their own, <HHMMSS>-alert.mp4 with its own frame list
(Phase 42): a recording like any other (the player, the blurred stream for viewers, Download), marked `kept` in the
table and deleted after `alert_keep_days` instead of keep_days. The frames are copied as they are (OpenCV reads, the
same FFmpeg writer writes), so each frame keeps its time on the picture and its line in the frame list.
"""
from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from . import privacy
from .config import resolve

log = logging.getLogger(__name__)

ENCODERS = ("h264_nvenc", "h264_mf", "libopenh264")
QUEUE_FRAMES = 20          # raw frames waiting for FFmpeg (1280x720: about 55 MB)
DISK_CHECK_S = 60
_encoder_cache: dict[tuple[str, str], str | None] = {}


def ffmpeg_path(cfg: dict) -> Path:
    return resolve(cfg["recording"]["ffmpeg"])


def folder(cfg: dict) -> Path:
    return resolve(cfg["recording"]["dir"])


def encoder_args(encoder: str, quality: int) -> list[str]:
    """Quality as NVENC's cq (lower = better); the other encoders get settings measured to give similar files."""
    if encoder == "h264_nvenc":
        return ["-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"]
    if encoder == "h264_mf":
        return ["-rate_control", "quality", "-quality", str(max(1, min(100, 116 - 2 * quality)))]
    return ["-b:v", f"{int(2000 * 2 ** ((28 - quality) / 5))}k"]   # libopenh264: a bit rate (2 Mbit/s at 28)


def pick_encoder(ffmpeg: Path, wanted: str = "auto", run=subprocess.run) -> str | None:
    """The first H.264 encoder that works on this PC (a graphics card without NVENC makes h264_nvenc fail)."""
    key = (str(ffmpeg), wanted)
    if key in _encoder_cache:
        return _encoder_cache[key]
    found = None
    for enc in (ENCODERS if wanted == "auto" else (wanted,)):
        cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
               "testsrc=size=256x144:rate=10:duration=1", "-c:v", enc, *encoder_args(enc, 28), "-f", "null", "-"]
        try:
            if run(cmd, capture_output=True, timeout=30).returncode == 0:
                found = enc
                break
        except (OSError, subprocess.SubprocessError):
            break   # FFmpeg itself does not start
    _encoder_cache[key] = found
    return found


class FFmpegWriter:
    """One video file: raw BGR frames in through a pipe, fragmented MP4 out."""

    def __init__(self, ffmpeg: Path, path: Path, size: tuple[int, int], fps: float, encoder: str, quality: int):
        self.path = path
        self._err = tempfile.TemporaryFile()
        w, h = size
        cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", encoder, *encoder_args(encoder, quality),
               "-g", str(int(fps * 2)), "-pix_fmt", "yuv420p",
               "-movflags", "+frag_keyframe+empty_moov+default_base_moof", str(path)]
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err,
                                      creationflags=flags)

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(frame.tobytes())

    def close(self, timeout: float = 30) -> tuple[bool, str]:
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            code = self._proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            code = -1
        self._err.seek(0)
        message = self._err.read().decode(errors="replace").strip()[-300:]
        self._err.close()
        return code == 0, message


class Recorder:
    """Records one camera. feed() is called by the camera loop with every frame it reads."""

    def __init__(self, cfg: dict, camera: str, db, encoder: str, writer_factory=None, drop_when_full: bool = True):
        rc = cfg["recording"]
        self.cfg, self.rc, self.camera, self.db, self.encoder = cfg, rc, camera, db, encoder
        self.folder = folder(cfg)
        self.fps = float(rc["fps"])
        self.blur = bool(cfg["storage"].get("blur_faces", False))
        self.min_free = float(cfg["storage"]["min_free_gb"]) * 1e9
        self.writer_factory = writer_factory or (
            lambda path, size: FFmpegWriter(ffmpeg_path(cfg), path, size, self.fps, encoder, int(rc["quality"])))
        self._next_due = -1e9
        self._buffered: deque = deque()        # (monotonic, wall) of the frames in the writer's memory buffer
        self._buffer: deque = deque()          # the writer's: (monotonic, wall, jpeg, heads) before a recording
        self._current: dict | None = None      # the file being written: id, start (wall), opened (monotonic)
        self._last_person = -1e9
        self._disk_ok, self._disk_checked = True, -1e9
        self.dropped = 0
        self.drop_when_full = drop_when_full   # tests feed faster than real time and wait instead
        self.paused = ""                       # why recording is paused (disk almost full), for the Health page
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self._thread = threading.Thread(target=self._writer_loop, name=f"recorder-{camera}", daemon=True)
        self._thread.start()

    # --- camera thread ---------------------------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._current is not None

    def feed(self, frame: np.ndarray, people: bool, heads: list[tuple], now: float, wall: datetime) -> None:
        """Only bookkeeping here: shrinking, blurring and JPEG-making happen in the writer thread, so the camera
        loop is not slowed (the camera loop never changes a frame after reading it)."""
        if now < self._next_due:
            return
        self._next_due = max(self._next_due + 1 / self.fps, now - 1 / self.fps)   # on schedule, no catching up
        if self.rc["when"] == "always":
            people = True
        if people:
            self._last_person = now
        if self._current is None:
            if not (people and self._can_record(now)):
                if self.rc["when"] == "people":
                    self._buffered.append((now, wall))
                    while self._buffered and now - self._buffered[0][0] > self.rc["before_s"]:
                        self._buffered.popleft()
                    self._put(("buffer", frame, heads, now, wall))
                return
            self._open(self._buffered[0][1] if self._buffered else wall, frame.shape, now)
            self._buffered.clear()
            self._put(("flush",), block=True)
        self._put(("frame", frame, heads, wall), block=not self.drop_when_full)
        if now - self._last_person > self.rc["after_s"]:
            self._close()
        elif now - self._current["opened"] >= self.rc["max_file_min"] * 60:
            self._close()
            self._open(wall + timedelta(seconds=1 / self.fps), frame.shape, now)   # the next frame starts the next file

    def close(self, wait_s: float = 60) -> None:
        """The camera stops: finish the file being written."""
        if self._current is not None:
            self._close()
        self._put(("stop",), block=True)
        self._thread.join(wait_s)

    def _size(self, shape: tuple) -> tuple[int, int]:
        """The recorded picture size: at most max_width wide, even numbers (H.264 wants them)."""
        h, w = shape[:2]
        scale = min(1.0, self.rc["max_width"] / w)
        return int(w * scale) // 2 * 2, int(h * scale) // 2 * 2

    def _prepare(self, frame: np.ndarray, heads: list) -> tuple[np.ndarray, list]:
        """Writer thread: shrink, scale the head boxes, blur faces when storage.blur_faces."""
        h, w = frame.shape[:2]
        nw, nh = self._size(frame.shape)
        small = frame if (nw, nh) == (w, h) else cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        scale = nw / w
        boxes = [[int(round(v * scale)) for v in b] for b in heads]
        if self.blur and boxes:
            small = privacy.blur_boxes(small, boxes)
        return small, boxes

    def _can_record(self, now: float) -> bool:
        if now - self._disk_checked >= DISK_CHECK_S:
            self._disk_checked = now
            self.folder.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.folder).free
            ok = free >= self.min_free
            if ok != self._disk_ok:
                log.warning("[%s] recording %s", self.camera, "resumed" if ok else
                            f"paused: only {free / 1e9:.1f} GB free (storage.min_free_gb)")
            self._disk_ok = ok
            self.paused = "" if ok else f"disk almost full ({free / 1e9:.1f} GB free)"
        return self._disk_ok

    def _open(self, start: datetime, shape: tuple, now: float) -> None:
        day = self.folder / self.camera / start.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        path = day / f"{start:%H%M%S}.mp4"
        n = 2
        while path.exists():
            path = day / f"{start:%H%M%S}_{n}.mp4"
            n += 1
        rel = path.relative_to(self.folder).as_posix()
        rec_id = self.db.add_recording(self.camera, rel, start, self.encoder)
        self._current = {"id": rec_id, "start": start, "opened": now}
        self._put(("open", rec_id, path, self._size(shape), self._current["start"]), block=True)
        log.info("[%s] recording to %s", self.camera, rel)

    def _close(self) -> None:
        self._put(("close",), block=True)
        self._current = None

    def _put(self, item: tuple, block: bool = False) -> None:
        try:
            self._queue.put(item, block=block, timeout=10 if block else None)
        except queue.Full:
            self.dropped += 1
            if self.dropped in (1, 100) or self.dropped % 1000 == 0:
                log.warning("[%s] recording: the disk is too slow, %d frames dropped", self.camera, self.dropped)

    # --- writer thread ---------------------------------------------------------------------------

    def _writer_loop(self) -> None:
        f = None   # the open file: writer, sidecar, id, path, start, frames, last time
        while True:
            item = self._queue.get()
            try:
                if item[0] == "stop":
                    break
                if item[0] == "open":
                    _, rec_id, path, size, start = item
                    f = {"writer": self.writer_factory(path, size), "id": rec_id, "path": path, "start": start,
                         "size": size, "frames": 0, "t": 0.0,
                         # Line-buffered: each frame's line is on disk at once, like the frame itself in FFmpeg's
                         # pipe, so a crash or power cut leaves the two matching.
                         "side": open(path.with_suffix(".jsonl"), "w", encoding="utf-8", buffering=1)}
                    f["side"].write(json.dumps({"w": size[0], "h": size[1], "fps": self.fps}) + "\n")
                elif item[0] == "buffer":
                    _, frame, heads, now, wall = item
                    small, boxes = self._prepare(frame, heads)
                    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        self._buffer.append((now, wall, buf.tobytes(), boxes))
                    while self._buffer and now - self._buffer[0][0] > self.rc["before_s"]:
                        self._buffer.popleft()
                elif item[0] == "flush":
                    for _, wall, jpeg, boxes in self._buffer if f is not None else ():
                        self._write(f, cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), boxes, wall)
                    self._buffer.clear()
                elif item[0] == "frame" and f is not None:
                    _, frame, heads, wall = item
                    self._write(f, *self._prepare(frame, heads), wall)
                elif item[0] == "close" and f is not None:
                    self._finish(f)
                    f = None
            except Exception:
                log.exception("[%s] recording failed; this file is closed", self.camera)
                if f is not None:
                    self._finish(f)
                    f = None
        if f is not None:
            self._finish(f)

    def _write(self, f: dict, image: np.ndarray, boxes: list, wall: datetime) -> None:
        if image.shape[1] != f["size"][0] or image.shape[0] != f["size"][1]:
            image = cv2.resize(image, f["size"])   # the camera changed size mid-file
        f["writer"].write(stamp(image, f"{self.camera}  {wall:%Y-%m-%d %H:%M:%S}"))
        f["t"] = max(f["t"], (wall - f["start"]).total_seconds())
        f["side"].write(json.dumps([round(f["t"], 2), boxes]) + "\n")
        f["frames"] += 1

    def _finish(self, f: dict) -> None:
        ok, message = f["writer"].close()
        f["side"].close()
        path = f["path"]
        size = path.stat().st_size if path.exists() else 0
        if not size or not f["frames"]:
            log.warning("[%s] recording %s failed: %s", self.camera, path.name, message or "no frames")
            path.unlink(missing_ok=True)
            path.with_suffix(".jsonl").unlink(missing_ok=True)
            self.db.delete_recordings([f["id"]])
            return
        if not ok:
            log.warning("[%s] FFmpeg reported a problem with %s: %s", self.camera, path.name, message)
        self.db.update_recording(f["id"], frames=f["frames"], size_bytes=size,
                                 ended_at=f["start"] + timedelta(seconds=f["t"] + 1 / self.fps))


def stamp(image: np.ndarray, text: str) -> np.ndarray:
    """The camera and the time written on the picture (on a copy: the image may be the camera loop's own), so
    the time is right in the player, in downloads and in viewers' streams, however unevenly the camera ran."""
    out = image.copy()
    h = out.shape[0]
    size = max(0.45, h / 1100)
    width = 2 if h >= 600 else 1
    # The dark outline is the same text, same thickness, moved 2 px each way: OpenCV spaces the letters by their
    # thickness, so a thicker outline under thin text drifted apart along the line and the time looked doubled.
    for dx in (-2, 0, 2):
        for dy in (-2, 0, 2):
            if dx or dy:
                cv2.putText(out, text, (10 + dx, h - 12 + dy), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), width,
                            cv2.LINE_AA)
    cv2.putText(out, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), width, cv2.LINE_AA)
    return out


# --- housekeeping (run.py) -------------------------------------------------------------------------

def sidecar_times(path: Path) -> list[float]:
    """Each frame's time in a recording, from its .jsonl file (up to a line cut short by a crash)."""
    side = path.with_suffix(".jsonl")
    if not side.exists():
        return []
    times = []
    with open(side, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                break
            if isinstance(row, list):
                times.append(row[0])
    return times


def video_frames(path: Path, ffprobe: Path, run=subprocess.run) -> int | None:
    """How many frames really are in a video file (FFmpeg loses its last second or so when it is killed)."""
    if not ffprobe.exists():
        return None
    try:
        out = run([str(ffprobe), "-v", "error", "-count_packets", "-show_entries", "stream=nb_read_packets",
                   "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=120)
        return int(out.stdout.strip().split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def repair(db, cfg: dict, run=subprocess.run) -> int:
    """At startup: recordings left open by a crash or power cut get their end from their frame file, cut to the
    frames the video really has (ffprobe); empty ones are removed. Returns how many were closed."""
    base = folder(cfg)
    ffprobe = ffmpeg_path(cfg).with_name("ffprobe.exe" if ffmpeg_path(cfg).suffix == ".exe" else "ffprobe")
    closed = 0
    for r in db.open_recordings():
        path = base / r["path"]
        times = sidecar_times(path) if path.exists() and path.stat().st_size else []
        real = video_frames(path, ffprobe, run) if times else None
        if real is not None:
            times = times[:real]
        if not times:
            path.unlink(missing_ok=True)
            path.with_suffix(".jsonl").unlink(missing_ok=True)
            db.delete_recordings([r["id"]])
            continue
        start = datetime.fromisoformat(r["started_at"])
        db.update_recording(r["id"], ended_at=start + timedelta(seconds=times[-1]), size_bytes=path.stat().st_size,
                            frames=len(times))
        closed += 1
    if closed:
        log.info("recordings: closed %d left open by the previous run", closed)
    return closed


def frame_rows(path: Path) -> tuple[dict, list[tuple[float, list]]]:
    """A recording's frame list: its header and (time in the file, head boxes) per frame (up to a line cut short)."""
    header, rows = {}, []
    side = path.with_suffix(".jsonl")
    if not side.exists():
        return header, rows
    with open(side, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                break
            if isinstance(row, dict):
                header = row
            elif isinstance(row, list):
                rows.append((row[0], row[1]))
    return header, rows


def alert_stretches(alerts_s: list[float], duration: float, before: float, after: float,
                    join_s: float = 5.0) -> list[tuple[float, float]]:
    """[(from, to)] seconds in a file around each alert, within the file; overlapping or close ones joined."""
    out: list[list[float]] = []
    for t in sorted(alerts_s):
        a, b = max(0.0, t - before), min(duration, t + after)
        if b <= a:
            continue
        if out and a <= out[-1][1] + join_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def keep_around_alerts(db, cfg: dict, rec: dict, writer_factory=None) -> list[int]:
    """Copy the minutes around the alerts in a recording about to be deleted into parts of their own. Returns their
    ids (none: no alerts, keeping is off, or FFmpeg is missing)."""
    rc, base = cfg["recording"], folder(cfg)
    if float(rc.get("alert_keep_days", 0)) <= 0 or rec.get("kept") or not rec.get("ended_at"):
        return []
    path = base / rec["path"]
    start, end = datetime.fromisoformat(rec["started_at"]), datetime.fromisoformat(rec["ended_at"])
    alerts = db.alert_times(rec["camera"], start, end + timedelta(seconds=1))
    header, rows = frame_rows(path)
    if not alerts or not rows or not path.exists():
        return []
    stretches = alert_stretches([(a - start).total_seconds() for a in alerts], rows[-1][0],
                                float(rc.get("alert_before_s", 60)), float(rc.get("alert_after_s", 120)))
    fps = float(header.get("fps") or rc["fps"])
    if writer_factory is None:
        ffmpeg = ffmpeg_path(cfg)
        encoder = pick_encoder(ffmpeg, rc["encoder"]) if ffmpeg.exists() else None
        if encoder is None:
            log.warning("recordings: FFmpeg is missing, so the minutes around %d alerts in %s are not kept",
                        len(alerts), rec["path"])
            return []
        writer_factory = lambda p, size: FFmpegWriter(ffmpeg, p, size, fps, encoder, int(rc["quality"]))  # noqa: E731
    kept = []
    for a, b in stretches:
        first = next((i for i, (t, _) in enumerate(rows) if t >= a), None)
        if first is None:
            continue
        last = max(i for i, (t, _) in enumerate(rows) if t <= b) if rows[first][0] <= b else first
        t0 = rows[first][0]
        part_start = start + timedelta(seconds=t0)
        rel = Path(rec["path"]).parent / f"{part_start:%H%M%S}-alert.mp4"
        n = 2
        while (base / rel).exists():   # two parts starting in the same second
            rel = Path(rec["path"]).parent / f"{part_start:%H%M%S}-alert{n}.mp4"
            n += 1
        out = base / rel
        cap = cv2.VideoCapture(str(path))
        writer, written, size = None, 0, None
        try:
            with open(out.with_suffix(".jsonl"), "w", encoding="utf-8") as side:
                for i in range(last + 1):
                    if i < first:
                        if not cap.grab():
                            break
                        continue
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if writer is None:
                        size = (frame.shape[1], frame.shape[0])
                        writer = writer_factory(out, size)
                        side.write(json.dumps({"w": size[0], "h": size[1], "fps": fps, "kept": "alert",
                                               "from": rec["path"]}) + "\n")
                    writer.write(frame)
                    side.write(json.dumps([round(rows[i][0] - t0, 2), rows[i][1]]) + "\n")
                    written += 1
        finally:
            cap.release()
            ok, message = writer.close() if writer is not None else (False, "no frames")
        if not written or not out.exists() or not out.stat().st_size:
            log.warning("recordings: could not keep %s (%s)", rel.as_posix(), message)
            out.unlink(missing_ok=True)
            out.with_suffix(".jsonl").unlink(missing_ok=True)
            continue
        rec_id = db.add_recording(rec["camera"], rel.as_posix(), part_start, rec.get("encoder") or "")
        db.update_recording(rec_id, frames=written, size_bytes=out.stat().st_size, kept="alert",
                            ended_at=part_start + timedelta(seconds=rows[first + written - 1][0] - t0 + 1 / fps))
        kept.append(rec_id)
    if kept:
        log.info("recordings: kept %d part(s) around %d alert(s) of %s", len(kept), len(alerts), rec["path"])
    return kept


def prune(db, cfg: dict, now: datetime | None = None, writer_factory=None) -> int:
    """Delete recordings older than keep_days (kept parts: alert_keep_days), then the oldest while all of them are
    over max_gb; first keeping the minutes around alerts in each one deleted."""
    rc, base = cfg["recording"], folder(cfg)
    old = db.recordings_to_prune(float(rc["keep_days"]), int(float(rc["max_gb"]) * 1e9), now,
                                 float(rc.get("alert_keep_days", 0)))
    for r in old:
        if not r.get("kept"):
            try:
                keep_around_alerts(db, cfg, r, writer_factory)
            except Exception:   # never stop the deleting: the disk must not fill up
                log.exception("recordings: keeping the minutes around alerts in %s failed", r["path"])
        path = base / r["path"]
        path.unlink(missing_ok=True)
        path.with_suffix(".jsonl").unlink(missing_ok=True)
        from .timelapse import drop_if_alone
        drop_if_alone(path.parent)
        try:
            path.parent.rmdir()   # the day's folder, once empty
        except OSError:
            pass
    db.delete_recordings([r["id"] for r in old])
    if old:
        log.info("recordings: deleted %d old files (%.1f GB)", len(old), sum(r["size_bytes"] for r in old) / 1e9)
    return len(old)

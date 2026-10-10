"""Short video clips around alerts, and a short video of each person.

Each camera keeps its last `before_s` seconds of the live picture in memory (the JPEGs already made for the live
view, `fps` of them per second: about 5 MB per camera). When an alert fires, frames keep being collected for
`after_s` seconds, then the clip is written as WebM (VP8: built into OpenCV and played by every current
browser) by a background thread, so the camera never waits. A clip is written under a temporary name and
renamed when complete, so a half-written clip is never served. Clips show the labels and boxes, like the
Live page; with storage.blur_faces they are made from the blurred picture.

Person videos (person_videos): each person's visit is recorded from 1 s before they appear to 1 s after they
leave, at most max_s seconds (their arrival, which shows the most), smaller than alert clips (5 fps, 480 px):
about 1-2 MB each, since OpenCV's VP8 writer spends about width x height / 8 bytes on every frame (no quality setting). A person who turns out to be noise (too few frames) is dropped before anything is written.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def clip_name(event_id: int) -> str:
    """Path of an alert's clip, relative to the picture folder."""
    return f"clips/alert_{event_id}.webm"


def person_video_name(track_id: int, visit: int) -> str:
    """Path of a person's video for one visit (a returning person gets another), relative to the picture folder."""
    return f"clips/person_{track_id}_{visit}.webm"


class ClipRecorder:
    def __init__(self, crop_dir: Path, cfg: dict, person_cfg: dict | None = None):
        self.dir = crop_dir
        self.before_s, self.after_s = cfg["before_s"], cfg["after_s"]
        self.fps, self.max_width = cfg["fps"], cfg["max_width"]
        self.person = person_cfg if person_cfg and person_cfg.get("enabled") else None
        if self.person:
            self.before_s = max(self.before_s, self.person["before_s"])
        self._buffer: deque[tuple[float, bytes]] = deque()
        self._recording: list[dict] = []
        self._people: dict = {}   # key -> person recording still open (the person is in view)
        self._next_due = -1e9
        self._closed = False
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._writer, name="clip-writer", daemon=True)
        self._thread.start()

    def add_frame(self, jpeg: bytes, now: float | None = None) -> None:
        """Called by the camera loop with each new live picture."""
        now = time.monotonic() if now is None else now
        with self._lock:
            # Keep frames on a schedule of `fps` per second on average: a 15 fps camera then gives 10, not 7.5
            # (keeping "at least 0.1 s apart" would take every other frame). After a pause, no catching up.
            if now >= self._next_due:
                self._next_due = max(self._next_due + 1 / self.fps, now - 1 / self.fps)
                self._buffer.append((now, jpeg))
                for rec in self._recording:
                    rec["frames"].append((now, jpeg))
                for rec in self._people.values():
                    if now <= rec["cap"]:
                        rec["frames"].append((now, jpeg))
            while self._buffer and now - self._buffer[0][0] > self.before_s:
                self._buffer.popleft()
            done = [r for r in self._recording if now >= r["until"]]
            self._recording = [r for r in self._recording if now < r["until"]]
        for rec in done:
            self._queue.put(rec)

    def request(self, event_id: int, now: float | None = None) -> None:
        """An alert fired: save the last before_s seconds and the next after_s seconds."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._closed:   # a late labeling result after the camera stopped: no picture to show
                return
            self._recording.append({"event_id": event_id, "frames": list(self._buffer),
                                    "until": now + self.after_s})

    # --- person videos ---------------------------------------------------------------------------

    def start_person(self, key, now: float | None = None) -> None:
        """A person appeared: keep frames from `before_s` ago on, until finish_person or drop_person."""
        if self.person is None:
            return
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._closed:
                return
            pre = [(t, j) for t, j in self._buffer if now - t <= self.person["before_s"]]
            self._people[key] = {"frames": pre, "cap": now + self.person["max_s"], "person": True}

    def finish_person(self, key, name: str, now: float | None = None) -> None:
        """The person left and is kept: their video is written after `after_s` more seconds (or at the cap)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            rec = self._people.pop(key, None)
            if rec is None:
                return
            rec.update(name=name, until=min(now + self.person["after_s"], rec["cap"]))
            if self._closed or now >= rec["until"]:
                self._queue.put(rec)
            else:
                self._recording.append(rec)   # collects the last second like an alert clip, then is written

    def drop_person(self, key) -> None:
        """Noise, or deleted by the user while in view: nothing is written."""
        with self._lock:
            self._people.pop(key, None)

    def close(self, wait_s: float = 60) -> None:
        """The camera stops: write what the open clips have so far, then stop the writer."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            done, self._recording = self._recording, []
            self._people.clear()   # people still in view are finished by the camera loop before this
        for rec in done:
            self._queue.put(rec)
        self._queue.put(None)
        self._thread.join(timeout=wait_s)

    def _writer(self) -> None:
        while True:
            rec = self._queue.get()
            if rec is None:
                return
            try:
                self._write(rec)
            except Exception:
                log.exception("could not save the clip %s", rec.get("name") or f"of alert #{rec.get('event_id')}")

    def _write(self, rec: dict) -> None:
        frames = rec["frames"]
        person = rec.get("person", False)
        fps_wanted = self.person["fps"] if person else self.fps
        max_width = self.person["max_width"] if person else self.max_width
        if person and frames:   # fewer frames than the alert-clip buffer keeps: every n-th, on a schedule
            kept, due = [], -1e9
            for t, j in frames:
                if t >= due:
                    kept.append((t, j))
                    due = max(due + 1 / fps_wanted, t - 1 / fps_wanted)
            frames = kept
        if not frames:
            return
        images = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for _, j in frames]
        images = [im for im in images if im is not None]
        if not images:
            return
        h, w = images[0].shape[:2]
        if w > max_width:
            w, h = max_width, int(h * max_width / w)
        w, h = w - w % 2, h - h % 2                        # encoders want even sizes
        span = frames[-1][0] - frames[0][0]
        # The real rate (a slow camera gives fewer frames), so the clip plays at normal speed.
        fps = min(fps_wanted, max(1.0, (len(frames) - 1) / span)) if span > 0 else fps_wanted
        final = self.dir / (rec["name"] if person else clip_name(rec["event_id"]))
        final.parent.mkdir(parents=True, exist_ok=True)
        part = final.with_name(final.stem + ".part.webm")
        writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("the VP8 video encoder is not available")
        for im in images:
            writer.write(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) if im.shape[:2] != (h, w) else im)
        writer.release()
        os.replace(part, final)
        if person:
            log.debug("saved %s (%.0f s, %d frames)", rec["name"], span, len(images))
        else:
            log.info("saved the clip of alert #%s (%.0f s, %d frames)", rec["event_id"], span, len(images))

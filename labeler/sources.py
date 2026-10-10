"""Frame sources: webcam, video file, still image, or RTSP/HTTP stream."""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

# Without this, Media Foundation takes ~8 s to open many USB webcams.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import cv2  # noqa: E402
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FrameSource:
    def __init__(self, source: str | int, loop: bool = True):
        self.source = source
        self.loop = loop
        self.ended = False
        text = str(source)
        if text.isdigit():
            self.kind = "camera"
        elif text.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
            self.kind = "stream"
        elif Path(text).suffix.lower() in IMAGE_EXTS:
            self.kind = "image"
        else:
            self.kind = "video"

        self._image: np.ndarray | None = None
        self._cap: cv2.VideoCapture | None = None
        self._latest: np.ndarray | None = None
        self._latest_id = 0
        self._read_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._next_t = 0.0

        if self.kind == "image":
            self._image = cv2.imread(text)
            if self._image is None:
                raise FileNotFoundError(f"cannot read image {text}")
            self._period = 1 / 15
        elif self.kind == "video":
            if not Path(text).exists():
                raise FileNotFoundError(f"video not found: {text}")
            self._cap = cv2.VideoCapture(text)
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 25
            self._period = 1 / fps
        else:
            # Live sources: read on a thread and keep only the newest frame, so a slow
            # pipeline never falls behind real time.
            self._thread = threading.Thread(target=self._reader, name="frame-reader", daemon=True)
            self._thread.start()

    def _open_live(self) -> cv2.VideoCapture:
        if self.kind == "camera":
            # Media Foundation on Windows: DirectShow returned black frames from this PC's webcam.
            backend = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(self.source), backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
        return cap

    def _reader(self) -> None:
        while not self._stop.is_set():
            cap = self._open_live()
            if not cap.isOpened():
                log.warning("cannot open %s %s; retrying in 3 s", self.kind, self.source)
                time.sleep(3)
                continue
            log.info("opened %s %s", self.kind, self.source)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log.warning("lost %s %s; reconnecting", self.kind, self.source)
                    break
                with self._lock:
                    self._latest = frame
                    self._latest_id += 1
            cap.release()
            time.sleep(1)

    def read(self) -> np.ndarray | None:
        """Next frame, or None if none is ready yet (check `ended` for end of file)."""
        if self.kind in ("camera", "stream"):
            with self._lock:
                if self._latest_id == self._read_id:
                    frame = None
                else:
                    frame, self._read_id = self._latest, self._latest_id
            if frame is None:
                time.sleep(0.005)
            return frame

        # Files play back at their native rate, like a live camera.
        now = time.monotonic()
        if now < self._next_t:
            time.sleep(self._next_t - now)
        self._next_t = max(now, self._next_t) + self._period

        if self.kind == "image":
            return self._image.copy()
        ok, frame = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                self.ended = True
                return None
        return frame

    def release(self) -> None:
        self._stop.set()
        if self._cap is not None:
            self._cap.release()


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".wmv", ".mts", ".m2ts", ".ts", ".webm", ".3gp"}


class RecordedVideo:
    """A recorded video for the Import page: read as fast as the computer can (no waiting between frames, no
    looping), every `stride`-th frame, each with its position in the file in seconds. Positions come from the
    file's own timestamps, which is right for phone videos with a changing frame rate."""

    def __init__(self, path: str | Path, stride: int = 1):
        self.path = Path(path)
        self.stride = max(1, int(stride))
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"cannot open the video {self.path.name}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_s = self.frame_count / self.fps if self.frame_count else 0.0

    def frames(self, stop: threading.Event | None = None):
        """(frame, seconds into the video) for every stride-th frame. Skipped frames are only grabbed, not
        converted, which is quicker."""
        index = 0
        try:
            while stop is None or not stop.is_set():
                if index % self.stride:
                    if not self._cap.grab():
                        return
                    index += 1
                    continue
                ok, frame = self._cap.read()
                if not ok:
                    return
                pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                # Some files report no position (0 after the first frame): count frames instead.
                yield frame, (pos_ms / 1000 if pos_ms > 0 or index == 0 else index / self.fps)
                index += 1
        finally:
            self._cap.release()


def probe_video(path: str | Path) -> dict:
    """Frame rate, frame count and length of a video file, without reading it."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open the video {Path(path).name}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"fps": round(fps, 2), "frames": n, "duration_s": round(n / fps, 1) if n else 0.0,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def is_blank(frame: np.ndarray) -> bool:
    """True when the picture has no detail at all (lens cover, camera busy, or still starting)."""
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    return float(small.std()) < 3.0


def list_cameras(max_index: int = 5) -> list[tuple[int, str]]:
    """Probe camera indexes and report which ones give a real picture."""
    backend = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame, start = None, time.monotonic()
        while time.monotonic() - start < 3:  # give auto-exposure time to settle
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            found.append((idx, "opens, but sends no frames"))
        elif is_blank(frame):
            found.append((idx, f"{frame.shape[1]}x{frame.shape[0]}, but the picture is blank"))
        else:
            found.append((idx, f"{frame.shape[1]}x{frame.shape[0]}, picture OK"))
    return found

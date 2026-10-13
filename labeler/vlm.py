"""Attribute labeling with a local vision-language model served by Ollama."""
from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import cv2
import numpy as np
import requests

from . import schema

log = logging.getLogger(__name__)

# {images} names exactly the pictures sent. Measured 2026-10-08: listing them ("Images: 1 = ...; 2 = ...")
# instead of "Image 2, if present, is ..." scored higher on both answer keys (see README, Accuracy).
IMAGES_KIOSK = {False: "Images: 1 = the whole person.",
                True: "Images: 1 = the whole person; 2 = a close-up of their face."}
PROMPT_KIOSK = """You are labeling a person seen by a camera.
{images}
Describe only what is clearly visible. Answer "unknown" whenever something is not clearly visible.
age_range is an estimate of apparent age, not a fact.
apparent_gender: answer "man" or "woman" when the face, hair, build or clothing make it reasonably clear;
answer "unknown" only when you really cannot tell.
upper_* is clothing on the upper body, lower_* is clothing on the legs; use "unknown" for legs that are out of view.
If the head is visible and nothing is on it, headwear is "none". If the eyes are visible without glasses, eyewear is
"none". If the face is visible and uncovered, face_mask is "no".
For carrying, answer "yes" only for an item you can actually see the person holding or wearing as a bag.
A lanyard, badge, tie or clothing is not a carried item.
A "t-shirt" has no collar and no buttons, with short or long sleeves; a "shirt" has a collar.
Thin, rimless or clear glasses count as "glasses"; dark lenses are "sunglasses"."""

PROMPT_SECURITY = """You are labeling a person seen by a surveillance camera.
Describe only what is clearly visible. Answer "unknown" whenever something is not clearly visible.
age_group is a rough apparent estimate. upper_* is clothing on the upper body, lower_* is clothing on the legs.
If the head is visible and nothing is on it, headwear is "none".
For carrying, answer "yes" only for an item you can actually see the person holding or wearing as a bag."""

# Added to either prompt when mobility aids are labeled (schema.OPTIONAL_FIELDS). The look-alikes named here are
# the likely mistakes: chairs, umbrellas and anything pushed on wheels.
PROMPT_MOBILITY = """
mobility_aid: what the person uses to sit or walk, if anything. Answer a mobility aid (wheelchair, walking stick,
crutches, walker) only when the person is clearly using it for support. Poles held while skiing or hiking are "ski or
hiking poles"; bats, rackets, boards and other sport gear are "other sports equipment"; a seat that is not a
wheelchair is "an ordinary chair or bench"; something pushed that is not a walker frame is "a trolley, pram or
luggage"; a person who stands behind, pushes or stands next to a wheelchair that someone else sits in is "pushing or
standing next to a wheelchair". Answer "none" when the person uses nothing."""


def _b64(image: np.ndarray, max_side: int) -> str:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("could not encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class VLMClient:
    def __init__(self, cfg: dict, mode: str, labels: dict | None = None):
        self.host = cfg["host"].rstrip("/")
        self.model = cfg["model"]
        self.timeout = cfg["timeout_s"]
        self.mode = mode
        self.labels = labels or {}   # the `labels:` config section: optional fields turned on or off
        self._send_think = True

    def prompt(self, mode: str, with_face: bool) -> str:
        prompt = (PROMPT_KIOSK.replace("{images}", IMAGES_KIOSK[with_face]) if mode == "kiosk"
                  else PROMPT_SECURITY)
        if "mobility_aid" in schema.mode_fields(mode, self.labels):
            prompt += PROMPT_MOBILITY
        return prompt

    def available(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            r.raise_for_status()
        except requests.RequestException as e:
            return False, f"Ollama is not reachable at {self.host} ({e.__class__.__name__})"
        names = {m.get("name") for m in r.json().get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names:
            return False, f"model {self.model!r} is not pulled; run: ollama pull {self.model}"
        return True, "ok"

    def label(self, person: np.ndarray, face: np.ndarray | None, mode: str | None = None) -> dict[str, str]:
        """Label one person. Retries once if Ollama fails (it occasionally restarts its model process)."""
        try:
            return self._label(person, face, mode or self.mode)
        except (requests.ConnectionError, requests.Timeout) as e:
            log.warning("VLM request failed (%s); retrying once", e.__class__.__name__)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code < 500:
                raise
            log.warning("Ollama returned %s; retrying once", e.response.status_code)
        time.sleep(2)
        return self._label(person, face, mode or self.mode)

    def _label(self, person: np.ndarray, face: np.ndarray | None, mode: str) -> dict[str, str]:
        images = [_b64(person, 512)]
        if face is not None and mode == "kiosk":
            images.append(_b64(face, 256))
        body = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": self.prompt(mode, len(images) > 1),
                "images": images,
            }],
            "format": schema.json_schema(mode, self.labels),
            "stream": False,
            # The JSON answer is ~100 tokens; the cap stops a runaway generation early.
            "options": {"temperature": 0, "num_predict": 400},
            "keep_alive": "30m",
        }
        if self._send_think:
            body["think"] = False
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        if r.status_code == 400 and "think" in r.text and self._send_think:
            # Models without a thinking mode reject the flag entirely.
            self._send_think = False
            body.pop("think")
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return schema.clean(json.loads(content), mode, self.labels)


Job = tuple[int, list[tuple[np.ndarray, np.ndarray | None]], str | None]
ResultCallback = Callable[[int, list[dict[str, str]]], None]


class VLMWorker:
    """Runs VLM calls in the background so the camera loop never waits.

    Up to `workers` frames are sent to Ollama at once (Ollama needs OLLAMA_NUM_PARALLEL >= workers).
    Measured on an RTX 4070 SUPER: 2 at once labels people 1.9x faster than 1; 3 or 4 add nothing."""

    def __init__(self, client: VLMClient, on_result: ResultCallback, max_queue: int = 32, workers: int = 1):
        self.client = client
        self.on_result = on_result
        self.workers = max(1, workers)
        self.jobs: queue.Queue[Job | None] = queue.Queue(maxsize=max_queue)
        self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="vlm")
        self._slots = threading.Semaphore(self.workers)   # frames in flight
        self._idle = threading.Condition()
        self._jobs_in_flight = 0
        self.thread = threading.Thread(target=self._dispatch, name="vlm-dispatch", daemon=True)
        self.thread.start()

    @property
    def pending(self) -> int:
        """Jobs waiting for a free slot (not counting the ones being labeled)."""
        return self.jobs.qsize()

    def submit(self, track_id: int, frames: list[tuple[np.ndarray, np.ndarray | None]],
               mode: str | None = None) -> bool:
        try:
            self.jobs.put_nowait((track_id, frames, mode))
            return True
        except queue.Full:
            log.warning("VLM queue full; track %s will only get color labels", track_id)
            return False

    def _dispatch(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            track_id, frames, mode = job
            if not frames:
                self._deliver(track_id, [])
                continue
            state = {"results": [None] * len(frames), "left": len(frames), "lock": threading.Lock()}
            with self._idle:
                self._jobs_in_flight += 1
            for i, (person, face) in enumerate(frames):
                self._slots.acquire()   # waits here while all slots are busy, so the queue stays meaningful
                self._pool.submit(self._label_one, track_id, i, person, face, mode, state)

    def _label_one(self, track_id: int, i: int, person, face, mode, state: dict) -> None:
        try:
            state["results"][i] = self.client.label(person, face, mode)
        except Exception as e:  # keep going on any model/network error
            log.error("VLM labeling failed for track %s: %s", track_id, e)
        finally:
            self._slots.release()
        with state["lock"]:
            state["left"] -= 1
            last = state["left"] == 0
        if last:
            self._deliver(track_id, [r for r in state["results"] if r is not None])
            with self._idle:
                self._jobs_in_flight -= 1
                self._idle.notify_all()

    def _deliver(self, track_id: int, results: list[dict[str, str]]) -> None:
        try:
            # Called even with no results, so the pipeline knows this person is free to retry.
            self.on_result(track_id, results)
        except Exception:
            log.exception("storing VLM result for track %s failed", track_id)

    def close(self, wait_s: float = 0) -> bool:
        """Stop after the queued jobs finish, waiting at most wait_s.
        Returns False if jobs were still running when the wait ran out."""
        deadline = time.monotonic() + max(wait_s, 0.1)
        try:
            self.jobs.put(None, timeout=max(wait_s, 1))
        except queue.Full:
            return False
        self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._idle:
            while self._jobs_in_flight and time.monotonic() < deadline:
                self._idle.wait(timeout=max(0.0, deadline - time.monotonic()))
            finished = not self.thread.is_alive() and self._jobs_in_flight == 0
        self._pool.shutdown(wait=False)
        return finished

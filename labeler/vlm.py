"""Attribute labeling with a local vision-language model served by Ollama."""
from __future__ import annotations

import base64
import itertools
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


# Vehicles (labeler/vehicles.py): one picture, type, color, make and model. "not a vehicle" lets the model reject
# what the detector mistook for one (a lit shop sign was a "car" on MOT16-04).
# Measured on Stanford Cars (2026-10-13): with "not a vehicle" as one of the types, 8% of clear car photos (SUVs,
# minivans, pickups) were answered so (18% after adding "SUV" and a stricter sentence); a separate yes/no question
# first ("vehicle_in_picture") is asked instead.
VEHICLE_TYPES = ["car", "SUV", "van", "pickup", "lorry", "bus", "motorbike", "bicycle", "unknown"]
VEHICLE_COLORS = [c for c in schema.COLORS if c != "unknown"] + ["silver", "unknown"]
PROMPT_VEHICLE = """You are describing a vehicle seen by a security camera. The picture is cut out around it.
vehicle_in_picture: "yes" if a car, van, lorry, bus, motorbike or bicycle is in the picture; "no" only for a sign,
a poster, a building or a person without one.
type: what the vehicle is. A minivan or MPV is a "van".
color: the main color of its body (not windows, wheels or the road); "silver" for a metallic light gray.
make: the brand (for example Perodua, Proton, Toyota, Honda, Yamaha), only if the badge, logo or shape makes it
clear; otherwise "unknown". model: the model name (for example Myvi, Saga, Vios, City), only if clear; otherwise
"unknown". Do not guess from the color alone."""


def vehicle_schema() -> dict:
    props = {"vehicle_in_picture": {"type": "string", "enum": ["yes", "no"]},
             "type": {"type": "string", "enum": VEHICLE_TYPES},
             "color": {"type": "string", "enum": VEHICLE_COLORS},
             "make": {"type": "string"}, "model": {"type": "string"}}
    return {"type": "object", "properties": props, "required": list(props)}


def clean_vehicle(raw: dict) -> dict[str, str]:
    """Off-list answers become "unknown"; make and model are short names (a sentence is not a name)."""
    out = {"type": "unknown"}
    if str(raw.get("vehicle_in_picture", "yes")).strip().lower() == "no":
        return {"type": "not a vehicle", "color": "unknown", "make": "unknown", "model": "unknown"}
    for name, allowed in (("type", VEHICLE_TYPES), ("color", VEHICLE_COLORS)):
        v = str(raw.get(name, "unknown")).strip()
        v = {a.lower(): a for a in allowed}.get(v.lower(), "unknown")
        out[name] = v
    for name in ("make", "model"):
        v = " ".join(str(raw.get(name, "")).split())[:40]
        bad = not v or v.lower() in ("unknown", "none", "n/a", "unclear") or len(v.split()) > 3 or not any(c.isalpha() for c in v)
        out[name] = "unknown" if bad else v
    return out


# Scene questions (labeler/scene.py): one yes/no question about the whole picture of a camera.
SCENE_ANSWERS = ["yes", "no", "unsure"]
SCENE_SIDE = 1024   # the whole picture, larger than a person's (512): a gate or a light is a small part of it
PROMPT_SCENE = """This is one picture from a camera at a home or a shop. Answer the question about this whole picture with
"yes" or "no". Say "unsure" only if the picture does not show enough to tell (too dark, blurred, or the place asked
about is not in view).
Question: {question}"""
# A question about one area of the picture (Phase 27): the area cut out ("part") or outlined on the whole picture.
PROMPT_SCENE_PART = """This is part of a picture from a camera at a home or a shop: the area the question is about. Answer the
question about this area with "yes" or "no". Say "unsure" only if the picture does not show enough to tell (too dark,
blurred, or the place asked about is not in view).
Question: {question}"""
PROMPT_SCENE_OUTLINE = """This is one picture from a camera at a home or a shop. The question is only about the area inside the
red outline: ignore everything outside it. Answer with "yes" or "no". Say "unsure" only if the picture does not show
enough to tell (too dark, blurred, or the place asked about is not in view).
Question: {question}"""
SCENE_PROMPTS = {"whole": PROMPT_SCENE, "part": PROMPT_SCENE_PART, "outline": PROMPT_SCENE_OUTLINE}


def scene_schema() -> dict:
    return {"type": "object", "properties": {"answer": {"type": "string", "enum": SCENE_ANSWERS}},
            "required": ["answer"]}


def clean_scene(raw: dict) -> str:
    answer = str(raw.get("answer", "")).strip().lower()
    return answer if answer in SCENE_ANSWERS else "unsure"


# Safety incidents (labeler/incidents.py, Phase 44): a fast check saw something; the vision model looks at a few
# pictures from the last seconds (oldest first) and confirms. "what" is its few words, shown with the alert.
INCIDENT_ANSWERS = SCENE_ANSWERS
INCIDENT_SIDE = 768   # each of several pictures; several at 1024 px would slow every answer
PROMPT_INCIDENT = """{pictures} from a security camera{area}.
Answer the question about what they show with "yes" or "no". Say "unsure" only if the pictures do not show enough to
tell (too dark, blurred, hidden). Judge only what is there and what people do, never how a person looks.
Then say in a few words (at most 12) what you see that decided your answer.
Question: {question}"""


def incident_prompt(n: int, gap_s: float, area: str, question: str) -> str:
    """The check's prompt: n pictures about gap_s apart; area: the area's name when the pictures are cut out."""
    pictures = (f"These are {n} pictures, oldest first, about {max(1, round(gap_s))} seconds apart," if n > 1
                else "This is a picture")
    return PROMPT_INCIDENT.format(pictures=pictures, area=f', cut out around the area "{area}"' if area else "",
                                  question=question)


def incident_schema() -> dict:
    props = {"answer": {"type": "string", "enum": INCIDENT_ANSWERS}, "what": {"type": "string"}}
    return {"type": "object", "properties": props, "required": list(props)}


def clean_incident(raw: dict) -> dict[str, str]:
    answer = str(raw.get("answer", "")).strip().lower()
    what = " ".join(str(raw.get("what", "")).split())[:120]
    return {"answer": answer if answer in INCIDENT_ANSWERS else "unsure", "what": what}


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

    def label(self, person: np.ndarray, face: np.ndarray | None, mode=None) -> dict[str, str]:
        """Label one person (or, with mode "vehicle", one vehicle; with mode ("scene", question), answer a question
        about a whole picture). Retries once if Ollama fails (it occasionally restarts its model process)."""
        if mode == "vehicle":
            return self._retry(lambda: self._label_vehicle(person))
        if isinstance(mode, tuple) and mode[0] == "scene":
            way = mode[2] if len(mode) > 2 else "whole"
            return self._retry(lambda: self._ask_scene(person, mode[1], way))
        if isinstance(mode, tuple) and mode[0] == "incident":   # person: the list of pictures; mode[1]: the prompt
            return self._retry(lambda: self._ask_incident(person, mode[1]))
        return self._retry(lambda: self._label(person, face, mode or self.mode))

    def _retry(self, call):
        try:
            return call()
        except (requests.ConnectionError, requests.Timeout) as e:
            log.warning("VLM request failed (%s); retrying once", e.__class__.__name__)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code < 500:
                raise
            log.warning("Ollama returned %s; retrying once", e.response.status_code)
        time.sleep(2)
        return call()

    def _chat(self, prompt: str, images: list[str], fmt: dict) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt, "images": images}],
                "format": fmt, "stream": False, "options": {"temperature": 0, "num_predict": 200}, "keep_alive": "30m"}
        if self._send_think:
            body["think"] = False
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        if r.status_code == 400 and "think" in r.text and self._send_think:
            self._send_think = False
            body.pop("think")
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])

    def _ask_scene(self, picture: np.ndarray, question: str, way: str = "whole") -> dict[str, str]:
        started = time.monotonic()
        answer = clean_scene(self._chat(SCENE_PROMPTS[way].format(question=question), [_b64(picture, SCENE_SIDE)],
                                        scene_schema()))
        return {"answer": answer, "seconds": f"{time.monotonic() - started:.2f}"}

    def _ask_incident(self, pictures: list, prompt: str) -> dict[str, str]:
        started = time.monotonic()
        out = clean_incident(self._chat(prompt, [_b64(p, INCIDENT_SIDE) for p in pictures], incident_schema()))
        out["seconds"] = f"{time.monotonic() - started:.2f}"
        return out

    def _label_vehicle(self, picture: np.ndarray) -> dict[str, str]:
        return clean_vehicle(self._chat(PROMPT_VEHICLE, [_b64(picture, 512)], vehicle_schema()))

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

# Who goes first (Phase 44): one queue, lowest number first, in arrival order within a number. Critical safety
# checks jump ahead of people's labels; warnings and slow rounds wait behind them.
CRITICAL, PEOPLE, WARNING, ROUND = 0, 10, 20, 30
_STOP = 99


class VLMWorker:
    """Runs VLM calls in the background so the camera loop never waits.

    Up to `workers` frames are sent to Ollama at once (Ollama needs OLLAMA_NUM_PARALLEL >= workers).
    Measured on an RTX 4070 SUPER: 2 at once labels people 1.9x faster than 1; 3 or 4 add nothing.
    Jobs wait in one priority queue (CRITICAL .. ROUND); `max_queue` limits the jobs waiting at PEOPLE and below, so a
    busy camera never fills the queue for a critical safety check."""

    def __init__(self, client: VLMClient, on_result: ResultCallback, max_queue: int = 32, workers: int = 1):
        self.client = client
        self.on_result = on_result
        self.workers = max(1, workers)
        self.max_queue = max_queue
        self.jobs: queue.PriorityQueue = queue.PriorityQueue()   # (priority, order, job or None)
        self._order = itertools.count()
        self._waiting: dict[int, int] = {}   # priority -> jobs waiting
        self._count_lock = threading.Lock()
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

    @property
    def idle(self) -> bool:
        """Nothing waiting and nothing being labeled: scene questions go only then (people and vehicles first)."""
        with self._idle:
            return self.jobs.qsize() == 0 and self._jobs_in_flight == 0

    def waiting(self, priority: int) -> int:
        """Jobs waiting at this priority."""
        with self._count_lock:
            return self._waiting.get(priority, 0)

    def submit(self, track_id: int, frames: list[tuple[np.ndarray, np.ndarray | None]],
               mode: str | None = None, priority: int = PEOPLE) -> bool:
        with self._count_lock:
            if priority >= PEOPLE and sum(n for p, n in self._waiting.items() if p >= PEOPLE) >= self.max_queue:
                log.warning("VLM queue full; job %s is dropped (people: only color labels)", track_id)
                return False
            self._waiting[priority] = self._waiting.get(priority, 0) + 1
        self.jobs.put((priority, next(self._order), (track_id, frames, mode)))
        return True

    def _dispatch(self) -> None:
        while True:
            # A free slot first, then the job: so the job taken is the most urgent one waiting at that moment (taken
            # first, it would sit here while all slots are busy, and a critical check sent meanwhile would wait).
            self._slots.acquire()
            priority, _, job = self.jobs.get()
            if job is None:
                self._slots.release()
                return
            with self._count_lock:
                self._waiting[priority] -= 1
            track_id, frames, mode = job
            if not frames:
                self._slots.release()
                self._deliver(track_id, [])
                continue
            state = {"results": [None] * len(frames), "left": len(frames), "lock": threading.Lock()}
            with self._idle:
                self._jobs_in_flight += 1
            for i, (person, face) in enumerate(frames):
                if i:
                    self._slots.acquire()   # the first frame has its slot already
                try:
                    self._pool.submit(self._label_one, track_id, i, person, face, mode, state)
                except RuntimeError:   # closed meanwhile (close() waited long enough): no answer for the rest
                    self._slots.release()
                    with self._idle:
                        self._jobs_in_flight -= 1
                        self._idle.notify_all()
                    self._deliver(track_id, [])
                    return

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
        self.jobs.put((_STOP, next(self._order), None))   # after every job already waiting
        self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._idle:
            while self._jobs_in_flight and time.monotonic() < deadline:
                self._idle.wait(timeout=max(0.0, deadline - time.monotonic()))
            finished = not self.thread.is_alive() and self._jobs_in_flight == 0
        self._pool.shutdown(wait=False)
        return finished

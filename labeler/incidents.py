"""Safety incidents (Phase 44): fast checks raise a *suspect*; the vision model confirms it before anything alerts.

The user's choice (2026-10-20, docs/PHASE44-49-PLAN.md): fast checks every frame or second (body points, YOLO
objects, simple rules: later phases) only say "this may be a fall / fire / fight"; the vision model then looks at a few
pictures from the last seconds and answers yes / no / unsure with a few words. Three parts here:

- FrameMemory: each safety camera keeps one picture a second for the last `memory_s` seconds (shrunk to `memory_side`
  px, ~1.5 MB each at 960 px), so a check sees what happened just before, also when recording is off.
- CATEGORIES: what each kind of incident is called, its level (critical / warning) and the question asked. The first
  wording; each later phase measures its category and may change it.
- IncidentAlerts (step 3): a "yes" becomes an alert for each enabled rule of that kind (built-in: one per kind,
  alerts.ensure_incident_rules), with the newest picture (and a face-blurred copy for viewers) and a clip; "no" and
  "unsure" alert nothing and only show on the Health page.
- Direct alerts (Phase 49): ConfirmQueue.direct alerts at once through the same listeners, for a pattern the vision
  model cannot see in a few seconds (someone pacing, in a restricted area at night) or the camera covered; with a
  question, the vision model is asked afterwards what the person is doing and its words are added to the alert's
  message (IncidentAlerts.add_words). It never stops the alert. At most one a `cooldown_s` per camera, kind, area and
  sign: a real run alerted 29 times in 90 s as people walked one after another into a restricted area.
- ConfirmQueue: suspects wait here (critical first, then oldest); one is sent to the vision model at a time, at most
  `confirm_per_min` a minute, in the shared priority queue (labeler/vlm.py: critical checks, then warnings, then
  people's labels, then the slow rounds). A critical check never waits for a lower one: it goes even while a warning
  is being checked, and the per-minute limit does not hold it back. The per-minute limit holds only while people's
  labels wait (the user's choice, Phase 50): when nothing waits, checks use the vision model's idle time. A suspect
  older than `max_wait_s` is dropped and counted; the same camera, category and area is
  not checked again within `cooldown_s` of being sent. The answer goes to the listeners (step 3: alerts) and stays in
  `recent` for the Health page.

Vision model jobs for incidents use their own numbers, -(INCIDENT_BASE + n), between the vehicles' (minus the vehicle's
number) and the scene questions' (below -2e9), so run.py routes the answers by number.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .vlm import CRITICAL, PEOPLE, ROUND, WARNING, incident_prompt

log = logging.getLogger(__name__)

INCIDENT_BASE = 1_000_000_000
MAX_FRAMES = 3
LEVELS = {"critical": CRITICAL, "warning": WARNING, "round": ROUND}   # level -> vision model priority

# category -> (level, words, question). Questions ask about what is there and what people do, never how they look.
CATEGORIES = {
    "person_down": ("critical", "person needing help",
                    "Is a person lying on the floor, collapsed, slumped or in distress, who may need help?"),
    "fall": ("critical", "fall or collision",
             "Did a person fall, or did a person collide with something or someone (a wheelchair or trolley tipping "
             "over counts)?"),
    "fire": ("critical", "fire or smoke",
             "Is there fire (flames) or smoke in a place where it should not be, not a normal stove, candle or "
             "fireplace?"),
    "fight": ("critical", "aggression",
              "Are people fighting, hitting, pushing someone hard, throwing things at someone, or threatening "
              "someone with gestures?"),
    "weapon": ("critical", "dangerous object",
               "Is a person holding a knife, gun or other weapon in a threatening way (not using a knife for food or "
               "work)?"),
    "child": ("critical", "child in danger",
              "Is a young child alone without an adult nearby, climbing furniture, or in a kitchen or cooking area?"),
    "floor": ("warning", "floor hazard",
              "Is there something dangerous on the floor where people walk: a spill, broken glass, dropped food or an "
              "object in the way?"),
    "exit": ("warning", "blocked exit or aisle", "Is the exit, door or aisle blocked by objects, furniture or people?"),
    "equipment": ("warning", "unsafe equipment",
                  "Is there unsafe equipment: an overturned table or chair, a broken chair, cables across the floor, or "
                  "something hanging loose?"),
    "pests": ("warning", "animals or pests", "Is there a rat, mouse, cockroach or bird near food or where food is made?"),
    "crowd": ("warning", "unusual crowding",
              "Is the place dangerously crowded: people packed tightly, pushing, or someone pressed against a barrier?"),
    "suspicious": ("warning", "suspicious behaviour",
                   "Is a person doing something suspicious: trying doors or car doors, looking into windows or cars, "
                   "climbing a fence or wall, or hiding something? Judge only what they do."),
}


# A fall the fall check saw (labeler/falls.py) is checked for what is visible, not for need: asked "does this person
# need help?", the model called a real fall "exercising" (UR Fall fall-15, Phase 45 tuning half).
FALL_QUESTION = "Is a person lying or sitting on the floor (not on a bed, sofa or chair)?"


def job_id(n: int) -> int:
    return -(INCIDENT_BASE + n)


def number_of(job: int) -> int | None:
    return -job - INCIDENT_BASE if -2 * INCIDENT_BASE < job <= -INCIDENT_BASE else None


class FrameMemory:
    """The last few seconds of one camera, one picture every `every_s`, shrunk to at most `side` px."""

    def __init__(self, keep_s: float = 10.0, every_s: float = 1.0, side: int = 960):
        self.keep_s, self.every_s, self.side = float(keep_s), float(every_s), int(side)
        self._frames: deque = deque()   # (monotonic time, wall time, picture, head boxes in the picture)
        self._lock = threading.Lock()

    def due(self, now: float) -> bool:
        with self._lock:
            return not self._frames or now - self._frames[-1][0] >= self.every_s

    def add(self, frame: np.ndarray, wall: datetime, now: float, heads=()) -> bool:
        """Keep this frame if one is due; True when kept. heads: people's head boxes in the frame (for the viewers'
        blurred copy of an alert's picture)."""
        if not self.due(now):
            return False
        h, w = frame.shape[:2]
        scale = self.side / max(h, w)
        pic = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 \
            else frame.copy()
        k = min(1.0, scale)
        boxes = [tuple(round(v * k) for v in b[:4]) for b in heads]
        with self._lock:
            self._frames.append((now, wall, pic, boxes))
            while self._frames and now - self._frames[0][0] > self.keep_s:
                self._frames.popleft()
        return True

    def pick(self, n: int, span_s: float) -> list[tuple[datetime, np.ndarray, list]]:
        """n pictures (wall time, picture, head boxes) spread over the last span_s seconds, oldest first, the newest
        always included."""
        with self._lock:
            frames = list(self._frames)
        if not frames:
            return []
        newest = frames[-1][0]
        inside = [f for f in frames if newest - f[0] <= span_s] or frames[-1:]
        if len(inside) <= n:
            chosen = inside
        else:
            idx = sorted({round(i * (len(inside) - 1) / (n - 1)) for i in range(n)}) if n > 1 else [len(inside) - 1]
            chosen = [inside[i] for i in idx]
        return [(wall, pic, heads) for _, wall, pic, heads in chosen]

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


@dataclass
class Suspect:
    camera: str
    category: str
    reason: str                          # what the fast check saw ("lying still 20 s")
    area: dict | None = None             # the zone it is about (cut out for the vision model), or the whole picture
    people: list = field(default_factory=list)   # track ids involved
    pictures: list = field(default_factory=list)  # [(wall time, picture, head boxes)], oldest first
    at: datetime = field(default_factory=datetime.now)
    queued: float = 0.0                  # monotonic time it was queued
    number: int = 0
    crop: tuple | None = None            # (x1, y1, x2, y2) fractions: the person it is about, cut out with context
    question: str | None = None          # instead of the category's own question
    level_as: str | None = None          # instead of the category's level ("round": a slow round, after people)
    on_done: object = None               # called with the answer, or None when no answer came (failed, dropped):
                                         # the caller alerts itself (falls: only a clear "no" stops the alert)
    alert_if_yes: bool | None = None     # None: "yes" alerts unless on_done is set; False: a first look (floor
                                         # hazards: noted, the second "yes" alerts); True: alerts, on_done too
    cut: tuple | None = None             # (margin, smallest share) around crop instead of a person's (floor: tighter)

    @property
    def level(self) -> str:
        return self.level_as or CATEGORIES[self.category][0]

    @property
    def key(self) -> tuple:
        """Checked again only after the cooldown: the same camera, kind, area and (for a person's check) person."""
        return self.camera, self.category, (self.area or {}).get("id"), tuple(self.people[:1])


def prompt_for(s: Suspect, gap_s: float) -> str:
    return incident_prompt(len(s.pictures), gap_s, (s.area or {}).get("name", ""),
                           s.question or CATEGORIES[s.category][2])


CROP_MARGIN = 0.75   # around a person: this share of their box's width and height on each side (the floor, a chair)
CROP_MIN = 0.35      # at least this share of the picture's width and height: enough of the room to judge


def person_crop(pic: np.ndarray, crop: tuple, margin: float = CROP_MARGIN, minimum: float = CROP_MIN) -> np.ndarray:
    """The part of the picture around a person: a far person is too small for the vision model in the whole picture.
    margin: this share of the box on each side; minimum: at least this share of the picture's width and height."""
    h, w = pic.shape[:2]
    x1, y1, x2, y2 = crop
    bw, bh = max(x2 - x1, minimum / (1 + 2 * margin)), max(y2 - y1, minimum / (1 + 2 * margin))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = bw * (0.5 + margin), bh * (0.5 + margin)
    a, b = max(0, int((cx - half_w) * w)), max(0, int((cy - half_h) * h))
    c, d = min(w, int((cx + half_w) * w)), min(h, int((cy + half_h) * h))
    return pic[b:d, a:c] if c - a > 8 and d - b > 8 else pic


def pictures_for(s: Suspect) -> list[np.ndarray]:
    """What the vision model sees: each picture, cut out around the person (crop) or the area when there is one
    (labeler/scene.py), else whole."""
    if s.crop:
        return [person_crop(p[1], s.crop, *(s.cut or ())) for p in s.pictures]
    if not s.area:
        return [p[1] for p in s.pictures]
    from .scene import area_picture
    return [area_picture(p[1], s.area["points"], "part") for p in s.pictures]


class ConfirmQueue:
    def __init__(self, cfg: dict, worker, memories: dict[str, FrameMemory], clock=time.monotonic):
        c = cfg["safety"]
        self.per_min, self.max_wait_s = int(c["confirm_per_min"]), float(c["max_wait_s"])
        self.cooldown_s, self.span_s = float(c["cooldown_s"]), float(c["frames_span_s"])
        # At most 3: Ollama's context is 4096 tokens and 4 pictures at 768 px need ~4263 (refused, measured with
        # tools/incident_bench.py). A larger context per request would make Ollama reload the model between checks
        # and people's labels.
        self.frames = max(1, min(MAX_FRAMES, int(c["frames"])))
        self.worker, self.memories, self.clock = worker, memories, clock
        self.waiting: list[Suspect] = []
        self.sent: dict[int, Suspect] = {}         # number -> the suspect being checked
        self.sent_at: deque = deque()              # monotonic times of the last minute's checks
        self.last_sent: dict[tuple, float] = {}    # (camera, category, area id) -> when last sent
        self.recent: deque = deque(maxlen=50)      # the latest answers, newest last
        self.listeners: list = []                  # called with (suspect, answer dict)
        self.counts = {"suspects": 0, "sent": 0, "yes": 0, "no": 0, "unsure": 0, "failed": 0, "dropped_old": 0,
                       "cooldown": 0, "no_pictures": 0, "direct": 0}
        self.waits: deque = deque(maxlen=200)      # seconds each sent suspect waited
        self.dropped_at: deque = deque(maxlen=500)  # when suspects were dropped (waited too long), for the Health page
        self._numbers = itertools.count(1)
        self._lock = threading.Lock()

    def owns(self, job: int) -> bool:
        return number_of(job) is not None

    def suspect(self, camera: str, category: str, reason: str, area: dict | None = None, people=(),
                pictures: list | None = None, crop: tuple | None = None, question: str | None = None,
                level: str | None = None, on_done=None, span_s: float | None = None,
                alert_if_yes: bool | None = None, cut: tuple | None = None, cooldown: bool = True) -> str:
        """A fast check saw something. Returns "queued", "cooldown" (checked just now) or "no pictures"; only
        when "queued" will on_done be called (with the answer, or None when none came). cooldown=False: not held back
        by the cooldown (a direct alert's question, limited by direct() itself)."""
        if category not in CATEGORIES:
            raise ValueError(f"unknown incident category {category!r}")
        now = self.clock()
        s = Suspect(camera, category, reason, area, list(people), queued=now, crop=crop, question=question,
                    level_as=level, on_done=on_done, alert_if_yes=alert_if_yes, cut=cut)
        with self._lock:
            self.counts["suspects"] += 1
            if cooldown and (now - self.last_sent.get(s.key, -1e9) < self.cooldown_s
                             or any(w.key == s.key for w in self.waiting)):
                self.counts["cooldown"] += 1
                return "cooldown"
        memory = self.memories.get(camera)
        span = self.span_s if span_s is None else min(float(span_s), memory.keep_s if memory else span_s)
        s.pictures = pictures if pictures is not None else (memory.pick(self.frames, span) if memory else [])
        if not s.pictures:
            with self._lock:
                self.counts["no_pictures"] += 1
            return "no pictures"
        s.at = s.pictures[-1][0]   # the camera's own time (a video's when measuring or importing)
        with self._lock:
            self.waiting.append(s)
        return "queued"

    def direct(self, camera: str, category: str, reason: str, area: dict | None = None, people=(),
               crop: tuple | None = None, level: str | None = None, pictures: list | None = None,
               describe: str | None = None, sign: str = "") -> list[dict]:
        """Alert now, without waiting for the vision model's yes (see the module text). pictures: the alert's picture
        (default: the newest from memory); describe: a question about what the person is doing, asked afterwards,
        whose words are added to the alerts; sign: what was seen ("pacing"), for the once-a-minute limit. Returns the
        alerts made ([] within the limit)."""
        if category not in CATEGORIES:
            raise ValueError(f"unknown incident category {category!r}")
        key = ("direct", camera, category, (area or {}).get("id"), sign)
        with self._lock:
            now = self.clock()
            if now - self.last_sent.get(key, -1e9) < self.cooldown_s:
                self.counts["cooldown"] += 1
                return []
            self.last_sent[key] = now
        memory = self.memories.get(camera)
        s = Suspect(camera, category, reason, area, list(people), queued=self.clock(), crop=crop, level_as=level)
        s.pictures = pictures if pictures is not None else (memory.pick(1, 0) if memory else [])
        if s.pictures:
            s.at = s.pictures[-1][0]
        made = []
        for listener in self.listeners:
            try:
                made += listener(s, {"answer": "yes", "what": "", "direct": True}) or []
            except Exception:
                log.exception("a safety check listener failed")
        with self._lock:
            self.counts["direct"] += 1
            self.recent.append({"camera": camera, "category": category, "level": s.level, "reason": reason,
                                "area": (area or {}).get("name"), "at": s.at.isoformat(timespec="seconds"),
                                "answer": "direct", "what": "", "seconds": 0.0})
        log.warning("[%s] safety alert without a check: %s (%s)", camera, category, reason)
        if describe and made:
            def words(answer, made=made):
                for listener in self.listeners:
                    add = getattr(listener, "add_words", None)
                    if add is not None and answer:
                        add(made, answer)
            self.suspect(camera, category, reason, area=area, people=people, crop=crop, question=describe,
                         alert_if_yes=False, on_done=words, cooldown=False)
        return made

    def _labels_waiting(self) -> bool:
        """People's labels wait in the vision model's queue (a stand-in worker without a queue: as if they did)."""
        waiting = getattr(self.worker, "waiting", None)
        return waiting is None or waiting(PEOPLE) > 0

    def tick(self) -> Suspect | None:
        """Drop suspects that waited too long; send the next one when allowed. Returns the one sent."""
        now = self.clock()
        with self._lock:
            old = [s for s in self.waiting if now - s.queued > self.max_wait_s]
            for s in old:
                self.waiting.remove(s)
                self.counts["dropped_old"] += 1
                self.dropped_at.append(now)
                log.warning("[%s] safety check dropped after %.0f s waiting: %s (%s)", s.camera, now - s.queued,
                            s.category, s.reason)
        for s in old:
            self._done(s, None)
        with self._lock:
            while self.sent_at and now - self.sent_at[0] >= 60:
                self.sent_at.popleft()
            if not self.waiting:
                return None
            s = min(self.waiting, key=lambda w: (LEVELS[w.level], w.queued))
            critical = s.level == "critical"
            if critical:   # never behind a lower one, nor held back by the per-minute limit
                if any(x.level == "critical" for x in self.sent.values()):
                    return None
            elif self.sent or (len(self.sent_at) >= self.per_min and self._labels_waiting()):
                return None
            self.waiting.remove(s)
            s.number = next(self._numbers)
            self.sent[s.number] = s
            self.sent_at.append(now)
            self.last_sent[s.key] = now
            self.counts["sent"] += 1
            self.waits.append(now - s.queued)
        times = [p[0] for p in s.pictures]
        gap = (times[-1] - times[0]).total_seconds() / (len(times) - 1) if len(times) > 1 else 0
        ok = self.worker.submit(job_id(s.number), [(pictures_for(s), None)], ("incident", prompt_for(s, gap)),
                                priority=LEVELS[s.level])
        if not ok:
            self.on_result(job_id(s.number), [])
        return s

    def on_result(self, job: int, results: list[dict]) -> None:
        n = number_of(job)
        with self._lock:
            s = self.sent.pop(n, None)
            if s is None:
                return
            failed = not results
            if failed:
                self.counts["failed"] += 1
                log.warning("[%s] safety check failed (vision model): %s", s.camera, s.category)
        if failed:
            self._done(s, None)
            return
        with self._lock:
            answer = results[0]
            self.counts[answer["answer"]] += 1
            self.recent.append({"camera": s.camera, "category": s.category, "level": s.level, "reason": s.reason,
                                "area": (s.area or {}).get("name"), "at": s.at.isoformat(timespec="seconds"),
                                "answer": answer["answer"], "what": answer.get("what", ""),
                                "seconds": float(answer.get("seconds", 0) or 0)})
        log.info("[%s] safety check %s (%s): %s - %s", s.camera, s.category, s.reason, answer["answer"],
                 answer.get("what", ""))
        for listener in self.listeners:
            try:
                listener(s, answer)
            except Exception:
                log.exception("a safety check listener failed")
        self._done(s, answer)

    @staticmethod
    def _done(s: Suspect, answer: dict | None) -> None:
        if s.on_done is not None:
            try:
                s.on_done(answer)
            except Exception:
                log.exception("[%s] handling a safety check's answer failed", s.camera)

    def state(self) -> dict:
        with self._lock:
            waits = sorted(self.waits)
            return {**self.counts, "cameras": sorted(self.memories), "waiting": len(self.waiting),
                    "checking": len(self.sent),
                    "per_min": self.per_min, "last_minute": len(self.sent_at),
                    "wait_median_s": round(waits[len(waits) // 2], 1) if waits else None,
                    "recent": list(self.recent)[-10:]}

    def dropped_within(self, seconds: float) -> int:
        """Suspects dropped (waited too long) in the last `seconds`."""
        now = self.clock()
        with self._lock:
            return sum(1 for t in self.dropped_at if now - t <= seconds)

    def loop(self, stop: threading.Event, every_s: float = 0.5) -> None:
        while not stop.wait(every_s):
            try:
                self.tick()
            except Exception:
                log.exception("safety checks: sending failed")


def picture_name(s: Suspect) -> str:
    return f"incident_{s.camera}_{s.at:%Y%m%d%H%M%S}_{s.number}.jpg"


class IncidentAlerts:
    """A ConfirmQueue listener: a confirmed incident (the vision model said "yes") becomes an alert."""

    def __init__(self, db, alerts, pipelines: list, crop_dir: Path, clips_on: bool = True):
        self.db, self.alerts, self.crop_dir, self.clips_on = db, alerts, Path(crop_dir), clips_on
        self.by_name = {p.name: p for p in pipelines}

    def _save_picture(self, s: Suspect) -> str | None:
        """The newest picture, whole (the area outlined when there is one), and a copy with faces blurred."""
        from . import privacy
        if not s.pictures:
            return None
        _, pic, heads = s.pictures[-1]
        if s.area:
            from .scene import area_picture
            show = lambda f: area_picture(f, s.area["points"], "outline")   # noqa: E731
        else:
            show = lambda f: f   # noqa: E731
        name = picture_name(s)
        try:
            cv2.imwrite(str(self.crop_dir / name), show(pic), [cv2.IMWRITE_JPEG_QUALITY, 85])
            path = privacy.blurred_path(self.crop_dir, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), show(privacy.blur_boxes(pic, heads)), [cv2.IMWRITE_JPEG_QUALITY, 85])
            return name
        except Exception:
            log.exception("safety alert: could not save the picture")
            return None

    def __call__(self, s: Suspect, answer: dict) -> list[dict]:
        alert = s.on_done is None if s.alert_if_yes is None else s.alert_if_yes   # on_done: the caller alerts (falls)
        if answer.get("answer") != "yes" or not alert:
            return []
        area = (s.area or {}).get("name")
        rules = self.alerts.check_incident(s.camera, s.category, area)
        if not rules:
            return []
        _, words, _ = CATEGORIES[s.category]
        level = s.level if s.level in ("critical", "warning") else CATEGORIES[s.category][0]   # not "round"
        picture = self._save_picture(s)
        made = []
        for rule in rules:
            what = answer.get("what") or s.reason
            message = f"{rule['name']}: {what}" + (f" (in {area})" if area else "") + \
                ("" if answer.get("direct") else f"; seen first: {s.reason}")
            event = self.db.record_incident_alert(rule["id"], s.camera, message, s.category, level,
                                                  answer.get("what", ""), picture)
            if event is None:
                continue
            log.warning("ALERT [%s] %s: %s", s.camera, words, what)
            p = self.by_name.get(s.camera)
            clips = getattr(p, "clips", None)
            if clips is not None and self.clips_on:
                clips.request(event["id"])
            made.append(event)
        return made

    def add_words(self, events: list[dict], answer: dict) -> None:
        """A direct alert (ConfirmQueue.direct): the vision model's words on what the person is doing, added."""
        what = (answer or {}).get("what")
        if not what:
            return
        for e in events:
            try:
                self.db.update_alert_message(e["id"], f"{e['message']}; the vision model: {what}")
            except Exception:
                log.exception("safety alert: could not add the vision model's words")

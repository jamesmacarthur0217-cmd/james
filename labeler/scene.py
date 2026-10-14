"""Scene questions: yes/no questions the vision model answers about a camera's picture every few minutes.

An admin writes them on the Alerts page ("Is the gate open?", "Is there a car in the driveway?"; table
scene_questions). Every `scene.check_every_s` the runner looks for a question that is due (its interval, 5 minutes by
default, since the last answer) and sends the camera's latest clean picture (no boxes or labels; Pipeline.scene_frame)
to the vision model (or only one area of it, drawn on the Live page: area_picture), but only while it has nothing
else to do: people and vehicles always go first, and only one question is out at a time. The answer is "yes", "no" or "unsure"; it is stored (scene_answers) with the time it took,
and the picture is kept as scene_<id>.jpg (with a copy for viewers in which people's faces are blurred).

An alert comes when `answers_in_a_row` answers in a row (2, the user's choice) are the one the question alerts on
("yes", or "no" for a question like "Is the gate closed?"); one wrong answer alone does not. It does not come again
until a different answer has come in between. A failed request is tried again after `retry_after_s`. The alert
(alert_events with scene_id, no rule) keeps its own copy of the picture (scene_<id>_<time>.jpg: the question's picture
is replaced at every answer), pops up on every page and gets a clip from the camera, like other alerts.

A question can be limited like an alert rule (Phase 27): only when the house is set to Away, and/or only at some hours
("22-06"). Outside its limits it is not asked at all, and its answers in a row start again: each night or each Away
period alerts by itself (the gate left open all day alerts again when the night starts).

Vision model jobs for questions use their own numbers (-(SCENE_BASE + question id)): people's are positive, vehicles'
are minus the vehicle's number, so they never mix (run.py routes answers by number).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

log = logging.getLogger(__name__)

SCENE_BASE = 2_000_000_000


def job_id(question_id: int) -> int:
    return -(SCENE_BASE + question_id)


def question_of(job: int) -> int | None:
    return -job - SCENE_BASE if job <= -SCENE_BASE else None


def asked_now(q: dict, now: datetime, mode: str) -> bool:
    """Whether a question's limits allow asking it now (mode: Home / Away, labeler/arming.py)."""
    from .alerts import in_hours
    from .arming import AWAY
    if q.get("only_away") and mode != AWAY:
        return False
    return not q.get("hours") or in_hours(q["hours"], now.hour)


def limits_since(q: dict, now: datetime, away_since: datetime | None) -> datetime | None:
    """When a question that may be asked now last began to be allowed (its hours started, the house went Away), so it
    is asked at once then and the Health page does not call it late for the time it was not asked; None without limits."""
    from datetime import timedelta
    starts = []
    if q.get("hours"):
        start = now.replace(hour=int(q["hours"].split("-")[0]) % 24, minute=0, second=0, microsecond=0)
        starts.append(start if start <= now else start - timedelta(days=1))
    if q.get("only_away") and away_since is not None:
        starts.append(away_since)
    return max(starts) if starts else None


def picture_name(question_id: int) -> str:
    return f"scene_{question_id}.jpg"


# How an area is shown to the vision model. Measured with tools/scene_area_check.py (2026-10-14, 784 items): the area
# cut out 94.4% right, outlined in red on the whole picture 90.2% (it still answers about the whole picture: someone
# lying outside the area, 37 of 37 wrong), the whole picture as before 66.5%. "outline" is kept for that tool only.
AREA_WAY = "part"
AREA_MARGIN = 0.1   # of the area's size, each side: a gate's posts or a car's bumper just outside the line still show


def area_picture(frame, points: list, way: str = "part", margin: float = AREA_MARGIN):
    """The picture sent for a question about one area (points: an area's polygon, fractions of the frame).

    "part": the area's box with a margin is cut out, and what lies outside the polygon within it is darkened, so the
    area keeps its full detail (a gate is a small part of a wide picture, which is shrunk to 1024 px when sent whole).
    "outline": the whole picture with the area outlined in red (the prompt says to look only inside it).
    """
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    poly = np.array([[round(x * w), round(y * h)] for x, y in points], dtype=np.int32)
    if way == "outline":
        out = frame.copy()
        cv2.polylines(out, [poly], True, (0, 0, 255), max(3, round(max(h, w) / 250)), cv2.LINE_AA)
        return out
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    mx, my = round((x2 - x1) * margin) + 2, round((y2 - y1) * margin) + 2
    x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    out = frame[y1:y2, x1:x2].copy()
    mask = np.zeros(out.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - [x1, y1]], 255)
    mask = cv2.dilate(mask, np.ones((2 * my + 1, 2 * mx + 1), np.uint8))   # the margin stays bright too
    out[mask == 0] = (out[mask == 0] * 0.35).astype(out.dtype)
    return out


class SceneRunner:
    def __init__(self, db, pipelines: list, worker, crop_dir, cfg: dict, clock=datetime.now, mono=time.monotonic,
                 on_alert=None, arming=None):
        self.db, self.pipelines, self.worker, self.crop_dir = db, pipelines, worker, crop_dir
        self.arming = arming                 # Home / Away (labeler/arming.py), for questions asked only when Away
        self.scfg = cfg["scene"]
        self.cfg_clips = bool(cfg["clips"]["enabled"])
        self.clock, self.mono = clock, mono
        self.on_alert = on_alert            # (question, picture name) -> None; step 3 makes it an alert
        self._lock = threading.Lock()
        self.pending: dict[int, tuple] = {}   # question -> (sent at, frame, heads, area points or None)
        self.failed_at: dict[int, float] = {}
        self.runs: dict[int, int] = {}       # question -> matching answers in a row
        self.alerted: set[int] = set()       # questions whose current run already alerted

    # --- choosing and sending ------------------------------------------------------------------------------------

    def _frame(self, camera: str):
        """(frame, wall time, heads) of a camera's latest clean picture, if recent enough."""
        p = next((p for p in self.pipelines if p.name == camera), None)
        got = getattr(p, "scene_frame", None) if p is not None else None
        if got is None or (self.clock() - got[1]).total_seconds() > float(self.scfg["frame_max_age_s"]):
            return None
        return got

    def _area(self, q: dict) -> list | None:
        """The points of the area a question is about, or None for the whole picture."""
        if not q.get("zone_id"):
            return None
        z = next((z for z in self.db.list_zones(q["camera"]) if z["id"] == q["zone_id"]), None)
        return z["points"] if z is not None and z["kind"] == "area" else None

    def due(self) -> list[dict]:
        """Enabled questions whose interval has passed since their last answer, oldest answer first."""
        from .arming import AWAY, HOME
        now, mono = self.clock(), self.mono()
        state = self.arming.state() if self.arming is not None else {"mode": HOME, "since": None}
        away_since = datetime.fromisoformat(state["since"]) if state["mode"] == AWAY and state["since"] else None
        out = []
        for q in self.db.list_scene_questions():
            if not q["enabled"] or q["id"] in self.pending:
                continue
            if not asked_now(q, now, state["mode"]):
                self.runs.pop(q["id"], None)       # outside its limits: the next window starts afresh
                self.alerted.discard(q["id"])
                continue
            if mono - self.failed_at.get(q["id"], -1e18) < float(self.scfg["retry_after_s"]):
                continue
            last = datetime.fromisoformat(q["last_at"]) if q["last_at"] else None
            since = limits_since(q, now, away_since)
            if last is not None and since is not None and last < since:
                last = None                        # a new night or Away period: the first answer comes at once
            if last is None or (now - last).total_seconds() >= q["every_s"]:
                out.append(q)
        return sorted(out, key=lambda q: q["last_at"] or "")

    def tick(self) -> int | None:
        """Send one due question if the vision model is free; returns its id."""
        if self.worker is None or self.pending or not self.worker.idle:
            return None
        for q in self.due():
            got = self._frame(q["camera"])
            if got is None:
                continue   # the camera has no recent picture (off, or not running)
            frame, _, heads = got
            area = self._area(q)
            picture = frame if area is None else area_picture(frame, area, AREA_WAY)
            mode = ("scene", q["question"]) if area is None else ("scene", q["question"], AREA_WAY)
            with self._lock:
                self.pending[q["id"]] = (self.mono(), frame, heads, area)
            if self.worker.submit(job_id(q["id"]), [(picture, None)], mode):
                return q["id"]
            with self._lock:
                self.pending.pop(q["id"], None)
            return None
        return None

    def loop(self, stop: threading.Event) -> None:
        while not stop.wait(float(self.scfg["check_every_s"])):
            try:
                self.tick()
            except Exception:
                log.exception("scene questions: checking failed")

    # --- answers ---------------------------------------------------------------------------------------------------

    def owns(self, job: int) -> bool:
        return question_of(job) is not None

    def on_result(self, job: int, results: list[dict]) -> None:
        qid = question_of(job)
        with self._lock:
            sent = self.pending.pop(qid, None)
        if not results:   # the vision model failed: try again later
            self.failed_at[qid] = self.mono()
            return
        answer = results[0].get("answer", "unsure")
        seconds = float(results[0].get("seconds") or 0) or None
        q = next((x for x in self.db.list_scene_questions() if x["id"] == qid), None)
        if q is None:
            return   # deleted meanwhile
        at = self.clock()
        name = self._save_picture(qid, sent) if sent is not None else q["last_picture"]
        self.db.update_scene_question(qid, last_answer=answer, last_at=at, last_picture=name)
        self.db.add_scene_answer(qid, answer, at, seconds)
        log.info("scene question [%s] %r: %s (%.1f s)", q["camera"], q["question"], answer, seconds or 0)
        if answer == q["alert_on"]:
            self.runs[qid] = self.runs.get(qid, 0) + 1
        else:
            self.runs[qid] = 0
            self.alerted.discard(qid)
        if self.runs[qid] >= int(self.scfg["answers_in_a_row"]) and qid not in self.alerted:
            self.alerted.add(qid)
            log.warning("SCENE [%s] %s -> %s (%d answers in a row)", q["camera"], q["question"], answer, self.runs[qid])
            self._alert(q, answer, name, at)
            if self.on_alert is not None:
                self.on_alert(q, name)

    def _alert(self, q: dict, answer: str, name: str | None, at: datetime) -> None:
        """Store the alert with its own copy of the picture, and ask the camera for a clip."""
        import shutil
        from . import privacy
        picture = None
        if name and (self.crop_dir / name).is_file():
            picture = f"scene_{q['id']}_{at:%Y%m%d%H%M%S}.jpg"
            shutil.copyfile(self.crop_dir / name, self.crop_dir / picture)
            blurred = privacy.blurred_path(self.crop_dir, name)
            if blurred.is_file():
                shutil.copyfile(blurred, privacy.blurred_path(self.crop_dir, picture))
        event = self.db.record_scene_alert(q["id"], q["camera"], f"{q['question']} → {answer}", picture)
        p = next((p for p in self.pipelines if p.name == q["camera"]), None)
        clips = getattr(p, "clips", None)
        if event is not None and clips is not None and self.cfg_clips:
            clips.request(event["id"])

    def _save_picture(self, qid: int, sent) -> str | None:
        import cv2
        from . import privacy
        _, frame, heads, area = sent
        shown = (lambda f: f) if area is None else (lambda f: area_picture(f, area, AREA_WAY))   # what the model saw
        name = picture_name(qid)
        try:
            cv2.imwrite(str(self.crop_dir / name), shown(frame), [cv2.IMWRITE_JPEG_QUALITY, 85])
            path = privacy.blurred_path(self.crop_dir, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), shown(privacy.blur_boxes(frame, heads)), [cv2.IMWRITE_JPEG_QUALITY, 85])
            return name
        except Exception:
            log.exception("scene question: could not save the picture")
            return None

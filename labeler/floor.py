"""Floor hazards, blocked exits and left objects (Phase 47 step 2): each walkway and exit area is compared with the
camera's empty picture, and a patch that changed and stays changed becomes a suspect for the vision model.

The empty picture (the user's choice: one per safety camera, saved as <crop_dir>/empty/<camera>.png, kept across
restarts) is how the camera looks with nobody in it. It is learned from `floor_learn_n` looks `floor_learn_every_s`
apart (15 x 2 s): the parts where people stand (their boxes, a bit wider) are left out, and each spot is the middle
value (median) of its looks without a person on it, so passers-by the detectors miss and their reflections drop out (an
average smeared them in: on a busy station floor a third of the floor then always "differed"). The full-size picture
takes each spot from the look closest to that middle value, so it stays sharp for the vision model. A spot with fewer
than `floor_learn_min` looks without a person waits and learns on its later looks. After that a spot follows slow
changes (daylight) by `floor_alpha` a look, but only while it matches; a spot that changed is held as it was, so a bag
or a spill never fades into the empty picture by itself.

Each look (at `floor_side` px, blurred, the current picture's brightness matched to the empty one's):
- a spot differs when one of its colors is off by more than `floor_diff`, lowered for a picture with little contrast
  (Phase 53: at night, IR or dark, everything differs less: a bag 9-41 instead of 40-94 by day, in proportion to the
  picture's contrast, while H.264 leaves little grain): `floor_diff` x the empty picture's contrast (the gray values'
  spread) / `floor_contrast_ref`, at most `floor_diff` and at least `floor_diff_min`; tiny specks are removed;
- a spot that differs keeps the time it began to differ; under a person it is not seen and keeps that time (someone
  standing on a spill does not clean it); seen again and matching, it starts afresh;
- spots that differed `floor_hold_s` (walkway, 30 s) or `exit_hold_s` (exit, 2 min), are seen now, and add up to at
  least `floor_min_size` of the picture are a suspect: "floor" on a walkway, "exit" in an exit area;
- when most of the picture differs at once (`floor_light_share`: the lights went on or off, the camera moved) nothing
  is counted; when that lasts `floor_light_s` the empty picture is learned again.

The vision model sees the empty picture and now (the user's choice; picture 1 = the area clear, picture 2 = now), cut
out closely around the change (CUT) and asked to compare them. The alert comes after two "yes" about `floor_again_s` apart (the user's choice): the first is a
first look (Suspect.alert_if_yes False), the second alerts. A "no" (a chair moved, a shadow) puts the change into the
empty picture, so it is not asked again. After an alert the change is "reported": not raised again while it stays; still
there after `floor_keep_h` hours, it becomes part of the empty picture (the daily refresh).

Left objects (a bag) are floor hazards on a walkway and blocked exits in an exit area; Phase 54 adds who left it
(Pipeline._left_by: the reason says "left by person #12, who walked away").
"""
from __future__ import annotations

import logging
import threading
import warnings
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .zones import AREA

log = logging.getLogger(__name__)

ROLE_CATEGORY = {"walkway": "floor", "exit": "exit"}
# Asked to compare (measured on the tune half, tools/make_floor_videos.py: "is there something new that is a danger"
# saw 9 of 13 changes, this 11 of 13 with the tighter cut, no false "yes" on 8 spots people walked over).
QUESTIONS = {
    "floor": "Picture 1 shows this floor earlier; picture 2 shows the same place now. Compare them: is there something "
             "on the floor in picture 2 that was not there in picture 1 (a stain, puddle or spilled liquid, broken "
             "glass, dropped food, a bag, a box, furniture)? People, their shadows and a change of light do not count.",
    "exit": "Picture 1 shows this exit or aisle earlier; picture 2 shows the same place now. Compare them: is there "
            "something in picture 2 that was not there in picture 1 and is in the way (boxes, a bag, furniture, a "
            "trolley)? People walking through, shadows and a change of light do not count.",
}
# Phase 53 (the user's choice): at night the vision model said "no" more often to a change it was shown.
NIGHT_NOTE = (" The pictures may be dark, grainy or black and white (a camera at night): look for the shape of an object "
              "even when it is faint.")
QUESTIONS = {k: q + NIGHT_NOTE for k, q in QUESTIONS.items()}
# The cut-out around the change: 60% of its size on each side, at least a quarter of the picture (a person's check
# takes 75% and a third: the change was then too small a part of the picture to be seen).
CUT = (0.6, 0.26)
WORDS = {"floor": "on the floor of", "exit": "in the exit"}


def _box_mask(shape: tuple, boxes: list, frame_shape: tuple, margin: float) -> np.ndarray:
    """True where a person may be: their boxes (pixels of the frame) widened by margin of their width each side and
    half of that below (feet, shadow), at the check size."""
    h, w = shape
    fh, fw = frame_shape[:2]
    sx, sy = w / fw, h / fh
    mask = np.zeros((h, w), np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        bw, bh = x2 - x1, y2 - y1
        a, c = int((x1 - bw * margin) * sx), int(np.ceil((x2 + bw * margin) * sx))
        t, d = int((y1 - bh * margin / 2) * sy), int(np.ceil((y2 + bh * margin / 2) * sy))
        mask[max(0, t):max(0, d), max(0, a):max(0, c)] = 1
    return mask.astype(bool)


class FloorWatch:
    def __init__(self, cfg: dict, path: Path | None = None):
        self.every_s = float(cfg.get("floor_every_s", 1.0))
        self.side = int(cfg.get("floor_side", 480))
        self.keep_side = int(cfg.get("memory_side", 960))
        self.diff = float(cfg.get("floor_diff", 30))
        self.contrast_ref = float(cfg.get("floor_contrast_ref", 50))
        self.diff_min = float(cfg.get("floor_diff_min", 10))
        self.min_size = float(cfg.get("floor_min_size", 0.0004))
        self.hold = {"floor": float(cfg.get("floor_hold_s", 30)), "exit": float(cfg.get("exit_hold_s", 120))}
        self.alpha = float(cfg.get("floor_alpha", 0.02))
        self.learn_n = int(cfg.get("floor_learn_n", 15))
        self.learn_every_s = float(cfg.get("floor_learn_every_s", 2.0))
        self.learn_min = int(cfg.get("floor_learn_min", 5))
        self.light_share = float(cfg.get("floor_light_share", 0.4))
        self.light_s = float(cfg.get("floor_light_s", 10))
        self.again_s = float(cfg.get("floor_again_s", 60))
        self.keep_s = float(cfg.get("floor_keep_h", 24)) * 3600
        self.save_s = float(cfg.get("floor_save_s", 600))
        self.margin = float(cfg.get("floor_margin", 0.2))
        self.path = Path(path) if path else None
        self.last_look = -1e9
        self.empty: np.ndarray | None = None   # the empty picture, float32 at keep_side (what is saved and shown)
        self.seen = self.since = self.known = None   # per spot at the check size: looks seen, differs since, reported
        self.small_shape: tuple | None = None
        self._frame_shape: tuple = ()
        self.learned = False
        self._stack: list = []          # while learning: (picture at keep_side, at the check size, spots seen)
        self._stacked_at = -1e9
        self.light_since: float | None = None
        self.relearned = 0
        self.learned_at: datetime | None = None   # when the empty picture was last learned whole (the safety page)
        self.saved_at = -1e9
        self.saved_wall: datetime | None = None
        self.areas: dict = {}       # area id -> {"waiting", "yes_at", "mask"}
        self.answers: deque = deque()   # (area id, answer or None), from the confirm queue's thread
        self._masks: tuple = ((), {})   # (key of the areas, {area id: mask})
        self._last = None               # the latest look's (picture at keep_side, seen spots, differing spots)
        self._lock = threading.Lock()   # state() is read by the web thread

    def due(self, now: float) -> bool:
        return now - self.last_look >= self.every_s

    # The empty picture -------------------------------------------------------------------------------------------

    def _start(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        k = min(1.0, self.keep_side / max(h, w))
        big = (round(h * k), round(w * k))
        s = self.side / max(h, w)
        self.small_shape = (max(8, round(h * s)), max(8, round(w * s)))
        self.empty = np.zeros((*big, 3), np.float32)
        self.seen = np.zeros(self.small_shape, np.uint16)
        self._forget()
        self.learned, self._stack = False, []
        if self.path is not None and self.path.is_file():
            pic = cv2.imread(str(self.path))
            if pic is not None and abs(pic.shape[1] / pic.shape[0] - w / h) < 0.02:
                self.empty = cv2.resize(pic, (big[1], big[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
                self.seen[:] = self.learn_min
                self.learned = True
                self.saved_wall = datetime.fromtimestamp(self.path.stat().st_mtime)
                self.learned_at = self.learned_at or self.saved_wall   # from an earlier run: when it was saved
                log.info("floor and exits: the empty picture from %s", self.path)

    def _forget(self) -> None:
        self.since = np.full(self.small_shape, np.nan)
        self.known = np.full(self.small_shape, np.nan)
        self.areas = {}

    def relearn(self) -> None:
        """Learn the empty picture again (the lights changed, the camera moved)."""
        if self.seen is not None:
            self.seen[:] = 0
            self._forget()
            self.learned, self._stack = False, []
            self.relearned += 1

    def _learn(self) -> None:
        """The empty picture from the learning looks: each spot's median over its looks without a person."""
        bigs = np.stack([x[0] for x in self._stack])
        smalls = np.stack([x[1] for x in self._stack]).astype(np.float32)
        seen = np.stack([x[2] for x in self._stack])
        with warnings.catch_warnings():   # a spot never seen: all NaN, filled below
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(seen[..., None], smalls, np.nan), axis=0)
        med = np.where(np.isnan(med), np.median(smalls, axis=0), med)
        dist = np.abs(smalls - med).sum(axis=3) + np.where(seen, 0, 1e6)
        idx = cv2.resize(dist.argmin(axis=0).astype(np.uint8), (bigs.shape[2], bigs.shape[1]),
                         interpolation=cv2.INTER_NEAREST)
        self.empty = np.take_along_axis(bigs, idx[None, :, :, None], axis=0)[0].astype(np.float32)
        self.seen = seen.sum(axis=0).astype(np.uint16)
        self.learned, self._stack = True, []

    def threshold(self, ref: np.ndarray) -> float:
        """How much a spot must differ: floor_diff, lowered in proportion for an empty picture with little contrast
        (night), not below floor_diff_min (see the module text)."""
        contrast = float(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).std())
        return max(self.diff_min, self.diff * min(1.0, contrast / self.contrast_ref))

    def ready_share(self, mask: np.ndarray | None = None) -> float:
        if self.seen is None:
            return 0.0
        known = self.seen >= self.learn_min
        return float(known[mask].mean()) if mask is not None and mask.any() else float(known.mean())

    def save(self, wall: datetime | None = None) -> None:
        if self.path is None or self.empty is None or self.ready_share() < 0.5:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.path), np.clip(self.empty, 0, 255).astype(np.uint8))
            self.saved_wall = wall or datetime.now()
        except Exception:
            log.exception("floor and exits: could not save the empty picture")

    def picture(self) -> np.ndarray | None:
        """The empty picture now (for the vision model and the tools)."""
        return None if self.empty is None else np.clip(self.empty, 0, 255).astype(np.uint8)

    # One look ----------------------------------------------------------------------------------------------------

    def _area_masks(self, zones: list[dict]) -> dict:
        watched = [z for z in zones if z.get("kind") == AREA and z.get("role") in ROLE_CATEGORY]
        key = (self.small_shape, *((z["id"], z["role"], tuple(map(tuple, z["points"]))) for z in watched))
        if key != self._masks[0]:
            h, w = self.small_shape
            masks = {}
            for z in watched:
                m = np.zeros((h, w), np.uint8)
                pts = np.array([[x * w, y * h] for x, y in z["points"]], np.int32)
                cv2.fillPoly(m, [pts], 1)
                masks[z["id"]] = (z, m.astype(bool))
            self._masks = (key, masks)
        return self._masks[1]

    def update(self, now: float, frame: np.ndarray, people: list, zones: list[dict],
               wall: datetime | None = None) -> list[dict]:
        """One look. people: boxes (pixels) of everyone seen now. Returns the suspects to raise:
        {"area", "category", "reason", "crop" (fractions), "first_look", "before" (wall time the change began)}."""
        self.last_look = now
        if not any(z.get("kind") == AREA and z.get("role") in ROLE_CATEGORY for z in zones):
            return []
        with self._lock:
            if self.empty is None or frame.shape[:2] != self._frame_shape:
                self._frame_shape = frame.shape[:2]
                self._start(frame)
            masks = self._area_masks(zones)
            self._take_answers(now)
            out = self._look(now, frame, people, masks, wall)
        if self.ready_share() >= 0.95 and now - self.saved_at >= self.save_s:
            self.saved_at = now
            self.save(wall)
        return out

    def _look(self, now: float, frame: np.ndarray, people: list, masks: dict, wall) -> list[dict]:
        h, w = self.small_shape
        big = cv2.resize(frame, (self.empty.shape[1], self.empty.shape[0]), interpolation=cv2.INTER_AREA)
        small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
        visible = ~_box_mask((h, w), people, frame.shape, self.margin)
        if not self.learned:
            if now - self._stacked_at >= self.learn_every_s:
                self._stacked_at = now
                self._stack.append((big, small, visible))
                if len(self._stack) >= self.learn_n:
                    self._learn()
                    self.learned_at = wall or datetime.now()
            return []
        cur = cv2.GaussianBlur(small, (5, 5), 0)
        ref = cv2.convertScaleAbs(cv2.GaussianBlur(cv2.resize(self.empty, (w, h), interpolation=cv2.INTER_AREA),
                                                   (5, 5), 0))
        ready = self.seen >= self.learn_min
        both = visible & ready
        if both.sum() > 50:   # match the brightness (a camera's own exposure changes) before comparing
            gray = lambda p: cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)[both]   # noqa: E731
            gain = float(np.median(gray(ref)) / max(1.0, float(np.median(gray(cur)))))
            cur = cv2.convertScaleAbs(cur, alpha=min(1.5, max(0.67, gain)))
        b, g, r = cv2.split(cv2.absdiff(cur, ref))
        differs = (cv2.max(cv2.max(b, g), r) > self.threshold(ref)).astype(np.uint8)
        differs = cv2.morphologyEx(differs, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        self._last = (big, visible, differs)

        # Most of the picture changed at once: the lights, not the floor.
        share = float(differs[both].mean()) if both.sum() > 50 else 0.0
        if share > self.light_share:
            self.light_since = now if self.light_since is None else self.light_since
            if now - self.light_since >= self.light_s:
                log.info("floor and exits: the picture changed as a whole (%.0f%%): learning the empty picture again",
                         share * 100)
                self.relearn()
                self.light_since = None
            return []
        self.light_since = None

        # Learn and follow: new spots take the mean of their first looks; known ones follow slowly while they match.
        size = (self.empty.shape[1], self.empty.shape[0])
        # Not next to a change either: the soft edge of a spill differs too little to count, and following it drew a
        # dark ring in the empty picture, so the vision model saw the spill's shape in both pictures.
        near = cv2.dilate(differs.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        follow = cv2.resize((both & ~near).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        cv2.accumulateWeighted(big, self.empty, self.alpha, mask=follow)
        learning = visible & ~ready
        if learning.any():
            self.seen[learning] += 1
            alpha = np.zeros((h, w), np.float32)
            alpha[learning] = 1.0 / self.seen[learning]
            a = cv2.resize(alpha, size, interpolation=cv2.INTER_NEAREST)[..., None]
            self.empty += (big.astype(np.float32) - self.empty) * a

        # How long each spot has differed; seen and matching again: afresh (and no longer "reported").
        start = both & differs & np.isnan(self.since)
        self.since[start] = now
        clear = both & ~differs
        self.since[clear] = np.nan
        self.known[clear] = np.nan
        old = ~np.isnan(self.known) & (now - np.nan_to_num(self.known, nan=now) >= self.keep_s)
        if old.any():   # reported long ago and still there: it belongs to the place now
            self._absorb(old)

        out = []
        minimum = self.min_size * h * w
        for zid, (z, m) in masks.items():
            category = ROLE_CATEGORY[z["role"]]
            st = self.areas.setdefault(zid, {"waiting": False, "yes_at": None, "mask": None, "pause": -1e9})
            if st["waiting"] or now < st["pause"]:
                continue
            held = m & both & ~np.isnan(self.since) & np.isnan(self.known)
            held &= now - np.nan_to_num(self.since, nan=now) >= self.hold[category]
            if held.sum() < minimum:
                if st["yes_at"] is not None and now - st["yes_at"] >= 3 * self.again_s:
                    st["yes_at"] = None   # gone before the second look
                continue
            if st["yes_at"] is not None and now - st["yes_at"] < self.again_s:
                continue
            ys, xs = np.nonzero(held)
            began = float(np.nanmin(self.since[held]))
            st.update(waiting=True, mask=held)
            secs = int(now - began)
            took = f"{secs // 60} min {secs % 60} s" if secs >= 60 else f"{secs} s"
            out.append({"area": z, "category": category, "first_look": st["yes_at"] is None,
                        "reason": f"something new {WORDS[category]} {z['name']} for {took} "
                                  f"({max(1, round(100 * held.sum() / max(1, m.sum())))}% of the area)",
                        "crop": (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h),
                        "before": (wall - timedelta(seconds=now - began)) if wall is not None else None})
        return out

    def _absorb(self, spots: np.ndarray) -> None:
        """Put these spots, as they look now, into the empty picture."""
        if self._last is None:
            return
        big, visible, _ = self._last
        spots = spots & visible
        m = cv2.resize(spots.astype(np.uint8), (self.empty.shape[1], self.empty.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        self.empty[m] = big[m]
        self.since[spots] = np.nan
        self.known[spots] = np.nan

    # Answers -----------------------------------------------------------------------------------------------------

    def answered(self, area_id, answer: dict | None) -> None:
        """The confirm queue's answer (None: none came; also when the check was not sent). Any thread."""
        self.answers.append((area_id, answer))

    def _take_answers(self, now: float) -> None:
        while self.answers:
            zid, answer = self.answers.popleft()
            st = self.areas.get(zid)
            if st is None:
                continue
            st["waiting"] = False
            said = (answer or {}).get("answer")
            if said not in ("yes", "no"):   # unsure, not sent (checked just now), dropped, failed: again later
                st["pause"] = now + self.again_s / 2
            elif said == "no":
                self._absorb(self._whole(st["mask"]))
                st["yes_at"] = None
            elif said == "yes" and st["yes_at"] is None:
                st["yes_at"] = now   # a first look: asked again after floor_again_s
            elif said == "yes":
                self.known[self._whole(st["mask"])] = now   # the second yes alerted: reported
                st["yes_at"] = None

    def _whole(self, mask: np.ndarray) -> np.ndarray:
        """The whole change the checked spots belong to: every spot that differs and touches them. Spots someone stood
        on began to differ later and were not checked yet; without this the rest of the same bag was asked about again
        and alerted a second time (found in a real run on a test copy)."""
        differ = (~np.isnan(self.since) | mask).astype(np.uint8)
        _, comp = cv2.connectedComponents(differ)
        ids = np.unique(comp[mask])
        return np.isin(comp, ids[ids > 0])

    def state(self) -> dict:
        """For the Health page."""
        with self._lock:
            masks = self._masks[1]
            watched = np.zeros(self.small_shape, bool) if self.small_shape else None
            for _, m in masks.values():
                watched |= m
            changing = int(sum(1 for st in self.areas.values() if st["waiting"] or st["yes_at"] is not None))
            return {"areas": [z["name"] for z, _ in masks.values()],
                    "learning": None if self.learned else f"{len(self._stack)} of {self.learn_n} looks",
                    "ready": round(self.ready_share(watched if watched is not None and watched.any() else None), 2),
                    "saved": self.saved_wall.isoformat(timespec="minutes") if self.saved_wall else None,
                    "checking": changing,
                    "reported": int(np.count_nonzero(~np.isnan(self.known))) if self.known is not None else 0,
                    "relearned": self.relearned,
                    "learned_at": self.learned_at.isoformat(timespec="seconds") if self.learned_at else None}

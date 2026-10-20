"""Fire and smoke (Phase 46): the app's own detector, trained on D-Fire (tools/train_fire.py; data/models/fire), looks
at each safety camera's picture once a second (`safety.fire_every_s`, the user's choice) and raises a critical "fire
or smoke" suspect when it sees fire or smoke on `fire_hold` of the last `fire_window` looks. The vision model confirms
it from 3 pictures over the last 8 seconds (smoke spreads, flames flicker) with the part around the boxes.

Cooking areas (the user's choice): a box that lies mostly (`fire_inside` of its area) inside an area with the role
"cooking" is normal (a stove, an oven, steam from a pot) and ignored; one that reaches past the area counts, and the
reason says so. After a suspect, nothing is raised again until nothing was seen for `fire_rearm_s`.

D-Fire's classes: 0 = smoke, 1 = fire. Measured on D-Fire's test part (single photos): mAP50 smoke 0.83, fire 0.71,
9.5 ms a picture at 640 px (README "Fire and smoke").
"""
from __future__ import annotations

from collections import deque

from .zones import AREA, inside

SMOKE, FIRE = 0, 1
WORDS = {SMOKE: "smoke", FIRE: "flames"}
GRID = 5   # how much of a box is inside an area: the share of a 5 x 5 grid of points in it


def share_inside(box: tuple, shape: tuple, points: list) -> float:
    """The share of a box (pixels) inside an area (fractions of the picture)."""
    h, w = shape[:2]
    x1, y1, x2, y2 = box[:4]
    pts = [((x1 + (x2 - x1) * (i + 0.5) / GRID) / w, (y1 + (y2 - y1) * (j + 0.5) / GRID) / h)
           for i in range(GRID) for j in range(GRID)]
    return sum(inside(p, points) for p in pts) / len(pts)


class FireWatch:
    def __init__(self, cfg: dict):
        self.every_s = float(cfg.get("fire_every_s", 1.0))
        self.window = int(cfg.get("fire_window", 5))
        self.hold = int(cfg.get("fire_hold", 3))
        self.inside = float(cfg.get("fire_inside", 0.8))
        self.rearm_s = float(cfg.get("fire_rearm_s", 60))
        self.looks: deque = deque(maxlen=self.window)   # (time, kinds seen, union box or None)
        self.last_look = -1e9
        self.last_seen = -1e9
        self.fired = False
        self.past: set = set()   # cooking areas something reached past, in the current window

    def due(self, now: float) -> bool:
        return now - self.last_look >= self.every_s

    def update(self, now: float, dets: list[tuple], shape: tuple, zones: list[dict]) -> tuple | None:
        """dets: (class, confidence, box in pixels) from one look. Returns (reason, crop as fractions) when a suspect
        should be raised, else None."""
        self.last_look = now
        cooking = [z for z in zones if z.get("kind") == AREA and z.get("role") == "cooking"]
        kept, past = [], set()
        for cls, conf, box in dets:
            shares = [(share_inside(box, shape, z["points"]), z["name"]) for z in cooking]
            if any(s >= self.inside for s, _ in shares):
                continue   # inside a cooking area: normal
            past |= {name for s, name in shares if s > 0}
            kept.append((cls, box))
        union = None
        if kept:
            xs1, ys1, xs2, ys2 = zip(*[b[:4] for _, b in kept])
            union = (min(xs1), min(ys1), max(xs2), max(ys2))
            self.last_seen = now
        self.looks.append((now, {cls for cls, _ in kept}, union))
        self.past = past if kept else self.past
        if self.fired and now - self.last_seen >= self.rearm_s:
            self.fired = False
        seen = [x for x in self.looks if x[1]]
        if self.fired or len(seen) < self.hold:
            return None
        self.fired = True
        kinds = set().union(*(x[1] for x in seen))
        what = " and ".join(WORDS[k] for k in (FIRE, SMOKE) if k in kinds)
        reason = f"{what} seen {len(seen)} of the last {len(self.looks)} s"
        if self.past:
            reason += f", reaching past the cooking area {', '.join(sorted(self.past))}"
        h, w = shape[:2]
        boxes = [x[2] for x in seen]
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        return reason, (x1 / w, y1 / h, x2 / w, y2 / h)

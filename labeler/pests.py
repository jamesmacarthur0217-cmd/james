"""Mice and rats in food areas (Phase 55): the app's own detector (tools/train_pests.py; data/models/pests) looks at
each safety camera's cooking, food preparation and dining areas about once a second (`safety.pest_every_s`).

The design agreed with the user (2026-10-30, docs/PHASE50-55-PLAN.md): only those areas are looked at, each cut out
of the picture and enlarged to `pest_imgsz` (1280) px, because a mouse across a room is only ~40 px wide: in the
whole picture the detector found 1% of such mice, in an enlarged area 61% (tools/pest_check.py, tune half). A box
counts when its middle is inside the area's outline. Seen on `pest_hold` of the last `pest_window` looks of an area,
the vision model is asked about the looks where it was seen (up to 3, full size, cut out around the boxes as measured)
and one "yes" alerts (a mouse runs away: there is no second look). Then that area is not raised again for
`pest_rearm_s` (pitfall 40: one alert a minute per camera and area).

The detector's one class, "rodent", is Open Images' "Mouse": rats and mice, but also squirrels and hamsters.
Cockroaches are not looked for (no training boxes, and too small for a room camera). Measured on Open Images photos
(README "Our own pest model"): close-up mice 97% (night look 90%); far, by day, 55% confirmed by the vision model with
no false "yes"; far at night 11%.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime

import numpy as np

from .zones import AREA, ROLES, inside

FOOD_ROLES = ("cooking", "food_prep", "dining")
# The question the vision model was measured with (tools/pest_check.py); it says what is not a pest.
QUESTION = ("Is there a live rat, mouse or cockroach in these pictures (not a toy, a computer mouse, a drawing or a "
            "picture on a screen)?")
CUT = (0.75, 0.35)   # as measured: around the boxes, at least this share of the area's size (person_crop)


def food_areas(zones: list[dict]) -> list[dict]:
    return [z for z in zones if z.get("kind") == AREA and z.get("role") in FOOD_ROLES]


def area_box(points: list, shape: tuple) -> tuple[int, int, int, int]:
    """The area's box in pixels (x1, y1, x2, y2), within the picture."""
    h, w = shape[:2]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x1, y1 = max(0, int(min(xs) * w)), max(0, int(min(ys) * h))
    x2, y2 = min(w, int(np.ceil(max(xs) * w))), min(h, int(np.ceil(max(ys) * h)))
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def in_area(box: tuple, shape: tuple, points: list) -> bool:
    h, w = shape[:2]
    return inside(((box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h), points)


class PestWatch:
    def __init__(self, cfg: dict):
        self.every_s = float(cfg.get("pest_every_s", 1.0))
        self.window = int(cfg.get("pest_window", 3))
        self.hold = int(cfg.get("pest_hold", 2))
        self.rearm_s = float(cfg.get("pest_rearm_s", 60))
        self.looks: dict = {}    # area id -> deque of (time, best confidence, union box or None, (wall time, picture))
        self.raised: dict = {}   # area id -> when last raised
        self.last_look = -1e9

    def due(self, now: float) -> bool:
        return now - self.last_look >= self.every_s

    def update(self, now: float, wall: datetime, frame: np.ndarray, found: dict, areas: list[dict]) -> list[tuple]:
        """found: area id -> [(confidence, box in pixels)] from one look at each area. Returns (area, reason, crop as
        fractions, pictures, cut) for each area to raise now."""
        self.last_look = now
        out = []
        ids = {z["id"] for z in areas}
        for gone in set(self.looks) - ids:   # an area deleted or its role changed
            del self.looks[gone]
        for z in areas:
            seen = found.get(z["id"]) or []
            looks = self.looks.setdefault(z["id"], deque(maxlen=self.window))
            if seen:
                xs1, ys1, xs2, ys2 = zip(*[b[:4] for _, b in seen])
                union = (min(xs1), min(ys1), max(xs2), max(ys2))
                looks.append((now, max(c for c, _ in seen), union, (wall, frame.copy())))
            else:
                looks.append((now, 0.0, None, None))
            hits = [x for x in looks if x[2] is not None]
            if len(hits) < self.hold or now - self.raised.get(z["id"], -1e9) < self.rearm_s:
                continue
            self.raised[z["id"]] = now
            h, w = frame.shape[:2]
            x1, y1 = min(x[2][0] for x in hits), min(x[2][1] for x in hits)
            x2, y2 = max(x[2][2] for x in hits), max(x[2][3] for x in hits)
            ax1, ay1, ax2, ay2 = area_box(z["points"], frame.shape)
            share = max((ax2 - ax1) / w, (ay2 - ay1) / h)
            reason = (f"a mouse or rat seen {len(hits)} of the last {len(looks)} looks in {z.get('name') or 'an area'} "
                      f"({ROLES.get(z.get('role'), 'food area')}; up to {max(x[1] for x in hits):.2f})")
            pictures = [(x[3][0], x[3][1], []) for x in hits][-3:]
            out.append((z, reason, (x1 / w, y1 / h, x2 / w, y2 / h), pictures, (CUT[0], CUT[1] * share)))
            looks.clear()   # the pictures are no longer needed
        return out

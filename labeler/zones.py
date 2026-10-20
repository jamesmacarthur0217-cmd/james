"""Zones (areas) and counting lines drawn on a camera's picture.

Points are fractions of the frame (0..1), so they still fit after a change of camera resolution.

- Area: a polygon. A person "enters" when their anchor point has been inside it for `min_frames` frames in a
  row, and "exits" the same way, or when they leave the view. The time in between is their dwell time.
- Line: two points a -> b. Crossing it towards the side its arrow points to is "in", the other way "out".
  The arrow is drawn on the live picture (the side to the right of a -> b on the screen).

Requiring a few frames in a row keeps someone standing on a border from being counted again and again.
A person's anchor is the middle of their box on kiosk cameras (feet are usually out of view) and their feet
(the bottom middle) on security cameras.

Area roles (Phase 44): an area can say what the place is, so the safety checks know what is normal there (a knife or
a flame in the cooking area; someone lying on the bed). The roles are fixed here because later checks depend on their
meaning; `safety.roles` in the config picks which ones the Live page offers. "bed" is the old "lying is normal here"
(zones.lying_ok is kept the same, so fall checks and old databases work unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

AREA, LINE = "area", "line"
LINE_MARGIN = 0.05   # a crossing counts slightly beyond the line's ends too
BED = "bed"
# role -> what the Live page calls it ("" = no role: a plain area for counting).
ROLES = {"walkway": "walkway or aisle", "exit": "exit or emergency exit", "cooking": "cooking area",
         "food_prep": "food preparation", "dining": "dining or seating", "play": "children's play area",
         "restricted": "restricted (staff or no entry)", BED: "bed or sofa (lying is normal here)"}


def role_at(zones: list[dict], pt: tuple[float, float]) -> str:
    """The role of the first area with a role that this point is in ("" when none)."""
    return next((z.get("role") or "" for z in zones
                 if z["kind"] == AREA and z.get("role") and inside(pt, z["points"])), "")


def check_role(kind: str, role: str | None) -> str:
    """A valid role for a zone ("" for none); only areas have one."""
    role = (role or "").strip()
    if role and role not in ROLES:
        raise ValueError(f"unknown area role {role!r} (one of: {', '.join(ROLES)})")
    if role and kind != AREA:
        raise ValueError("only areas have a role, not counting lines")
    return role


def anchor(box: tuple, shape: tuple, mode: str) -> tuple[float, float]:
    """The point that decides where a person is, as fractions of the frame."""
    h, w = shape[:2]
    x1, y1, x2, y2 = box[:4]
    y = y2 if mode == "security" else (y1 + y2) / 2
    return ((x1 + x2) / 2 / w, y / h)


def inside(pt: tuple[float, float], polygon: list) -> bool:
    """Point in polygon (ray casting)."""
    x, y = pt
    hit = False
    n = len(polygon)
    for i in range(n):
        (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def lying_ok(zones: list[dict], pt: tuple[float, float]) -> bool:
    """True inside an area marked "lying is normal here" (a bed, a sofa): no fall alerts start there."""
    return any(z["kind"] == AREA and z.get("lying_ok") and inside(pt, z["points"]) for z in zones)


def side(pt: tuple[float, float], a: tuple, b: tuple) -> int:
    """+1 on the "in" side of line a -> b, -1 on the "out" side, 0 when beside the line's ends."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return 0
    t = ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / length2
    if not -LINE_MARGIN <= t <= 1 + LINE_MARGIN:
        return 0
    cross = dx * (pt[1] - a[1]) - dy * (pt[0] - a[0])
    return 1 if cross > 0 else -1 if cross < 0 else 0


@dataclass
class PersonZones:
    """Where one person is, relative to the camera's zones."""
    inside: dict[int, tuple[str, datetime]] = field(default_factory=dict)   # area id -> (name, entered at)
    line_side: dict[int, int] = field(default_factory=dict)                 # line id -> confirmed side
    pending: dict[int, tuple[object, int]] = field(default_factory=dict)    # zone id -> (new state, frames)


class ZoneCounter:
    """Turns each person's position into enter/exit (areas) and in/out (lines) events."""

    def __init__(self, zones: list[dict] | None = None, min_frames: int = 3):
        self.zones = zones or []
        self.min_frames = max(1, min_frames)

    def set_zones(self, zones: list[dict]) -> None:
        self.zones = list(zones)   # one assignment: the camera thread sees the old or the new list

    def _held(self, person: PersonZones, zid: int, state, current) -> bool:
        """True once `state` (different from `current`) has held for min_frames frames in a row."""
        if state == current:
            person.pending.pop(zid, None)
            return False
        prev, frames = person.pending.get(zid, (None, 0))
        frames = frames + 1 if prev == state else 1
        if frames >= self.min_frames:
            person.pending.pop(zid, None)
            return True
        person.pending[zid] = (state, frames)
        return False

    def update(self, person: PersonZones, pt: tuple[float, float], at: datetime) -> list[dict]:
        events = []
        for z in self.zones:
            zid, name = z["id"], z["name"]
            if z["kind"] == AREA:
                now_in = inside(pt, z["points"])
                was_in = zid in person.inside
                if self._held(person, zid, now_in, was_in):
                    if now_in:
                        person.inside[zid] = (name, at)
                        events.append(_event(z, "enter", at))
                    else:
                        _, since = person.inside.pop(zid)
                        events.append(_event(z, "exit", at, (at - since).total_seconds()))
            else:
                s = side(pt, *z["points"])
                if s == 0:
                    person.pending.pop(zid, None)   # beside the line: no crossing, keep the last side
                    continue
                confirmed = person.line_side.get(zid)
                if self._held(person, zid, s, confirmed):
                    person.line_side[zid] = s
                    if confirmed is not None:   # the first side seen is where they came from, not a crossing
                        events.append(_event(z, "in" if s > 0 else "out", at))
        return events

    def leave(self, person: PersonZones, at: datetime) -> list[dict]:
        """The person left the view: they exit every area they were in."""
        events = [{"zone_id": zid, "zone_name": name, "kind": "exit", "at": at,
                   "dwell_s": (at - since).total_seconds()} for zid, (name, since) in person.inside.items()]
        person.inside.clear()
        person.pending.clear()
        person.line_side.clear()
        return events

    def draw(self, frame: np.ndarray, people: list[PersonZones]) -> None:
        """Outline each zone on the picture, with how many people are in each area now."""
        h, w = frame.shape[:2]
        color = (230, 200, 40)
        for z in self.zones:
            pts = np.array([(x * w, y * h) for x, y in z["points"]], np.int32)
            if z["kind"] == AREA:
                cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
                n = sum(z["id"] in p.inside for p in people)
                label = f"{z['name']}: {n}"
            else:
                (ax, ay), (bx, by) = pts
                cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
                mx, my = (ax + bx) // 2, (ay + by) // 2
                nx, ny = -(by - ay), bx - ax                 # towards the "in" side
                norm = max(1.0, float(np.hypot(nx, ny)))
                tip = (int(mx + nx / norm * 28), int(my + ny / norm * 28))
                cv2.arrowedLine(frame, (mx, my), tip, color, 2, cv2.LINE_AA, tipLength=0.4)
                cv2.putText(frame, "in", (tip[0] + 4, tip[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA)
                label = z["name"]
            x, y = int(pts[:, 0].min()), max(16, int(pts[:, 1].min()) - 6)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _event(z: dict, kind: str, at: datetime, dwell_s: float | None = None) -> dict:
    return {"zone_id": z["id"], "zone_name": z["name"], "kind": kind, "at": at, "dwell_s": dwell_s}


def check_points(kind: str, points: list) -> list[tuple[float, float]]:
    """Validate points from the web page: an area needs 3+ points, a line exactly 2, all within the frame."""
    if kind not in (AREA, LINE):
        raise ValueError("kind must be 'area' or 'line'")
    try:
        pts = [(float(x), float(y)) for x, y in points]
    except (TypeError, ValueError):
        raise ValueError("points must be [x, y] pairs") from None
    if any(not (0 <= v <= 1) for p in pts for v in p):
        raise ValueError("points must be fractions of the picture (0 to 1)")
    if kind == AREA and len(pts) < 3:
        raise ValueError("an area needs at least 3 points")
    if kind == LINE and (len(pts) != 2 or pts[0] == pts[1]):
        raise ValueError("a line needs 2 different points")
    return pts

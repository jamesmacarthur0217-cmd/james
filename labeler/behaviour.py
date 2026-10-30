"""Suspicious behaviour (Phase 49): what people do, never how they look (no age, gender, clothing, skin or face
covering; no face recognition; no automatic action: an alert shows the picture, clip and reason, a person decides).

One BehaviourWatch per safety camera, fed like SafetyWatch with each person's box (labeler/pipeline.py). The user's
choices (2026-10-23): four signs, moments confirmed by the vision model, patterns alerted directly at night (`night_hours`,
22-06) or when the house is Away, with the vision model's words on what the person is doing added afterwards:

- climbing a fence or wall (a moment; the vision model confirms): a person at an area with the role "fence" (the
  middle or the feet of their box inside it) whose feet rise `climb_rise` x their usual height above where they
  usually were (the median of the last 10 s) while the body does not get smaller (walking away from the camera also
  lifts the feet in the picture, but shrinks the body), for `climb_hold_s`. Critical at night or when Away, else a
  warning. Once until the feet are down again.
- in a restricted area (a pattern): the feet inside an area with the role "restricted" for `restricted_s`, at night or
  when Away; once per visit (outside again for `restricted_rearm_s` re-arms it).
- pacing back and forth (a pattern): inside any area, the walk turns back `pace_turns` times within `pace_window_s`,
  each leg at least `pace_span` body heights (along the way the person moves most); at night or when Away.
- at parked cars (a moment, Phase 54; the user's choice): a wrist inside a parked car's box (car, van or lorry,
  motorbike) while the person stands at it, hardly moving, for `car_hold_s`; or stopping (`car_stop_s`, hardly moving)
  at `car_count` different parked cars within `car_window_s`. Only at night or when Away; the vision model looks once
  (CAR_QUESTION) and only its "yes" alerts; at most one car check a camera every `car_rearm_s`.
- following through a door (a pattern, Phase 54; the user's choice): on a counting line with the role "door", two
  people crossing it the same way within `follow_s` (someone following another through, as when slipping in behind a
  resident); at night or when Away, alerted directly once a minute per door (by day two friends do it all the time).
- the camera covered or turned (camera()): a tiny gray picture once a second; "now" is the middle value of each spot
  over the last NOW_LOOKS looks and "before" over the looks 10-15 s ago, so people walking past drop out (comparing
  single pictures failed on a busy street: big people close to the lens changed the edges as much as a turned camera).
  Covered = now is flat and its edges are gone (no edge where one was), and before was not, for `camera_hold_s` (only
  "flat" took a dark night picture with the light dimmed for a covered camera: contrast 21 -> 4, but the edges stay
  where they were, match 0.9; covered 0.0; Phase 53, tools/night_check.py); turned = most of the edges moved
  (a light switched on or off keeps them where they were), for `camera_hold_s`. The camera alerts at any time, but
  only when a person was seen in the `camera_seen_s` before (else it stays a camera problem,
  labeler/picture_check.py).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from .zones import AREA, inside

CLIMB_QUESTION = ("Is a person climbing over or onto a fence, wall or gate (not stairs, a ladder used for work, or "
                  "children on a playground)? Judge only what they do.")
DESCRIBE_QUESTION = ("Is a person here doing something other than walking or standing normally (for example trying "
                     "doors or windows, looking into places, hiding, or carrying something away)? Judge only what they "
                     "do, never how they look.")
CAR_QUESTION = ("Is a person trying to open a car's door, reaching into a car, or looking into its windows (not getting "
                "into their own car normally, or just walking past)? Judge only what they do.")
CAR_KINDS = ("car", "lorry", "motorbike")   # the parked vehicles the car signs look at (labeler/vehicles.py names)
WRISTS = (9, 10)                              # COCO body points: left and right wrist
FEET_WINDOW_S = 10.0
THUMB = (64, 36)     # the camera check looks at a tiny gray picture
NOW_LOOKS = 5        # "now": the middle value of the last 5 looks
REF_S = (10.0, 15.0)  # "before": the looks this long ago


@dataclass
class _Person:
    feet: deque = field(default_factory=deque)     # (t, box bottom, height)
    walk: dict = field(default_factory=dict)       # area id -> deque of (t, x, y, height) while inside
    box: tuple = ()
    last: float = 0.0
    climb_since: float | None = None
    fired: set = field(default_factory=set)
    restricted_in: dict = field(default_factory=dict)    # area id -> since when inside
    restricted_out: dict = field(default_factory=dict)   # area id -> since when outside (after a visit)
    paced: dict = field(default_factory=dict)      # area id -> when pacing was raised
    wrists: list = field(default_factory=list)     # the wrists seen this frame (pixels)
    spots: deque = field(default_factory=deque)    # (t, feet x, feet y, height) for FEET_WINDOW_S
    car_hand: dict = field(default_factory=dict)   # car key -> (since when a hand is at it, feet x, feet y then)
    car_stops: list = field(default_factory=list)  # (t, car key) of each stop at a parked car


def _edges(gray: np.ndarray) -> np.ndarray:
    """The picture's edges, as a unit vector (brightness does not matter; where the edges are does)."""
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).ravel()
    mag -= mag.mean()
    n = float(np.linalg.norm(mag))
    return mag / n if n > 1e-6 else mag


class BehaviourWatch:
    def __init__(self, cfg: dict):
        self.night_hours = str(cfg.get("night_hours", "22-06"))
        self.climb_rise = float(cfg.get("climb_rise", 0.25))
        self.climb_hold_s = float(cfg.get("climb_hold_s", 1.0))
        self.restricted_s = float(cfg.get("restricted_s", 3.0))
        self.restricted_rearm_s = float(cfg.get("restricted_rearm_s", 60.0))
        self.pace_window_s = float(cfg.get("pace_window_s", 180.0))
        self.pace_turns = int(cfg.get("pace_turns", 4))
        self.pace_span = float(cfg.get("pace_span", 1.5))
        self.camera_every_s = float(cfg.get("camera_every_s", 1.0))
        self.camera_hold_s = float(cfg.get("camera_hold_s", 3.0))
        self.camera_seen_s = float(cfg.get("camera_seen_s", 30.0))
        self.covered_std = float(cfg.get("covered_std", 6.0))
        self.moved_corr = float(cfg.get("moved_corr", 0.35))
        self.follow_s = float(cfg.get("follow_s", 2.0))
        self.car_hold_s = float(cfg.get("car_hold_s", 5.0))
        self.car_stop_s = float(cfg.get("car_stop_s", 3.0))
        self.car_count = int(cfg.get("car_count", 2))
        self.car_window_s = float(cfg.get("car_window_s", 120.0))
        self.car_rearm_s = float(cfg.get("car_rearm_s", 60.0))
        self.car_last = -1e9          # when a car sign was last raised (one a camera every car_rearm_s)
        self.crossings: dict = {}     # door line id -> deque of (t, key, "in" or "out"), the last follow_s
        self.people: dict = {}
        self.looks: deque = deque()   # (t, edges, std of the gray picture)
        self.last_look = -1e9
        self.camera_since: dict = {"covered": None, "turned": None}
        self.camera_fired: set = set()

    def night(self, wall: datetime) -> bool:
        from .alerts import in_hours
        return in_hours(self.night_hours, wall.hour)

    # People ------------------------------------------------------------------------------------------------------

    def update(self, key, t: float, box: tuple, shape: tuple, zones: list[dict], kpts=None) -> None:
        p = self.people.setdefault(key, _Person())
        x1, y1, x2, y2 = box[:4]
        h = max(y2 - y1, 1.0)
        p.box, p.last = tuple(box[:4]), t
        p.feet.append((t, y2, h))
        while p.feet and t - p.feet[0][0] > FEET_WINDOW_S:
            p.feet.popleft()
        p.spots.append((t, (x1 + x2) / 2, y2, h))
        while p.spots and t - p.spots[0][0] > FEET_WINDOW_S:
            p.spots.popleft()
        p.wrists = [(float(kpts[i][0]), float(kpts[i][1])) for i in WRISTS
                    if kpts is not None and len(kpts) > i and kpts[i][2] >= 0.3]
        H, W = shape[:2]
        feet = ((x1 + x2) / 2 / W, y2 / H)
        for z in zones:
            if z.get("kind") != AREA:
                continue
            walk = p.walk.setdefault(z["id"], deque())
            if inside(feet, z["points"]):
                if not walk or t - walk[-1][0] >= 0.5:   # twice a second is enough to follow a walk
                    walk.append((t, (x1 + x2) / 2, y2, h))
            while walk and t - walk[0][0] > self.pace_window_s:
                walk.popleft()

    def forget(self, key) -> None:
        self.people.pop(key, None)

    def _at(self, p: _Person, area: dict, shape: tuple) -> bool:
        H, W = shape[:2]
        x1, y1, x2, y2 = p.box
        pts = [((x1 + x2) / 2 / W, (y1 + y2) / 2 / H), ((x1 + x2) / 2 / W, y2 / H)]
        return any(inside(pt, area["points"]) for pt in pts)

    def _climbing(self, t: float, p: _Person) -> bool:
        if len(p.feet) < 5:
            return False
        bottoms = sorted(b for _, b, _ in p.feet)
        heights = sorted(h for _, _, h in p.feet)
        usual_bottom, usual_h = bottoms[len(bottoms) // 2], heights[len(heights) // 2]
        _, bottom, h = p.feet[-1]
        up = usual_bottom - bottom >= self.climb_rise * usual_h and h >= 0.85 * usual_h
        if not up:
            p.climb_since = None
            if usual_bottom - bottom < 0.1 * usual_h:
                p.fired.discard("climb")
            return False
        p.climb_since = t if p.climb_since is None else p.climb_since
        if t - p.climb_since >= self.climb_hold_s and "climb" not in p.fired:
            p.fired.add("climb")
            return True
        return False

    def turns(self, walk) -> int:
        """How often a walk turned back, each leg at least pace_span body heights (along x or y, whichever it moved
        most)."""
        if len(walk) < 3:
            return 0
        xs, ys = [w[1] for w in walk], [w[2] for w in walk]
        vals = xs if max(xs) - min(xs) >= max(ys) - min(ys) else ys
        span = self.pace_span * float(np.median([w[3] for w in walk]))
        ext, direction, n = vals[0], 0, 0
        for v in vals[1:]:
            if direction == 0:
                if abs(v - ext) >= span:
                    direction, ext = (1 if v > ext else -1), v
            elif direction > 0:
                if v > ext:
                    ext = v
                elif ext - v >= span:
                    n, direction, ext = n + 1, -1, v
            else:
                if v < ext:
                    ext = v
                elif v - ext >= span:
                    n, direction, ext = n + 1, 1, v
        return n

    def tick(self, t: float, zones: list[dict], shape: tuple, watchful: bool) -> list[tuple]:
        """(key, sign, reason, area) for each sign now: "climb" (confirm it), "restricted" and "pacing" (alert
        directly; only raised when watchful: at night or Away)."""
        out = []
        areas = [z for z in zones if z.get("kind") == AREA]
        for key, p in self.people.items():
            if t - p.last > 1.0:
                continue
            for z in areas:
                if z.get("role") == "fence" and self._at(p, z, shape) and self._climbing(t, p):
                    out.append((key, "climb", f"climbing at {z['name']} (feet raised)", z))
                if z.get("role") == "restricted":
                    out += self._restricted(t, key, p, z, shape, watchful)
                walk = p.walk.get(z["id"])
                if walk and watchful and t - p.paced.get(z["id"], -1e9) > self.pace_window_s:
                    n = self.turns(walk)
                    if n >= self.pace_turns:
                        p.paced[z["id"]] = t
                        minutes = max(1, round((walk[-1][0] - walk[0][0]) / 60))
                        out.append((key, "pacing", f"walking back and forth in {z['name']} ({n} turns in "
                                                   f"{minutes} min)", z))
        for key in [k for k, p in self.people.items() if t - p.last > self.pace_window_s]:
            del self.people[key]
        return out

    def _restricted(self, t: float, key, p: _Person, z: dict, shape: tuple, watchful: bool) -> list[tuple]:
        H, W = shape[:2]
        feet = ((p.box[0] + p.box[2]) / 2 / W, p.box[3] / H)
        zid = z["id"]
        if inside(feet, z["points"]):
            p.restricted_out.pop(zid, None)
            since = p.restricted_in.setdefault(zid, t)
            if watchful and t - since >= self.restricted_s and f"restricted {zid}" not in p.fired:
                p.fired.add(f"restricted {zid}")
                return [(key, "restricted", f"in the restricted area {z['name']} for {int(t - since)} s", z)]
        else:
            p.restricted_in.pop(zid, None)
            out_since = p.restricted_out.setdefault(zid, t)
            if t - out_since >= self.restricted_rearm_s:
                p.fired.discard(f"restricted {zid}")
        return []

    # Parked cars -------------------------------------------------------------------------------------------------

    def _still(self, t: float, p: _Person, seconds: float) -> bool:
        """Seen for `seconds` and the feet moved less than half a body height over them."""
        spots = [s for s in p.spots if t - s[0] <= seconds]
        if not spots or t - spots[0][0] < seconds - 0.5:
            return False
        xs, ys = [s[1] for s in spots], [s[2] for s in spots]
        return max(max(xs) - min(xs), max(ys) - min(ys)) < 0.5 * min(s[3] for s in spots)

    @staticmethod
    def _at_car(p: _Person, car: tuple) -> bool:
        """The person stands at the car: their feet beside it (within a quarter of its width) and around its bottom."""
        cx1, cy1, cx2, cy2 = car[:4]
        cw, ch = cx2 - cx1, cy2 - cy1
        fx, fy = (p.box[0] + p.box[2]) / 2, p.box[3]
        return cx1 - 0.25 * cw <= fx <= cx2 + 0.25 * cw and cy1 + 0.4 * ch <= fy <= cy2 + 0.4 * (p.box[3] - p.box[1])

    def cars(self, t: float, parked: list[tuple], watchful: bool) -> list[tuple]:
        """(person key, sign, reason, car box) for a car sign now; parked: (car key, box, kind) of the parked cars in
        view. Only when watchful (night or Away); at most one a camera every car_rearm_s."""
        if not watchful:
            for p in self.people.values():
                p.car_hand.clear()
                p.car_stops.clear()
            return []
        out = []
        for key, p in self.people.items():
            if t - p.last > 1.0 or not p.box:
                continue
            for ck, box, kind in parked:
                at = self._at_car(p, box)
                pad = 0.05 * (box[2] - box[0])
                hand = at and any(box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad
                                  for x, y in p.wrists)
                fx, fy, h = (p.box[0] + p.box[2]) / 2, p.box[3], p.box[3] - p.box[1]
                since, sx, sy = p.car_hand.get(ck, (t, fx, fy))
                if hand and math.hypot(fx - sx, fy - sy) < 0.5 * h:   # the hand stays, and so do they (no stroll)
                    p.car_hand[ck] = (since, sx, sy)
                    if t - since >= self.car_hold_s:
                        out.append((key, "car", f"a hand at a parked {kind} for {int(t - since)} s", box))
                        p.car_hand.clear()
                else:
                    p.car_hand.pop(ck, None)
                if at and self._still(t, p, self.car_stop_s) and (not p.car_stops or p.car_stops[-1][1] != ck):
                    p.car_stops.append((t, ck))
            p.car_stops = [s for s in p.car_stops if t - s[0] <= self.car_window_s]
            visited = list(dict.fromkeys(ck for _, ck in p.car_stops))
            if len(visited) >= self.car_count:
                box = next((b for ck, b, _ in parked if ck == visited[-1]), p.box)   # the last car (else the person)
                out.append((key, "cars", f"stopped at {len(visited)} parked vehicles in a row", box))
                p.car_stops.clear()
        if not out or t - self.car_last < self.car_rearm_s:
            return []
        self.car_last = t
        return out[:1]

    # Doors -------------------------------------------------------------------------------------------------------

    def crossed(self, t: float, key, zone: dict, direction: str) -> tuple | None:
        """Someone crossed a counting line ("in" or "out"). On a "door" line: (other key, reason) when someone else
        crossed it the same way within follow_s before (following them through), else None."""
        if zone.get("role") != "door" or direction not in ("in", "out"):
            return None
        recent = self.crossings.setdefault(zone["id"], deque())
        while recent and t - recent[0][0] > self.follow_s:
            recent.popleft()
        before = next(((ti, k) for ti, k, d in reversed(recent) if d == direction and k != key), None)
        recent.append((t, key, direction))
        if before is None:
            return None
        return before[1], (f"two people went {direction} through {zone['name']} {t - before[0]:.1f} s apart (someone "
                           f"following another through)")

    # The camera --------------------------------------------------------------------------------------------------

    def camera_due(self, t: float) -> bool:
        return t - self.last_look >= self.camera_every_s

    def camera(self, t: float, frame: np.ndarray) -> tuple[str, str] | None:
        """("covered" or "turned", reason) when the camera was just covered or moved (see the module text)."""
        self.last_look = t
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA)
        self.looks.append((t, gray))
        while self.looks and t - self.looks[0][0] > REF_S[1] + 1:
            self.looks.popleft()
        old = [g for ti, g in self.looks if REF_S[0] <= t - ti <= REF_S[1]]
        if len(old) < 3 or len(self.looks) < NOW_LOOKS:
            return None
        now = np.median(np.stack([g for _, g in list(self.looks)[-NOW_LOOKS:]]), axis=0).astype(np.uint8)
        before = np.median(np.stack(old), axis=0).astype(np.uint8)
        match = float(_edges(now) @ _edges(before))
        had_picture = float(before.std()) >= 3 * self.covered_std
        covered = had_picture and float(now.std()) < self.covered_std and match < self.moved_corr
        turned = had_picture and not covered and match < self.moved_corr
        out = None
        for kind, now_on in (("covered", covered), ("turned", turned)):
            if not now_on:
                self.camera_since[kind] = None
                continue
            since = self.camera_since[kind] = self.camera_since[kind] or t
            if t - since >= self.camera_hold_s and kind not in self.camera_fired:
                self.camera_fired.add(kind)
                out = (kind, "the camera was covered" if kind == "covered" else "the camera was turned or moved")
        if not covered and not turned and match > 0.8:
            self.camera_fired.clear()   # the same view for a while again: ready for the next time
        return out

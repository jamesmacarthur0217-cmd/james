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
        self.people: dict = {}
        self.looks: deque = deque()   # (t, edges, std of the gray picture)
        self.last_look = -1e9
        self.camera_since: dict = {"covered": None, "turned": None}
        self.camera_fired: set = set()

    def night(self, wall: datetime) -> bool:
        from .alerts import in_hours
        return in_hours(self.night_hours, wall.hour)

    # People ------------------------------------------------------------------------------------------------------

    def update(self, key, t: float, box: tuple, shape: tuple, zones: list[dict]) -> None:
        p = self.people.setdefault(key, _Person())
        x1, y1, x2, y2 = box[:4]
        h = max(y2 - y1, 1.0)
        p.box, p.last = tuple(box[:4]), t
        p.feet.append((t, y2, h))
        while p.feet and t - p.feet[0][0] > FEET_WINDOW_S:
            p.feet.popleft()
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

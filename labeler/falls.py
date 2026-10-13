"""Fall detection from the body points the detector already finds (YOLO11-pose, COCO keypoints).

A fall here is: a person who was upright is down a moment later (within `drop_s`), their head having dropped a
large part of their standing height (`min_drop`), and who then stays down for `after_s`. Lying down slowly
(going to bed, lying on a mat) is not a fall: the upright frames are too long ago when lying starts. Getting up
again (standing for `getup_s`) before `after_s` cancels it. Areas marked "lying is normal here" (a bed, a sofa)
never start a fall.

"Down" is lying (the line from the middle of the shoulders to the middle of the hips closer to horizontal than
`lying_angle`, or a box clearly wider than tall) or collapsed (less than `low_ratio` of the person's standing
height: someone lying towards the camera looks short, not flat). "Standing" again needs an upright torso and most
of the standing height. A fall is confirmed only if, while down, the box was typically clearly wider than tall
(`min_width`): kneeling or crouching on the floor (looking under a bed) is not lying.

What the tracker does around a fall, found on the UR Fall Detection videos and checked on MOT16 (tools/fall_check.py):
- it gives the person a new number as they fall, sometimes in the very frame the old number is last seen, or gives
  out an old number again: the history moves to the number still in view (_inherit, _new_number_here, _merge);
- it loses a person lying at the bottom edge of the picture, half cut off: someone who was coming down fast and was
  no longer upright when they vanished there counts as down ("fell out of sight"). Near that edge the box's shape
  says nothing (the edge cuts it), so only the body points decide there: someone walking out of the bottom of the
  picture stays upright and is not a fall (MOT16-04, a street filmed from high up);
- one fallen person can get two overlapping boxes: one alert;
- the box flickers while they fall: decisions use the highest head and tallest standing box of the last seconds.
Cameras looking straight down (from the ceiling) do not work with this: from above, standing and lying look alike.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

UPRIGHT, LYING, UNSURE = "upright", "lying", "unsure"
MIN_CONF = 0.3        # body points below this are not used
KEEP_S = 10.0         # history kept per person (standing height comes from it)
EDGE = 0.04           # "at the bottom edge": the box's bottom within this share of the picture height


def _mid(kpts: np.ndarray, a: int, b: int) -> tuple[float, float] | None:
    pts = [kpts[i, :2] for i in (a, b) if kpts[i, 2] >= MIN_CONF]
    return tuple(np.mean(pts, axis=0)) if pts else None


def posture(kpts: np.ndarray | None, box: tuple, lying_angle: float = 60.0, upright_angle: float = 35.0,
            cut: bool = False) -> str:
    """UPRIGHT, LYING or UNSURE for one person in one frame (from the torso, else the box shape).
    cut: the box touches the picture's edge, so its shape says nothing (only head and shoulders may be in view)."""
    x1, y1, x2, y2 = box[:4]
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    if w / h >= 1.3 and not cut:
        return LYING   # nobody upright is this wide (a person lying diagonally can have a "vertical" torso)
    if kpts is not None:
        shoulders, hips = _mid(kpts, L_SHOULDER, R_SHOULDER), _mid(kpts, L_HIP, R_HIP)
        if shoulders and hips:
            dx, dy = hips[0] - shoulders[0], hips[1] - shoulders[1]
            if math.hypot(dx, dy) >= 0.12 * max(w, h):   # a torso too short to trust: seen end-on
                angle = math.degrees(math.atan2(abs(dx), abs(dy)))   # 0 = vertical, 90 = horizontal
                if angle >= lying_angle:
                    return LYING
                if angle <= upright_angle:
                    return UPRIGHT
                return UNSURE
    if h / w >= 1.6 and not cut:
        return UPRIGHT
    return UNSURE


@dataclass
class _Track:
    history: deque = field(default_factory=deque)     # (t, posture, top y, height, center x, center y, width)
    last_t: float = 0.0
    last_box: tuple = (0.0, 0.0, 0.0, 0.0)
    at_bottom: bool = False                           # last box touched the bottom edge of the picture
    down_since: float | None = None                   # a fall started here
    standing_since: float | None = None               # standing again since (cancels after getup_s)
    fired: bool = False                               # confirmed (once, until they get up)
    fired_at: float | None = None
    born: float | None = None                         # first seen under this number
    lying_ok: bool = False                            # last seen where lying is normal (a bed, a sofa)
    out_of_sight: bool = False                        # the fall continues unseen (lost at the bottom edge)


class FallWatch:
    """Fall state of every person on one camera. Call update() for each person in each frame and tick() once per
    frame (it confirms falls of people no longer seen)."""

    def __init__(self, cfg: dict):
        self.after_s = float(cfg["after_s"])
        self.drop_s = float(cfg["drop_s"])
        self.min_drop = float(cfg["min_drop"])
        self.getup_s = float(cfg["getup_s"])
        self.lying_angle = float(cfg["lying_angle"])
        self.low_ratio = float(cfg.get("low_ratio", 0.5))
        self.lost_s = float(cfg.get("lost_s", 0.5))
        self.min_width = float(cfg.get("min_width", 1.25))
        self.tracks: dict = {}

    # --- per person, per frame ---------------------------------------------------------------------------

    def update(self, key, t: float, box: tuple, kpts: np.ndarray | None, lying_ok: bool = False,
               frame_h: float | None = None) -> dict | None:
        """Returns {"since": t} the frame a fall is confirmed (once per fall), else None.
        lying_ok: the person is in an area where lying is normal (no new fall starts there).
        frame_h: the picture's height, to notice people lost at the bottom edge."""
        new = key not in self.tracks
        tr = self.tracks.setdefault(key, _Track())
        if new or t - tr.last_t > self.lost_s:   # a new number, or an old one the tracker gave out again
            self._inherit(tr, t, box)
        at_bottom = frame_h is not None and box[3] >= frame_h * (1 - EDGE)
        p = posture(kpts, box, self.lying_angle, cut=at_bottom)
        top, height = float(box[1]), float(box[3] - box[1])
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        tr.history.append((t, p, top, height, cx, cy, float(box[2] - box[0])))
        while tr.history and t - tr.history[0][0] > KEEP_S:
            tr.history.popleft()
        tr.last_t, tr.last_box, tr.out_of_sight = t, tuple(box[:4]), False
        tr.at_bottom, tr.lying_ok = at_bottom, lying_ok
        tr.born = t if tr.born is None else tr.born

        tall = self._standing_height(tr, t)
        # Collapsed (short): not at the bottom edge, where the edge itself cuts the box short.
        down = p == LYING or (tall is not None and height <= self.low_ratio * tall and not tr.at_bottom)
        standing = p == UPRIGHT and (tall is None or height >= (1 - self.min_drop / 2) * tall)

        if tr.down_since is None:
            if down and not lying_ok and self._dropped(tr, t, top):
                tr.down_since, tr.standing_since = t, None
            return None
        if standing:
            tr.standing_since = tr.standing_since if tr.standing_since is not None else t
            if t - tr.standing_since >= self.getup_s:   # got up: over (and a later fall counts again)
                tr.down_since, tr.standing_since, tr.fired = None, None, False
            return None
        tr.standing_since = None
        return self._confirm(tr, t)

    def tick(self, t: float) -> list[tuple[object, dict]]:
        """Once per frame: [(key, {"since": t})] for falls confirmed although the person is no longer seen."""
        events = []
        for key, tr in list(self.tracks.items()):
            if tr.last_t >= t:
                continue
            if t - tr.last_t > KEEP_S + self.after_s:
                del self.tracks[key]   # long gone
                continue
            if t - tr.last_t < self.lost_s:
                continue
            here = self._new_number_here(key, tr)
            if here is not None:
                # Still in view under a number the tracker gave out as this one vanished: it carries on.
                self._merge(tr, here)
                del self.tracks[key]
                continue
            if tr.lying_ok:
                continue
            # Lost at the bottom edge while falling (head coming down fast): the fall continues out of sight.
            # A person still upright while the edge cuts them off is walking out of the picture (MOT16-04).
            if (tr.down_since is None and tr.at_bottom and any(h[1] != UPRIGHT for h in list(tr.history)[-3:])
                    and self._dropped(tr, tr.last_t, tr.last_box[1], tr.last_box[3])):
                tr.down_since, tr.out_of_sight = tr.last_t, True
            if tr.down_since is not None and (tr.out_of_sight or tr.at_bottom):
                tr.out_of_sight = True
                e = self._confirm(tr, t)
                if e:
                    events.append((key, e))
        return events

    # --- helpers -----------------------------------------------------------------------------------------

    def _confirm(self, tr: _Track, t: float) -> dict | None:
        if tr.fired or t - tr.down_since < self.after_s:
            return None
        # Lying, not kneeling or crouching: while down, the person's box is typically clearly wider than tall
        # (UR Fall videos: fallen 1.1-4.0 times, people looking under a bed or crawling under a desk 0.8-1.2).
        # Not for a fall out of sight: its few boxes are from the middle of the fall.
        shapes = [h[6] / max(h[3], 1.0) for h in tr.history if h[0] >= tr.down_since]
        if not tr.out_of_sight and shapes and float(np.median(shapes)) < self.min_width:
            return None
        tr.fired, tr.fired_at = True, t
        if self._same_fall_as_another(tr, t):
            return None   # one fallen person seen as two overlapping boxes: one alert
        return {"since": tr.down_since}

    def _same_fall_as_another(self, tr: _Track, t: float) -> bool:
        """Another fall, confirmed and still going on (seen in the last 2 s), where this one is (boxes overlap)."""
        x1, y1, x2, y2 = tr.last_box
        for other in self.tracks.values():
            if other is tr or not other.fired or other.down_since is None or t - other.last_t > 2.0:
                continue
            ox1, oy1, ox2, oy2 = other.last_box
            if ox1 <= x2 and x1 <= ox2 and oy1 <= y2 and y1 <= oy2:
                return True
        return False

    def _standing_height(self, tr: _Track, t: float) -> float | None:
        hs = [h[3] for h in tr.history if h[1] == UPRIGHT and t - h[0] <= KEEP_S]
        return max(hs) if hs else None

    def _dropped(self, tr: _Track, t: float, top: float, bottom: float | None = None) -> bool:
        """Upright within drop_s before t, and the head (box top) came down min_drop of standing height.
        With `bottom`: the feet also stayed where they were (someone walking out of the bottom of the picture
        moves down as a whole; seen on MOT16-04, a street filmed from high up)."""
        standing = [(h[2], h[3]) for h in tr.history if h[1] == UPRIGHT and 0 <= t - h[0] <= self.drop_s]
        if not standing:
            return False
        tall = max(hh for _, hh in standing)
        highest = min(tp for tp, _ in standing)
        if bottom is not None and bottom - max(tp + hh for tp, hh in standing) > self.min_drop / 2 * tall:
            return False
        return (top - highest) >= self.min_drop * tall

    def _inherit(self, tr: _Track, t: float, box: tuple) -> None:
        """A new (or reused) number for someone the tracker just lost close by: continue their history, when it
        is fresher than this number's own."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best, best_d = None, None
        for other in self.tracks.values():
            if (other is tr or not other.history or not 0 < t - other.last_t <= self.drop_s
                    or other.last_t <= tr.last_t):
                continue
            ox = (other.last_box[0] + other.last_box[2]) / 2
            oy = (other.last_box[1] + other.last_box[3]) / 2
            reach = max(other.last_box[3] - other.last_box[1], other.last_box[2] - other.last_box[0])
            d = math.hypot(cx - ox, cy - oy)
            if d <= reach and (best_d is None or d < best_d):
                best, best_d = other, d
        if best is not None:
            tr.history = deque(best.history)
            tr.down_since, tr.fired = best.down_since, best.fired
            best.down_since, best.out_of_sight = None, False   # the fall is followed by the new number now

    def _new_number_here(self, key, tr: _Track) -> _Track | None:
        """A person first seen around when this one vanished (within drop_s), close to where it was, and seen
        since: most likely the same person under a new number. The tracker sometimes starts the new number in
        the same frame the old one is last seen, so _inherit could not take the history then."""
        x1, y1, x2, y2 = tr.last_box
        reach = max(x2 - x1, y2 - y1, 1.0)
        best, best_d = None, None
        for k, other in self.tracks.items():
            if k == key or other.last_t <= tr.last_t or other.born is None or other.born < tr.last_t - self.drop_s:
                continue
            ox = (other.last_box[0] + other.last_box[2]) / 2
            oy = (other.last_box[1] + other.last_box[3]) / 2
            d = math.hypot(ox - (x1 + x2) / 2, oy - (y1 + y2) / 2)
            if d <= reach and (best_d is None or d < best_d):
                best, best_d = other, d
        return best

    def _merge(self, old: _Track, new: _Track) -> None:
        """The person vanished as `old` continues as `new`: one history, and a fall already under way."""
        new.history = deque(sorted(list(old.history) + list(new.history), key=lambda h: h[0]))
        if new.down_since is None and old.down_since is not None:
            new.down_since, new.fired, new.fired_at = old.down_since, old.fired, old.fired_at

    def state(self, key) -> str | None:
        """"fallen" once confirmed, "down" while a fall waits for after_s, else None (for the live picture)."""
        tr = self.tracks.get(key)
        if tr is None or tr.down_since is None:
            return None
        return "fallen" if tr.fired else "down"

    def forget(self, key) -> None:
        self.tracks.pop(key, None)

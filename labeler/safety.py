"""Quick checks for people (Phase 45): signs in the body points and boxes the camera already has, every frame, that a
person may need help. They only raise a suspect; the vision model confirms it (labeler/incidents.py).

One SafetyWatch per safety camera (Pipeline.safety_watch), fed like FallWatch with each person's box and body points
and the role of the area they stand in. For each person it keeps the last few minutes and, once a frame, looks for:

- lying still: the posture is lying (labeler/falls.py posture: the torso near horizontal, or a box much wider than
  tall) on most frames for `down_s` seconds, and the person has not moved: someone who lay down slowly or collapsed
  without a fall the fall check would see. Question: does this person need help?
- slumped: not lying, the head below the line of the shoulders (bent forward or to the side) on most frames for
  `slumped_s` seconds, not moving: someone slumped in a chair or against a wall.
- still for long: not moved for `still_s` seconds in any posture: the slow round for distress (a person can be unwell
  without looking slumped). Sent at the lowest priority, after people's labels, with its own question.

"Not moved": the middle of the box stayed within `still_move` of the person's height of where it is now. Each sign is
raised once per still period (moving again re-arms it). Nothing is raised in an area with the role "bed" (lying,
sleeping and resting are normal there).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .falls import LYING, MIN_CONF, posture
from .geometry import L_EAR, L_EYE, L_SHOULDER, NOSE, R_EAR, R_EYE, R_SHOULDER

DISTRESS_QUESTION = ("Does this person look unwell, collapsed, unconscious or in distress (not simply resting, "
                     "reading, working or sleeping in a bed)?")
SHARE = 0.8   # a sign must be there on this share of the frames of its time


def head_low(kpts: np.ndarray | None) -> bool | None:
    """The head below the line of the shoulders (image y grows downwards); None when not seen well enough."""
    if kpts is None:
        return None
    shoulders = [kpts[i, 1] for i in (L_SHOULDER, R_SHOULDER) if kpts[i, 2] >= MIN_CONF]
    head = [kpts[i, 1] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kpts[i, 2] >= MIN_CONF]
    if not shoulders or not head:
        return None
    return float(np.mean(head)) > float(np.mean(shoulders))


@dataclass
class _Person:
    hist: deque = field(default_factory=deque)   # (t, centre x, centre y, height, lying, head low)
    role: str = ""
    box: tuple = ()
    last: float = 0.0
    fired: set = field(default_factory=set)


class SafetyWatch:
    def __init__(self, cfg: dict):
        self.down_s = float(cfg.get("down_s", 20))
        self.slumped_s = float(cfg.get("slumped_s", 30))
        self.still_s = float(cfg.get("still_s", 300))
        self.still_move = float(cfg.get("still_move", 0.08))
        self.keep_s = max(self.down_s, self.slumped_s, self.still_s) + 5
        self.people: dict = {}

    def update(self, key, t: float, box: tuple, kpts: np.ndarray | None, role: str = "") -> None:
        p = self.people.setdefault(key, _Person())
        x1, y1, x2, y2 = box[:4]
        p.hist.append((t, (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1.0), posture(kpts, box) == LYING,
                       head_low(kpts)))
        while p.hist and t - p.hist[0][0] > self.keep_s:
            p.hist.popleft()
        p.role, p.box, p.last = role, tuple(box[:4]), t

    def forget(self, key) -> None:
        self.people.pop(key, None)

    def mark(self, key, sign: str = "down") -> None:
        """This sign is handled already for this still period (a fall: no second "lying still" check)."""
        if key in self.people:
            self.people[key].fired.add(sign)

    @staticmethod
    def _still_for(p: _Person, move: float) -> float:
        """How long the person has stayed where they are now (within `move` x their height)."""
        t, cx, cy, h, *_ = p.hist[-1]
        since = t
        for ti, x, y, hi, *_ in reversed(p.hist):
            if abs(x - cx) > move * h or abs(y - cy) > move * h:
                break
            since = ti
        return t - since

    @staticmethod
    def _share(p: _Person, since: float, index: int) -> float:
        frames = [f[index] for f in p.hist if f[0] >= since and f[index] is not None]
        return sum(bool(v) for v in frames) / len(frames) if frames else 0.0

    def tick(self, t: float) -> list[tuple]:
        """(key, category, reason, level or None, question or None) for each sign seen now."""
        out = []
        for key, p in self.people.items():
            if not p.hist or t - p.last > 1.0 or p.role == "bed":
                continue
            still = self._still_for(p, self.still_move)
            if still < 2.0:
                p.fired.clear()   # moving: every sign can be raised again
                continue
            lying = self._share(p, t - self.down_s, 4) >= SHARE
            low = self._share(p, t - self.slumped_s, 5) >= SHARE
            if still >= self.down_s and lying and "down" not in p.fired:
                p.fired.add("down")
                out.append((key, "person_down", f"lying still {int(still)} s", None, None))
            elif still >= self.slumped_s and low and not lying and "slumped" not in p.fired:
                p.fired.add("slumped")
                out.append((key, "person_down", f"slumped, head down, still {int(still)} s", None, None))
            elif still >= self.still_s and "still" not in p.fired:
                p.fired.add("still")
                out.append((key, "person_down", f"not moved for {int(still // 60)} min", "round", DISTRESS_QUESTION))
        return out

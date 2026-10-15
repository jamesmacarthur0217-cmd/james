"""The help sign (Phase 32): one hand held above the head for a few seconds, as a silent call for help.

From the body points the detector already finds (YOLO11-pose, COCO keypoints), no extra model. In one frame a hand is
**up** when its wrist is above the top of the head by a margin (a share of the person's torso length, so it works
near and far) and its elbow is above its shoulder (an arm stretched up, not a hand scratching the head). The sign is
a hand up held for `hold_s` seconds (3 s, the user's choice), with short gaps allowed (a missed frame, a body point
not found for a moment); one alert per sign: the hand must be down for `rearm_s` before a new sign counts.

The head's top is the highest of the nose, eyes and ears; when none is found the person is not checked (seen from
behind, a hand in front of the face: the camera cannot tell where the head is). `both_ok`: whether two hands up
count too (stretching does that; measured with tools/gesture_check.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import L_EAR, L_EYE, L_HIP, L_SHOULDER, NOSE, R_EAR, R_EYE, R_HIP, R_SHOULDER

L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10
MIN_CONF = 0.3
DEFAULTS = {"hold_s": 3.0, "gap_s": 0.5, "rearm_s": 2.0, "margin": 0.15, "both_ok": True}


def _pt(kpts: np.ndarray, i: int) -> tuple[float, float] | None:
    return (float(kpts[i, 0]), float(kpts[i, 1])) if kpts[i, 2] >= MIN_CONF else None


def _torso(kpts: np.ndarray) -> float | None:
    """Shoulders to hips (the body's size, whatever the distance); from the shoulders' width when the hips are hidden."""
    ls, rs, lh, rh = (_pt(kpts, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
    shoulders = [p for p in (ls, rs) if p]
    hips = [p for p in (lh, rh) if p]
    if shoulders and hips:
        s, h = np.mean(shoulders, axis=0), np.mean(hips, axis=0)
        return float(math.hypot(*(h - s)))
    if ls and rs:
        return 1.5 * math.hypot(ls[0] - rs[0], ls[1] - rs[1])
    return None


def hands_up(kpts: np.ndarray | None, margin: float = DEFAULTS["margin"]) -> list[str]:
    """Which hands are up in one frame: [], ["left"], ["right"] or both."""
    if kpts is None or len(kpts) < 17:
        return []
    head = [p[1] for p in (_pt(kpts, i) for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)) if p]
    torso = _torso(kpts)
    if not head or not torso or torso < 5:
        return []
    top = min(head) - margin * torso   # image y grows downwards: "above" is smaller
    out = []
    for side, wrist, elbow, shoulder in (("left", L_WRIST, L_ELBOW, L_SHOULDER), ("right", R_WRIST, R_ELBOW, R_SHOULDER)):
        w, e, s = _pt(kpts, wrist), _pt(kpts, elbow), _pt(kpts, shoulder)
        if w and e and s and w[1] < top and e[1] < s[1]:
            out.append(side)
    return out


@dataclass
class _Hold:
    since: float | None = None     # when the hand went up (this sign)
    last_up: float = -1e9          # the last frame with a hand up
    last_down: float = -1e9        # the last frame with no hand up
    fired: bool = False            # this sign already alerted


class GestureWatch:
    """Help signs of every person on one camera: call update() for each person in each frame."""

    def __init__(self, cfg: dict | None = None):
        c = {**DEFAULTS, **(cfg or {})}
        self.hold_s, self.gap_s, self.rearm_s = float(c["hold_s"]), float(c["gap_s"]), float(c["rearm_s"])
        self.margin, self.both_ok = float(c["margin"]), bool(c["both_ok"])
        self.people: dict = {}

    def update(self, key, t: float, kpts: np.ndarray | None) -> dict | None:
        """{"since": t, "hand": "left"/"right"/"both"} the frame a sign is confirmed (once per sign), else None."""
        h = self.people.setdefault(key, _Hold())
        up = hands_up(kpts, self.margin)
        if len(up) == 2 and not self.both_ok:
            up = []
        if not up:
            h.last_down = t
            if h.since is not None and t - h.last_up > self.gap_s:
                h.since = None                              # the hand came down
            if t - h.last_up >= self.rearm_s:
                h.fired = False                             # down long enough: a new sign may come
            return None
        if h.since is None or t - h.last_up > self.gap_s:
            h.since = t
        h.last_up = t
        if not h.fired and t - h.since >= self.hold_s:
            h.fired = True
            return {"since": h.since, "hand": "both" if len(up) == 2 else up[0]}
        return None

    def forget(self, key) -> None:
        self.people.pop(key, None)

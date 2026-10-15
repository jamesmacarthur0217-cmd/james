"""Camera picture check (Phase 30): what is wrong with a camera's picture, in plain words, and what to do.

From frames the camera already has (no extra model):
  covered    no detail at all (lens cover, camera busy): sources.is_blank, as the Health "blank picture" problem
  dark       too dark to see clothes and faces (mean brightness, share of near-black pixels)
  bright     washed out (share of near-white pixels)
  blurry     out of focus or a dirty lens: little fine detail for the picture's contrast (the variance of the
             Laplacian divided by the variance of the picture, so a plain wall and a busy street compare)
  backlight  people much darker than what is behind them (a window or the sun behind them)
And from the people the camera recorded (tracks.in_view_s / facing_s, age labels), on kiosk cameras:
  faces      people rarely face the camera, or their faces are too small for the age model (camera too high,
             too far, or to the side)

The limits (LIMITS) were set with tools/picture_check_eval.py on one half of COCO and MOT16 pictures made worse on
purpose and reported on the other half (README "Camera picture check").
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

WIDTH = 640   # pictures are compared at this width, so the camera's resolution does not change the numbers

# Set 2026-10-15 by tools/picture_check_eval.py --tune (the most caught with at most 1% of good pictures flagged).
LIMITS = {
    "dark_mean": 24.0,        # mean brightness (0-255) below this ...
    "dark_share": 0.72,       # ... or this share of pixels below 35
    "bright_share": 0.25,     # share of pixels above 245
    "blur_ratio": 0.004,      # Laplacian variance / picture variance below this
    "backlight_ratio": 0.36,  # people's brightness / background's below this ...
    "backlight_bg": 140.0,    # ... with a background at least this bright
}
# Faces (kiosk cameras), from the people recorded; chosen, not measured (no public footage with such cameras):
FACES = {"min_people": 10, "facing_share": 0.15, "ages_share": 0.3}

ADVICE = {
    "covered": "The picture is blank: open the lens cover, or close other apps that use the camera.",
    "dark": "The picture is too dark: turn on a light, or use a camera with night vision (infrared).",
    "bright": "The picture is washed out: turn the camera away from the window or the sun, or lower its exposure.",
    "blurry": "The picture is blurry: clean the lens, and check the focus (some webcams focus by hand).",
    "backlight": "People are dark against a bright background: move the camera so the light comes from behind it.",
    "faces": "Faces are rarely seen: put the camera at about eye height, facing where people look, not high up.",
    "small_faces": "Faces are too small for ages: move the camera closer to where people stand.",
}


def stats(frame: np.ndarray, people: list | tuple = ()) -> dict:
    """Numbers about one frame. people: boxes (x1, y1, x2, y2) in the frame's pixels."""
    h, w = frame.shape[:2]
    scale = WIDTH / w
    small = cv2.resize(frame, (WIDTH, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    var = float(gray.var())
    out = {"mean": float(gray.mean()), "dark_share": float((gray < 35).mean()),
           "bright_share": float((gray > 245).mean()),
           "blur_ratio": float(cv2.Laplacian(gray, cv2.CV_64F).var()) / var if var > 1 else 0.0}
    if len(people):
        mask = np.zeros(gray.shape, bool)
        for x1, y1, x2, y2 in people:
            mask[max(0, int(y1 * scale)):int(y2 * scale), max(0, int(x1 * scale)):int(x2 * scale)] = True
        if 0.01 < mask.mean() < 0.9:
            out["people_mean"], out["background_mean"] = float(gray[mask].mean()), float(gray[~mask].mean())
    return out


def findings(st: dict, blank: bool = False, limits: dict = LIMITS) -> list[str]:
    """What is wrong with one frame (codes of ADVICE), from stats()."""
    if blank:
        return ["covered"]
    out = []
    if st["mean"] < limits["dark_mean"] or st["dark_share"] > limits["dark_share"]:
        out.append("dark")
    elif st["bright_share"] > limits["bright_share"]:
        out.append("bright")
    if st["blur_ratio"] < limits["blur_ratio"] and "dark" not in out:   # a dark picture has little detail anyway
        out.append("blurry")
    bg, pm = st.get("background_mean"), st.get("people_mean")
    if bg is not None and bg >= limits["backlight_bg"] and pm < limits["backlight_ratio"] * bg:
        out = [f for f in out if f != "bright"] + ["backlight"]   # the bright background is the backlight
    return out


def check_frame(frame: np.ndarray, people: list | tuple = (), limits: dict = LIMITS) -> list[str]:
    from .sources import is_blank
    return findings(stats(frame, people), is_blank(frame), limits)


class PictureWatch:
    """One camera's checks over time (Pipeline.run checks a frame every `every_s`). A finding *lasts* when it was in at
    least `lasting_share` of the checks of the last `window_s`, with at least half the window checked: one dark minute
    (a light switched off and on) or one blurry moment (someone close to the lens) is not reported. "covered" is left
    to the Health "blank picture" problem, which reports it already."""

    def __init__(self, cfg: dict, clock=time.monotonic, wall=datetime.now):
        self.every_s, self.window_s = float(cfg["every_s"]), float(cfg["window_s"])
        self.share = float(cfg["lasting_share"])
        self.clock, self.wall = clock, wall
        self._checks: deque = deque()          # (monotonic, wall, findings)
        self._last_at = -1e18
        self._lock = threading.Lock()

    def due(self) -> bool:
        return self.clock() - self._last_at >= self.every_s

    def add(self, found: list[str]) -> None:
        now = self.clock()
        with self._lock:
            self._last_at = now
            self._checks.append((now, self.wall(), list(found)))
            while self._checks and now - self._checks[0][0] > self.window_s:
                self._checks.popleft()

    def check(self, frame: np.ndarray, people: list | tuple = ()) -> list[str]:
        found = check_frame(frame, people)
        self.add(found)
        return found

    def lasting(self) -> dict[str, str]:
        """{finding: since when (wall time, ISO)} for the findings that last."""
        with self._lock:
            checks = list(self._checks)
        if not checks or len(checks) < max(2, 0.5 * self.window_s / self.every_s):
            return {}
        out = {}
        for code in {c for _, _, found in checks for c in found} - {"covered"}:
            if sum(code in found for _, _, found in checks) >= self.share * len(checks):
                first = next(w for _, w, found in checks if code in found)
                out[code] = first.isoformat(timespec="seconds")
        return out

    def state(self) -> dict:
        with self._lock:
            last = self._checks[-1] if self._checks else None
        lasting = self.lasting()
        return {"checked_at": last[1].isoformat(timespec="seconds") if last else None,
                "findings": last[2] if last else [], "lasting": lasting,
                "advice": {code: ADVICE[code] for code in lasting}}


def faces(people: list[dict], limits: dict = FACES) -> tuple[list[str], dict]:
    """From a kiosk camera's recent people (db.people_between: in_view_s, facing_s, attributes): whether faces are
    seen enough. Returns (findings, numbers); nothing with fewer than limits["min_people"] people."""
    seen = [p for p in people if (p.get("in_view_s") or 0) > 0]
    if len(seen) < limits["min_people"]:
        return [], {"people": len(seen)}
    facing = sum(p.get("facing_s") or 0 for p in seen) / sum(p["in_view_s"] for p in seen)
    aged = sum(((p.get("attributes") or {}).get("age_range") or {}).get("value") not in (None, "unknown")
               for p in seen) / len(seen)
    numbers = {"people": len(seen), "facing_share": round(facing, 2), "ages_share": round(aged, 2)}
    if facing < limits["facing_share"]:
        return ["faces"], numbers
    if aged < limits["ages_share"]:
        return ["small_faces"], numbers
    return [], numbers

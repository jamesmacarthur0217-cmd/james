"""Camera health: notice when a camera sends no picture, a blank picture, or runs slowly.

A problem must last a while before it counts (health.*_after_s), so a short hiccup or a camera reconnecting is
not reported; the first seconds after start are ignored (webcams send blank frames while warming up). Each problem
is stored with when it started and ended (table camera_events), shown on the Live page and the Alerts page.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

PROBLEMS = {
    "offline": "no picture from the camera",
    "blank": "the picture is blank (lens covered, or another app is using the camera)",
    "slow": "running slowly",
}


class HealthMonitor:
    def __init__(self, db, cfg: dict):
        self.db = db
        self.cfg = cfg
        self._since: dict[tuple[str, str], float] = {}   # (camera, problem) -> first seen (monotonic)
        self._open: dict[tuple[str, str], dict] = {}     # (camera, problem) -> stored event
        self._lock = threading.Lock()
        closed = db.close_open_camera_events()            # left open by an earlier run that was stopped
        if closed:
            log.info("closed %d camera problems left open by the previous run", closed)

    def problems_now(self, p, now: float) -> set[str]:
        """What is wrong with this camera right now (before the time thresholds)."""
        if now - p.started_mono < self.cfg["startup_grace_s"]:
            return set()
        if now - p.last_frame_at > self.cfg["offline_after_s"]:
            return {"offline"}                           # blank/slow mean nothing without frames
        out = set()
        if p.blank:
            out.add("blank")
        if p.fps < self.cfg["min_fps"]:
            out.add("slow")
        return out

    def check(self, pipelines, now: float | None = None) -> list[dict]:
        """Open and close problems; returns the problems that just started."""
        now = time.monotonic() if now is None else now
        started = []
        with self._lock:
            for p in pipelines:
                current = self.problems_now(p, now)
                for kind in PROBLEMS:
                    key = (p.name, kind)
                    if kind in current:
                        first = self._since.setdefault(key, now)
                        wait = 0 if kind == "offline" else self.cfg[f"{kind}_after_s"]
                        if key not in self._open and now - first >= wait:
                            message = PROBLEMS[kind] + (f" ({p.fps:.1f} fps)" if kind == "slow" else "")
                            event = self.db.open_camera_event(p.name, kind, message)
                            self._open[key] = event
                            started.append(event)
                            log.warning("CAMERA PROBLEM [%s] %s", p.name, message)
                    else:
                        self._since.pop(key, None)
                        if key in self._open:
                            self.db.close_camera_event(self._open.pop(key)["id"])
                            log.info("[%s] camera problem over: %s", p.name, PROBLEMS[kind])
        return started

    def current(self) -> list[dict]:
        """Problems going on now, for the Live page."""
        with self._lock:
            return sorted(self._open.values(), key=lambda e: e["id"])


def health_loop(monitor: HealthMonitor, pipelines, stop: threading.Event, every_s: float = 2.0) -> None:
    while not stop.wait(every_s):
        try:
            monitor.check(pipelines)
        except Exception:
            log.exception("camera health check failed")

"""Unusual quiet (Phase 39): an alert when nobody is seen at a time of day someone usually is.

The usual hours come from the camera's own history (labeler/routine.py): an hour in which someone was seen on at
least 80% of the counted days of that kind (weekday / weekend). At the end of each usual hour (e.g. 09:00 for the hour
08:00-09:00: "usually up by 08:00, nobody by 09:00") the watch checks whether anyone was seen in it; if not, the
built-in rule "Unusual quiet" fires, once a day per camera, with what is usual in its message. When someone is seen
again the message says when.

An hour counts only if the camera watched all of it: the watch saw the camera working from its start (so nothing is
alerted about hours before the app started, or during a camera problem), and with "armed = home" the house was Home
from its start. While a camera is still learning (fewer than 14 counted days) nothing is usual, so nothing fires.
Checked once a minute (quiet.check_every_s, shared with "No movement").
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, time, timedelta

from .alerts import ARMED
from .routine import MAX_SPAN_H, kind_of

log = logging.getLogger(__name__)


def _value(rule: dict, name: str) -> str | None:
    return next((v for n, v in rule["conditions"] if n == name), None)


class UnusualWatch:
    def __init__(self, db, alerts, pipelines: list, routine, arming=None, health=None, clock=datetime.now):
        self.db, self.alerts, self.pipelines, self.routine = db, alerts, pipelines, routine
        self.arming, self.health, self.clock = arming, health, clock
        self.started: dict[str, datetime] = {}       # camera -> when this watch first saw it working
        self.problem_end: dict[str, datetime] = {}   # camera -> the last time it was not working
        self.fired: dict[tuple, object] = {}         # (rule id, camera) -> the day it fired
        self.open: dict[tuple, dict] = {}            # (rule id, camera) -> {"event", "message", "start"}
        # (rule id, camera, day) -> the usual hours already judged: each once, at its end (the camera keeps only its
        # last sighting, so an hour judged fine must not be judged again later from a newer one)
        self.judged: dict[tuple, set[int]] = {}
        self._lock = threading.Lock()

    def _armed(self) -> tuple[str, datetime | None]:
        from .arming import HOME
        if self.arming is None:
            return HOME, None
        s = self.arming.state()
        return s["mode"], datetime.fromisoformat(s["since"]) if s["since"] else None

    def _problems(self) -> set[str]:
        return {e["camera"] for e in (self.health.current() if self.health is not None else [])}

    @staticmethod
    def applies(rule: dict, p) -> bool:
        return rule["camera"] == p.name if rule["camera"] else getattr(p, "mode", "") == "kiosk"

    def seen_between(self, p, start: datetime, end: datetime) -> bool:
        """Was anyone in view on this camera at some time in [start, end)? (A person in view across `start` counts.)"""
        last = getattr(p, "person_seen_wall", None)
        if last is not None and start <= last < end:
            return True
        return any(b >= start for a, b in self.db.track_spans(p.name, start - timedelta(hours=MAX_SPAN_H), end))

    def _watched(self, p, start: datetime, armed: str | None, mode_since: datetime | None) -> bool:
        if self.started.get(p.name, start + timedelta(seconds=1)) > start:
            return False   # the app (or the camera) was not watching yet when the hour began
        if self.problem_end.get(p.name, start) > start:
            return False   # the camera had a problem during the hour
        return not (armed and mode_since is not None and mode_since > start)   # switched to Home after it began

    def tick(self) -> list[dict]:
        """Fire for usual hours just over with nobody seen; note when someone is seen again. Returns new alerts."""
        now = self.clock()
        today = now.date()
        rules = self.alerts.unusual_rules()
        problems = self._problems()
        mode, mode_since = self._armed()
        made = []
        with self._lock:
            self.judged = {k: v for k, v in self.judged.items() if k[2] == today}   # forget earlier days
            for p in self.pipelines:
                if not getattr(p, "running", False) or p.name in problems:
                    self.problem_end[p.name] = now
                    continue
                self.started.setdefault(p.name, now)
            for rule in rules:
                armed = _value(rule, ARMED)
                for p in self.pipelines:
                    if not self.applies(rule, p):
                        continue
                    key = (rule["id"], p.name)
                    if key in self.open:
                        seen = getattr(p, "person_seen_wall", None)
                        if seen is not None and seen >= self.open[key]["start"]:
                            item = self.open.pop(key)
                            self.db.update_alert_message(item["event"], f"{item['message']}; seen at {seen:%H:%M}")
                    if self.fired.get(key) == today or not getattr(p, "running", False) or p.name in problems:
                        continue
                    if armed and armed != mode:
                        continue
                    judged = self.judged.setdefault((rule["id"], p.name, today), set())
                    for h in self.routine.usual_hours(p.name, today):
                        start = datetime.combine(today, time(h))
                        end = start + timedelta(hours=1)
                        if end > now:
                            break
                        if h in judged:
                            continue
                        judged.add(h)
                        if not self._watched(p, start, armed, mode_since) or self.seen_between(p, start, end):
                            continue
                        message = self._message(rule, p.name, today, h)
                        event = self.db.record_alert(rule["id"], None, p.name, message, repeat=True)
                        self.fired[key] = today   # once a day per camera, even if storing it failed
                        if event is not None:
                            self.open[key] = {"event": event["id"], "message": message, "start": start}
                            log.warning("ALERT [%s] %s", p.name, message)
                            made.append(event)
                        break
        return made

    def _message(self, rule: dict, camera: str, day, h: int) -> str:
        r = self.routine.get(camera)
        k = kind_of(day)
        share, days = r["share"][k][h], r["asked"][k][h]
        return (f"{rule['name']}: nobody seen on {camera} between {h:02d}:00 and {(h + 1) % 24:02d}:00; on "
                f"{'weekdays' if k == 'weekday' else 'weekends'} someone usually is (on {share:.0%} of {days} days)")

    def loop(self, stop: threading.Event, every_s: float = 60.0) -> None:
        while not stop.wait(every_s):
            try:
                self.tick()
            except Exception:
                log.exception("unusual quiet: checking failed")

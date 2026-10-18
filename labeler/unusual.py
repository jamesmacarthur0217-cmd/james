"""Unusual quiet (Phase 39): an alert when nobody has been seen for a stretch of the day in which, in this camera's
own history, someone (nearly) always was.

The quiet stretch is today's time since the last sighting on the camera (or since the camera, the house set to Home,
or the day began, whichever is later). Its whole hours are compared with the history (labeler/routine.py): on how
many of the counted days of that kind (weekday / weekend, Home in those hours) was someone seen in them? When that is
at least routine.quiet_share (97%: on 40 weekdays at most one without), it is unusual and the built-in rule
"Unusual quiet" fires, once a day per camera, with what is usual in its message; when someone is seen again the
message says when. E.g. usually up by 08:00 and nobody seen by 09:00: the stretch 00:00-09:00 had someone on every
day, so it fires at 09:00.

Measured (tools/unusual_check.py): the first version judged each hour usual on 80% of days on its own; a household at
home all day (seen in each hour on 90% of days) then had an alert on 19 of 24 normal days. A whole stretch is
compared instead, and it must be at least routine.min_quiet_h (3) whole hours: with 40 days of history a 2-hour gap
that happens 1 day in 100 still looks like "never". On made-up households not used for tuning (4 x 24 normal days
each): 1 false alert for a regular household, 1 for one that sleeps in, 1 for one at home all day, 6 for an irregular
one; every day with nobody up was found, 1-2 hours after the first usual hour began.

While a camera is still learning (fewer than 14 counted days) nothing fires. Checked once a minute
(quiet.check_every_s, shared with "No movement").
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
    def __init__(self, db, alerts, pipelines: list, routine, arming=None, health=None, clock=datetime.now,
                 share: float = 0.97, min_hours: int = 3):
        self.db, self.alerts, self.pipelines, self.routine = db, alerts, pipelines, routine
        self.arming, self.health, self.clock, self.share, self.min_hours = arming, health, clock, share, min_hours
        self.started: dict[str, datetime] = {}       # camera -> when this watch first saw it working
        self.problem_end: dict[str, datetime] = {}   # camera -> the last time it was not working
        self.fired: dict[tuple, object] = {}         # (rule id, camera) -> the day it fired
        self.open: dict[tuple, dict] = {}            # (rule id, camera) -> {"event", "message", "start"}
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

    def last_seen(self, p, since: datetime, now: datetime) -> datetime | None:
        """The last time anyone was in view on this camera since `since` (in view now counts as now)."""
        times = [b for a, b in self.db.track_spans(p.name, since - timedelta(hours=MAX_SPAN_H), now) if b >= since]
        live = getattr(p, "person_seen_wall", None)
        if live is not None and live >= since:
            times.append(live)
        return max(times) if times else None

    def quiet_since(self, p, now: datetime, armed: str | None, mode_since: datetime | None) -> datetime:
        """Since when this camera has watched today without seeing anyone."""
        day = datetime.combine(now.date(), time())
        starts = [day, self.started.get(p.name, now)]
        if p.name in self.problem_end:
            starts.append(self.problem_end[p.name])
        if armed and mode_since is not None:
            starts.append(mode_since)
        seen = self.last_seen(p, day, now)
        if seen is not None:
            starts.append(seen)
        return max(starts)

    def check_one(self, rule: dict, p, now: datetime, armed, mode_since) -> str | None:
        """The alert's message if the quiet stretch on this camera is unusual now, else None."""
        since = self.quiet_since(p, now, armed, mode_since)
        h1 = since.hour + (since.minute > 0 or since.second > 0)   # whole hours only
        h2 = now.hour
        if h2 - h1 < self.min_hours:
            return None
        got = self.routine.stretch(p.name, now.date(), h1, h2)
        if got is None:
            return None   # still learning
        yes, days = got
        if days < max(1, self.routine.min_days // 2) or yes / days < self.share:
            return None
        k = kind_of(now.date())
        return (f"{rule['name']}: nobody seen on {p.name} since {since:%H:%M}; on "
                f"{'weekdays' if k == 'weekday' else 'weekends'} someone is seen between {h1:02d}:00 and "
                f"{h2:02d}:00 on {yes} of {days} days")

    def tick(self) -> list[dict]:
        """Fire for unusual quiet stretches; note when someone is seen again. Returns the new alerts."""
        now = self.clock()
        today = now.date()
        rules = self.alerts.unusual_rules()
        problems = self._problems()
        mode, mode_since = self._armed()
        made = []
        with self._lock:
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
                        seen = self.last_seen(p, self.open[key]["start"], now)
                        if seen is not None and seen > self.open[key]["start"]:
                            item = self.open.pop(key)
                            self.db.update_alert_message(item["event"], f"{item['message']}; seen at {seen:%H:%M}")
                    if self.fired.get(key) == today or not getattr(p, "running", False) or p.name in problems:
                        continue
                    if armed and armed != mode:
                        continue
                    message = self.check_one(rule, p, now, armed, mode_since)
                    if message is None:
                        continue
                    event = self.db.record_alert(rule["id"], None, p.name, message, repeat=True)
                    self.fired[key] = today   # once a day per camera, even if storing it failed
                    if event is not None:
                        self.open[key] = {"event": event["id"], "message": message, "start": now}
                        log.warning("ALERT [%s] %s", p.name, message)
                        made.append(event)
        return made

    def loop(self, stop: threading.Event, every_s: float = 60.0) -> None:
        while not stop.wait(every_s):
            try:
                self.tick()
            except Exception:
                log.exception("unusual quiet: checking failed")

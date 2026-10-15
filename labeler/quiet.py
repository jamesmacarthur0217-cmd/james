"""No movement (Phase 31): an alert when nobody has been seen on a camera for a long time while someone should be in.

For someone living alone (an elderly parent): a rule "quiet = 240" (minutes) fires when nobody was seen on the camera
for 4 hours. The built-in rule "No movement" (labeler/alerts.py QUIET_RULE; the user's choice) also has "hours =
08-22" and "armed = home": the time counts only while those hold, so it starts again at 08:00 each day and when the
house is switched to Home. A rule without a camera watches the kiosk cameras (a gate camera sees nobody for hours).

The time also counts only while the camera works: from the camera's start and from the end of a camera problem (no
picture, a blank picture: labeler/health.py reports those itself). One alert per quiet stretch; when someone is seen
again the alert's message says when and how long it was quiet. Checked once a minute (quiet.check_every_s).
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta

from .alerts import ARMED, HOURS, QUIET, in_hours, nice_seconds

log = logging.getLogger(__name__)


def _value(rule: dict, name: str) -> str | None:
    return next((v for n, v in rule["conditions"] if n == name), None)


def _duration(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60} h {m % 60} min" if m >= 60 else f"{m} min"


class QuietWatch:
    def __init__(self, db, alerts, pipelines: list, arming=None, health=None, clock=datetime.now):
        self.db, self.alerts, self.pipelines = db, alerts, pipelines
        self.arming, self.health, self.clock = arming, health, clock
        self.started: dict[str, datetime] = {}        # camera -> when this watch first saw it working
        self.problem_end: dict[str, datetime] = {}    # camera -> the last time it had a problem
        self.open: dict[tuple, dict] = {}             # (rule id, camera) -> {"start", "event", "message"}
        self._lock = threading.Lock()

    def _problems(self) -> set[str]:
        return {e["camera"] for e in (self.health.current() if self.health is not None else [])}

    def _armed(self) -> tuple[str, datetime | None]:
        from .arming import HOME
        if self.arming is None:
            return HOME, None
        s = self.arming.state()
        return s["mode"], datetime.fromisoformat(s["since"]) if s["since"] else None

    @staticmethod
    def applies(rule: dict, p) -> bool:
        return rule["camera"] == p.name if rule["camera"] else getattr(p, "mode", "") == "kiosk"

    def quiet_since(self, rule: dict, p, now: datetime, mode_since: datetime | None) -> datetime:
        """Since when the camera has watched, with the rule's hours and Home/Away holding, and seen nobody."""
        starts = [self.started.get(p.name, now)]
        if getattr(p, "person_seen_wall", None):
            starts.append(p.person_seen_wall)
        if p.name in self.problem_end:
            starts.append(self.problem_end[p.name])
        hours = _value(rule, HOURS)
        if hours:
            start = now.replace(hour=int(hours.split("-")[0]) % 24, minute=0, second=0, microsecond=0)
            starts.append(start if start <= now else start - timedelta(days=1))
        if _value(rule, ARMED) and mode_since is not None:
            starts.append(mode_since)
        return max(starts)

    def tick(self) -> list[dict]:
        """Open and close "no movement" alerts; returns the alerts just made."""
        now = self.clock()
        rules = self.alerts.quiet_rules()
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
                for p in self.pipelines:
                    if not self.applies(rule, p):
                        continue
                    key = (rule["id"], p.name)
                    seen = getattr(p, "person_seen_wall", None)
                    if key in self.open and seen and seen > self.open[key]["start"]:
                        self._close(key, seen)
                    if not getattr(p, "running", False) or p.name in problems:
                        continue
                    hours, armed = _value(rule, HOURS), _value(rule, ARMED)
                    if (hours and not in_hours(hours, now.hour)) or (armed and armed != mode):
                        continue
                    start = self.quiet_since(rule, p, now, mode_since)
                    if key in self.open and self.open[key]["start"] == start:
                        continue   # this stretch already alerted
                    minutes = int(_value(rule, QUIET))
                    if (now - start).total_seconds() < minutes * 60:
                        continue
                    message = f"{rule['name']}: nobody seen on {p.name} for {nice_seconds(minutes * 60)} (since {start:%H:%M})"
                    event = self.db.record_alert(rule["id"], None, p.name, message, repeat=True)
                    if event is not None:
                        self.open[key] = {"start": start, "event": event["id"], "message": message}
                        log.warning("ALERT [%s] %s", p.name, message)
                        made.append(event)
        return made

    def _close(self, key: tuple, seen: datetime) -> None:
        item = self.open.pop(key)
        quiet = (seen - item["start"]).total_seconds()
        self.db.update_alert_message(item["event"], f"{item['message']}; seen again at {seen:%H:%M}, after "
                                                    f"{_duration(quiet)}")
        log.info("[%s] someone seen again after %s quiet", key[1], _duration(quiet))

    def loop(self, stop: threading.Event, every_s: float = 60.0) -> None:
        while not stop.wait(every_s):
            try:
                self.tick()
            except Exception:
                log.exception("no movement: checking failed")

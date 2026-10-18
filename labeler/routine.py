"""The usual day (Phase 39): when someone is usually seen on each kiosk camera, learned from its own history.

For each hour of the day, separately for weekdays (Monday-Friday) and the weekend, the share of days on which
someone was seen on the camera in that hour, over the last `routine.weeks` weeks (8). Someone is "seen in an hour"
from the person's first to last sight (at most MAX_SPAN_H hours: a record left open counts at its start).

Only days that say something count:
- a day on which the camera saw nobody at all is left out (the app was off, or nobody lived there that day: either
  way it tells nothing about the usual day), as Today at a glance does;
- an hour in which the house was set to Away is left out of that hour (the "Unusual quiet" rule only works when Home).

Until there are `routine.min_days` (14) such days the camera is "still learning": no hour is usual yet, so the rule
(Phase 39 step 2) cannot fire on too little history.
"""
from __future__ import annotations

import threading
from datetime import date, datetime, time, timedelta

KINDS = ("weekday", "weekend")
MAX_SPAN_H = 6


def kind_of(d: date) -> str:
    return "weekend" if d.weekday() >= 5 else "weekday"


def _at(value) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def away_spans(db, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """When the house was set to Away within [start, end), from the Home / Away history (labeler/arming.py)."""
    changes = sorted(((_at(h["at"]), h["mode"]) for h in db.arming_history(limit=5000)), key=lambda c: c[0])
    spans, since, mode = [], None, "home"
    for at, new in changes:
        if new == mode:
            continue
        if new == "away":
            since = at
        elif since is not None:
            spans.append((since, at))
            since = None
        mode = new
    if mode == "away" and since is not None:
        spans.append((since, end))
    return [(max(a, start), min(b, end)) for a, b in spans if b > start and a < end]


def _away_at(spans, t: datetime) -> bool:
    return any(a <= t < b for a, b in spans)


def learn(db, camera: str, now: datetime, weeks: int = 8, min_days: int = 14) -> dict:
    """The usual day of one camera from the `weeks` weeks before today (today is not finished, so not counted)."""
    today = now.date()
    first = today - timedelta(weeks=weeks)
    start, end = datetime.combine(first, time()), datetime.combine(today, time())
    seen: dict[date, set[int]] = {}   # day -> the hours someone was seen
    for a, b in db.track_spans(camera, start, end):
        b = min(b, a + timedelta(hours=MAX_SPAN_H))
        t = a.replace(minute=0, second=0, microsecond=0)
        while t <= b:
            if t >= start and t < end:
                seen.setdefault(t.date(), set()).add(t.hour)
            t += timedelta(hours=1)
    away = away_spans(db, start, end)
    days = {k: 0 for k in KINDS}
    asked = {k: [0] * 24 for k in KINDS}   # days that count for that hour (Home at its middle)
    yes = {k: [0] * 24 for k in KINDS}     # of which someone was seen
    for d, hours in seen.items():
        k = kind_of(d)
        days[k] += 1
        for h in range(24):
            if _away_at(away, datetime.combine(d, time(h, 30))):
                continue
            asked[k][h] += 1
            yes[k][h] += h in hours
    total = sum(days.values())
    return {"camera": camera, "days": days, "total_days": total, "min_days": min_days,
            "learning": total < min_days, "since": min(seen).isoformat() if seen else None,
            "share": {k: [round(yes[k][h] / asked[k][h], 3) if asked[k][h] else None for h in range(24)]
                      for k in KINDS},
            "asked": asked}


class Routine:
    """The usual day of each kiosk camera, learned again once an hour (or when asked for a new day)."""

    def __init__(self, db, cfg: dict, cameras: list[str], clock=datetime.now):
        c = cfg["routine"]
        self.weeks, self.min_days, self.usual_share = int(c["weeks"]), int(c["min_days"]), float(c["usual_share"])
        self.db, self.cameras, self.clock = db, list(cameras), clock
        self._cache: dict[str, tuple[datetime, dict]] = {}
        self._lock = threading.Lock()

    def get(self, camera: str) -> dict:
        now = self.clock()
        with self._lock:
            hit = self._cache.get(camera)
            if hit and hit[0].date() == now.date() and now - hit[0] < timedelta(hours=1):
                return hit[1]
        got = learn(self.db, camera, now, self.weeks, self.min_days)
        with self._lock:
            self._cache[camera] = (now, got)
        return got

    def usual_hours(self, camera: str, day: date) -> list[int]:
        """The hours of that kind of day in which someone is usually seen (none while still learning). An hour
        needs at least half the counted days of its kind behind it, so a rare Away hour does not decide alone."""
        r = self.get(camera)
        if r["learning"]:
            return []
        k = kind_of(day)
        need = max(1, r["days"][k] // 2)
        return [h for h, s in enumerate(r["share"][k])
                if s is not None and s >= self.usual_share and r["asked"][k][h] >= need]

    def state(self) -> list[dict]:
        """For the Health page: each camera learning (n of 14 days) or its usual hours today."""
        today = self.clock().date()
        out = []
        for cam in self.cameras:
            r = self.get(cam)
            out.append({"camera": cam, "learning": r["learning"], "days": r["total_days"], "min_days": r["min_days"],
                        "usual_hours": self.usual_hours(cam, today), "kind": kind_of(today)})
        return out

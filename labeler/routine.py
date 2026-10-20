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

People are deleted after 30 days (storage.retention_days), so each finished day is also kept as a small summary
(table routine_days: the hours someone was seen and how many people; nothing about who): `store_days`, called at the
start of every retention pass (at start-up and hourly, before old people are deleted). The usual day reads the
summaries and, for days not summed up yet, the people themselves.
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


def hours_by_day(db, camera: str, start: datetime, end: datetime) -> dict[date, tuple[set[int], int]]:
    """From the people still kept: day -> (the hours someone was seen, how many people started that day)."""
    seen: dict[date, tuple[set[int], int]] = {}
    for a, b in db.track_spans(camera, start - timedelta(hours=MAX_SPAN_H), end):
        b = min(b, a + timedelta(hours=MAX_SPAN_H))
        if start <= a < end:
            hours, n = seen.get(a.date(), (set(), 0))
            seen[a.date()] = (hours, n + 1)
        t = a.replace(minute=0, second=0, microsecond=0)
        while t <= b:
            if start <= t < end:
                seen.setdefault(t.date(), (set(), 0))[0].add(t.hour)
            t += timedelta(hours=1)
    return seen


def _mask(hours: set[int]) -> int:
    return sum(1 << h for h in hours)


def _hours(mask: int) -> set[int]:
    return {h for h in range(24) if mask >> h & 1}


def store_days(db, cameras: list[str], now: datetime, weeks: int = 8) -> int:
    """Sum up each finished day (before today) of these cameras that has people and no summary yet; drop summaries
    older than the window and a week more. Returns how many days were stored."""
    today = now.date()
    start, end = datetime.combine(today - timedelta(weeks=weeks), time()), datetime.combine(today, time())
    stored = 0
    for cam in cameras:
        have = db.routine_days(cam, start, end)
        days = {datetime.combine(d, time()): (_mask(hours), n)
                for d, (hours, n) in hours_by_day(db, cam, start, end).items()
                if n > 0 and datetime.combine(d, time()) not in have}
        stored += db.add_routine_days(cam, days)
    db.cleanup_routine_days(start - timedelta(weeks=1))
    return stored


def learn(db, camera: str, now: datetime, weeks: int = 8, min_days: int = 14) -> dict:
    """The usual day of one camera from the `weeks` weeks before today (today is not finished, so not counted):
    the stored day summaries, and the people still kept for days not summed up yet."""
    today = now.date()
    first = today - timedelta(weeks=weeks)
    start, end = datetime.combine(first, time()), datetime.combine(today, time())
    seen: dict[date, set[int]] = {d.date(): _hours(mask) for d, (mask, n) in db.routine_days(camera, start, end).items()
                                  if n > 0}   # day -> the hours someone was seen
    for d, (hours, n) in hours_by_day(db, camera, start, end).items():
        if d not in seen and n > 0:
            seen[d] = hours
    away = away_spans(db, start, end)
    days = {k: 0 for k in KINDS}
    asked = {k: [0] * 24 for k in KINDS}   # days that count for that hour (Home at its middle)
    yes = {k: [0] * 24 for k in KINDS}     # of which someone was seen
    per_day = {}                            # day -> (kind, hours seen, hours Away): for a whole quiet stretch
    for d, hours in seen.items():
        k = kind_of(d)
        days[k] += 1
        away_hours = {h for h in range(24) if _away_at(away, datetime.combine(d, time(h, 30)))}
        per_day[d] = (k, hours, away_hours)
        for h in range(24):
            if h in away_hours:
                continue
            asked[k][h] += 1
            yes[k][h] += h in hours
    total = sum(days.values())
    return {"camera": camera, "days": days, "total_days": total, "min_days": min_days,
            "learning": total < min_days, "since": min(seen).isoformat() if seen else None,
            "share": {k: [round(yes[k][h] / asked[k][h], 3) if asked[k][h] else None for h in range(24)]
                      for k in KINDS},
            "asked": asked, "per_day": per_day}


def hour_ranges(hours: list[int]) -> str:
    """[7, 8, 12, 18, 19] -> "07:00-09:00, 12:00-13:00, 18:00-20:00"."""
    out, run = [], []
    for h in sorted(hours):
        if run and h != run[-1] + 1:
            out.append(run)
            run = []
        run.append(h)
    if run:
        out.append(run)
    return ", ".join(f"{r[0]:02d}:00-{(r[-1] + 1) % 24:02d}:00" for r in out)


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
        return self.usual_for(camera, kind_of(day))

    @staticmethod
    def need(r: dict, kind: str) -> int:
        """Counted days an hour needs behind it to be judged (fewer: drawn hollow on the chart)."""
        return max(1, r["days"][kind] // 2)

    def usual_for(self, camera: str, kind: str) -> list[int]:
        r = self.get(camera)
        if r["learning"]:
            return []
        need = self.need(r, kind)
        return [h for h, s in enumerate(r["share"][kind])
                if s is not None and s >= self.usual_share and r["asked"][kind][h] >= need]

    def chart(self, camera: str) -> dict:
        """For the usual-day chart (Phase 43): per kind of day, the share and the counted days of each hour, the
        days needed behind an hour, and the usual hours. JSON-ready (no per-day detail)."""
        r = self.get(camera)
        return {"camera": camera, "learning": r["learning"], "days": r["days"], "total_days": r["total_days"],
                "min_days": r["min_days"], "since": r["since"], "share": r["share"], "asked": r["asked"],
                "need": {k: self.need(r, k) for k in KINDS},
                "usual_hours": {k: self.usual_for(camera, k) for k in KINDS}}

    def stretch(self, camera: str, day: date, h1: int, h2: int) -> tuple[int, int] | None:
        """For the hours h1 .. h2-1 of that kind of day: (days on which someone was seen in them, days that count:
        Home in all of them). None while still learning."""
        r = self.get(camera)
        if r["learning"]:
            return None
        k = kind_of(day)
        hours = set(range(h1, h2))
        counted = [seen for kind, seen, away in r["per_day"].values() if kind == k and not hours & away]
        return sum(bool(seen & hours) for seen in counted), len(counted)

    def state(self) -> list[dict]:
        """For the Health page: each camera learning (n of 14 days) or its usual hours today."""
        today = self.clock().date()
        out = []
        for cam in self.cameras:
            r = self.get(cam)
            out.append({"camera": cam, "learning": r["learning"], "days": r["total_days"], "min_days": r["min_days"],
                        "usual_hours": self.usual_hours(cam, today), "kind": kind_of(today)})
        return out

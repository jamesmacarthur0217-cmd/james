"""Home / Away: whether the house is empty, so alert rules can say "only when Away" (labeler/alerts.py).

Two ways to change it. A button on the pages (admins and viewers: family members switch it from their phone) and a
weekly schedule (config `arming.schedule`, e.g. "mon-fri 08:00-18:00" = Away on weekdays from 08:00 until 18:00).
A button press holds until the schedule next changes: pressing Home at 07:00 is overruled at 08:00, pressing Away
at 19:00 lasts until the next 08:00 (Away anyway) and then the 18:00 change to Home.

Every change is a row in the `arming` table (who, when, why: "button" or "schedule"); the newest row is the current
state, Home when there is none. A scheduled change is written when it is first noticed, with the time it was due, so
the history is right also when the app was off at that moment.
"""
from __future__ import annotations

import logging
import re
import threading
from datetime import date, datetime, time, timedelta

log = logging.getLogger(__name__)

HOME, AWAY = "home", "away"
MODES = (HOME, AWAY)
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
BUTTON, SCHEDULE = "button", "schedule"
_ENTRY = re.compile(r"^\s*(\S+)\s+(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")
# A weekly schedule repeats every 7 days, so its last change is never further back than this.
_LOOK_DAYS = 8


class Window:
    """Away on these weekdays (0 = Monday) from `start` until `end`, minutes after midnight; an end at or before
    the start runs past midnight into the next day."""

    def __init__(self, days: frozenset[int], start: int, end: int, text: str):
        self.days, self.start, self.end, self.text = days, start, end, text

    def spans(self, day: date) -> tuple[datetime, datetime] | None:
        """(from, until) of the stretch that starts on `day`, if this window is on that weekday."""
        if day.weekday() not in self.days:
            return None
        midnight = datetime.combine(day, time())
        end = self.end if self.end > self.start else self.end + 24 * 60
        return midnight + timedelta(minutes=self.start), midnight + timedelta(minutes=end)


def _days(text: str) -> frozenset[int]:
    if text == "daily":
        return frozenset(range(7))
    out = set()
    for part in text.split(","):
        ends = part.split("-")
        if len(ends) > 2 or any(e not in DAYS for e in ends):
            raise ValueError(f"unknown day {part!r} (use mon, tue, wed, thu, fri, sat, sun, a range like mon-fri, "
                             f"or daily)")
        a, b = DAYS.index(ends[0]), DAYS.index(ends[-1])
        out.update((a + i) % 7 for i in range((b - a) % 7 + 1))   # fri-mon runs over the weekend
    return frozenset(out)


def parse_entry(text: str) -> Window:
    """"mon-fri 08:00-18:00", "sat,sun 09:00-13:00", "daily 23:00-06:00" (into the next morning)."""
    m = _ENTRY.match(str(text).lower())
    if not m:
        raise ValueError(f"{text!r}: write days and times like mon-fri 08:00-18:00")
    h1, m1, h2, m2 = (int(g) for g in m.groups()[1:])
    if h1 > 23 or m1 > 59 or h2 > 24 or m2 > 59 or (h2 == 24 and m2):
        raise ValueError(f"{text!r}: times go from 00:00 to 24:00")
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if start == end:   # 00:00-24:00 is the whole day
        raise ValueError(f"{text!r}: the start and the end are the same time")
    return Window(_days(m.group(1)), start, end, str(text).strip())


def parse_schedule(entries) -> list[Window]:
    if not entries:
        return []
    if isinstance(entries, str) or not isinstance(entries, list):
        raise ValueError("arming.schedule is a list, e.g. [\"mon-fri 08:00-18:00\"]")
    return [parse_entry(e) for e in entries]


def scheduled_mode(windows: list[Window], at: datetime) -> str:
    for w in windows:
        for back in (0, 1):   # a stretch from yesterday evening can still be running
            span = w.spans(at.date() - timedelta(days=back))
            if span and span[0] <= at < span[1]:
                return AWAY
    return HOME


def _edges(windows: list[Window], first: date, days: int) -> list[datetime]:
    out = set()
    for i in range(days):
        for w in windows:
            span = w.spans(first + timedelta(days=i))
            if span:
                out.update(span)
    return sorted(out)


def last_change(windows: list[Window], now: datetime) -> datetime | None:
    """When the schedule last switched (at or before `now`); None if it never switches (empty, or Away all week)."""
    for edge in reversed(_edges(windows, now.date() - timedelta(days=_LOOK_DAYS), _LOOK_DAYS + 1)):
        if edge <= now and scheduled_mode(windows, edge) != scheduled_mode(windows, edge - timedelta(seconds=1)):
            return edge
    return None


def next_change(windows: list[Window], now: datetime) -> tuple[datetime, str] | None:
    """(when, to which mode) the schedule switches next after `now`."""
    for edge in _edges(windows, now.date() - timedelta(days=1), _LOOK_DAYS + 1):
        if edge > now:
            mode = scheduled_mode(windows, edge)
            if mode != scheduled_mode(windows, edge - timedelta(seconds=1)):
                return edge, mode
    return None


class Arming:
    """The current Home / Away state, kept in memory (alert checks ask often) and written to the database."""

    def __init__(self, db, cfg: dict, clock=datetime.now):
        self.db = db
        self.clock = clock
        self.windows = parse_schedule(cfg.get("arming", {}).get("schedule"))
        self._lock = threading.Lock()
        self._last = db.latest_arming()   # {"mode", "at", "by", "why"} or None

    def _follow_schedule(self, now: datetime) -> None:
        """Write the schedule's change if one came due since the last change (button or schedule)."""
        if not self.windows:
            return
        due = last_change(self.windows, now)
        if self._last is not None and (due is None or self._last["at"] >= due):
            return   # nothing new from the schedule since the last change: a button press holds
        mode = scheduled_mode(self.windows, now)
        if mode != (self._last["mode"] if self._last else HOME):
            self._last = self.db.add_arming(mode, due or now, SCHEDULE, SCHEDULE)
            log.info("Home/Away: %s (schedule)", mode)

    def current(self) -> str:
        with self._lock:
            self._follow_schedule(self.clock())
            return self._last["mode"] if self._last else HOME

    def set(self, mode: str, by: str) -> bool:
        """A button press. False if it was already so (nothing written)."""
        if mode not in MODES:
            raise ValueError(f"mode is {' or '.join(MODES)}")
        with self._lock:
            now = self.clock()
            self._follow_schedule(now)
            if mode == (self._last["mode"] if self._last else HOME):
                return False
            self._last = self.db.add_arming(mode, now, by, BUTTON)
            log.info("Home/Away: %s (by %s)", mode, by)
            return True

    def state(self) -> dict:
        """For the pages: the mode, since when and by whom, the schedule and its next change."""
        with self._lock:
            now = self.clock()
            self._follow_schedule(now)
            last = self._last
            nxt = next_change(self.windows, now)
        return {"mode": last["mode"] if last else HOME,
                "since": last["at"].isoformat(timespec="seconds") if last else None,
                "by": last["by"] if last else None, "why": last["why"] if last else None,
                "schedule": [w.text for w in self.windows],
                "next": {"mode": nxt[1], "at": nxt[0].isoformat(timespec="seconds")} if nxt else None}

"""Measure "Unusual quiet" (labeler/unusual.py) on made-up households: false alerts on normal days, and how late it
alerts on a bad day (nobody up all day).

    python tools/unusual_check.py [--learn-weeks 8] [--test-days 28] [--seed 1]

Each household makes its people day by day (time ranges someone is in view). The first `--learn-weeks` weeks are only
history; then for `--test-days` days the real UnusualWatch ticks every 5 minutes from 05:00 to 23:55 on a scratch SQLite
database (made and removed here), with the day's people added at its start (each hour is judged only at its end, so
people later in the day are not seen early). Every 7th test day is a bad day: nobody at all. Always Home.
Prints per household: the usual weekday hours, false alerts (normal days with an alert), and for bad days the time of
the alert after the first usual hour began.
"""
from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labeler.alerts import AlertEngine  # noqa: E402
from labeler.config import DEFAULTS  # noqa: E402
from labeler.db import Database, Track  # noqa: E402
from labeler.routine import Routine, hour_ranges  # noqa: E402
from labeler.unusual import UnusualWatch  # noqa: E402


def at(d: date, h: int, m: int = 0) -> datetime:
    return datetime.combine(d, time(h, m))


def visit(d, rng, h0, m0, h1, m1, jitter=15):
    """One stay from h0:m0 to h1:m1, both moved by up to `jitter` minutes."""
    a = at(d, h0, m0) + timedelta(minutes=rng.randint(-jitter, jitter))
    b = at(d, h1, m1) + timedelta(minutes=rng.randint(-jitter, jitter))
    return [(a, max(b, a + timedelta(minutes=2)))]


def regular(d, rng):
    """Up 07:10-08:40, lunch on weekdays, evenings out 1 day in 20; the weekend starts at 09:15."""
    out = []
    if d.weekday() < 5:
        out += visit(d, rng, 7, 10, 8, 40) + visit(d, rng, 12, 20, 12, 50, 10)
    else:
        out += visit(d, rng, 9, 15, 10, 30)
    if rng.random() > 0.05:
        out += visit(d, rng, 18, 0, 21, 30)
    return out


def sleeps_in(d, rng):
    """Like regular, but on 3 days in 10 up only at 08:45."""
    out = regular(d, rng)
    if d.weekday() < 5 and rng.random() < 0.3:
        out = [s for s in out if s[0].hour >= 9] + visit(d, rng, 8, 45, 9, 40, 10)
    return out


def at_home(d, rng):
    """Home all day (retired): seen in each hour 07-22 on 90% of days, a few minutes each time."""
    out = []
    for h in range(7, 23):
        if rng.random() < 0.9:
            a = at(d, h, rng.randint(0, 50))
            out.append((a, a + timedelta(minutes=rng.randint(2, 9))))
    return out


def irregular(d, rng):
    """1-4 visits at any hour from 07 to 22."""
    out = []
    for _ in range(rng.randint(1, 4)):
        a = at(d, rng.randint(7, 22), rng.randint(0, 59))
        out.append((a, a + timedelta(minutes=rng.randint(5, 50))))
    return out


def shifts(d, rng):
    """Early shift one week (home 15:00-22:00), late shift the next (home 07:00-13:00)."""
    if d.isocalendar()[1] % 2:
        return visit(d, rng, 15, 0, 22, 0, 20)
    return visit(d, rng, 7, 0, 13, 0, 20)


HOUSEHOLDS = {"regular": regular, "sleeps in": sleeps_in, "home all day": at_home, "irregular": irregular,
              "shift work": shifts}


class Cam:
    name, mode, running, person_seen_wall = "cam", "kiosk", True, None


def run(name, make, learn_weeks, test_days, seed, share, min_hours=1):
    rng = random.Random(seed)
    first_test = date(2026, 3, 2)   # a Monday
    tmp = Path(tempfile.mkdtemp(prefix="vl_unusual_"))
    db = Database("sqlite:///" + (tmp / "t.db").as_posix())
    cam_id = db.get_or_create_camera("cam", "x", "kiosk")

    def add(spans):
        with db.Session.begin() as s:
            s.add_all(Track(camera_id=cam_id, tracker_id=0, started_at=a, ended_at=b, status="done") for a, b in spans)

    add([s for n in range(learn_weeks * 7, 0, -1) for s in make(first_test - timedelta(days=n), rng)])
    engine = AlertEngine(db)
    engine.ensure_unusual_rule()
    clock = [at(first_test, 5)]
    routine = Routine(db, copy.deepcopy(DEFAULTS), ["cam"], clock=lambda: clock[0])
    watch = UnusualWatch(db, engine, [Cam()], routine, clock=lambda: clock[0], share=share, min_hours=min_hours)
    usual = hour_ranges(routine.usual_hours("cam", first_test)) or "none"
    false, normal, bad = 0, 0, []
    for n in range(test_days):
        d = first_test + timedelta(days=n)
        is_bad = n % 7 == 3
        if not is_bad:
            add(make(d, rng))
        clock[0] = at(d, 5)
        watch.tick()
        made, when = [], None
        while clock[0] < at(d, 23, 55):
            clock[0] += timedelta(minutes=5)
            new = watch.tick()
            if new and when is None:
                when = clock[0]
            made += new
        if is_bad:
            hours = routine.usual_hours("cam", d)   # how late: from the first usual hour (80%) of the day
            bad.append((d, when - at(d, hours[0]) if when and hours else None))
        else:
            normal += 1
            false += bool(made)
    db.engine.dispose()
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return usual, false, normal, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn-weeks", type=int, default=8)
    ap.add_argument("--test-days", type=int, default=28)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--share", type=float, default=DEFAULTS["routine"]["quiet_share"])
    ap.add_argument("--min-hours", type=int, default=DEFAULTS["routine"]["min_quiet_h"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    print(f"learning {args.learn_weeks} weeks, testing {args.test_days} days (every 7th: nobody all day), seed "
          f"{args.seed}, share {args.share}, at least {args.min_hours} quiet hours")
    for name, make in HOUSEHOLDS.items():
        usual, false, normal, bad = run(name, make, args.learn_weeks, args.test_days, args.seed, args.share, args.min_hours)
        found = [b for b in bad if b[1] is not None]
        lates = ", ".join(f"{b[1].total_seconds() / 3600:.0f} h" for b in found)
        print(f"{name:13s} usual Mon: {usual:45s} false alerts {false}/{normal} normal days; "
              f"bad days found {len(found)}/{len(bad)}" + (f" ({lates} after the first usual hour began)" if found else ""))


if __name__ == "__main__":
    main()

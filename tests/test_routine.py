"""The usual day (Phase 39 step 1): learned from made-up histories: a regular household, an irregular one, days the
app was off, Away hours, still learning."""
import copy
import random
from datetime import date, datetime, time, timedelta

from labeler.config import DEFAULTS
from labeler.db import Track
from labeler.routine import Routine, away_spans, kind_of, learn

TODAY = date(2026, 10, 19)          # a Monday
NOW = datetime.combine(TODAY, time(10, 0))


def _people(db, camera_id, spans):
    """Many people at once (one transaction): spans = [(start, end)]."""
    with db.Session.begin() as s:
        for i, (a, b) in enumerate(spans):
            s.add(Track(camera_id=camera_id, tracker_id=i, started_at=a, ended_at=b, status="done"))


def _regular(days=56, skip=(), seed=1):
    """Up 07:10-08:40 every day, lunch 12:20-12:50 on weekdays only, evening 18:00-21:30; ~1 in 20 days the evening
    is missed (out). The weekend starts later: up at 09:15."""
    rng = random.Random(seed)
    spans = []
    for n in range(1, days + 1):
        d = TODAY - timedelta(days=n)
        if d in skip:
            continue
        at = lambda h, m: datetime.combine(d, time(h, m))
        if kind_of(d) == "weekday":
            spans += [(at(7, 10), at(8, 40)), (at(12, 20), at(12, 50))]
        else:
            spans.append((at(9, 15), at(10, 30)))
        if rng.random() > 0.05:
            spans.append((at(18, 0), at(21, 30)))
    return spans


def test_a_regular_household(db, camera):
    _people(db, camera, _regular())
    r = learn(db, "test-cam", NOW)
    assert r["days"] == {"weekday": 40, "weekend": 16} and not r["learning"] and r["since"] == "2026-08-24"
    wd, we = r["share"]["weekday"], r["share"]["weekend"]
    assert wd[7] == wd[8] == 1.0 and wd[12] == 1.0 and wd[10] == 0.0 and wd[3] == 0.0
    assert we[7] == 0.0 and we[9] == 1.0 and we[12] == 0.0
    assert 0.85 <= wd[19] <= 1.0   # the evening, missed now and then
    cfg = copy.deepcopy(DEFAULTS)
    routine = Routine(db, cfg, ["test-cam"], clock=lambda: NOW)
    assert routine.usual_hours("test-cam", TODAY) == [7, 8, 12, 18, 19, 20, 21]
    assert routine.usual_hours("test-cam", date(2026, 10, 24)) == [9, 10, 18, 19, 20, 21]   # a Saturday
    assert routine.state() == [{"camera": "test-cam", "learning": False, "days": 56, "min_days": 14,
                                "usual_hours": [7, 8, 12, 18, 19, 20, 21], "kind": "weekday"}]


def test_an_irregular_household_has_few_or_no_usual_hours(db, camera):
    rng = random.Random(7)
    spans = []
    for n in range(1, 57):
        d = TODAY - timedelta(days=n)
        for _ in range(rng.randint(1, 4)):   # a few visits at any hour from 07 to 22
            a = datetime.combine(d, time(rng.randint(7, 22), rng.randint(0, 59)))
            spans.append((a, a + timedelta(minutes=rng.randint(5, 50))))
    _people(db, camera, spans)
    routine = Routine(db, copy.deepcopy(DEFAULTS), ["test-cam"], clock=lambda: NOW)
    assert routine.usual_hours("test-cam", TODAY) == [] and routine.usual_hours("test-cam", date(2026, 10, 24)) == []


def test_days_with_nobody_do_not_count_and_still_learning(db, camera):
    off = {TODAY - timedelta(days=n) for n in range(1, 57) if n > 10}   # the app ran only the last 10 days
    _people(db, camera, _regular(skip=off))
    r = learn(db, "test-cam", NOW)
    assert r["total_days"] == 10 and r["learning"]
    assert r["share"]["weekday"][7] == 1.0   # the days that ran still say when someone is up
    routine = Routine(db, copy.deepcopy(DEFAULTS), ["test-cam"], clock=lambda: NOW)
    assert routine.usual_hours("test-cam", TODAY) == []   # too little history: nothing is usual yet
    assert routine.state()[0]["days"] == 10


def test_away_hours_are_left_out(db, camera):
    _people(db, camera, _regular())
    # Away every weekday 08:00-18:00 for the last 8 weeks (at work): those hours say nothing.
    for n in range(1, 57):
        d = TODAY - timedelta(days=n)
        if kind_of(d) == "weekday":
            db.add_arming("away", datetime.combine(d, time(8, 0)), "test", "schedule")
            db.add_arming("home", datetime.combine(d, time(18, 0)), "test", "schedule")
    r = learn(db, "test-cam", NOW)
    assert r["asked"]["weekday"][12] == 0 and r["share"]["weekday"][12] is None
    assert r["asked"]["weekday"][7] == 40 and r["asked"]["weekend"][12] == 16
    spans = away_spans(db, datetime.combine(TODAY - timedelta(days=3), time()), NOW)
    assert spans == [(datetime(2026, 10, 16, 8), datetime(2026, 10, 16, 18))]   # Friday only (Sat/Sun stayed Home)


def test_a_record_left_open_counts_at_its_start_only(db, camera):
    d = TODAY - timedelta(days=1)
    _people(db, camera, [(datetime.combine(d, time(23, 30)), None)])
    assert db.track_spans("test-cam", datetime.combine(d, time()), NOW)[0][1] == datetime.combine(d, time(23, 30))
    r = learn(db, "test-cam", NOW, min_days=1)
    assert r["share"]["weekend"][23] == 1.0 and r["share"]["weekend"][0] == 0.0   # a Sunday

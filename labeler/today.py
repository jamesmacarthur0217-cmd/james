"""Today at a glance (the strip at the top of the Live page): how many people today against a usual day, the
busiest hour, alerts and camera problems today, and the newest backup.

"Usual" is the average of the same weekday in the last 4 weeks, counting only days with at least one person
(a day the app was off says nothing about how busy the place is). Viewers see the people numbers only.
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timedelta

from . import backups

WEEKS = 4


def summary(db, now: datetime | None = None, admin: bool = True) -> dict:
    now = now or datetime.now()
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    starts = db.track_starts(day, now)
    hours = Counter(t.hour for t in starts)
    busiest = max(hours.items(), key=lambda kv: (kv[1], -kv[0])) if hours else None
    weeks = []   # (whole day, by this time of day) for the same weekday in earlier weeks with people
    for w in range(1, WEEKS + 1):
        then = day - timedelta(weeks=w)
        whole = db.count_tracks(then, then + timedelta(days=1))
        if whole:
            # So far today vs. that day by the same time: a morning is not compared with whole days.
            weeks.append((whole, db.count_tracks(then, now - timedelta(weeks=w))))
    out = {"date": day.date().isoformat(), "people": len(starts),
           "usual_by_now": round(sum(b for _, b in weeks) / len(weeks), 1) if weeks else None,
           "usual_day": round(sum(a for a, _ in weeks) / len(weeks), 1) if weeks else None,
           "usual_from_days": len(weeks),
           "busiest_hour": {"hour": busiest[0], "people": busiest[1]} if busiest else None}
    if admin:
        problems = [e for e in db.list_camera_events(200) if e["started_at"] >= day.isoformat()]
        newest = backups.newest()
        out.update(alerts=db.count_alerts(day),
                   camera_problems=len(problems), camera_problems_now=sum(1 for e in problems if not e["ended_at"]),
                   backup_age_h=round((time.time() - newest.stat().st_mtime) / 3600, 1) if newest else None,
                   postgres=db.is_postgres)
    return out

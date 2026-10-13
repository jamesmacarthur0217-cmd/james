"""Summaries for the reports page and CSV export."""
from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import datetime, timedelta

from . import schema

HIDDEN = ("unknown", "none", "no")
# Attributes shown on the reports page, in this order.
REPORT_FIELDS = ["apparent_gender", "age_range", "age_group", "upper_color", "upper_type", "lower_color",
                 "lower_type", "headwear", "eyewear", "carrying", "mobility_aid"]
CSV_FIELDS = sorted({f for fields in schema.MODE_FIELDS.values() for f in fields})


def buckets(start: datetime, end: datetime) -> tuple[str, list[datetime]]:
    """Hourly buckets for up to 3 days, daily beyond that."""
    if end - start <= timedelta(days=3):
        step, first = timedelta(hours=1), start.replace(minute=0, second=0, microsecond=0)
        unit = "hour"
    else:
        step, first = timedelta(days=1), start.replace(hour=0, minute=0, second=0, microsecond=0)
        unit = "day"
    out, t = [], first
    while t < end:
        out.append(t)
        t += step
    return unit, out


GENDERS = ["man", "woman"]
AGE_GROUPS = ["child", "teen", "adult", "senior"]
# Time-in-view buckets: (from seconds, to seconds, label).
DWELL_BINS = [(0, 10, "under 10 s"), (10, 30, "10-30 s"), (30, 60, "30 s - 1 min"), (60, 120, "1-2 min"),
              (120, 300, "2-5 min"), (300, float("inf"), "5 min or more")]


def age_group(attrs: dict) -> str | None:
    """A person's age group: stored directly (security cameras) or from their age range (kiosk cameras)."""
    group = attrs.get("age_group", {}).get("value")
    if group in AGE_GROUPS:
        return group
    return schema.AGE_RANGE_TO_GROUP.get(attrs.get("age_range", {}).get("value"))


def hours_of_day(people: list[dict]) -> list[dict]:
    """Per hour of the day (0-23, over all days in the range): how many people, by gender and age group."""
    rows = [{"hour": h, "people": 0, **{k: 0 for k in GENDERS + AGE_GROUPS}} for h in range(24)]
    for p in people:
        row = rows[datetime.fromisoformat(p["started_at"]).hour]
        row["people"] += 1
        gender = p["attributes"].get("apparent_gender", {}).get("value")
        if gender in GENDERS:
            row[gender] += 1
        group = age_group(p["attributes"])
        if group:
            row[group] += 1
    return rows


def dwell_histogram(people: list[dict]) -> list[dict]:
    """How many people stayed how long (only finished people have a time in view)."""
    counts = Counter()
    for p in people:
        s = p.get("in_view_s")
        if s:
            counts[next(label for lo, hi, label in DWELL_BINS if lo <= s < hi)] += 1
    return [{"label": label, "count": counts[label]} for _, _, label in DWELL_BINS]


def engagement(people: list[dict], looked_s: float = 3.0) -> dict:
    """Time in view and facing the camera, over the people who have finished (only they have times)."""
    timed = [p for p in people if p.get("in_view_s")]
    total = sum(p["in_view_s"] for p in timed)
    return {
        "people": len(timed),
        "avg_in_view_s": round(total / len(timed), 1) if timed else None,
        "facing_share": round(sum(p.get("facing_s") or 0 for p in timed) / total, 3) if total else None,
        "looked": sum((p.get("facing_s") or 0) >= looked_s for p in timed),
        "looked_s": looked_s,
    }


def summarize(people: list[dict], start: datetime, end: datetime, looked_s: float = 3.0) -> dict:
    unit, slots = buckets(start, end)
    counts = Counter()
    for p in people:
        t = datetime.fromisoformat(p["started_at"])
        key = t.replace(minute=0, second=0, microsecond=0) if unit == "hour" else \
            t.replace(hour=0, minute=0, second=0, microsecond=0)
        counts[key] += 1
    attributes: dict[str, list[dict]] = {}
    for name in REPORT_FIELDS:
        c = Counter()
        for p in people:
            value = p["attributes"].get(name, {}).get("value")
            if value is None or value in HIDDEN:
                continue
            if name in schema.LIST_FIELDS:
                c.update(value.split(","))
            else:
                c[value] += 1
        if c:
            attributes[name] = [{"value": v, "count": n} for v, n in c.most_common()]
    return {
        "start": start.isoformat(timespec="minutes"),
        "end": end.isoformat(timespec="minutes"),
        "total": len(people),
        "unit": unit,
        "timeline": [{"t": s.isoformat(timespec="minutes"), "count": counts.get(s, 0)} for s in slots],
        "attributes": attributes,
        "cameras": dict(Counter(p.get("camera") or "?" for p in people)),
        "engagement": engagement(people, looked_s),
        "days": max(1, round((end - start) / timedelta(days=1))),
        "hours_of_day": hours_of_day(people),
        "dwell": dwell_histogram(people),
    }


def to_csv(people: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "camera", "started_at", "ended_at", "status", "in_view_s", "facing_s", *CSV_FIELDS,
                "picture"])
    for p in people:
        attrs = p["attributes"]
        w.writerow([p["id"], p.get("camera") or "", p["started_at"], p["ended_at"] or "", p["status"],
                    "" if p.get("in_view_s") is None else p["in_view_s"],
                    "" if p.get("facing_s") is None else p["facing_s"],
                    *[attrs.get(f, {}).get("value", "") for f in CSV_FIELDS], p.get("crop") or ""])
    return buf.getvalue()

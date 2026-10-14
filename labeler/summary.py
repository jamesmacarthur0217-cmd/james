"""What happened in a period: "while you were away", last night, today, or any start and end (Phase 25).

Read from the existing tables, nothing new is stored: people (how many, per camera, the first few with pictures),
vehicles and animals (per kind, with pictures), alerts of every kind, scene questions whose answer changed ("Is the
gate open?: no -> yes"), camera problems, and all of it on one timeline in time order with a link to the recording.

Periods: "away" = from the last switch to Away until the switch back to Home (or now, still Away), from the Home /
Away history (labeler/arming.py); "night" = 18:00 yesterday until 08:00 today (until now if earlier); "today" =
since midnight. Viewers get the same without clips and camera problems (as on the other pages: Today at a glance
shows them people numbers only); their pictures come blurred from /crops/ as everywhere.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, time, timedelta

PERIODS = ("away", "night", "today")
NIGHT_FROM, NIGHT_UNTIL = 18, 8
PEOPLE_SHOWN = 12
TIMELINE_MAX = 300


def away_period(db, now: datetime) -> tuple[datetime, datetime] | None:
    """(start, end) of the newest Away stretch: its switch to Away until the next switch (to Home), or now."""
    history = db.arming_history(limit=100)   # newest first
    for i, h in enumerate(history):
        if h["mode"] == "away":
            start = datetime.fromisoformat(h["at"])
            end = datetime.fromisoformat(history[i - 1]["at"]) if i > 0 else now
            return start, min(end, now)
    return None


def period(db, name: str, now: datetime) -> tuple[datetime, datetime] | None:
    if name == "away":
        return away_period(db, now)
    today = datetime.combine(now.date(), time())
    if name == "night":
        start = today - timedelta(days=1) + timedelta(hours=NIGHT_FROM)
        return start, min(today + timedelta(hours=NIGHT_UNTIL), now)
    if name == "today":
        return today, now
    raise ValueError(f"period is one of {', '.join(PERIODS)}")


def describe_person(p: dict) -> str:
    """"man, 30-39, red t-shirt" from a person's labels (what is known)."""
    a = p.get("attributes") or {}
    v = lambda n: (a.get(n) or {}).get("value")   # noqa: E731
    known = lambda x: x and x not in ("unknown", "none")   # noqa: E731
    top = " ".join(x for x in (v("upper_color"), v("upper_type")) if known(x))
    return ", ".join(x for x in (v("apparent_gender"), v("age_group"), top) if known(x)) or "a person"


def build(db, start: datetime, end: datetime, admin: bool = True, clip_for=None) -> dict:
    """The summary of [start, end)."""
    people = db.people_between(start, end)
    vehicles = db.list_vehicles(limit=1000, start=start, end=end)
    animals = db.list_animals(limit=1000, start=start, end=end)
    alerts = db.alerts_between(start, end)
    changes = scene_changes(db, start, end)
    problems = db.camera_events_between(start, end) if admin else []
    timeline = []
    for p in people:
        timeline.append({"at": p["started_at"], "camera": p["camera"], "type": "person", "id": p["id"],
                         "text": describe_person(p), "picture": p["crop"]})
    for v in vehicles:
        timeline.append({"at": v["started_at"], "camera": v["camera"], "type": "vehicle", "id": v["id"],
                         "text": " ".join(x for x in (v["color"], v["kind"]) if x), "picture": v["crop"]})
    for a in animals:
        timeline.append({"at": a["started_at"], "camera": a["camera"], "type": "animal", "id": a["id"],
                         "text": " ".join(x for x in (a["color"], a["kind"]) if x), "picture": a["crop"]})
    for e in alerts:
        e["clip"] = clip_for(e["id"]) if (admin and clip_for) else None
        timeline.append({"at": e["created_at"], "camera": e["camera"], "type": "alert", "id": e["id"],
                         "text": e["message"], "picture": e["crop"], "clip": e["clip"]})
    for c in changes:
        timeline.append({"at": c["at"], "camera": c["camera"], "type": "scene", "id": c["question_id"],
                         "text": f"{c['question']}: {c['before'] or '–'} → {c['after']}", "picture": None})
    for pr in problems:
        timeline.append({"at": pr["started_at"], "camera": pr["camera"], "type": "camera", "id": pr["id"],
                         "text": pr["message"], "picture": None})
    timeline.sort(key=lambda x: x["at"])
    return {
        "start": start.isoformat(timespec="seconds"), "end": end.isoformat(timespec="seconds"),
        "counts": {"people": len(people), "vehicles": len(vehicles), "animals": len(animals), "alerts": len(alerts),
                   "scene_changes": len(changes), "camera_problems": len(problems)},
        "people_per_camera": dict(Counter(p["camera"] for p in people)),
        "people": [{"id": p["id"], "camera": p["camera"], "at": p["started_at"], "text": describe_person(p),
                    "picture": p["crop"]} for p in people[:PEOPLE_SHOWN]],
        "vehicle_kinds": dict(Counter(v["kind"] for v in vehicles).most_common()),
        "animal_kinds": dict(Counter(a["kind"] for a in animals).most_common()),
        "alerts": alerts, "scene_changes": changes, "camera_problems": problems,
        "timeline": timeline[-TIMELINE_MAX:], "timeline_cut": max(0, len(timeline) - TIMELINE_MAX),
    }


def scene_changes(db, start: datetime, end: datetime) -> list[dict]:
    """Scene answers in [start, end) that differ from the question's answer before ("unsure" is no answer)."""
    out = []
    for q in db.scene_answer_history(start, end):
        before = q["before"]
        for a in q["answers"]:
            if a["answer"] == "unsure":
                continue
            if a["answer"] != before:
                out.append({"question_id": q["id"], "question": q["question"], "camera": q["camera"],
                            "at": a["at"], "before": before, "after": a["answer"]})
            before = a["answer"]
    return sorted(out, key=lambda c: c["at"])

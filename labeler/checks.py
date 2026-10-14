"""Checks ("was this right?"): an admin says right / wrong / not sure about what the system found on their own cameras.

The answer key (Answer key page) measures people's labels; this measures the rest, on the user's cameras instead of
public videos: vehicles ("a white car?"), animals ("a dog?"), alerts of every kind (falls, loitering, scene questions,
vehicle, animal and person rules: "was this alert right?") and scene questions' latest answers. A wrong vehicle or
animal can get a correction ("it was a cat", "not an animal").

The Check page shows one item at a time: the kind with the fewest checks so far first (so every kind gets some), the
newest unchecked item of that kind, from the last `DAYS` days. Items already checked, or skipped on the page, are left
out. Checks stay when the item is deleted later (they are the measurement).
"""
from __future__ import annotations

from datetime import datetime, timedelta

KINDS = ("vehicle", "animal", "alert", "scene")
RIGHT = ("yes", "no", "unsure")
DAYS = 30
VEHICLE_KINDS = ["car", "SUV", "van", "pickup", "lorry", "bus", "motorbike", "bicycle"]
ANIMAL_KINDS = ["dog", "cat", "bird", "horse", "sheep", "cow"]
CORRECTIONS = {"vehicle": ["not a vehicle", *VEHICLE_KINDS], "animal": ["not an animal", *ANIMAL_KINDS]}


def alert_what(event: dict, conditions: list) -> str:
    """Which kind of alert: fall, loitering, scene, vehicle, animal or person."""
    names = {n for n, _ in conditions}
    if event.get("scene_id"):
        return "scene"
    if event.get("fall") or "fallen" in names:
        return "fall"
    if "stayed" in names:
        return "loitering"
    if event.get("animal_id"):
        return "animal"
    if event.get("vehicle_id"):
        return "vehicle"
    return "person"


def _recent(iso: str | None, now: datetime) -> bool:
    return bool(iso) and datetime.fromisoformat(iso) >= now - timedelta(days=DAYS)


def candidates(db, kind: str, now: datetime | None = None, clip_for=None) -> list[dict]:
    """Unchecked items of one kind, newest first. clip_for(alert id) -> clip file name or None."""
    now = now or datetime.now()
    done = db.checked_ids(kind)
    out = []
    if kind == "vehicle":
        for v in db.list_vehicles(limit=300):
            if v["id"] not in done and v["crop"] and _recent(v["started_at"], now):
                what = " ".join(x for x in (v["color"], v["kind"]) if x)
                out.append({"kind": kind, "id": v["id"], "camera": v["camera"], "at": v["started_at"], "what": v["kind"],
                            "picture": v["crop"], "question": f"Is this a {what}?",
                            "corrections": [c for c in CORRECTIONS[kind] if c != v["kind"]]})
    elif kind == "animal":
        for a in db.list_animals(limit=300):
            if a["id"] not in done and a["crop"] and _recent(a["started_at"], now):
                out.append({"kind": kind, "id": a["id"], "camera": a["camera"], "at": a["started_at"], "what": a["kind"],
                            "picture": a["crop"], "question": f"Is this a {a['kind']}?",
                            "corrections": [c for c in CORRECTIONS[kind] if c != a["kind"]]})
    elif kind == "alert":
        rules = {r["id"]: r["conditions"] for r in db.list_rules()}
        for e in db.list_alerts(limit=300):
            if e["id"] not in done and _recent(e["created_at"], now):
                what = alert_what(e, rules.get(e["rule_id"], []))
                out.append({"kind": kind, "id": e["id"], "camera": e["camera"], "at": e["created_at"], "what": what,
                            "picture": e["crop"], "clip": clip_for(e["id"]) if clip_for else None,
                            "question": f"Was this {what} alert right? “{e['message']}”", "corrections": []})
    elif kind == "scene":
        for a in db.latest_scene_answers():
            if a["answer_id"] not in done and a["picture"] and _recent(a["at"], now):
                out.append({"kind": kind, "id": a["answer_id"], "camera": a["camera"], "at": a["at"], "what": a["answer"],
                            "picture": a["picture"],
                            "question": f"“{a['question']}” was answered “{a['answer']}”. Is that right?",
                            "corrections": []})
    return out


def next_item(db, skip: set[tuple[str, int]] | None = None, now: datetime | None = None, clip_for=None) -> dict | None:
    """The kind with the fewest checks first (among kinds with something to check), its newest unchecked item."""
    skip = skip or set()
    counts = db.check_counts()
    for kind in sorted(KINDS, key=lambda k: (counts.get(k, 0), KINDS.index(k))):
        items = [i for i in candidates(db, kind, now, clip_for) if (i["kind"], i["id"]) not in skip]
        if items:
            return items[0]
    return None

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


# --- results -----------------------------------------------------------------------------------------------------

FEW = 10   # fewer right + wrong answers than this: shown, but marked "few checks yet"
ALERT_LABELS = {"fall": "fall alerts", "loitering": "loitering alerts", "scene": "scene alerts",
                "vehicle": "vehicle alerts", "animal": "animal alerts", "person": "person alerts"}
ORDER = ["vehicles", "animals", "scene answers", *ALERT_LABELS.values()]


def label(check: dict) -> str:
    """What a check measures, in words: "vehicles", "animals", "scene answers", "fall alerts", ..."""
    if check["kind"] == "alert":
        return ALERT_LABELS.get(check["what"], "alerts")
    return {"vehicle": "vehicles", "animal": "animals", "scene": "scene answers"}[check["kind"]]


def results(rows: list[dict]) -> list[dict]:
    """Per measured thing, all cameras first ("camera": None), then each camera: right, wrong, not sure, the share
    right of those answered right or wrong, and the most common mistakes ("dog -> cat": 3)."""
    from collections import Counter, defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in rows:
        groups[(label(c), None)].append(c)
        groups[(label(c), c["camera"])].append(c)
    out = []
    for (what, camera), cs in groups.items():
        right = sum(c["right"] == "yes" for c in cs)
        wrong = sum(c["right"] == "no" for c in cs)
        mistakes = Counter(f"{c['what']} → {c['correction']}" for c in cs if c["right"] == "no" and c["correction"])
        out.append({"what": what, "camera": camera, "right": right, "wrong": wrong,
                    "unsure": sum(c["right"] == "unsure" for c in cs),
                    "share": round(right / (right + wrong), 3) if right + wrong else None, "few": right + wrong < FEW,
                    "mistakes": [{"what": k, "n": n} for k, n in mistakes.most_common(3)]})
    rank = {w: i for i, w in enumerate(ORDER)}
    return sorted(out, key=lambda r: (rank.get(r["what"], 99), r["camera"] is not None, r["camera"] or ""))


def summary_line(rows: list[dict]) -> str | None:
    """One line for the Health page: every thing checked, all cameras together."""
    parts = []
    for r in results(rows):
        if r["camera"] is None and r["right"] + r["wrong"]:
            n = r["right"] + r["wrong"]
            parts.append(f"{r['what']} {r['right']} of {n} right" + (" (few checks yet)" if r["few"]
                                                                     else f" ({r['share']:.0%})"))
    return "; ".join(parts) or None


# --- "often wrong" (step 3): notes and suggestions, never automatic changes ---------------------------------------

OFTEN_MIN = 10      # at least this many answered right or wrong ...
OFTEN_SHARE = 0.7   # ... and fewer than this share right: worth a note


def _often(right: int, wrong: int) -> bool:
    return right + wrong >= OFTEN_MIN and right / (right + wrong) < OFTEN_SHARE


def flags(db) -> dict:
    """{"rules": {rule id: note}, "scene": {question id: note}, "kinds": [suggestion, ...]}: things often wrong."""
    from collections import Counter, defaultdict
    rows = db.list_checks()
    sources = db.alert_sources([c["item_id"] for c in rows if c["kind"] == "alert"])
    questions = db.scene_answer_questions([c["item_id"] for c in rows if c["kind"] == "scene"])
    per_rule, per_question = defaultdict(Counter), defaultdict(Counter)
    for c in rows:
        if c["right"] == "unsure":
            continue
        if c["kind"] == "alert" and c["item_id"] in sources:
            src = sources[c["item_id"]]
            if src["rule_id"] is not None:
                per_rule[src["rule_id"]][c["right"]] += 1
            if src["scene_id"] is not None:
                per_question[src["scene_id"]][c["right"]] += 1
        elif c["kind"] == "scene" and c["item_id"] in questions:
            per_question[questions[c["item_id"]]][c["right"]] += 1

    def note(counts: Counter) -> str:
        return f"often wrong: {counts['yes']} of {counts['yes'] + counts['no']} right (Check page)"

    out = {"rules": {rid: note(n) for rid, n in per_rule.items() if _often(n["yes"], n["no"])},
           "scene": {qid: note(n) for qid, n in per_question.items() if _often(n["yes"], n["no"])}, "kinds": []}
    groups = defaultdict(list)
    for c in rows:
        if c["kind"] in ("vehicle", "animal") and c["right"] != "unsure":
            groups[(c["kind"], c["camera"], c["what"])].append(c)
    for (kind, camera, what), cs in sorted(groups.items()):
        right = sum(c["right"] == "yes" for c in cs)
        wrong = len(cs) - right
        if not _often(right, wrong):
            continue
        fixes = Counter(c["correction"] for c in cs if c["right"] == "no" and c["correction"])
        text = f"“{what}” on {camera}: {right} of {right + wrong} right"
        if fixes:
            top, n = fixes.most_common(1)[0]
            if n * 2 >= wrong:
                text += f", mostly {top}"
                if top.startswith("not a"):
                    text += (f"; if {camera} never sees real {kind}s, turn {kind}s off for it (Settings page, the "
                             f"{'Vehicles' if kind == 'vehicle' else 'Animals'} column)")
        out["kinds"].append({"kind": kind, "camera": camera, "what": what, "text": text})
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

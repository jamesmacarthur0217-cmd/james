"""Natural-language search: "who wore a red top after 3pm on the entrance camera?"

The local model (the same one Ollama runs for labeling) only turns the question into a small, structured
search plan: a time range, a camera, attribute conditions from the fixed vocabulary, and a free-text
appearance description. The plan is validated against the vocabulary and executed by normal code, so a
confused answer can at worst give a poor search, never touch the database in other ways.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta

import numpy as np
import requests

from . import schema
from .db import Database

log = logging.getLogger(__name__)

EARLIEST = datetime(2000, 1, 1)
ALL_VALUES = {**schema.FIELDS, **schema.LIST_FIELDS}

PROMPT = """You turn a question about people seen by security or kiosk cameras into a search plan.
Current date and time: {now} ({weekday}).
Cameras: {cameras}.
Attributes and their allowed values:
{vocab}

Rules:
- Use "conditions" only with the attributes and values listed above (for example upper_color = red).
- Describe anything else about appearance (patterns, logos, objects, styles, colors of things not listed)
  in "appearance", in a few words like "man in a striped scarf". Leave it empty if the conditions say it all.
- "start" and "end" are local times as YYYY-MM-DDTHH:MM. "after 3pm" means today 15:00 to now unless another
  day is named. "yesterday" means yesterday 00:00 to 23:59. Leave both empty when no time is mentioned.
- "camera" is one of the camera names, or empty for all cameras.

Question: {question}"""


def plan_schema(cameras: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "start": {"type": "string"},
            "end": {"type": "string"},
            "camera": {"type": "string", "enum": [""] + cameras},
            "conditions": {"type": "array", "items": {
                "type": "object",
                "properties": {"attribute": {"type": "string", "enum": sorted(ALL_VALUES)},
                               "value": {"type": "string"}},
                "required": ["attribute", "value"]}},
            "appearance": {"type": "string"},
        },
        "required": ["start", "end", "camera", "conditions", "appearance"],
    }


# Words that count as mentioning a value. A condition is only kept if the question mentions it: small models
# otherwise add filters nobody asked for (e.g. "headwear = none"), which would hide the right people.
SYNONYMS = {
    "man": {"man", "men", "male", "males", "guy", "guys", "boy", "boys", "gentleman", "gentlemen"},
    "woman": {"woman", "women", "female", "females", "lady", "ladies", "girl", "girls"},
    "child": {"child", "children", "kid", "kids", "toddler", "toddlers", "baby", "babies"},
    "teen": {"teen", "teens", "teenager", "teenagers"},
    "senior": {"senior", "seniors", "elderly", "old"},
    "adult": {"adult", "adults"},
    "glasses": {"glasses", "spectacles", "eyeglasses"},
    "sunglasses": {"sunglasses", "shades"},
    "cap": {"cap", "caps", "baseball"},
    "hat": {"hat", "hats"},
    "helmet": {"helmet", "helmets"},
    "headscarf": {"headscarf", "headscarves", "hijab"},
    "hood": {"hood", "hooded"},
    "yes": {"mask", "masks", "masked"},
    "trousers": {"trousers", "pants", "slacks"},
    "jeans": {"jeans", "denim"},
    "t-shirt": {"t-shirt", "tshirt", "tee", "tees"},
    "jacket": {"jacket", "jackets", "coat", "coats", "blazer"},
    "hoodie": {"hoodie", "hoodies"},
    "sweater": {"sweater", "sweaters", "jumper", "pullover", "cardigan"},
    "backpack": {"backpack", "backpacks", "rucksack"},
    "handbag": {"handbag", "handbags", "purse"},
    "shopping bag": {"shopping"},
    "luggage": {"luggage", "suitcase", "suitcases", "trolley"},
    "phone": {"phone", "phones", "mobile", "smartphone"},
    "gray": {"gray", "grey"},
    "purple": {"purple", "violet"},
    "beige": {"beige", "tan", "khaki", "cream"},
    "multicolor": {"multicolor", "multicolored", "multicolour", "colorful", "colourful"},
}
AGE_BINS = [(12, "0-12"), (17, "13-17"), (24, "18-24"), (34, "25-34"), (44, "35-44"), (54, "45-54"), (64, "55-64")]


def _words(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()))
    return words | {w for word in words for w in word.split("-")}


def _mentioned(name: str, value: str, words: set[str]) -> bool:
    if value in ("none", "no", "unknown"):
        return False
    if name == "age_range":
        return False   # handled from numbers in the question, see _age_from_question
    options = SYNONYMS.get(value, {value})
    return any(o in words or all(part in words for part in o.split()) for o in options)


def _age_from_question(question: str) -> str | None:
    """'30 years old', 'aged 30', '30-year-old' -> '25-34'."""
    m = re.search(r"\b(\d{1,2})\s*(?:-\s*)?(?:years?|yrs?|y/?o)\b|\baged?\s+(\d{1,2})\b", question.lower())
    if not m:
        return None
    age = int(m.group(1) or m.group(2))
    return next((label for top, label in AGE_BINS if age <= top), "65+")


def _parse_time(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text.strip()) if text and text.strip() else None
    except ValueError:
        return None


def clean_plan(raw: dict, cameras: list[str], now: datetime, question: str = "") -> dict:
    """Keep only conditions the vocabulary allows AND the question mentions; other descriptive words go
    to the appearance search. The camera must be named in the question."""
    words = _words(question)
    conditions, extra = [], []
    for c in raw.get("conditions") or []:
        name, value = str(c.get("attribute", "")), str(c.get("value", "")).strip().lower()
        valid = name in ALL_VALUES and value in ALL_VALUES[name]
        if valid and (not question or _mentioned(name, value, words)):
            if [name, value] not in conditions:
                conditions.append([name, value])
        elif value and not valid:
            extra.append(value)
    age = _age_from_question(question)
    if age:
        conditions = [c for c in conditions if c[0] != "age_range"] + [["age_range", age]]
    appearance = str(raw.get("appearance") or "").strip()
    for word in extra:
        if word not in appearance.lower():
            appearance = f"{appearance} {word}".strip()
    start, end = _parse_time(raw.get("start", "")), _parse_time(raw.get("end", ""))
    if start and not end:
        end = now
    if end and not start:
        start = EARLIEST
    if start and end and end < start:
        start, end = end, start
    camera = raw.get("camera") if raw.get("camera") in cameras else ""
    if camera and question and camera.lower() not in question.lower():
        camera = ""   # the model picked a camera nobody asked for
    return {"start": start, "end": end, "camera": camera, "conditions": conditions, "appearance": appearance}


def explain(plan: dict) -> str:
    parts = []
    if plan["start"] and plan["start"] > EARLIEST:
        parts.append(f"from {plan['start']:%d %b %H:%M} to {plan['end']:%d %b %H:%M}")
    elif plan["end"]:
        parts.append(f"until {plan['end']:%d %b %H:%M}")
    parts.append(f"camera {plan['camera']}" if plan["camera"] else "all cameras")
    parts += [f"{n.replace('_', ' ')} = {v}" for n, v in plan["conditions"]]
    if plan["appearance"]:
        parts.append(f"looks like “{plan['appearance']}”")
    return "Searched: " + "; ".join(parts)


class QueryAgent:
    def __init__(self, cfg: dict, db: Database, embedder=None, cameras: list[str] | None = None):
        self.host = cfg["vlm"]["host"].rstrip("/")
        self.model = cfg["vlm"]["model"]
        self.db = db
        self.embedder = embedder
        self.cameras = cameras or []

    def _ask_llm(self, question: str, now: datetime) -> dict:
        vocab = "\n".join(f"- {n}: {', '.join(v for v in vals if v != 'unknown')}" for n, vals in ALL_VALUES.items())
        prompt = PROMPT.format(now=f"{now:%Y-%m-%d %H:%M}", weekday=f"{now:%A}",
                               cameras=", ".join(self.cameras) or "none", vocab=vocab, question=question)
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "format": plan_schema(self.cameras), "stream": False, "think": False,
                "options": {"temperature": 0, "num_predict": 300}, "keep_alive": "30m"}
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=60)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])

    def plan(self, question: str, now: datetime | None = None) -> dict:
        now = now or datetime.now()
        try:
            raw = self._ask_llm(question, now)
        except Exception as e:  # model unavailable or odd answer: fall back to an appearance search
            log.warning("could not interpret the question (%s); searching by appearance only", e)
            raw = {"appearance": question}
        return clean_plan(raw, self.cameras, now, question)

    def run(self, question: str, limit: int = 30, now: datetime | None = None) -> dict:
        now = now or datetime.now()
        plan = self.plan(question, now)
        start, end = plan["start"] or EARLIEST, plan["end"] or now + timedelta(minutes=1)
        candidates = self.db.people_between(start, end, plan["camera"] or None, limit=5000, newest_first=True)
        people = [p for p in candidates if _matches(plan["conditions"], p["attributes"])]
        total = len(people)          # people whose labels match; look-alikes are extra
        explanation = explain(plan)
        if plan["appearance"] and self.embedder is not None and candidates:
            query = self.embedder.text(plan["appearance"])
            vectors = self.db.vectors_for([p["id"] for p in candidates], self.embedder.model_name)
            ranked = sorted(((float(vectors[p["id"]] @ query), p) for p in candidates if p["id"] in vectors),
                            key=lambda sp: sp[0], reverse=True)
            for sim, p in ranked:
                p["similarity"] = round(sim, 4)
            matched = {p["id"] for p in people}
            people = [p for _, p in ranked if p["id"] in matched]
            if len(people) < limit and plan["conditions"]:
                # Labels can be wrong (a beige coat labeled "orange t-shirt"), so when few people match the
                # conditions, also show the closest look-alikes whose labels differ, clearly marked.
                extra = [p for _, p in ranked if p["id"] not in matched][:min(10, limit - len(people))]
                for p in extra:
                    p["labels_differ"] = True
                if extra:
                    explanation += f" (plus {len(extra)} that look similar but are labeled differently)"
                people += extra
        return {"question": question, "explanation": explanation,
                "plan": {**plan, "start": plan["start"] and plan["start"].isoformat(timespec="minutes"),
                         "end": plan["end"] and plan["end"].isoformat(timespec="minutes")},
                "total": total, "people": people[:limit]}


def _matches(conditions: list[list[str]], attrs: dict) -> bool:
    for name, value in conditions:
        current = attrs.get(name, {}).get("value")
        if name in schema.LIST_FIELDS:
            if value not in (current or "").split(","):
                return False
        elif current != value:
            return False
    return True

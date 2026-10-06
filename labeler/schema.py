"""The fixed attribute vocabulary. The VLM must answer with these values only,
so the database stays consistent and searchable."""
from __future__ import annotations

COLORS = [
    "black", "white", "gray", "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "beige", "multicolor", "unknown",
]

FIELDS: dict[str, list[str]] = {
    "age_range": ["0-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "unknown"],
    "age_group": ["child", "teen", "adult", "senior", "unknown"],
    "apparent_gender": ["man", "woman", "unknown"],
    "upper_type": ["t-shirt", "shirt", "jacket", "hoodie", "sweater", "dress", "uniform", "other", "unknown"],
    "upper_color": COLORS,
    "lower_type": ["trousers", "jeans", "shorts", "skirt", "dress", "other", "unknown"],
    "lower_color": COLORS,
    "headwear": ["none", "cap", "hat", "helmet", "headscarf", "hood", "other", "unknown"],
    "eyewear": ["none", "glasses", "sunglasses", "unknown"],
    "face_mask": ["yes", "no", "unknown"],
}

# Multi-valued fields: stored as a sorted comma-separated string ("none" when empty).
LIST_FIELDS: dict[str, list[str]] = {
    "carrying": ["backpack", "handbag", "shopping bag", "luggage", "phone", "umbrella"],
}

MODE_FIELDS: dict[str, list[str]] = {
    "kiosk": [
        "age_range", "apparent_gender", "upper_type", "upper_color",
        "lower_type", "lower_color", "headwear", "eyewear", "face_mask", "carrying",
    ],
    "security": [
        "age_group", "apparent_gender", "upper_type", "upper_color",
        "lower_type", "lower_color", "headwear", "carrying",
    ],
}

# Order of attributes in the on-screen label.
DISPLAY_ORDER = ["age_range", "age_group", "apparent_gender", "upper_color", "upper_type", "carrying"]


def _key(item: str) -> str:
    return item.replace(" ", "_")


def json_schema(mode: str) -> dict:
    props: dict[str, dict] = {}
    for name in MODE_FIELDS[mode]:
        if name in LIST_FIELDS:
            # One yes/no per item: small models invent items when asked to fill a list.
            items = {_key(item): {"type": "string", "enum": ["yes", "no"]} for item in LIST_FIELDS[name]}
            props[name] = {"type": "object", "properties": items, "required": list(items)}
        else:
            props[name] = {"type": "string", "enum": FIELDS[name]}
    return {"type": "object", "properties": props, "required": list(props)}


def clean(raw: dict, mode: str) -> dict[str, str]:
    """Coerce a model answer onto the vocabulary; anything off-vocabulary becomes 'unknown'."""
    out: dict[str, str] = {}
    for name in MODE_FIELDS[mode]:
        value = raw.get(name)
        if name in LIST_FIELDS:
            answers = value if isinstance(value, dict) else {}
            items = sorted(item for item in LIST_FIELDS[name]
                           if str(answers.get(_key(item), "no")).strip().lower() == "yes")
            out[name] = ",".join(items) if items else "none"
        else:
            value = str(value).strip().lower() if value is not None else "unknown"
            out[name] = value if value in FIELDS[name] else "unknown"
    return out


def allowed_values(name: str) -> list[str] | None:
    """Allowed values for review dropdowns; None for free-form list fields."""
    if name in LIST_FIELDS:
        return None
    return FIELDS.get(name)

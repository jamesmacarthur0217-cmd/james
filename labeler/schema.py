"""The fixed attribute vocabulary. The VLM must answer with these values only,
so the database stays consistent and searchable."""
from __future__ import annotations

COLORS = [
    "black", "white", "gray", "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "beige", "multicolor", "unknown",
]

FIELDS: dict[str, list[str]] = {
    # Decades, like people say ("about 30" = 30-39) and like the age model's training data.
    "age_range": ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+", "unknown"],
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

# Kiosk cameras store an age range, security cameras an age group; this converts one to the other.
AGE_RANGE_TO_GROUP = {"0-9": "child", "10-19": "teen", "20-29": "adult", "30-39": "adult", "40-49": "adult",
                      "50-59": "adult", "60-69": "senior", "70+": "senior"}

# Multi-valued fields: stored as a sorted comma-separated string ("none" when empty).
LIST_FIELDS: dict[str, list[str]] = {
    "carrying": ["backpack", "handbag", "shopping bag", "luggage", "phone", "umbrella"],
    # Visible mobility aids only: a camera cannot see a disability, only an aid being used.
    "mobility_aid": ["wheelchair", "walking stick", "crutches", "walker"],
}

# Fields asked as ONE choice that includes look-alikes ("decoys"), stored as "none" when one is picked. Measured
# 2026-10-13 on 200 COCO people: asked yes/no per aid, the model called ski and hiking poles a walking stick
# (8 of 8 "aids" were wrong, 5 still after the prompt said they are not); a separate answer for them absorbs that.
DECOYS: dict[str, list[str]] = {
    "mobility_aid": ["ski or hiking poles", "other sports equipment", "an ordinary chair or bench",
                     "a trolley, pram or luggage", "pushing or standing next to a wheelchair"],
}

# Fields a setting can turn off (field -> option in the `labels:` config section). Off = not asked, not stored.
# Mobility aids are health-related data, so the user can choose not to keep them.
OPTIONAL_FIELDS = {"mobility_aid": "mobility_aids"}

MODE_FIELDS: dict[str, list[str]] = {
    "kiosk": [
        "age_range", "apparent_gender", "upper_type", "upper_color",
        "lower_type", "lower_color", "headwear", "eyewear", "face_mask", "carrying", "mobility_aid",
    ],
    "security": [
        "age_group", "apparent_gender", "upper_type", "upper_color",
        "lower_type", "lower_color", "headwear", "carrying", "mobility_aid",
    ],
}

# Order of attributes in the on-screen label.
DISPLAY_ORDER = ["age_range", "age_group", "apparent_gender", "upper_color", "upper_type", "carrying", "mobility_aid"]


def _key(item: str) -> str:
    return item.replace(" ", "_")


def mode_fields(mode: str, labels: dict | None = None) -> list[str]:
    """The fields labeled in this mode, without the ones turned off in the `labels:` config section."""
    labels = labels or {}
    return [n for n in MODE_FIELDS[mode] if n not in OPTIONAL_FIELDS or labels.get(OPTIONAL_FIELDS[n], True)]


def json_schema(mode: str, labels: dict | None = None) -> dict:
    props: dict[str, dict] = {}
    for name in mode_fields(mode, labels):
        if name in DECOYS:
            props[name] = {"type": "string", "enum": ["none", *LIST_FIELDS[name], *DECOYS[name], "unknown"]}
        elif name in LIST_FIELDS:
            # One yes/no per item: small models invent items when asked to fill a list.
            items = {_key(item): {"type": "string", "enum": ["yes", "no"]} for item in LIST_FIELDS[name]}
            props[name] = {"type": "object", "properties": items, "required": list(items)}
        else:
            props[name] = {"type": "string", "enum": FIELDS[name]}
    return {"type": "object", "properties": props, "required": list(props)}


def clean(raw: dict, mode: str, labels: dict | None = None) -> dict[str, str]:
    """Coerce a model answer onto the vocabulary; anything off-vocabulary becomes 'unknown'."""
    out: dict[str, str] = {}
    for name in mode_fields(mode, labels):
        value = raw.get(name)
        if name in DECOYS:
            value = str(value).strip().lower() if value is not None else "unknown"
            out[name] = value if value in (*LIST_FIELDS[name], "unknown") else "none"   # a decoy = no aid
        elif name in LIST_FIELDS:
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

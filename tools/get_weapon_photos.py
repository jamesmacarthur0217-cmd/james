"""Download Open Images photos with a handgun, rifle or shotgun, to measure the dangerous-object check (Phase 47), or
with mice (the animal) for the pest check (Phase 47 step 3; Open Images has no rat or cockroach class).

    python tools/get_weapon_photos.py                 # into data/datasets/openimages_weapons
    python tools/get_weapon_photos.py --kind pests    # into data/datasets/openimages_pests (with the user's OK)

Open Images V7 (Google): the photos are CC BY 2.0 (each photo's author), the boxes CC BY 4.0. Used for measuring
only. The validation and test box lists (25 + 77 MB) are read as they download and not kept; only the matching photos
are saved (drawings and toys left out), with their boxes in boxes.json. Downloaded with the user's OK (2026-10-21;
the mice too). Stops before saving photos past --max-mb.
Then: python tools/weapon_check.py photos --folder data/datasets/openimages_weapons --vlm
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUTS = {"weapons": ROOT / "data" / "datasets" / "openimages_weapons",
        "pests": ROOT / "data" / "datasets" / "openimages_pests"}
BOXES = "https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv"
IMAGE = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"   # the bucket the official downloader uses
KINDS = {"weapons": {"/m/0gxl3": "handgun", "/m/06c54": "rifle", "/m/06nrc": "shotgun"},
         "pests": {"/m/04rmv": "mouse"}}   # the animal; "Computer mouse" is /m/020lf
CLASSES = KINDS["weapons"]


def boxes(split: str) -> dict[str, list[dict]]:
    """image id -> weapon boxes (fractions of the picture), read line by line from the streamed list."""
    found: dict[str, list[dict]] = defaultdict(list)
    with requests.get(BOXES.format(split=split), stream=True, timeout=60) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for row in csv.DictReader(r.iter_lines(decode_unicode=True)):
            kind = CLASSES.get(row["LabelName"])
            if kind:
                found[row["ImageID"]].append({"kind": kind, "depiction": row["IsDepiction"] == "1",
                                              "box": [float(row[k]) for k in ("XMin", "YMin", "XMax", "YMax")]})
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=list(KINDS), default="weapons")
    ap.add_argument("--out", help="default: data/datasets/openimages_<kind>")
    ap.add_argument("--max-mb", type=float, default=150, help="stop before the photos pass this size")
    args = ap.parse_args()
    global CLASSES
    CLASSES = KINDS[args.kind]
    out = Path(args.out or OUTS[args.kind])
    out.mkdir(parents=True, exist_ok=True)
    index, kinds = {}, Counter()
    for split in ("validation", "test"):
        found = boxes(split)
        print(f"{split}: {len(found)} photos with a {' or '.join(CLASSES.values())}", flush=True)
        for image_id, items in found.items():
            if all(i["depiction"] for i in items):
                continue   # drawings, toys, posters
            name = f"{split}_{image_id}.jpg"
            path = out / name
            if not path.exists():
                if sum(q.stat().st_size for q in out.glob("*.jpg")) / 1e6 > args.max_mb:
                    print(f"stopped: over {args.max_mb:.0f} MB (ask before downloading more)")
                    break
                r = requests.get(IMAGE.format(split=split, id=image_id), timeout=60)
                if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
                    print(f"  {name}: HTTP {r.status_code}, skipped")
                    continue
                path.write_bytes(r.content)
            index[name] = items
            kinds.update({i["kind"] for i in items})
    (out / "boxes.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.glob("*.jpg")) / 1e6
    print(f"{len(index)} photos ({dict(kinds)}), {size:.0f} MB in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

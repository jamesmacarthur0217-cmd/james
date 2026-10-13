"""Download Open Images photos with a wheelchair or a crutch, to measure the mobility-aid label.

    python tools/get_aid_photos.py                 # into data/datasets/openimages_aids

Open Images V7 (Google): the photos are CC BY 2.0 (each photo's author), the boxes CC BY 4.0. Used for measuring
only. The validation and test box lists (25 + 77 MB) are read as they download and not kept; only the matching
photos are saved, with their boxes in boxes.json. Downloaded with the user's OK (2026-10-13).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "datasets" / "openimages_aids"
BOXES = "https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv"
IMAGE = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"   # the bucket the official downloader uses
CLASSES = {"/m/0qmmr": "wheelchair", "/m/05441v": "crutches"}


def boxes(split: str) -> dict[str, list[dict]]:
    """image id -> aid boxes (fractions of the picture), read line by line from the streamed list."""
    found: dict[str, list[dict]] = defaultdict(list)
    with requests.get(BOXES.format(split=split), stream=True, timeout=60) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for row in csv.DictReader(r.iter_lines(decode_unicode=True)):
            aid = CLASSES.get(row["LabelName"])
            if aid:
                found[row["ImageID"]].append({"aid": aid, "box": [float(row[k]) for k in ("XMin", "YMin", "XMax", "YMax")],
                                              "depiction": row["IsDepiction"] == "1"})
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for split in ("validation", "test"):
        found = boxes(split)
        print(f"{split}: {len(found)} photos with a wheelchair or crutch", flush=True)
        for image_id, items in found.items():
            if all(i["depiction"] for i in items):
                continue   # drawings, signs and toys: not a person using an aid
            name = f"{split}_{image_id}.jpg"
            path = out / name
            if not path.exists():
                r = requests.get(IMAGE.format(split=split, id=image_id), timeout=60)
                if r.status_code != 200:
                    print(f"  {name}: HTTP {r.status_code}, skipped")
                    continue
                path.write_bytes(r.content)
            index[name] = items
    (out / "boxes.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.glob("*.jpg")) / 1e6
    print(f"{len(index)} photos, {size:.0f} MB in {out}")


if __name__ == "__main__":
    sys.exit(main())

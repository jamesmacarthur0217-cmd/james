"""Open Images training photos for our own gun detector (Phase 47 step 1; the user's choice after YOLOE found only
23-36% of guns).

    python tools/get_gun_training.py [--negatives 1000] [--max-gb 1.5]     # into data/datasets/oi_guns

Open Images V6 (Google): photos CC BY 2.0 (each photo's author), boxes CC BY 4.0. The training box list
(oidv6-train-annotations-bbox.csv, 2.26 GB) is read as it downloads and not kept. Saved: every training photo with a
handgun, rifle or shotgun box (drawings and toys left out), plus --negatives photos of people with no gun (a fixed
random sample) so the model also learns what is not a gun, each with a YOLO label file (0 = handgun, 1 = rifle,
2 = shotgun; empty for the photos without). The validation and test photos (tools/get_weapon_photos.py) are not
touched: they stay the test set. Stops when the photos pass --max-gb. Downloaded with the user's OK (2026-10-21:
about 3.5 GB of traffic, about 1.3 GB kept).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "datasets" / "oi_guns"
BOXES = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
IMAGE = "https://open-images-dataset.s3.amazonaws.com/train/{id}.jpg"
CLASSES = {"/m/0gxl3": 0, "/m/06c54": 1, "/m/06nrc": 2}   # handgun, rifle, shotgun
PERSON = "/m/01g317"
NAMES = {0: "handgun", 1: "rifle", 2: "shotgun"}


def read_list(pool_size: int, seed: int = 47) -> tuple[dict, list]:
    """(image id -> [(class, x1, y1, x2, y2)] for the guns, a random pool of image ids with a person), streamed."""
    guns: dict[str, list] = defaultdict(list)
    pool, seen_people = [], 0
    rng = random.Random(seed)
    last_person = None
    with requests.get(BOXES, stream=True, timeout=60) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for n, row in enumerate(csv.DictReader(r.iter_lines(decode_unicode=True)), 1):
            cls = CLASSES.get(row["LabelName"])
            if cls is not None and row["IsDepiction"] != "1":
                guns[row["ImageID"]].append((cls, *(float(row[k]) for k in ("XMin", "YMin", "XMax", "YMax"))))
            elif row["LabelName"] == PERSON and row["ImageID"] != last_person:
                last_person = row["ImageID"]
                seen_people += 1   # reservoir sample over the images with a person
                if len(pool) < pool_size:
                    pool.append(row["ImageID"])
                elif (j := rng.randrange(seen_people)) < pool_size:
                    pool[j] = row["ImageID"]
            if n % 2_000_000 == 0:
                print(f"  {n:,} boxes read; {len(guns)} photos with a gun so far", flush=True)
    return guns, pool


def save(image_id: str, boxes: list) -> int:
    path = OUT / "images" / f"{image_id}.jpg"
    if not path.exists():
        r = requests.get(IMAGE.format(id=image_id), timeout=60)
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
            return 0
        path.write_bytes(r.content)
    lines = [f"{c} {(a + b) / 2:.6f} {(y1 + y2) / 2:.6f} {b - a:.6f} {y2 - y1:.6f}" for c, a, y1, b, y2 in boxes]
    (OUT / "labels" / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--negatives", type=int, default=1000)
    ap.add_argument("--max-gb", type=float, default=1.5)
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    guns, pool = read_list(args.negatives * 3)
    negatives = [i for i in pool if i not in guns][: args.negatives]
    counts = defaultdict(int)
    for boxes in guns.values():
        for c in {b[0] for b in boxes}:
            counts[NAMES[c]] += 1
    print(f"{len(guns)} training photos with a gun ({dict(counts)}), {len(negatives)} without", flush=True)
    jobs = [(i, b) for i, b in guns.items()] + [(i, []) for i in negatives]
    total = done = 0
    with ThreadPoolExecutor(12) as ex:
        for size in ex.map(lambda j: save(*j), jobs):
            total += size
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} photos, {total / 1e9:.2f} GB", flush=True)
            if total > args.max_gb * 1e9:
                print(f"stopped at {args.max_gb} GB ({done} photos): ask before going on", flush=True)
                ex.shutdown(cancel_futures=True)
                return 2
    print(f"{done} photos, {total / 1e9:.2f} GB in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

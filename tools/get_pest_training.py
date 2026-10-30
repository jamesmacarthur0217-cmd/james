"""Open Images training photos for our own pest (rodent) detector (Phase 55; YOLOE found only 3% of mice in Phase 47).

    python tools/get_pest_training.py [--computer-mice 500] [--kitchens 500] [--max-gb 1.0]   # into data/datasets/oi_pests

Open Images V6 (Google): photos CC BY 2.0 (each photo's author), boxes CC BY 4.0. The training box list
(oidv6-train-annotations-bbox.csv, 2.26 GB) is read as it downloads and not kept. Saved: every training photo with a
mouse box (the animal; drawings and toys left out), plus photos without one so the model learns what is not a mouse:
--computer-mice photos with a computer mouse and --kitchens photos of kitchens (oven, sink, stove, fridge, counter...),
each a fixed random sample. Photos where a mouse is only drawn are left out of both. Each photo gets a YOLO label file
(0 = rodent; empty for the photos without). The validation and test photos (tools/get_weapon_photos.py --kind pests)
are not touched: they stay the test set. Stops when the photos pass --max-gb. Downloaded with the user's OK
(2026-10-30: about 3.3 GB of traffic, about 1 GB kept, said beforehand).
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
OUT = ROOT / "data" / "datasets" / "oi_pests"
BOXES = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
IMAGE = "https://open-images-dataset.s3.amazonaws.com/train/{id}.jpg"
MOUSE = "/m/04rmv"
COMPUTER_MOUSE = "/m/020lf"
KITCHEN = {"/m/0130jx": "sink", "/m/029bxz": "oven", "/m/02wv84t": "gas stove", "/m/040b_t": "refrigerator",
           "/m/0642b4": "cupboard", "/m/0b3fp9": "countertop", "/m/0fx9l": "microwave oven",
           "/m/0h8n5zk": "kitchen table", "/m/0h99cwc": "kitchen appliance"}


def read_list() -> tuple[dict, set, set, set]:
    """(image id -> [(0, x1, y1, x2, y2)] for real mice, ids with a drawn mouse, ids with a computer mouse, ids with
    something from a kitchen), streamed."""
    mice: dict[str, list] = defaultdict(list)
    drawn, computer, kitchen = set(), set(), set()
    with requests.get(BOXES, stream=True, timeout=60) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for n, row in enumerate(csv.DictReader(r.iter_lines(decode_unicode=True)), 1):
            label, image = row["LabelName"], row["ImageID"]
            if label == MOUSE:
                if row["IsDepiction"] == "1":
                    drawn.add(image)
                else:
                    mice[image].append((0, *(float(row[k]) for k in ("XMin", "YMin", "XMax", "YMax"))))
            elif label == COMPUTER_MOUSE:
                computer.add(image)
            elif label in KITCHEN:
                kitchen.add(image)
            if n % 2_000_000 == 0:
                print(f"  {n:,} boxes read; {len(mice)} photos with a mouse so far", flush=True)
    return mice, drawn, computer, kitchen


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
    ap.add_argument("--computer-mice", type=int, default=500)
    ap.add_argument("--kitchens", type=int, default=500)
    ap.add_argument("--max-gb", type=float, default=1.0)
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    mice, drawn, computer, kitchen = read_list()
    rng = random.Random(55)
    taken = set(mice) | drawn
    computer = sorted(computer - taken)
    rng.shuffle(computer)
    computer = computer[: args.computer_mice]
    kitchen = sorted(kitchen - taken - set(computer))
    rng.shuffle(kitchen)
    kitchen = kitchen[: args.kitchens]
    print(f"{len(mice)} training photos with a mouse ({sum(len(b) for b in mice.values())} boxes; {len(drawn)} with "
          f"only drawn mice left out), {len(computer)} with a computer mouse, {len(kitchen)} kitchens", flush=True)
    (OUT / "negatives.txt").write_text("".join(f"computer_mouse {i}\n" for i in computer)
                                       + "".join(f"kitchen {i}\n" for i in kitchen), encoding="utf-8")
    jobs = [(i, b) for i, b in mice.items()] + [(i, []) for i in computer + kitchen]
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

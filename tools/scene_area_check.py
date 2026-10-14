"""Measure scene questions about one area (Phase 27): whole picture vs the area cut out vs the area outlined.

    python tools/scene_area_check.py --config <test_config.yaml> [--coco 150] [--ways whole,part,outline] [--json <file>]

Each item is a picture, an area of it (a rectangle, fractions of the picture) and the true answer for that area.
Asked three ways, exactly as the app asks (labeler/scene.py area_picture, labeler/vlm.py SCENE_PROMPTS):
  whole    the whole picture and the question as typed: what the app did before areas (it cannot tell the area apart)
  part     the area cut out with a margin, the rest of the cut-out darkened
  outline  the whole picture with the area outlined in red, "only inside the red outline"

Sets (nothing was tuned on them; the prompts and the margin were written before the first run):
  coco-car    "Is there a car?" on COCO val2017 photos with a car: an area around a car (1.6x its box) = yes, an area of
              the same size touching no car, lorry, bus or motorbike = no (a car elsewhere in the picture)
  mot04-*     MOT16-04 (fixed camera, a parked car, one picture a second): the car's parking spot = yes; the walkway
              and the bicycle stand = no
  mot13-small MOT16-13 (1920x1080): far cars (marked, at least half visible, under 6% of the picture high) with an
              area 3x their box = yes; shows what shrinking the whole picture to 1024 px loses
  urfd-floor  "Is someone lying on the floor?" on UR Fall (the color half, one picture a second), the floor (lower
              55%) as the area: the marks decide (as tools/scene_check.py; lying down on purpose left out)
  urfd-wall   the same question, the wall (top 35%) as the area, on pictures where someone lies on the floor = no
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from labeler.config import load_config  # noqa: E402
from labeler.scene import area_picture  # noqa: E402
from scene_check import MOT, URFD, urfd_marks  # noqa: E402

COCO = ROOT / "data" / "datasets" / "coco"
CAR_Q, LYING_Q = "Is there a car?", "Is someone lying on the floor?"
VEHICLES = {2, 3, 5, 7}   # COCO car, motorbike, bus, lorry (YOLO class numbers)


def rect(x1, y1, x2, y2) -> list:
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def grow(cx, cy, w, h, k) -> list:
    return rect(cx - w * k / 2, cy - h * k / 2, cx + w * k / 2, cy + h * k / 2)


def overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def coco_items(n: int, seed: int = 27) -> list[dict]:
    names = sorted(p.stem for p in (COCO / "labels" / "val2017").glob("*.txt"))
    random.Random(seed).shuffle(names)
    items = []
    for name in names:
        boxes = []
        for line in open(COCO / "labels" / "val2017" / f"{name}.txt"):
            c, cx, cy, w, h = line.split()
            boxes.append((int(c), float(cx), float(cy), float(w), float(h)))
        cars = [b for b in boxes if b[0] == 2 and b[3] * b[4] >= 0.015]
        if not cars:
            continue
        c, cx, cy, w, h = max(cars, key=lambda b: b[3] * b[4])
        w, h = max(w * 1.6, 0.15), max(h * 1.6, 0.15)
        yes = rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        w, h = yes[2][0] - yes[0][0], yes[2][1] - yes[0][1]
        taken = [(b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2) for b in boxes if b[0] in VEHICLES]
        spots = [(x / 20, y / 20) for x in range(0, 21) for y in range(0, 21) if x / 20 + w <= 1 and y / 20 + h <= 1]
        random.Random(name).shuffle(spots)
        free = next(((x, y) for x, y in spots if not any(overlaps((x, y, x + w, y + h), t) for t in taken)), None)
        if free is None:
            continue
        img = str(COCO / "val2017" / f"{name}.jpg")
        items.append({"set": "coco-car", "question": CAR_Q, "image": img, "area": yes, "truth": "yes"})
        items.append({"set": "coco-car", "question": CAR_Q, "image": img, "area": rect(free[0], free[1], free[0] + w,
                                                                                     free[1] + h), "truth": "no"})
        if len(items) >= 2 * n:
            break
    return items


def mot_items() -> list[dict]:
    items = []
    areas = (("mot04-car", rect(0.58, 0.25, 0.80, 0.58), "yes"), ("mot04-walkway", rect(0.36, 0.35, 0.56, 0.75), "no"),
             ("mot04-bicycles", rect(0.62, 0.02, 0.80, 0.25), "no"))
    for frame in range(1, 1051, 30):
        img = str(MOT / "MOT16-04" / "img1" / f"{frame:06d}.jpg")
        for name, area, truth in areas:
            items.append({"set": name, "question": CAR_Q, "image": img, "area": area, "truth": truth})
    cars = defaultdict(list)
    for line in open(MOT / "MOT16-13" / "gt" / "gt.txt"):
        f = line.split(",")
        if int(f[7]) == 3 and float(f[8]) >= 0.5 and float(f[5]) < 0.06 * 1080:
            cars[int(f[0])].append(tuple(map(float, f[2:6])))
    for frame in range(1, 751, 25):
        if cars.get(frame):
            x, y, w, h = min(cars[frame], key=lambda b: b[3])
            items.append({"set": "mot13-small", "question": CAR_Q, "image": str(MOT / "MOT16-13" / "img1" / f"{frame:06d}.jpg"),
                          "area": grow((x + w / 2) / 1920, (y + h / 2) / 1080, w / 1920, h / 1080, 3), "truth": "yes"})
    return items


def urfd_items(every_s: float = 1.0) -> list[dict]:
    items = []
    step = round(30 * every_s)
    for video, marks in sorted(urfd_marks().items()):
        for frame in sorted(marks)[::step]:
            mark, fall = marks[frame], video.startswith("fall")
            if mark == 0 or (not fall and mark == 1):
                continue   # falling, or lying down on purpose: neither
            base = {"question": LYING_Q, "video": video, "frame": frame}
            items.append({**base, "set": "urfd-floor", "area": rect(0, 0.45, 1, 1), "truth": "yes" if mark == 1 else "no"})
            if mark == 1:
                items.append({**base, "set": "urfd-wall", "area": rect(0, 0, 1, 0.35), "truth": "no"})
    return items


def load(item: dict, caps: dict):
    if "image" in item:
        return cv2.imread(item["image"])
    cap = caps.get(item["video"])
    if cap is None:
        cap = caps[item["video"]] = cv2.VideoCapture(str(URFD / f"{item['video']}-cam0.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame"] - 1)
    ok, frame = cap.read()
    return frame[:, frame.shape[1] // 2:] if ok else None   # the color picture; the left half is a depth map


def report(items: list[dict], ways: list[str]) -> list[dict]:
    """Per set and true answer: right, wrong, unsure and seconds (median) for each way."""
    rows = []
    for name in dict.fromkeys(i["set"] for i in items):
        for truth in ("yes", "no"):
            sel = [i for i in items if i["set"] == name and i["truth"] == truth]
            if not sel:
                continue
            row = {"set": name, "truth": truth, "items": len(sel)}
            for way in ways:
                c = Counter(i[way]["answer"] for i in sel if way in i)
                row[way] = {"right": c[truth], "wrong": c["no" if truth == "yes" else "yes"], "unsure": c["unsure"],
                            "seconds": round(statistics.median(i[way]["seconds"] for i in sel if way in i), 2)}
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a test config (tools/make_test_config.py)")
    ap.add_argument("--coco", type=int, default=150, help="COCO photos (two items each)")
    ap.add_argument("--ways", default="whole,part,outline")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    from labeler.vlm import VLMClient
    cfg = load_config(args.config)
    client = VLMClient(cfg["vlm"], cfg["mode"], cfg["labels"])
    ok, why = client.available()
    if not ok:
        sys.exit(f"the vision model is not available: {why}")
    items = coco_items(args.coco) + mot_items() + urfd_items()
    ways = args.ways.split(",")
    print(f"{len(items)} items x {len(ways)} ways", flush=True)
    caps: dict = {}
    for way in ways:   # one whole pass per way (Ollama reuses work on a repeated picture: pitfall 20)
        whole_seen: dict = {}
        started_way = time.monotonic()
        for k, item in enumerate(items):
            key = (item.get("image") or (item["video"], item["frame"]), item["question"])
            if way == "whole" and key in whole_seen:   # the same picture and question: the same answer
                item[way] = whole_seen[key]
                continue
            img = load(item, caps)
            if img is None:
                continue
            pic = img if way == "whole" else area_picture(img, item["area"], way)
            started = time.monotonic()
            answer = client.label(pic, None, ("scene", item["question"], way))["answer"]
            item[way] = whole_seen[key] = {"answer": answer, "seconds": round(time.monotonic() - started, 3)}
            if (k + 1) % 100 == 0:
                print(f"{way} {k + 1}/{len(items)}", flush=True)
        print(f"{way}: {time.monotonic() - started_way:.0f} s", flush=True)
    rows = report(items, ways)
    print(f"\n{'set':16s} {'truth':5s} {'n':>4s}  " + "  ".join(f"{w:>22s}" for w in ways))
    for r in rows:
        cells = "  ".join(f"{r[w]['right']:>4d} right {r[w]['wrong']:>3d} wr {r[w]['unsure']:>2d} un {r[w]['seconds']:4.2f}s"
                          for w in ways)
        print(f"{r['set']:16s} {r['truth']:5s} {r['items']:>4d}  {cells}")
    for way in ways:
        n = sum(r["items"] for r in rows)
        right = sum(r[way]["right"] for r in rows)
        print(f"all {way}: {right}/{n} right ({100 * right / n:.1f}%)")
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "items": items}, indent=1))


if __name__ == "__main__":
    main()

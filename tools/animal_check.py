"""Measure animal detection (Phase 21): COCO photos, and false alarms on videos without animals.

    python tools/animal_check.py coco [--imgsz 640]           # dogs, cats, birds found per confidence, COCO val2017
    python tools/animal_check.py videos [--out <folder>]      # animal records on MOT16 and UR Fall (no animals there)

coco: every COCO val2017 photo (data/datasets/coco/val2017 + labels/val2017), the vehicle detector asked for the
animals, matched to the marked ones at IoU 0.5 per kind. Counted at three confidences: 0.5 is what the tracker needs to
start following something new (labeler/trackers/vehicles.yaml, new_track_thresh), 0.4 to keep following it
(track_high_thresh), 0.3 for comparison. Small = under 32x32 pixels (COCO's split: far away). "wrong" = boxes that match
no marked animal of that kind. One photo is not a video: on a camera an animal needs one frame at 0.5 and then keeps
being found for vehicles.min_frames; photos give the chance per frame.

--person-filter (both): an animal box that is the same box as a person found by the person detector (IoU >=
--person-iou; people at the person tracker's 0.5) is left out. Tried 2026-10-14 against a person crawling on all fours
taken for a dog, not kept in the app: it also dropped real animals (README "Animals").

videos: the camera's own way: every 3rd frame of a 30 fps video (10 a second), the second detector every 2nd of those
(vehicles.every_n_frames), its tracker, and the vehicle watcher with the animals' names (labeler/animals.py). The MOT16
street videos (data/datasets/mot16/train_videos) and the UR Fall indoor videos (data/datasets/urfd; their right half,
the color picture: the left half is a depth map) have no animals, so every animal record is a false alarm; with --out
its best picture is saved there to look at.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from vehicle_check import KINDS as VEHICLE_KINDS, iou  # noqa: E402

COCO = ROOT / "data" / "datasets" / "coco"
ANIMALS = {14: "bird", 15: "cat", 16: "dog"}
SMALL = 32 * 32
CONFS = (0.5, 0.4, 0.3)


def truth(label: Path, w: int, h: int) -> list[tuple[int, tuple]]:
    out = []
    if label.exists():
        for line in label.read_text().splitlines():
            c, cx, cy, bw, bh = line.split()
            if int(c) in ANIMALS:
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                out.append((int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
    return out


def match(gts: list, dets: list) -> tuple[set[int], set[int]]:
    """(indexes of marked animals found, indexes of boxes used): same kind, IoU >= 0.5, best first."""
    found, used = set(), set()
    for gi, (gc, gb) in enumerate(gts):
        best, best_i = 0.5, None
        for di, (dc, db, _) in enumerate(dets):
            if di in used or dc != gc:
                continue
            v = iou(gb, db)
            if v >= best:
                best, best_i = v, di
        if best_i is not None:
            found.add(gi)
            used.add(best_i)
    return found, used


PERSON_CONF = 0.5   # people the camera follows: the person tracker starts at 0.5 (labeler/trackers/people.yaml)


def not_people(dets: list[tuple], person_boxes: list[tuple], min_iou: float = 0.5) -> list[tuple]:
    """dets: [(tracker id, box, class, confidence)] without those that are the same box as a found person. Tried as a
    camera rule against people crawling on all fours (2026-10-14) and not kept: it also dropped real animals."""
    return [d for d in dets if all(iou(tuple(d[1][:4]), tuple(p[:4])) < min_iou for p in person_boxes)]


def people_in(pose, img, imgsz: int) -> list[tuple]:
    r = pose(img, conf=PERSON_CONF, imgsz=imgsz, verbose=False)[0]
    return [tuple(float(v) for v in b) for b in r.boxes.xyxy.cpu().numpy()]


def coco(args) -> None:
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "yolo11s.pt"))
    pose = YOLO(str(ROOT / "yolo11s-pose.pt")) if args.person_filter else None
    tally = defaultdict(Counter)   # (kind, size, conf) -> truth / found
    wrong = Counter()              # (kind, conf) -> boxes matching no marked animal
    photos_with = Counter()
    images = sorted((COCO / "val2017").glob("*.jpg"))
    for k, path in enumerate(images):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gts = truth(COCO / "labels" / "val2017" / f"{path.stem}.txt", w, h)
        for c in {gc for gc, _ in gts}:
            photos_with[ANIMALS[c]] += 1
        r = model(img, classes=list(ANIMALS), conf=min(CONFS), imgsz=args.imgsz, verbose=False)[0]
        all_dets = [(int(c), tuple(float(v) for v in b), float(p))
                    for c, b, p in zip(r.boxes.cls.tolist(), r.boxes.xyxy.cpu().numpy(), r.boxes.conf.tolist())]
        if pose is not None:   # as on the camera: not a box that is a person the person detector found
            kept = not_people([(i, d[1], d[0], d[2]) for i, d in enumerate(all_dets)], people_in(pose, img, args.imgsz),
                              args.person_iou)
            all_dets = [all_dets[k[0]] for k in kept]
        for conf in CONFS:
            dets = [d for d in all_dets if d[2] >= conf]
            found, used = match(gts, dets)
            for gi, (gc, gb) in enumerate(gts):
                size = "small" if (gb[2] - gb[0]) * (gb[3] - gb[1]) < SMALL else "medium/large"
                tally[(ANIMALS[gc], size, conf)]["truth"] += 1
                tally[(ANIMALS[gc], size, conf)]["found"] += gi in found
            for di, (dc, _, _) in enumerate(dets):
                if di not in used:
                    wrong[(ANIMALS[dc], conf)] += 1
        if (k + 1) % 1000 == 0:
            print(f"{k + 1}/{len(images)}", flush=True)
    print(f"\nCOCO val2017, {len(images)} photos, detector size {args.imgsz}"
          + (f", boxes that are a found person (IoU >= {args.person_iou}) left out" if args.person_filter else ""))
    print(f"{'kind':6s} {'conf':>5s} {'medium/large found':>22s} {'small found':>16s} {'wrong boxes':>12s}")
    for kind in ANIMALS.values():
        for conf in CONFS:
            big, small = tally[(kind, "medium/large", conf)], tally[(kind, "small", conf)]
            print(f"{kind:6s} {conf:>5.1f} {big['found']:>9d} of {big['truth']:<5d} ({big['found'] / max(big['truth'], 1):4.0%})"
                  f" {small['found']:>5d} of {small['truth']:<4d} ({small['found'] / max(small['truth'], 1):4.0%})"
                  f" {wrong[(kind, conf)]:>8d}")
    print("photos with: " + ", ".join(f"{k} {n}" for k, n in photos_with.items()))


class _Store:
    """Records in memory; with `out`, each record's best picture is saved there."""

    def __init__(self, out: Path | None, name: str):
        self.rows, self.out, self.name = {}, out, name

    def create(self, kind, started_at, tracker_id):
        self.rows[len(self.rows) + 1] = {"kind": kind}
        return len(self.rows)

    def update(self, animal_id, **fields):
        self.rows[animal_id].update(fields)

    def picture(self, animal_id, crop, heads=None):
        if self.out is not None:
            cv2.imwrite(str(self.out / f"{self.name}_{animal_id}_{self.rows[animal_id]['kind']}.jpg"), crop)

    def describe(self, animal_id, crop):
        pass

    def changed(self, animal_id):
        pass

    def zone_events(self, animal_id, events):
        pass


def videos(args) -> None:
    from ultralytics import YOLO
    from labeler.animals import kinds_map
    pose = YOLO(str(ROOT / "yolo11s-pose.pt")) if args.person_filter else None
    from labeler.config import DEFAULTS
    from labeler.vehicles import VehicleWatch
    vcfg, acfg = DEFAULTS["vehicles"], DEFAULTS["animals"]
    names = kinds_map(acfg["kinds"])
    classes = list(VEHICLE_KINDS) + list(names)   # one pass, as on a security camera with vehicles on
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    sets = [("MOT16", sorted((ROOT / "data" / "datasets" / "mot16" / "train_videos").glob("*.mp4"))),
            ("UR Fall", sorted((ROOT / "data" / "datasets" / "urfd").glob("*.mp4")))]
    for label, paths in sets:
        records, minutes, by_kind, with_any = 0, 0.0, Counter(), []
        for path in paths:
            model = YOLO(str(ROOT / vcfg["model"]))   # a fresh tracker per video
            store = _Store(args.out, path.stem)
            watch = VehicleWatch(acfg, store, acfg["kinds"], names=names, still_word="resting")
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            step = max(1, round(fps / 10)) * int(vcfg["every_n_frames"])
            n = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                if (n - 1) % step:
                    continue
                if label == "UR Fall":
                    frame = frame[:, frame.shape[1] // 2:]   # the color picture; the left half is a depth map
                r = model.track(frame, persist=True, tracker=str(ROOT / vcfg["tracker"]), classes=classes,
                                conf=0.1, imgsz=int(vcfg["imgsz"]), verbose=False)[0]
                dets = []
                if r.boxes is not None and r.boxes.id is not None:
                    dets = [d for d in zip(r.boxes.id.int().tolist(), [tuple(b) for b in r.boxes.xyxy.cpu().numpy()],
                                           r.boxes.cls.int().tolist(), r.boxes.conf.tolist()) if d[2] in names]
                if pose is not None and dets:
                    dets = not_people(dets, people_in(pose, frame, int(vcfg["imgsz"])), args.person_iou)
                t = (n - 1) / fps
                watch.update(t, datetime(2026, 1, 1) + timedelta(seconds=t), frame, dets)
            watch.close()
            minutes += n / fps / 60
            records += len(store.rows)
            by_kind.update(row["kind"] for row in store.rows.values())
            if store.rows:
                with_any.append(f"{path.stem} ({', '.join(r['kind'] for r in store.rows.values())})")
        print(f"\n{label}: {len(paths)} videos, {minutes:.1f} minutes, animal records {records} "
              f"({', '.join(f'{k} {n}' for k, n in by_kind.most_common()) or 'none'})")
        for line in with_any:
            print("  " + line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coco")
    c.add_argument("--imgsz", type=int, default=640)
    c.add_argument("--person-filter", action="store_true", help="leave out boxes that are a found person (tried, not used by the app)")
    c.add_argument("--person-iou", type=float, default=0.5)
    v = sub.add_parser("videos")
    v.add_argument("--person-filter", action="store_true", help="leave out boxes that are a found person (tried, not used by the app)")
    v.add_argument("--person-iou", type=float, default=0.5)
    v.add_argument("--out", type=Path, help="save each animal record's best picture here")
    args = ap.parse_args()
    {"coco": coco, "videos": videos}[args.cmd](args)


if __name__ == "__main__":
    main()

"""Measure vehicle detection (Phase 18): COCO photos, cost per frame, and counting on MOT16.

    python tools/vehicle_check.py coco [--imgsz 640] [--conf 0.35]   # found / wrong per kind, COCO val2017
    python tools/vehicle_check.py speed                              # ms per frame: people only vs people + vehicles

COCO val2017 (data/datasets/coco/val2017 + labels/val2017, YOLO boxes): every photo, vehicles matched to the marked
ones at IoU 0.5, per kind. Small = under 32x32 pixels (COCO's own split): far away, which a gate camera rarely needs;
shown apart. A vehicle found as another vehicle kind (a pickup as car instead of truck) counts as found, and is
listed as a mix-up.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COCO = ROOT / "data" / "datasets" / "coco"
KINDS = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "lorry"}   # COCO class -> our name ("truck" -> lorry)
SMALL = 32 * 32


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def truth(label: Path, w: int, h: int) -> list[tuple[int, tuple]]:
    out = []
    if label.exists():
        for line in label.read_text().splitlines():
            c, cx, cy, bw, bh = line.split()
            c = int(c)
            if c in KINDS:
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                out.append((c, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
    return out


def coco(args) -> None:
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "yolo11s.pt"))
    found = defaultdict(lambda: Counter())     # (kind, size) -> {"truth": n, "found": n}
    wrong = Counter()                          # kind -> detections matching nothing
    mixups = Counter()
    images = sorted((COCO / "val2017").glob("*.jpg"))
    for k, path in enumerate(images):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gts = truth(COCO / "labels" / "val2017" / f"{path.stem}.txt", w, h)
        r = model(img, classes=list(KINDS), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dets = [(int(c), tuple(float(v) for v in b)) for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.cpu().numpy())]
        used = set()
        # Same kind first, then any vehicle kind (a mix-up still found the vehicle).
        for same_kind in (True, False):
            for gi, (gc, gb) in enumerate(gts):
                if ("g", gi) in used:
                    continue
                best, best_i = 0.5, None
                for di, (dc, db) in enumerate(dets):
                    if di in used or (dc == gc) != same_kind:
                        continue
                    v = iou(gb, db)
                    if v >= best:
                        best, best_i = v, di
                if best_i is not None:
                    used |= {best_i, ("g", gi)}
                    if not same_kind:
                        mixups[(KINDS[gc], KINDS[dets[best_i][0]])] += 1
        for gi, (gc, gb) in enumerate(gts):
            size = "small" if (gb[2] - gb[0]) * (gb[3] - gb[1]) < SMALL else "medium/large"
            found[(KINDS[gc], size)]["truth"] += 1
            found[(KINDS[gc], size)]["found"] += ("g", gi) in used
        for di, (dc, _) in enumerate(dets):
            if di not in used:
                wrong[KINDS[dc]] += 1
        if (k + 1) % 1000 == 0:
            print(f"{k + 1}/{len(images)}", flush=True)
    print(f"\nCOCO val2017, {len(images)} photos, detector size {args.imgsz}, confidence {args.conf}")
    print(f"{'kind':10s} {'medium/large found':>20s} {'small found':>14s} {'wrong boxes':>12s}")
    for kind in KINDS.values():
        big, small = found[(kind, "medium/large")], found[(kind, "small")]
        print(f"{kind:10s} {big['found']:>9d} of {big['truth']:<5d} ({big['found'] / max(big['truth'], 1):4.0%})"
              f" {small['found']:>5d} of {small['truth']:<5d} {wrong[kind]:>8d}")
    if mixups:
        print("mix-ups (marked -> found as):", ", ".join(f"{a}->{b} {n}" for (a, b), n in mixups.most_common(8)))


def speed(args) -> None:
    from ultralytics import YOLO
    pose = YOLO(str(ROOT / "yolo11s-pose.pt"))
    det = YOLO(str(ROOT / "yolo11s.pt"))
    cap = cv2.VideoCapture(str(ROOT / "data" / "datasets" / "mot16" / "train_videos" / "MOT16-13.mp4"))
    frames = []
    while len(frames) < 300:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    tracker = str(ROOT / "labeler" / "trackers" / "people.yaml")

    def run(with_vehicles: bool, every: int) -> float:
        pose.predictor = det.predictor = None    # fresh trackers
        times = []
        for i, f in enumerate(frames):
            t = time.perf_counter()
            pose.track(f, persist=True, tracker=tracker, classes=[0], conf=0.1, imgsz=640, verbose=False)
            if with_vehicles and i % every == 0:
                det.track(f, persist=True, tracker=tracker, classes=list(KINDS), conf=0.1, imgsz=640, verbose=False)
            times.append(time.perf_counter() - t)
        return statistics.median(times[20:]) * 1000, statistics.mean(times[20:]) * 1000

    run(True, 1)   # warm-up
    print(f"{len(frames)} frames of MOT16-13 ({frames[0].shape[1]}x{frames[0].shape[0]}), detector size 640")
    for label, wv, every in (("people only", False, 1), ("people + vehicles, every frame", True, 1),
                             ("people + vehicles, every 2nd frame", True, 2), ("people + vehicles, every 3rd", True, 3)):
        med, mean = run(wv, every)
        print(f"  {label:36s} median {med:5.1f} ms, mean {mean:5.1f} ms per frame")


MOT_KINDS = {3: "car", 4: "bicycle", 5: "motorbike"}   # MOT16 classes in gt.txt


class _Store:
    """Records kept in memory (no database), for counting."""

    def __init__(self):
        self.rows = {}

    def create(self, kind, started_at, tracker_id):
        self.rows[len(self.rows) + 1] = {"kind": kind}
        return len(self.rows)

    def update(self, vehicle_id, **fields):
        self.rows[vehicle_id].update(fields)

    def picture(self, vehicle_id, crop):
        pass


def count(args) -> None:
    """One record per vehicle? MOT16 clips with marked vehicles: each marked vehicle (identity) is matched to the
    records whose boxes overlap it most (IoU >= 0.3 in at least a third of the frames both are seen)."""
    from datetime import datetime, timedelta
    from ultralytics import YOLO
    from labeler.config import DEFAULTS
    from labeler.vehicles import VehicleWatch
    cfg = {**DEFAULTS["vehicles"], **({"every_n_frames": args.every} if args.every else {}),
           **({"parked_after_s": args.parked} if args.parked else {}),
           **({"move_ratio": args.move} if args.move else {})}
    every = int(cfg["every_n_frames"])
    for seq in args.clips:
        gt_path = ROOT / "data" / "datasets" / "mot16" / "MOT16" / "train" / seq / "gt" / "gt.txt"
        gt = defaultdict(list)   # frame -> [(identity, box, kind)]
        for line in gt_path.read_text().splitlines():
            f, i, x, y, w, h, _, c, vis = line.split(",")[:9]
            if int(c) in MOT_KINDS and float(vis) >= 0.25:
                x, y, w, h = map(float, (x, y, w, h))
                gt[int(f)].append((int(i), (x, y, x + w, y + h), MOT_KINDS[int(c)]))
        model = YOLO(str(ROOT / cfg["model"]))
        store = _Store()
        watch = VehicleWatch(cfg, store)
        cap = cv2.VideoCapture(str(ROOT / "data" / "datasets" / "mot16" / "train_videos" / f"{seq}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        overlap = defaultdict(Counter)   # record -> identity -> frames overlapping
        seen_rec, seen_gt = Counter(), Counter()
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if (n - 1) % every:
                continue
            r = model.track(frame, persist=True, tracker=str(ROOT / cfg["tracker"]), classes=list(KINDS),
                            conf=0.1, imgsz=int(cfg["imgsz"]), verbose=False)[0]
            dets = []
            if r.boxes is not None and r.boxes.id is not None:
                dets = list(zip(r.boxes.id.int().tolist(), [tuple(b) for b in r.boxes.xyxy.cpu().numpy()],
                                r.boxes.cls.int().tolist(), r.boxes.conf.tolist()))
            t = (n - 1) / fps
            watch.update(t, datetime(2026, 1, 1) + timedelta(seconds=t), frame, dets)
            for gid, _, _ in gt.get(n, []):
                seen_gt[gid] += 1
            for st in watch.tracks.values():
                if st.db_id is None or st.last_t != t:
                    continue
                seen_rec[st.db_id] += 1
                for gid, gbox, _ in gt.get(n, []):
                    if iou(st.box, gbox) >= 0.3:
                        overlap[st.db_id][gid] += 1
        watch.close()
        ids = {gid for rows in gt.values() for gid, _, _ in rows}
        owner = {}   # record -> identity it follows
        for rec, c in overlap.items():
            gid, k = c.most_common(1)[0]
            if k >= seen_rec[rec] / 3:
                owner[rec] = gid
        per_gt = Counter(owner.values())
        found = [g for g in ids if per_gt[g]]
        print(f"\n{seq} ({n} frames, vehicles every {every}. frame): {len(ids)} marked vehicles "
              f"({', '.join(f'{k} {v}' for k, v in Counter(next(kd for rows in gt.values() for gi, _, kd in rows if gi == g) for g in ids).most_common())})")
        print(f"  found {len(found)} of {len(ids)}; records {len(store.rows)}: {len(owner)} follow a marked vehicle, "
              f"{len(store.rows) - len(owner)} do not (unmarked vehicles or wrong)")
        split = {g: k for g, k in per_gt.items() if k > 1}
        print(f"  marked vehicles split into several records: {len(split)}" + (f" {dict(split)}" if split else ""))
        parked = [r for r, row in store.rows.items() if row.get("parked")]
        print(f"  records parked: {len(parked)}; marked vehicles seen in most frames (standing): "
              f"{sum(1 for g in ids if seen_gt[g] >= 0.8 * n / every)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("coco")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    sub.add_parser("speed")
    p = sub.add_parser("count", help="one record per vehicle on MOT16 clips with marked vehicles")
    p.add_argument("clips", nargs="*", default=["MOT16-04", "MOT16-13", "MOT16-05"])
    p.add_argument("--every", type=int, default=0, help="vehicle detector on every n-th frame (default: config)")
    p.add_argument("--parked", type=float, default=0, help="parked after this many seconds (the clips are short)")
    p.add_argument("--move", type=float, default=0, help="moved = the centre went this share of its size away")
    args = ap.parse_args()
    {"coco": coco, "speed": speed, "count": count}[args.cmd](args)


if __name__ == "__main__":
    main()

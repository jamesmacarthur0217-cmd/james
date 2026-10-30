"""Pests (Phase 47 step 3): does YOLOE prompt-free find real mice, and what does it take for a mouse where there is
none? Then the vision model's answer to the pest question about the part around each box. Measured 2026-10-21: too
poor (3% of real rodents found, computer mice taken for mice), so pests were left out of the app (the user's choice);
kept to try a better model later. Phase 55: --model measures our own detector (tools/train_pests.py) instead.

    python tools/pest_check.py [--half tune|report|all] [--model data/models/pests/pests.pt] [--far | --area]
                               [--night] [--vlm] [--cut wide|tight|close]

Mice: Open Images validation and test photos with a mouse (the animal; tools/get_weapon_photos.py --kind pests, CC BY
2.0), their true boxes in boxes.json; a mouse is found when a pest box (rat, mouse, rodent, roach) overlaps a true one
(IoU >= 0.3) or has its middle inside it. Not mice (COCO val2017, here): photos with a computer mouse, a teddy bear, or
a kitchen (oven, sink, microwave, refrigerator); any pest box there is a false alarm. Halves (pitfall 21): the photos in
name order, even = tune, odd = report. --far: most Open Images mice are close-ups; this shrinks each photo so the
mouse (or, for COCO, the photo's largest marked thing) is about 40 px wide in a 1280 x 720 frame, the edges repeated,
as a mouse across a kitchen would look. --area: as --far, then only a food area is looked at, as the app does
(Phase 55): a third of the frame each way (427 x 240, about a kitchen counter) with the mouse (or the largest thing)
somewhere inside, the same place for a photo on every run. --night: the picture in the made-up infrared look of
tools/make_night_videos.py (black and white, flat, grainy), after --far and before the area is cut. Per threshold: mice
found, photos with a false pest; with --vlm at the app's threshold the vision model's yes / no / unsure on found mice
and on the false boxes (--cut: how much around the box).
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeler.config import load_config  # noqa: E402
from labeler.incidents import person_crop  # noqa: E402
from labeler.pests import QUESTION as PEST_QUESTION  # noqa: E402 (the app's question)
from tools.make_night_videos import night  # noqa: E402

# The words of YOLOE's fixed list for pests (geckos and lizards, the harmless cicak, left out on purpose).
PEST_WORDS = ("rat", "mouse", "rodent", "roach")
PEST_CONF = 0.3
MICE = ROOT / "data" / "datasets" / "openimages_pests"
COCO = ROOT / "data" / "datasets" / "coco"
NOT_MICE = {64: "computer mouse", 77: "teddy bear", 68: "microwave", 69: "oven", 71: "sink", 72: "refrigerator"}
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
CUTS = {"wide": (0.75, 0.35), "tight": (0.6, 0.26), "close": (1.0, 0.15)}   # (margin, smallest share): person_crop
FAR_W, FAR_H, FAR_PX = 1280, 720, 40
AREA_W, AREA_H = FAR_W // 3, FAR_H // 3


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix)


def far(img, box):
    """The photo shrunk so box (pixels) is FAR_PX wide, in a FAR_W x FAR_H frame with the edges repeated; returns
    (frame, box in the frame)."""
    h, w = img.shape[:2]
    k = min(1.0, FAR_PX / max(1.0, box[2] - box[0]))
    small = cv2.resize(img, (max(8, int(w * k)), max(8, int(h * k))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    top, left = max(0, (FAR_H - sh) // 2), max(0, (FAR_W - sw) // 2)
    frame = cv2.copyMakeBorder(small, top, max(0, FAR_H - sh - top), left, max(0, FAR_W - sw - left),
                               cv2.BORDER_REPLICATE)[:FAR_H, :FAR_W]
    return frame, [box[0] * k + left, box[1] * k + top, box[2] * k + left, box[3] * k + top]


def area(frame, box, name: str):
    """An AREA_W x AREA_H part of the frame with box (pixels) inside it at a place fixed by the photo's name; returns
    (part, its left, its top)."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    bw, bh = box[2] - box[0], box[3] - box[1]
    x = int(np.clip(box[0] - rng.uniform(0, max(0.0, AREA_W - bw)), 0, FAR_W - AREA_W))
    y = int(np.clip(box[1] - rng.uniform(0, max(0.0, AREA_H - bh)), 0, FAR_H - AREA_H))
    return frame[y:y + AREA_H, x:x + AREA_W], x, y


def cases(half: str, limit: int) -> list[dict]:
    """[{"path", "truth": [boxes in pixels], "mouse": bool}] for the half."""
    out = []
    index = json.loads((MICE / "boxes.json").read_text(encoding="utf-8"))
    for i, (name, items) in enumerate(sorted(index.items())):
        if half != "all" and i % 2 != (0 if half == "tune" else 1):
            continue
        img = cv2.imread(str(MICE / name))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = [[b["box"][0] * w, b["box"][1] * h, b["box"][2] * w, b["box"][3] * h] for b in items
                 if not b["depiction"]]
        if boxes:
            out.append({"path": MICE / name, "truth": boxes, "mouse": True})
    coco = []
    for txt in sorted((COCO / "labels" / "val2017").glob("*.txt")):
        rows = [r.split() for r in txt.read_text().splitlines() if r.strip()]
        if any(int(r[0]) in NOT_MICE for r in rows):
            coco.append((txt.stem, rows))
    for i, (stem, rows) in enumerate(coco[:limit * 2]):
        if half != "all" and i % 2 != (0 if half == "tune" else 1):
            continue
        out.append({"path": COCO / "val2017" / f"{stem}.jpg", "mouse": False,
                    "kind": Counter(NOT_MICE[int(r[0])] for r in rows if int(r[0]) in NOT_MICE).most_common(1)[0][0],
                    "rows": [list(map(float, r[1:5])) for r in rows]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--half", choices=["tune", "report", "all"], default="tune")
    ap.add_argument("--limit", type=int, default=300, help="COCO photos per half")
    ap.add_argument("--model", help="our own detector (tools/train_pests.py) instead of YOLOE")
    ap.add_argument("--conf", type=float, default=PEST_CONF, help="the threshold for --vlm and the words")
    ap.add_argument("--far", action="store_true")
    ap.add_argument("--area", action="store_true", help="--far, then only a food area around the mouse")
    ap.add_argument("--night", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640, help="the size the detector looks at (an area is enlarged)")
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--cut", choices=list(CUTS), default="wide")
    args = ap.parse_args()
    args.far = args.far or args.area
    cfg = load_config(Path(args.config))
    scfg = cfg["safety"]
    if args.model:
        from ultralytics import YOLO
        model = YOLO(str(ROOT / args.model))
        ids = list(model.names)
    else:
        from ultralytics import YOLOE
        model = YOLOE(str(ROOT / scfg["weapon_model"]))
        ids = [i for i, n in model.names.items() if n.lower() in PEST_WORDS]
    client = None
    if args.vlm:
        from labeler.vlm import VLMClient, incident_prompt
        client = VLMClient(cfg["vlm"], "security")
    conf_app = args.conf
    mice = [c for c in cases(args.half, args.limit) if c["mouse"]]
    not_mice = [c for c in cases(args.half, args.limit) if not c["mouse"]]
    found = Counter()
    false = Counter()
    false_kind = Counter()
    answers = {True: Counter(), False: Counter()}
    words = Counter()
    for c in mice + not_mice:
        img = cv2.imread(str(c["path"]))
        h, w = img.shape[:2]
        if args.far:
            if c["mouse"]:
                ref = c["truth"][0]
            else:   # the photo's largest marked thing stands in for the mouse's size
                cx, cy, bw, bh = max(c["rows"], key=lambda r: r[2] * r[3])
                ref = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            k_img, ref = far(img, ref)
            if c["mouse"]:
                c["truth"] = [far(img, b)[1] for b in c["truth"]]
            img = k_img
        if args.night:
            img = night("ir", img, np.random.default_rng(zlib.crc32(c["path"].name.encode())))
        if args.area:
            img, x, y = area(img, ref, c["path"].name)
            if c["mouse"]:
                c["truth"] = [[b[0] - x, b[1] - y, b[2] - x, b[3] - y] for b in c["truth"]]
        r = model.predict(img, classes=ids, conf=min(THRESHOLDS), imgsz=args.imgsz, verbose=False)[0]
        dets = [(model.names[int(cl)].lower(), float(p), list(b))
                for cl, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        if c["mouse"]:
            def hits(d):
                mx, my = (d[2][0] + d[2][2]) / 2, (d[2][1] + d[2][3]) / 2
                return any(iou(d[2], t) >= 0.3 or (t[0] <= mx <= t[2] and t[1] <= my <= t[3]) for t in c["truth"])
            good = [d for d in dets if hits(d)]
            best = max((d[1] for d in good), default=0.0)
            for t in THRESHOLDS:
                found[t] += best >= t
            pick = max(good, key=lambda d: d[1]) if best >= conf_app else None
            if pick:
                words[pick[0]] += 1
        else:
            best = max((d[1] for d in dets), default=0.0)
            for t in THRESHOLDS:
                false[t] += best >= t
            pick = max(dets, key=lambda d: d[1]) if best >= conf_app else None
            if pick:
                false_kind[c["kind"]] += 1
        if client is not None and pick is not None:
            H, W = img.shape[:2]
            crop = (pick[2][0] / W, pick[2][1] / H, pick[2][2] / W, pick[2][3] / H)
            pic = person_crop(img, crop, *CUTS[args.cut])
            got = client.label([pic], None, ("incident", incident_prompt(1, 0, "", PEST_QUESTION)))
            answers[c["mouse"]][got.get("answer")] += 1
            if not c["mouse"] or got.get("answer") != "yes":
                print(f"  {c['path'].name} ({'mouse' if c['mouse'] else c['kind']}): {pick[0]} {pick[1]:.2f} -> "
                      f"{got.get('answer')} ({got.get('what', '')})")
    how = "".join(f", {k}" for k in ("area", "far", "night") if getattr(args, k) and not (k == "far" and args.area))
    print(f"{args.model or 'YOLOE'}, half {args.half}{how}: {len(mice)} mouse photos, {len(not_mice)} without "
          f"({dict(Counter(c['kind'] for c in not_mice))})")
    for t in THRESHOLDS:
        print(f"  conf {t:.1f}: mice found {found[t]}/{len(mice)} ({found[t] / max(1, len(mice)):.0%}), "
              f"photos with a false pest {false[t]}/{len(not_mice)}")
    print(f"  at {conf_app}: words for the mice {dict(words)}; false pests on {dict(false_kind)}")
    if client is not None:
        print(f"  vision model ({args.cut} cut): on found mice {dict(answers[True])}, on false boxes {dict(answers[False])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Measure the help sign (Phase 32, labeler/gestures.py): false alarms on videos, hands up on photos.

    python tools/gesture_check.py videos [--both-ok yes|no|both]   # false alarms on everyday videos
    python tools/gesture_check.py photos --out <folder> [--n 600]   # hands up on COCO people: mosaics to judge by eye

videos: the UR Fall clips (30 falls and 40 everyday activities: sitting, lying down, picking things up; camera 0, the
color half) and the 7 MOT16 street videos, with the app's detector and tracker (the cached detections of
tools/fall_check.py; the frozen last picture it adds to fall clips is left out here). Nobody makes the sign in them,
so every sign found is a false alarm. Also counted: frames with a hand up at all (one frame is not an alert).

photos: COCO val2017 photos with people; the app's pose model on each, hands_up() on every person found. Two mosaics
to look at: every person the rule says has a hand up (are they right?) and a random sample of people with the rule's
answer (are hands up missed?). The numbers by eye go in the README.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from labeler.gestures import GestureWatch, hands_up  # noqa: E402

COCO = ROOT / "data" / "datasets" / "coco"
TAIL_S = 15.0   # tools/fall_check.py holds a fall clip's last picture this long (hold_s): not part of the video


def video_signs(dets: list, both_ok: bool, tail_from: float | None = None) -> tuple[list[dict], int, int]:
    """(signs found, frames with a hand up, frames) for one video's detections [(t, [(id, box, kpts)])]."""
    watch = GestureWatch({"both_ok": both_ok})
    signs, up_frames, n = [], 0, 0
    for t, people in dets:
        if tail_from is not None and t > tail_from:
            break
        n += 1
        up_frames += any(hands_up(k) for _, _, k in people)
        for tid, _, kpts in people:
            e = watch.update(tid, t, kpts)
            if e:
                signs.append({**e, "at": t, "id": tid})
    return signs, up_frames, n


def videos(args) -> None:
    import fall_check as fc
    kinds = {"fall0": "UR Fall falls", "adl": "UR Fall everyday", "walk": "MOT16 street"}
    for both in ([True, False] if args.both_ok == "both" else [args.both_ok == "yes"]):
        print(f"\ntwo hands up count too: {'yes' if both else 'no'}")
        for kind, label in kinds.items():
            total = up = frames = 0
            where = []
            for name, path, k in fc.videos():
                if k != kind:
                    continue
                dets = fc.detections(name, path, k, TAIL_S)
                end = dets[-1][0] - TAIL_S if k != "walk" and dets else None
                signs, u, n = video_signs(dets, both, end)
                total, up, frames = total + len(signs), up + u, frames + n
                where += [f"{name} at {s['at']:.1f} s ({s['hand']})" for s in signs]
            print(f"  {label:18s} signs (false alarms) {total}; frames with a hand up {up} of {frames}"
                  + (f": {', '.join(where)}" if where else ""))


def photos(args) -> None:
    from ultralytics import YOLO
    from labeler.config import DEFAULTS
    model = YOLO(str(ROOT / DEFAULTS["detector"]["model"]))
    names = sorted(p.stem for p in (COCO / "labels" / "val2017").glob("*.txt"))
    random.Random(32).shuffle(names)
    people = []
    for name in names[args.start: args.start + args.n]:
        if not any(line.startswith("0 ") for line in open(COCO / "labels" / "val2017" / f"{name}.txt")):
            continue
        img = cv2.imread(str(COCO / "val2017" / f"{name}.jpg"))
        r = model.predict(img, conf=0.5, verbose=False)[0]
        if r.keypoints is None:
            continue
        for box, kp in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            if y2 - y1 < 80:
                continue   # too small to judge by eye
            pad = int(0.15 * (y2 - y1))
            crop = img[max(0, y1 - pad):y2, max(0, x1 - pad):x2 + pad]
            people.append({"photo": name, "up": hands_up(kp), "crop": crop})
    up = [p for p in people if p["up"]]
    print(f"{len(people)} people in {args.n} photos; the rule says a hand is up for {len(up)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mosaic(up, out / "up.jpg")
    sample = random.Random(1).sample(people, min(args.sample, len(people)))
    for k in range(0, len(sample), 40):
        mosaic(sample[k:k + 40], out / f"sample_{k // 40 + 1}.jpg", numbered_from=k)
    print(f"mosaics in {out}: up.jpg ({len(up)} people), sample_*.jpg ({len(sample)} people, numbered)")


def mosaic(items: list[dict], path: Path, numbered_from: int | None = None, cols: int = 8) -> None:
    tiles = []
    for i, p in enumerate(items):
        t = cv2.resize(p["crop"], (150, 220))
        label = ("UP " + ",".join(p["up"]) if p["up"] else "-")
        if numbered_from is not None:
            label = f"{numbered_from + i + 1} {label}"
        cv2.rectangle(t, (0, 0), (150, 22), (0, 0, 0), -1)
        cv2.putText(t, label, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if p["up"] else (200, 200, 200), 1)
        tiles.append(t)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("videos")
    p.add_argument("--both-ok", choices=["yes", "no", "both"], default="both")
    p = sub.add_parser("photos")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=600, help="COCO photos to look at")
    p.add_argument("--start", type=int, default=0, help="skip this many (the first 1500 were looked at when the "
                                                          "upright rule was added: report on --start 1500)")
    p.add_argument("--sample", type=int, default=200, help="people in the random sample to judge by eye")
    args = ap.parse_args()
    {"videos": videos, "photos": photos}[args.cmd](args)


if __name__ == "__main__":
    main()

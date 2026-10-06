"""Cut people out of a folder of photos, using the same detector and crop rules as the live pipeline.

    python tools/extract_people.py data/datasets/coco/val2017 data/datasets/coco_people

Writes <out>/person/*.jpg, <out>/face/*.jpg (when the face is visible) and <out>/people.jsonl.
Only clear, reasonably large people are kept, since those are the ones the system labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO  # noqa: E402

from labeler import geometry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--min-height", type=int, default=160, help="minimum person box height in pixels")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--max-per-image", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "person").mkdir(parents=True, exist_ok=True)
    (out / "face").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(Path(__file__).resolve().parent.parent / "yolo11s-pose.pt"))
    images = sorted(Path(args.images).glob("*.jpg"))
    n = 0
    with open(out / "people.jsonl", "w", encoding="utf-8") as f:
        for i, path in enumerate(images):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            r = model(frame, classes=[0], conf=args.min_conf, verbose=False)[0]
            kept = 0
            order = r.boxes.conf.argsort(descending=True).tolist()
            for k in order:
                if kept >= args.max_per_image:
                    break
                box = tuple(r.boxes.xyxy[k].tolist())
                if box[3] - box[1] < args.min_height:
                    continue
                kpts = r.keypoints.data[k].cpu().numpy()
                person = geometry.crop(frame, box, pad=0.05)
                if person is None:
                    continue
                name = f"{path.stem}_{k}"
                cv2.imwrite(str(out / "person" / f"{name}.jpg"), person)
                face = geometry.crop(frame, geometry.face_box(kpts), pad=0.1)
                has_face = face is not None and min(face.shape[:2]) >= 32
                if has_face:
                    cv2.imwrite(str(out / "face" / f"{name}.jpg"), face)
                f.write(json.dumps({"id": name, "image": path.name, "box": [round(v, 1) for v in box],
                                    "conf": round(float(r.boxes.conf[k]), 3), "face": has_face}) + "\n")
                kept += 1
                n += 1
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(images)} images, {n} people", flush=True)
    print(f"done: {n} people from {len(images)} images -> {out}")


if __name__ == "__main__":
    main()

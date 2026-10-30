"""Train our own pest (rodent) detector (Phase 55; YOLOE found only 3% of mice in Phase 47) and measure it on the
photos it never saw.

    python tools/train_pests.py prepare      # lists from data/datasets/oi_pests; the test set from openimages_pests
    python tools/train_pests.py prepare-bw   # night (black-and-white) copies of the training photos
    python tools/train_pests.py train [--epochs 60] [--batch 16] [--scale 0.5] [--bw] [--name pests]
    python tools/train_pests.py test [--model data/models/pests/pests.pt]

Training: Open Images V6 training photos with a mouse box plus photos with a computer mouse or a kitchen and no mouse
(tools/get_pest_training.py); a tenth (fixed seed) kept apart to choose the best round. Test: the Open Images
validation and test photos (tools/get_weapon_photos.py --kind pests, 251 photos, drawings left out), never trained
on. One class, 0 = rodent (Open Images' "Mouse" is any small rodent: rats, mice, squirrels, hamsters...). Starts from
the COCO detector already here (yolo11s.pt), 640 px. --scale is Ultralytics' zoom range (0.5: 0.5x-1.5x); a larger
one shows the model more small, far-away mice (most photos are close-ups). The best round of a run goes to
data/models/pests/<name>.pt (git-ignored); the mice found in the app's way are measured by tools/pest_check.py --model.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "datasets" / "oi_pests"
TEST_SRC = ROOT / "data" / "datasets" / "openimages_pests"
TEST = ROOT / "data" / "datasets" / "oi_pests_test"
OUT = ROOT / "data" / "models" / "pests"


def prepare() -> None:
    imgs = sorted((TRAIN / "images").glob("*.jpg"))
    imgs = [p for p in imgs if (TRAIN / "labels" / f"{p.stem}.txt").is_file()]
    rng = random.Random(55)
    rng.shuffle(imgs)
    n_val = len(imgs) // 10
    (TRAIN / "val.txt").write_text("\n".join(p.as_posix() for p in sorted(imgs[:n_val])) + "\n", encoding="utf-8")
    (TRAIN / "train.txt").write_text("\n".join(p.as_posix() for p in sorted(imgs[n_val:])) + "\n", encoding="utf-8")
    truth = json.loads((TEST_SRC / "boxes.json").read_text(encoding="utf-8"))
    (TEST / "images").mkdir(parents=True, exist_ok=True)
    (TEST / "labels").mkdir(parents=True, exist_ok=True)
    tests = []
    for name, items in truth.items():
        src = TEST_SRC / name
        lines = [f"0 {(i['box'][0] + i['box'][2]) / 2:.6f} {(i['box'][1] + i['box'][3]) / 2:.6f} "
                 f"{i['box'][2] - i['box'][0]:.6f} {i['box'][3] - i['box'][1]:.6f}" for i in items if not i["depiction"]]
        if not src.is_file() or not lines:
            continue
        shutil.copyfile(src, TEST / "images" / name)
        (TEST / "labels" / f"{name[:-4]}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        tests.append((TEST / "images" / name).as_posix())
    (TRAIN / "test.txt").write_text("\n".join(sorted(tests)) + "\n", encoding="utf-8")
    (TRAIN / "data.yaml").write_text(
        f"path: {TRAIN.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: rodent\n",
        encoding="utf-8")
    print(f"train {len(imgs) - n_val}, val {n_val}, test {len(tests)} -> {TRAIN / 'data.yaml'}")


def prepare_bw() -> None:
    """Night copies of the training photos (the made-up infrared look of tools/make_night_videos.py) in
    oi_pests/bw, and data_bw.yaml training on both (the tenth kept apart is not copied)."""
    import cv2
    import numpy as np
    import zlib
    sys.path.insert(0, str(ROOT))
    from tools.make_night_videos import night
    (TRAIN / "bw" / "images").mkdir(parents=True, exist_ok=True)
    (TRAIN / "bw" / "labels").mkdir(parents=True, exist_ok=True)
    out = []
    for line in (TRAIN / "train.txt").read_text(encoding="utf-8").split():
        src = Path(line)
        dst = TRAIN / "bw" / "images" / src.name
        if not dst.exists():
            img = cv2.imread(str(src))
            cv2.imwrite(str(dst), night("ir", img, np.random.default_rng(zlib.crc32(src.name.encode()))),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        shutil.copyfile(TRAIN / "labels" / f"{src.stem}.txt", TRAIN / "bw" / "labels" / f"{src.stem}.txt")
        out.append(dst.as_posix())
    (TRAIN / "train_bw.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    (TRAIN / "data_bw.yaml").write_text(
        f"path: {TRAIN.as_posix()}\ntrain: [train.txt, train_bw.txt]\nval: val.txt\ntest: test.txt\n"
        "names:\n  0: rodent\n", encoding="utf-8")
    print(f"{len(out)} night copies -> {TRAIN / 'data_bw.yaml'}")


def train(epochs: int, batch: int, workers: int, scale: float, name: str, data: str = "data.yaml") -> None:
    from ultralytics import YOLO
    import ultralytics.engine.trainer as trainer
    trainer.check_amp = lambda model: True   # no yolo11n.pt download (see tools/train_fire.py)
    model = YOLO(str(ROOT / "yolo11s.pt"))
    model.train(data=str(TRAIN / data), epochs=epochs, imgsz=640, batch=batch, workers=workers, seed=55,
                scale=scale, patience=15, project=str(OUT / "runs"), name=name, exist_ok=True, verbose=False,
                plots=False)
    shutil.copyfile(OUT / "runs" / name / "weights" / "best.pt", OUT / f"{name}.pt")
    print(f"best round -> {OUT / f'{name}.pt'}")


def test(model_path: Path) -> None:
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    m = model.val(data=str(TRAIN / "data.yaml"), split="test", imgsz=640, batch=16, verbose=False, plots=False,
                  project=str(OUT / "runs"), name="test", exist_ok=True)
    p, r, ap50, _ = m.box.class_result(0)
    report = {"map50": round(float(ap50), 3), "map50_95": round(float(m.box.map), 3),
              "precision": round(float(p), 3), "recall": round(float(r), 3)}
    import cv2
    pic = cv2.imread(open(TRAIN / "test.txt").readline().strip())
    for _ in range(3):
        model.predict(pic, imgsz=640, verbose=False)
    t = time.perf_counter()
    for _ in range(20):
        model.predict(pic, imgsz=640, verbose=False)
    report["ms_per_picture"] = round((time.perf_counter() - t) / 20 * 1000, 1)
    (OUT / f"report_{model_path.stem}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("prepare-bw")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--scale", type=float, default=0.5)
    t.add_argument("--bw", action="store_true", help="also train on the night copies")
    t.add_argument("--name", default="pests")
    e = sub.add_parser("test")
    e.add_argument("--model", type=Path, default=OUT / "pests.pt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "prepare-bw": prepare_bw,
     "train": lambda: train(args.epochs, args.batch, args.workers, args.scale, args.name,
                            "data_bw.yaml" if args.bw else "data.yaml"),
     "test": lambda: test(args.model)}[args.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

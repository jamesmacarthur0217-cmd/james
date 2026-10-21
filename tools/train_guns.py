"""Train our own gun detector (Phase 47 step 1; the user's choice after YOLOE found only 23-36% of guns) and measure it
on the photos it never saw.

    python tools/train_guns.py prepare      # lists from data/datasets/oi_guns; the test set from openimages_weapons
    python tools/train_guns.py train [--epochs 50] [--batch 16]
    python tools/train_guns.py test [--model data/models/guns/guns.pt]

Training: Open Images V6 training photos with handgun, rifle and shotgun boxes plus photos of people without a gun
(tools/get_gun_training.py); a tenth (fixed seed) kept apart to choose the best round. Test: the Open Images
validation and test photos (tools/get_weapon_photos.py, 439 photos, drawings and toys left out), never trained on.
Classes 0 = handgun, 1 = rifle, 2 = shotgun. Starts from the COCO detector already here (yolo11s.pt), 640 px; the
best round goes to data/models/guns/guns.pt (git-ignored), the numbers to data/models/guns/report.json.
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
TRAIN = ROOT / "data" / "datasets" / "oi_guns"
TEST_SRC = ROOT / "data" / "datasets" / "openimages_weapons"
TEST = ROOT / "data" / "datasets" / "oi_guns_test"
OUT = ROOT / "data" / "models" / "guns"
KIND = {"handgun": 0, "rifle": 1, "shotgun": 2}
NAMES = {v: k for k, v in KIND.items()}


def prepare() -> None:
    imgs = sorted((TRAIN / "images").glob("*.jpg"))
    imgs = [p for p in imgs if (TRAIN / "labels" / f"{p.stem}.txt").is_file()]
    rng = random.Random(47)
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
        if not src.is_file():
            continue
        shutil.copyfile(src, TEST / "images" / name)
        lines = [f"{KIND[i['kind']]} {(i['box'][0] + i['box'][2]) / 2:.6f} {(i['box'][1] + i['box'][3]) / 2:.6f} "
                 f"{i['box'][2] - i['box'][0]:.6f} {i['box'][3] - i['box'][1]:.6f}" for i in items if not i["depiction"]]
        (TEST / "labels" / f"{name[:-4]}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        tests.append((TEST / "images" / name).as_posix())
    (TRAIN / "test.txt").write_text("\n".join(sorted(tests)) + "\n", encoding="utf-8")
    (TRAIN / "data.yaml").write_text(
        f"path: {TRAIN.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        "names:\n  0: handgun\n  1: rifle\n  2: shotgun\n", encoding="utf-8")
    print(f"train {len(imgs) - n_val}, val {n_val}, test {len(tests)} -> {TRAIN / 'data.yaml'}")


def train(epochs: int, batch: int, workers: int) -> None:
    from ultralytics import YOLO
    import ultralytics.engine.trainer as trainer
    trainer.check_amp = lambda model: True   # no yolo11n.pt download (see tools/train_fire.py)
    model = YOLO(str(ROOT / "yolo11s.pt"))
    model.train(data=str(TRAIN / "data.yaml"), epochs=epochs, imgsz=640, batch=batch, workers=workers, seed=47,
                patience=10, project=str(OUT / "runs"), name="guns", exist_ok=True, verbose=False, plots=False)
    shutil.copyfile(OUT / "runs" / "guns" / "weights" / "best.pt", OUT / "guns.pt")
    print(f"best round -> {OUT / 'guns.pt'}")


def test(model_path: Path) -> None:
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    m = model.val(data=str(TRAIN / "data.yaml"), split="test", imgsz=640, batch=16, verbose=False, plots=False,
                  project=str(OUT / "runs"), name="test", exist_ok=True)
    report = {"map50": round(float(m.box.map50), 3), "map50_95": round(float(m.box.map), 3), "classes": {}}
    for i, name in NAMES.items():
        p, r, ap50, _ = m.box.class_result(i)
        report["classes"][name] = {"precision": round(float(p), 3), "recall": round(float(r), 3),
                                   "map50": round(float(ap50), 3)}
    import cv2
    pic = cv2.imread(open(TRAIN / "test.txt").readline().strip())
    for _ in range(3):
        model.predict(pic, imgsz=640, verbose=False)
    t = time.perf_counter()
    for _ in range(20):
        model.predict(pic, imgsz=640, verbose=False)
    report["ms_per_picture"] = round((time.perf_counter() - t) / 20 * 1000, 1)
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--workers", type=int, default=4)
    e = sub.add_parser("test")
    e.add_argument("--model", type=Path, default=OUT / "guns.pt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "train": lambda: train(args.epochs, args.batch, args.workers),
     "test": lambda: test(args.model)}[args.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

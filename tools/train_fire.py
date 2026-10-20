"""Train the fire and smoke detector (Phase 46 step 1) on D-Fire and measure it on D-Fire's own test part.

    python tools/train_fire.py prepare --zip <D-Fire.zip>     # unpack into data/datasets/dfire, check, write data.yaml
    python tools/train_fire.py train [--epochs 40] [--batch 16] [--start yolo11s.pt]
    python tools/train_fire.py test [--model data/models/fire/fire_smoke.pt]

D-Fire (GAIA; github.com/gaiasd/DFireDataset; its license file: public domain, Creative Commons Zero v1.0): 21,527
pictures from surveillance cameras and the web, boxes in YOLO format, classes 0 = smoke, 1 = fire (D-Fire's order).
Downloaded with the user's OK (2026-10-20, 2.83 GB, through the browser: OneDrive refuses scripts). Unpacked into its own
folder (data/datasets/dfire, git-ignored); only pictures and .txt labels are taken from the archive.

prepare: train/ and test/ as D-Fire splits them; a tenth of train (fixed seed) is kept apart to choose the best round
(val), so the test part is only used once, at the end. train: Ultralytics from the COCO detector already on this PC
(yolo11s.pt: no download), 640 px; the best round is copied to data/models/fire/fire_smoke.pt (git-ignored, like the
other models; tools/models.py lists it). test: mAP50 and precision / recall per class on the test part, and the time per
picture; the numbers go to data/models/fire/report.json.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "datasets" / "dfire"
OUT = ROOT / "data" / "models" / "fire"
NAMES = {0: "smoke", 1: "fire"}
KEEP = {".jpg", ".jpeg", ".png", ".txt"}


def prepare(zip_path: Path) -> None:
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    unsafe = [n for n in names if n.startswith("/") or ".." in n.split("/") or ":" in n]
    if unsafe:
        sys.exit(f"unsafe paths in the archive, not unpacked: {unsafe[:3]}")
    take = [n for n in names if Path(n).suffix.lower() in KEEP]
    print(f"{len(names)} entries, {len(take)} pictures and labels taken")
    raw = DATA / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    z.extractall(raw, take)
    splits = {}
    for split in ("train", "test"):
        imgs = sorted(p for p in raw.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                      and split in [part.lower() for part in p.parts])
        splits[split] = imgs
        print(f"{split}: {len(imgs)} pictures")
    if not splits["train"] or not splits["test"]:
        sys.exit("no train/ and test/ folders found in the archive: look at its layout")
    rng = random.Random(46)
    train = splits["train"][:]
    rng.shuffle(train)
    n_val = len(train) // 10
    lists = {"val": sorted(train[:n_val]), "train": sorted(train[n_val:]), "test": splits["test"]}
    missing, boxes = 0, {0: 0, 1: 0}
    for split, imgs in lists.items():
        (DATA / f"{split}.txt").write_text("\n".join(p.as_posix() for p in imgs) + "\n", encoding="utf-8")
        for p in imgs:
            label = Path(str(p.parent).replace("images", "labels")) / (p.stem + ".txt")
            if not label.is_file():
                missing += 1
                continue
            for line in label.read_text().split("\n"):
                if line.strip():
                    boxes[int(line.split()[0])] = boxes.get(int(line.split()[0]), 0) + 1
    print(f"lists: train {len(lists['train'])}, val {len(lists['val'])}, test {len(lists['test'])}; pictures without a "
          f"label file (no fire, no smoke): {missing}; boxes: smoke {boxes.get(0, 0)}, fire {boxes.get(1, 0)}")
    (DATA / "data.yaml").write_text(
        f"path: {DATA.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        f"names:\n  0: smoke\n  1: fire\n", encoding="utf-8")
    print(f"-> {DATA / 'data.yaml'}")


def train(epochs: int, batch: int, start: str, workers: int) -> None:
    from ultralytics import YOLO
    import ultralytics.engine.trainer as trainer
    # Ultralytics checks half precision by downloading yolo11n.pt; no download without the user's OK, and the RTX 4070
    # SUPER here runs it fine (the app's own detectors already do).
    trainer.check_amp = lambda model: True
    model = YOLO(start)
    model.train(data=str(DATA / "data.yaml"), epochs=epochs, imgsz=640, batch=batch, workers=workers, seed=46,
                patience=10, project=str(OUT / "runs"), name="dfire", exist_ok=True, verbose=False, plots=False)
    best = OUT / "runs" / "dfire" / "weights" / "best.pt"
    shutil.copyfile(best, OUT / "fire_smoke.pt")
    print(f"best round -> {OUT / 'fire_smoke.pt'}")


def test(model_path: Path) -> None:
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    m = model.val(data=str(DATA / "data.yaml"), split="test", imgsz=640, batch=16, verbose=False, plots=False,
                  project=str(OUT / "runs"), name="test", exist_ok=True)
    report = {"map50": round(float(m.box.map50), 3), "map50_95": round(float(m.box.map), 3), "classes": {}}
    for i, name in NAMES.items():
        p, r, ap50, ap = m.box.class_result(i)
        report["classes"][name] = {"precision": round(float(p), 3), "recall": round(float(r), 3),
                                   "map50": round(float(ap50), 3)}
    import cv2
    pic = cv2.imread(open(DATA / "test.txt").readline().strip())
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
    p = sub.add_parser("prepare")
    p.add_argument("--zip", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--start", default=str(ROOT / "yolo11s.pt"))
    e = sub.add_parser("test")
    e.add_argument("--model", type=Path, default=OUT / "fire_smoke.pt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "prepare":
        prepare(args.zip)
    elif args.cmd == "train":
        train(args.epochs, args.batch, args.start, args.workers)
    else:
        test(args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())

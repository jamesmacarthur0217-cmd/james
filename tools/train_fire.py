"""Train the fire and smoke detector (Phase 46 step 1) on D-Fire and measure it on D-Fire's own test part.

    python tools/train_fire.py prepare --zip <D-Fire.zip>     # unpack into data/datasets/dfire, check, write data.yaml
    python tools/train_fire.py train [--epochs 40] [--batch 16] [--start yolo11s.pt]
    python tools/train_fire.py test [--model data/models/fire/fire_smoke.pt] [--night]
    python tools/train_fire.py bw                             # Phase 53: black-and-white copies, and night test copies
    python tools/train_fire.py train --bw                     # -> data/models/fire/fire_smoke_bw.pt (the old one stays)

D-Fire (GAIA; github.com/gaiasd/DFireDataset; its license file: public domain, Creative Commons Zero v1.0): 21,527
pictures from surveillance cameras and the web, boxes in YOLO format, classes 0 = smoke, 1 = fire (D-Fire's order).
Downloaded with the user's OK (2026-10-20, 2.83 GB, through the browser: OneDrive refuses scripts). Unpacked into its own
folder (data/datasets/dfire, git-ignored); only pictures and .txt labels are taken from the archive.

prepare: train/ and test/ as D-Fire splits them; a tenth of train (fixed seed) is kept apart to choose the best round
(val), so the test part is only used once, at the end. train: Ultralytics from the COCO detector already on this PC
(yolo11s.pt: no download), 640 px; the best round is copied to data/models/fire/fire_smoke.pt (git-ignored, like the
other models; tools/models.py lists it). test: mAP50 and precision / recall per class on the test part, and the time per
picture; the numbers go to data/models/fire/report.json.

Phase 53 (night; the user's choice): in infrared the model found 3 of D-Fire's 50 fire videos (39 by day): smoke in black
and white looks like fog. bw: every train and val picture gets a black-and-white copy (data/datasets/dfire/bw, its
labels beside it) with random night settings per picture (contrast, lift, darker edges, softness, grain), so the model
does not learn the one fixed look of the made-up night videos it is measured on; and the test part is copied once as
infrared and once as dark colour exactly as tools/make_night_videos.py makes the night videos (dfire/night_test).
train --bw trains on the pictures and their copies from the same start (yolo11s.pt, same seed) into
fire_smoke_bw.pt: the old model is kept, and the new one is used only if it is not worse by day. test --night
measures a model on the test part as it is, as infrared and as dark (report_<model>.json).
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


def _label_of(picture: Path) -> Path:
    return Path(str(picture.parent).replace("images", "labels")) / (picture.stem + ".txt")


def night_copy(img, rng):
    """A black-and-white night copy with random settings (for training; see the module text)."""
    import cv2
    import numpy as np
    from make_night_videos import _vignette
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = rng.uniform(0.4, 0.9)                                      # flatter contrast, some lift
    gray = k * gray + (1 - k) * float(gray.mean()) + rng.uniform(0, 30)
    if rng.random() < 0.7:                                         # an IR lamp: darker edges, more or less
        gray *= _vignette(gray.shape, rng.uniform(0.4, 0.9))
    soft = rng.uniform(0, 1.2)
    if soft > 0.3:
        gray = cv2.GaussianBlur(gray, (0, 0), soft)
    gray += rng.normal(0, rng.uniform(0, 10), gray.shape).astype(np.float32)   # grain
    g = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.merge([g, g, g])


def _copy_list(pictures: list[Path], folder: Path, make) -> list[Path]:
    """Each picture made into folder/images (make(picture, name) -> image), its label copied to folder/labels."""
    import cv2
    out = []
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(pictures, 1):
        dst = folder / "images" / (p.stem + ".jpg")
        if not dst.exists():
            img = cv2.imread(str(p))
            if img is None:
                continue
            cv2.imwrite(str(dst), make(img, p.name), [cv2.IMWRITE_JPEG_QUALITY, 90])
        label = _label_of(p)
        if label.is_file():
            shutil.copyfile(label, folder / "labels" / label.name)
        out.append(dst)
        if i % 2000 == 0:
            print(f"  {folder.name}: {i}/{len(pictures)}", flush=True)
    return out


def bw() -> None:
    """Black-and-white copies of train and val (random settings), and the test part as infrared and as dark colour
    (the night videos' fixed look); lists and data_*.yaml files beside data.yaml."""
    import zlib
    import numpy as np
    from make_night_videos import _rng, night
    read = lambda name: [Path(x) for x in (DATA / name).read_text(encoding="utf-8").split() if x]   # noqa: E731
    seeded = lambda name: np.random.default_rng(zlib.crc32(name.encode()))   # noqa: E731
    lists = {}
    for split in ("train", "val"):
        pictures = read(f"{split}.txt")
        copies = _copy_list(pictures, DATA / "bw" / split, lambda img, name: night_copy(img, seeded(name)))
        lists[f"{split}_bw"] = pictures + copies
        print(f"{split}: {len(pictures)} pictures + {len(copies)} black-and-white copies")
    for kind in ("ir", "dark"):
        lists[f"test_{kind}"] = _copy_list(read("test.txt"), DATA / "night_test" / kind,
                                           lambda img, name, kind=kind: night(kind, img, _rng(name, kind)))
        print(f"test as {kind}: {len(lists[f'test_{kind}'])} pictures")
    for name, pictures in lists.items():
        (DATA / f"{name}.txt").write_text("\n".join(p.as_posix() for p in pictures) + "\n", encoding="utf-8")
    names = "names:\n  0: smoke\n  1: fire\n"
    (DATA / "data_bw.yaml").write_text(f"path: {DATA.as_posix()}\ntrain: train_bw.txt\nval: val_bw.txt\n"
                                       f"test: test.txt\n{names}", encoding="utf-8")
    for kind in ("ir", "dark"):
        (DATA / f"data_{kind}.yaml").write_text(f"path: {DATA.as_posix()}\ntrain: train.txt\nval: val.txt\n"
                                                f"test: test_{kind}.txt\n{names}", encoding="utf-8")
    print(f"-> {DATA / 'data_bw.yaml'}, data_ir.yaml, data_dark.yaml")


def train(epochs: int, batch: int, start: str, workers: int, with_bw: bool = False) -> None:
    from ultralytics import YOLO
    import ultralytics.engine.trainer as trainer
    # Ultralytics checks half precision by downloading yolo11n.pt; no download without the user's OK, and the RTX 4070
    # SUPER here runs it fine (the app's own detectors already do).
    trainer.check_amp = lambda model: True
    data, name, target = ("data_bw.yaml", "dfire_bw", "fire_smoke_bw.pt") if with_bw else \
        ("data.yaml", "dfire", "fire_smoke.pt")
    model = YOLO(start)
    model.train(data=str(DATA / data), epochs=epochs, imgsz=640, batch=batch, workers=workers, seed=46,
                patience=10, project=str(OUT / "runs"), name=name, exist_ok=True, verbose=False, plots=False)
    best = OUT / "runs" / name / "weights" / "best.pt"
    shutil.copyfile(best, OUT / target)
    print(f"best round -> {OUT / target}")


def _measure(model, data: str) -> dict:
    m = model.val(data=str(DATA / data), split="test", imgsz=640, batch=16, verbose=False, plots=False,
                  project=str(OUT / "runs"), name="test", exist_ok=True)
    report = {"map50": round(float(m.box.map50), 3), "map50_95": round(float(m.box.map), 3), "classes": {}}
    for i, name in NAMES.items():
        p, r, ap50, ap = m.box.class_result(i)
        report["classes"][name] = {"precision": round(float(p), 3), "recall": round(float(r), 3),
                                   "map50": round(float(ap50), 3)}
    return report


def test(model_path: Path, night: bool = False) -> None:
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    report = _measure(model, "data.yaml")
    import cv2
    pic = cv2.imread(open(DATA / "test.txt").readline().strip())
    for _ in range(3):
        model.predict(pic, imgsz=640, verbose=False)
    t = time.perf_counter()
    for _ in range(20):
        model.predict(pic, imgsz=640, verbose=False)
    report["ms_per_picture"] = round((time.perf_counter() - t) / 20 * 1000, 1)
    if night:
        report["night"] = {kind: _measure(model, f"data_{kind}.yaml") for kind in ("ir", "dark")}
    name = "report.json" if model_path.name == "fire_smoke.pt" and not night else f"report_{model_path.stem}.json"
    (OUT / name).write_text(json.dumps(report, indent=1), encoding="utf-8")
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
    t.add_argument("--bw", action="store_true", help="with the black-and-white copies (run bw first)")
    e = sub.add_parser("test")
    e.add_argument("--model", type=Path, default=OUT / "fire_smoke.pt")
    e.add_argument("--night", action="store_true", help="also on the test part as infrared and as dark (run bw first)")
    sub.add_parser("bw")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "prepare":
        prepare(args.zip)
    elif args.cmd == "bw":
        bw()
    elif args.cmd == "train":
        train(args.epochs, args.batch, args.start, args.workers, args.bw)
    else:
        test(args.model, args.night)
    return 0


if __name__ == "__main__":
    sys.exit(main())

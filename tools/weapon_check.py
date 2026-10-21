"""Dangerous objects (Phase 47 step 1, labeler/weapons.py): how often is an everyday knife, pair of scissors or bat taken
for a threat, and how often does YOLOE see a gun where there is none?

    python tools/weapon_check.py coco [--limit 300] [--vlm]      # COCO photos with a person and a knife/scissors/bat
    python tools/weapon_check.py guns [--n 500]                 # YOLOE's gun, rifle, ... words on ordinary COCO photos
    python tools/weapon_check.py photos --folder <dir> [--vlm]  # a folder of photos that do show weapons (found?)

coco: the app's pose detector (wrists) and plain detector (objects, 1280 px) on each photo with a person and a knife,
scissors or a baseball bat (COCO val2017 markings); counted: photos where the object is at a wrist (labeler/weapons.py
at_wrist), and with --vlm the vision model's answer to the weapon question about the part around that person (the
app's own prompt): COCO shows cooking, eating, crafts and baseball, so every "yes" is a false alarm. guns: YOLOE
prompt-free on ordinary COCO photos (no guns marked in COCO): photos with a gun-like word. photos: any folder of
pictures showing weapons: found at a wrist, found at all, and the vision model's answers.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeler.config import load_config  # noqa: E402
from labeler.incidents import CATEGORIES, person_crop  # noqa: E402
from labeler.weapons import COCO_WEAPONS, YOLOE_WORDS, at_wrist  # noqa: E402

COCO = ROOT / "data" / "datasets" / "coco"


def models(cfg):
    from ultralytics import YOLO, YOLOE
    pose = YOLO(str(ROOT / cfg["detector"]["model"]))
    plain = YOLO(str(ROOT / (cfg.get("vehicles") or {}).get("model", "yolo11s.pt")))
    yoloe = YOLOE(str(ROOT / cfg["safety"]["weapon_model"]))
    ids = [i for i, n in yoloe.names.items() if n.lower() in YOLOE_WORDS]
    return pose, plain, yoloe, ids


def look(img, pose, plain, yoloe, ids, cfg):
    """(people [(box, kpts)], objects [(word, conf, box)]) as the camera would see them in one look."""
    s = cfg["safety"]
    rp = pose.predict(img, conf=cfg["detector"]["conf"], imgsz=cfg["detector"]["imgsz"], verbose=False)[0]
    people = [(tuple(b), k) for b, k in zip(rp.boxes.xyxy.tolist(), rp.keypoints.data.cpu().numpy())] \
        if rp.keypoints is not None else []
    ro = plain.predict(img, classes=list(COCO_WEAPONS), conf=s["weapon_conf"], imgsz=s["crowd_imgsz"], verbose=False)[0]
    objs = [(COCO_WEAPONS[int(c)], float(p), tuple(b)) for c, p, b in
            zip(ro.boxes.cls.tolist(), ro.boxes.conf.tolist(), ro.boxes.xyxy.tolist())]
    re = yoloe.predict(img, classes=ids, conf=s["weapon_conf"], imgsz=640, verbose=False)[0]
    objs += [(yoloe.names[int(c)].lower(), float(p), tuple(b)) for c, p, b in
             zip(re.boxes.cls.tolist(), re.boxes.conf.tolist(), re.boxes.xyxy.tolist())]
    return people, objs


def held(people, objs, reach):
    for word, conf, box in objs:
        for pbox, k in people:
            if at_wrist(box, k, pbox, reach):
                return word, pbox
    return None


def ask(client, img, pbox):
    from labeler.vlm import incident_prompt
    h, w = img.shape[:2]
    crop = person_crop(img, (pbox[0] / w, pbox[1] / h, pbox[2] / w, pbox[3] / h))
    return client.label([crop], None, ("incident", incident_prompt(1, 0, "", CATEGORIES["weapon"][2])))


def run_photos(paths, cfg, use_vlm):
    pose, plain, yoloe, ids = models(cfg)
    client = None
    if use_vlm:
        from labeler.vlm import VLMClient
        client = VLMClient(cfg["vlm"], "security", cfg["labels"])
    n = at_all = at_hand = 0
    words, answers, examples = Counter(), Counter(), []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        n += 1
        people, objs = look(img, pose, plain, yoloe, ids, cfg)
        at_all += bool(objs)
        got = held(people, objs, cfg["safety"]["weapon_reach"])
        if not got:
            continue
        at_hand += 1
        words[got[0]] += 1
        if client is not None:
            a = ask(client, img, got[1])
            answers[a["answer"]] += 1
            if a["answer"] == "yes":
                examples.append(f"{path.stem} ({got[0]}: {a['what']})")
    print(f"{n} photos; any object found {at_all}; at a wrist {at_hand} {dict(words)}")
    if client is not None:
        print(f"vision model on those at a wrist: {dict(answers)}")
        for e in examples[:15]:
            print("  yes:", e)


def coco_photos(classes: set[str], need_person: bool, limit: int, seed: int = 47) -> list[Path]:
    out = []
    for f in sorted((COCO / "labels" / "val2017").glob("*.txt")):
        c = {r.split()[0] for r in f.read_text().split("\n") if r.strip()}
        if (c & classes) and (not need_person or "0" in c):
            out.append(COCO / "val2017" / f"{f.stem}.jpg")
    random.Random(seed).shuffle(out)
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["coco", "guns", "photos"])
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--folder", type=Path)
    ap.add_argument("--vlm", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, use_settings=False)
    if args.what == "coco":
        run_photos(coco_photos({str(c) for c in COCO_WEAPONS}, True, args.limit), cfg, args.vlm)
    elif args.what == "guns":
        _, _, yoloe, ids = models(cfg)
        photos = [p for p in sorted((COCO / "val2017").glob("*.jpg"))]
        random.Random(47).shuffle(photos)
        hits = Counter()
        for p in photos[: args.n]:
            r = yoloe.predict(str(p), classes=ids, conf=cfg["safety"]["weapon_conf"], imgsz=640, verbose=False)[0]
            for c in {yoloe.names[int(c)].lower() for c in r.boxes.cls.tolist()}:
                hits[c] += 1
            if len(r.boxes):
                hits["photos"] += 1
        print(f"{args.n} ordinary COCO photos: with a gun-like word {hits.pop('photos', 0)}; words {dict(hits)}")
    else:
        if not args.folder:
            ap.error("photos needs --folder")
        run_photos(sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                   cfg, args.vlm)
    return 0


if __name__ == "__main__":
    sys.exit(main())

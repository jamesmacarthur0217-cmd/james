"""Dangerous objects (Phase 47 step 1, labeler/weapons.py): how often is an everyday knife, pair of scissors or bat taken
for a threat, and how often does YOLOE see a gun where there is none?

    python tools/weapon_check.py coco [--limit 300] [--vlm]      # COCO photos with a person and a knife/scissors/bat
    python tools/weapon_check.py people [--limit 300] [--vlm]    # ordinary COCO photos with people, no weapon
    python tools/weapon_check.py guns [--n 500]                 # YOLOE's gun, rifle, ... words on ordinary COCO photos
    python tools/weapon_check.py photos --folder <dir> [--vlm]  # a folder of photos that do show weapons (found?)
    python tools/weapon_check.py openimages [--vlm]             # Open Images guns with their true boxes

coco: the app's pose detector (wrists) and plain detector (objects, 1280 px) on each photo with a person and a knife,
scissors or a baseball bat (COCO val2017 markings); counted: photos where the object is at a wrist (labeler/weapons.py
at_wrist), and with --vlm the vision model's answer to the weapon question about the part around that person (the
app's own prompt): COCO shows cooking, eating, crafts and baseball, so every "yes" is a false alarm. guns: YOLOE
prompt-free on ordinary COCO photos (no guns marked in COCO): photos with a gun-like word. photos: any folder of
pictures showing weapons: found at a wrist, found at all, and the vision model's answers. openimages: the handgun,
rifle and shotgun photos of tools/get_weapon_photos.py with their true boxes: guns found at all (a YOLOE gun-like box
overlapping a true one, IoU >= 0.3), photos where a person really holds one (a wrist inside a true box, grown as the
app grows its own), of those the ones the app also has at a wrist, and the vision model's answers there.
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
from labeler.weapons import COCO_WEAPONS, GUN_CLASSES, GUNS, YOLOE_WORDS, at_wrist  # noqa: E402

COCO = ROOT / "data" / "datasets" / "coco"
GUN_MODEL: list = []   # our own gun detector, loaded by models()


def models(cfg):
    from ultralytics import YOLO, YOLOE
    pose = YOLO(str(ROOT / cfg["detector"]["model"]))
    plain = YOLO(str(ROOT / (cfg.get("vehicles") or {}).get("model", "yolo11s.pt")))
    yoloe = YOLOE(str(ROOT / cfg["safety"]["weapon_model"]))
    GUN_MODEL.append(YOLO(str(ROOT / cfg["safety"]["gun_model"])))   # our own gun detector
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
    rg = GUN_MODEL[-1].predict(img, conf=s["gun_conf"], imgsz=640, verbose=False)[0]
    objs += [(GUN_CLASSES.get(int(c), "gun"), float(p), tuple(b)) for c, p, b in
             zip(rg.boxes.cls.tolist(), rg.boxes.conf.tolist(), rg.boxes.xyxy.tolist())]
    return people, objs


def held(people, objs, reach):
    for word, conf, box in objs:
        for pbox, k in people:
            if at_wrist(box, k, pbox, reach):
                return word, pbox
    return None


def ask(client, img, pbox, word: str = ""):
    from labeler.vlm import incident_prompt
    from labeler.weapons import GUN_QUESTION, GUNS
    h, w = img.shape[:2]
    crop = person_crop(img, (pbox[0] / w, pbox[1] / h, pbox[2] / w, pbox[3] / h))
    question = GUN_QUESTION if word in GUNS else CATEGORIES["weapon"][2]   # as the camera asks it
    return client.label([crop], None, ("incident", incident_prompt(1, 0, "", question)))


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
            a = ask(client, img, got[1], got[0])
            answers[a["answer"]] += 1
            if a["answer"] == "yes":
                examples.append(f"{path.stem} ({got[0]}: {a['what']})")
    print(f"{n} photos; any object found {at_all}; at a wrist {at_hand} {dict(words)}")
    if client is not None:
        print(f"vision model on those at a wrist: {dict(answers)}")
        for e in examples[:15]:
            print("  yes:", e)


def iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def run_openimages(cfg, use_vlm, folder: Path) -> None:
    import json
    pose, plain, yoloe, ids = models(cfg)
    client = None
    if use_vlm:
        from labeler.vlm import VLMClient
        client = VLMClient(cfg["vlm"], "security", cfg["labels"])
    truth = json.loads((folder / "boxes.json").read_text(encoding="utf-8"))
    reach = cfg["safety"]["weapon_reach"]
    guns = found = photos_held = app_held = 0
    answers, yes_words, no_words = Counter(), [], []
    for name, items in truth.items():
        img = cv2.imread(str(folder / name))
        if img is None:
            continue
        h, w = img.shape[:2]
        real = [[b * s for b, s in zip(i["box"], (w, h, w, h))] for i in items if not i["depiction"]]
        people, objs = look(img, pose, plain, yoloe, ids, cfg)
        guns_found = [o for o in objs if o[0] in GUNS]
        guns += len(real)
        found += sum(any(iou(r, o[2]) >= 0.3 for o in guns_found) for r in real)
        holder = next((pbox for pbox, k in people for r in real if at_wrist(r, k, pbox, reach)), None)
        if holder is None:
            continue
        photos_held += 1
        got = held(people, objs, reach)
        if not got:
            continue
        app_held += 1
        if client is not None:
            a = ask(client, img, got[1], got[0])
            answers[a["answer"]] += 1
            (yes_words if a["answer"] == "yes" else no_words).append(f"{name[:-4]}: {a['what']}")
    print(f"{len(truth)} photos, {guns} true guns: found by the gun detector {found} ({found / max(guns, 1):.0%})")
    print(f"photos where a person really holds one: {photos_held}; the app has one at a wrist: {app_held} "
          f"({app_held / max(photos_held, 1):.0%})")
    if client is not None:
        print(f"vision model on those: {dict(answers)}")
        for line in yes_words[:6] + no_words[:10]:
            print("  ", line)


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
    ap.add_argument("what", choices=["coco", "people", "guns", "photos", "openimages"])
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--folder", type=Path)
    ap.add_argument("--vlm", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, use_settings=False)
    if args.what == "coco":
        run_photos(coco_photos({str(c) for c in COCO_WEAPONS}, True, args.limit), cfg, args.vlm)
    elif args.what == "people":   # ordinary COCO photos with people and no knife, scissors or bat: every alert is false
        weapons = {str(c) for c in COCO_WEAPONS}
        photos = [p for p in coco_photos({"0"}, True, 5000)
                  if not weapons & {r.split()[0] for r in (COCO / "labels" / "val2017" / f"{p.stem}.txt").read_text().split("\n") if r.strip()}]
        run_photos(photos[: args.limit], cfg, args.vlm)
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
    elif args.what == "openimages":
        run_openimages(cfg, args.vlm, args.folder or ROOT / "data" / "datasets" / "openimages_weapons")
    else:
        if not args.folder:
            ap.error("photos needs --folder")
        run_photos(sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                   cfg, args.vlm)
    return 0


if __name__ == "__main__":
    sys.exit(main())

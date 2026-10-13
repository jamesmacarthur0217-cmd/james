"""Measure the mobility-aid label (wheelchair, walking stick, crutches, walker).

    python tools/mobility_check.py scan                 # the VLM on every COCO person (cached); who got an aid
    python tools/mobility_check.py find                 # likely real aids by text search (SigLIP 2), to check by eye
    python tools/mobility_check.py speed --n 60         # seconds per person without and with the question
    python tools/mobility_check.py score                # against mobility_truth.json (checked by eye)
    python tools/mobility_check.py --set openimages crops    # people in the Open Images aid photos (get_aid_photos.py)
    python tools/mobility_check.py --set openimages scan / score

Two sets: `coco` (data/datasets/coco_people: 1,534 people, almost none with an aid: false alarms) and `openimages`
(data/datasets/openimages_aids: photos with a wheelchair or crutch, incl. helpers pushing one: caught and missed).
`scan`, `find` and `crops` write contact sheets (numbered pictures) with --sheet, for checking by eye. The truth
file (<set>/mobility_truth.json) maps person ids to the aid really in use ("none" = no aid, "?" = can't tell:
left out); a person missing from it counts as "none".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labeler import schema  # noqa: E402
from labeler.config import load_config  # noqa: E402
from labeler.vlm import PROMPT_KIOSK, PROMPT_MOBILITY, VLMClient  # noqa: E402

SETS = {"coco": ROOT / "data" / "datasets" / "coco_people", "openimages": ROOT / "data" / "datasets" / "openimages_aids"}
PEOPLE = SETS["coco"]   # set by --set
AIDS = schema.LIST_FIELDS["mobility_aid"]
QUERIES = {"wheelchair": "a photo of a person sitting in a wheelchair",
           "walking stick": "a photo of a person walking with a walking stick",
           "crutches": "a photo of a person walking on crutches",
           "walker": "a photo of an elderly person with a walking frame"}
ON, OFF = {"mobility_aids": True}, {"mobility_aids": False}


def people() -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, (PEOPLE / "people.jsonl").read_text("utf-8").splitlines())}


def images(pid: str, meta: dict) -> tuple[np.ndarray, np.ndarray | None]:
    person = cv2.imread(str(PEOPLE / "person" / f"{pid}.jpg"))
    face = cv2.imread(str(PEOPLE / "face" / f"{pid}.jpg")) if meta[pid]["face"] else None
    return person, face


def client(labels: dict) -> VLMClient:
    return VLMClient(load_config()["vlm"], "kiosk", labels)


def cache_path() -> Path:
    """Answers are kept per prompt and answer format: changing either gets fresh answers."""
    asked = PROMPT_KIOSK + PROMPT_MOBILITY + json.dumps(schema.json_schema("kiosk", ON), sort_keys=True)
    return PEOPLE / f"mobility-{hashlib.sha1(asked.encode()).hexdigest()[:6]}.jsonl"


def scanned() -> dict[str, dict]:
    path = cache_path()
    return {r["id"]: r for r in map(json.loads, path.read_text("utf-8").splitlines())} if path.exists() else {}


def sheet(ids: list[str], path: Path, captions: dict[str, str] | None = None, h: int = 200, per_row: int = 8) -> None:
    """Numbered pictures in a grid; the number is the position in `ids` (printed with the id)."""
    tiles = []
    for n, pid in enumerate(ids, 1):
        img = cv2.imread(str(PEOPLE / "person" / f"{pid}.jpg"))
        img = cv2.resize(img, (max(1, int(img.shape[1] * h / img.shape[0])), h))
        img = cv2.copyMakeBorder(img, 0, 0, 0, max(0, 150 - img.shape[1]), cv2.BORDER_CONSTANT, value=(40, 40, 40))
        img = img[:, :max(150, min(img.shape[1], 260))]
        text = captions.get(pid, str(n)) if captions else str(n)
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(img, text[:24], (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    rows = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        rows.append(np.hstack(row))
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    print(f"sheet: {path}")


def scan(args) -> None:
    meta, done = people(), scanned()
    ids = [p for p in meta if p.startswith(args.prefix)]
    ids = ids[:args.limit] if args.limit else ids
    vlm = client(ON)
    with open(cache_path(), "a", encoding="utf-8") as f:
        for k, pid in enumerate(ids, 1):
            if pid in done:
                continue
            t = time.perf_counter()
            labels = vlm.label(*images(pid, meta))
            done[pid] = {"id": pid, "labels": labels, "s": round(time.perf_counter() - t, 3)}
            f.write(json.dumps(done[pid]) + "\n")
            f.flush()
            if k % 100 == 0:
                print(f"{k}/{len(ids)}", flush=True)
    rows = [done[p] for p in ids if p in done]
    flagged = [r["id"] for r in rows if r["labels"].get("mobility_aid", "none") != "none"]
    print(f"{len(rows)} people labeled, median {statistics.median(r['s'] for r in rows):.2f} s per person")
    for aid in AIDS:
        print(f"  {aid:14s} {sum(aid in r['labels'].get('mobility_aid', '').split(',') for r in rows)}")
    print(f"any aid: {len(flagged)}")
    for n, pid in enumerate(flagged, 1):
        print(f"  {n:3d} {pid} {done[pid]['labels']['mobility_aid']}")
    if args.sheet and flagged:
        sheet(flagged, Path(args.sheet), {p: f"{n} {done[p]['labels']['mobility_aid']}" for n, p in enumerate(flagged, 1)})


def find(args) -> None:
    from labeler.embed import Embedder
    data = np.load(PEOPLE / "embeddings.npz")
    ids, vectors = [str(i) for i in data["ids"]], data["vectors"]
    emb = Embedder(load_config()["embeddings"])
    done = scanned()
    for aid, query in QUERIES.items():
        q = emb.text(query)
        order = np.argsort(-(vectors @ q))[:args.top]
        top = [ids[i] for i in order]
        print(f"\n{aid}: top {args.top} for {query!r}")
        for n, pid in enumerate(top, 1):
            answer = done.get(pid, {}).get("labels", {}).get("mobility_aid", "-")
            print(f"  {n:3d} {pid} vlm={answer}")
        if args.sheet:
            stem = Path(args.sheet)
            sheet(top, stem.with_name(f"{stem.stem}-{aid.replace(' ', '_')}{stem.suffix}"))


def speed(args) -> None:
    meta = people()
    ids = list(meta)[args.offset:args.offset + args.n]
    clients = {"without": client(OFF), "with": client(ON)}
    clients["with"].label(*images(ids[0], meta))   # warm-up: the model is loaded
    secs: dict[str, list[float]] = {k: [] for k in clients}
    # One whole pass each: asked twice in a row about the same picture, Ollama reuses its work on the picture,
    # and the second request looks much faster (measured: 2.07 s then 1.56 s whichever prompt came second).
    for name, c in clients.items():
        for pid in ids:
            t = time.perf_counter()
            c.label(*images(pid, meta))
            secs[name].append(time.perf_counter() - t)
    for name, s in secs.items():
        print(f"{name:8s} the question: median {statistics.median(s):.2f} s, mean {statistics.mean(s):.2f} s "
              f"({len(s)} people)")


def crops(args) -> None:
    """Every person in the Open Images aid photos, cut out like the app does (body; face from the body points)."""
    from ultralytics import YOLO
    from labeler import geometry
    boxes = json.loads((PEOPLE / "boxes.json").read_text("utf-8"))
    model = YOLO(str(ROOT / "yolo11s-pose.pt"))
    (PEOPLE / "person").mkdir(exist_ok=True)
    (PEOPLE / "face").mkdir(exist_ok=True)
    rows = []
    for name, aids in boxes.items():
        frame = cv2.imread(str(PEOPLE / name))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        result = model(frame, classes=[0], conf=0.4, imgsz=960, verbose=False)[0]
        kpts_all = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
        for i, xyxy in enumerate(result.boxes.xyxy.cpu().numpy()):
            box = tuple(float(v) for v in xyxy)
            if box[3] - box[1] < 120:
                continue   # too small to see an aid
            # Which aid box this person overlaps most (a hint for checking by eye, not the truth).
            hint, best = "none", 0.0
            for a in aids:
                ab = (a["box"][0] * w, a["box"][1] * h, a["box"][2] * w, a["box"][3] * h)
                ix = max(0.0, min(box[2], ab[2]) - max(box[0], ab[0]))
                iy = max(0.0, min(box[3], ab[3]) - max(box[1], ab[1]))
                share = ix * iy / max(1.0, (ab[2] - ab[0]) * (ab[3] - ab[1]))
                if share > best:
                    hint, best = a["aid"], share
            pid = f"{Path(name).stem}_{i}"
            person = geometry.crop(frame, box, pad=0.05)
            kpts = kpts_all[i] if kpts_all is not None else None
            face = geometry.crop(frame, geometry.face_box(kpts), pad=0.1)
            cv2.imwrite(str(PEOPLE / "person" / f"{pid}.jpg"), person)
            if face is not None:
                cv2.imwrite(str(PEOPLE / "face" / f"{pid}.jpg"), face)
            rows.append({"id": pid, "image": name, "box": [round(v, 1) for v in box], "face": face is not None,
                         "hint": hint if best >= 0.3 else "none", "overlap": round(best, 2)})
    (PEOPLE / "people.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} people in {len(boxes)} photos; overlapping an aid box: "
          f"{sum(r['hint'] != 'none' for r in rows)}")
    if args.sheet:
        for k in range(0, len(rows), 48):
            part = rows[k:k + 48]
            sheet([r["id"] for r in part], Path(args.sheet).with_name(f"{Path(args.sheet).stem}-{k // 48 + 1}.jpg"),
                  {r["id"]: f"{k + n} {r['hint']}" for n, r in enumerate(part, 1)})


def score(args) -> None:
    truth = json.loads((PEOPLE / "mobility_truth.json").read_text("utf-8"))
    done = {p: r for p, r in scanned().items() if p.startswith(args.prefix)}
    rows = [r for r in done.values() if truth.get(r["id"]) != "?"]
    caught = {aid: [0, 0] for aid in AIDS}   # aid -> [found, really there]
    false = []
    for r in rows:
        said = set(r["labels"].get("mobility_aid", "none").split(",")) - {"none", "unknown"}
        real = set(truth.get(r["id"], "none").split(",")) - {"none"}
        for aid in real:
            caught[aid][1] += 1
            caught[aid][0] += aid in said
        false += [(r["id"], aid) for aid in said - real]
    print(f"{len(rows)} people scanned ({len(done) - len(rows)} unclear left out), "
          f"{sum(v[1] for v in caught.values())} real aids in the truth file")
    for aid, (found, real) in caught.items():
        print(f"  {aid:14s} found {found} of {real}")
    print(f"false alarms: {len(false)} ({len({p for p, _ in false})} people of {len(rows)})")
    for pid, aid in false:
        print(f"  {pid} {aid}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=SETS, default="coco")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--prefix", default="", help="only people whose id starts with this (e.g. validation)")
    p.add_argument("--sheet", default="")
    p = sub.add_parser("find")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--sheet", default="")
    p = sub.add_parser("speed")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--offset", type=int, default=0)
    p = sub.add_parser("score")
    p.add_argument("--prefix", default="")
    p = sub.add_parser("crops")
    p.add_argument("--sheet", default="")
    args = ap.parse_args()
    global PEOPLE
    PEOPLE = SETS[args.set]
    {"scan": scan, "find": find, "speed": speed, "score": score, "crops": crops}[args.cmd](args)


if __name__ == "__main__":
    main()

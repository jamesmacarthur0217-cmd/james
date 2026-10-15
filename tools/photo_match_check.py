"""Measure search by photo against true identities (Phase 33): how alike the same person and different people are.

    python tools/photo_match_check.py --config <test_config.yaml> [--json <file>]

On the MOT16 videos with their marked people (gt.txt: who is who in every frame): for each person seen clearly
(at least 120 px tall, at least 80% visible) in two moments at least 5 s apart, the first moment's picture is the
"photo" and the other moments' pictures (averaged, like a recorded person's vector: Embedder.track_vector) the
"recorded person". Every photo is compared with every recorded person of the same video (the app's appearance model,
SigLIP 2; cosine similarity). Printed per similarity limit: how many photos keep their own person (found), how many
other people stay listed (false), and how often the right person is first.

The limit is chosen on MOT16-02, 04 and 05 (tune) and reported on 09, 10, 11 and 13 (test).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeler.config import load_config, load_dotenv  # noqa: E402

MOT = ROOT / "data" / "datasets" / "mot16" / "MOT16" / "train"
TUNE, TEST = ["MOT16-02", "MOT16-04", "MOT16-05"], ["MOT16-09", "MOT16-10", "MOT16-11", "MOT16-13"]
LIMITS = [round(x, 2) for x in np.arange(0.70, 0.96, 0.01)]


def people(seq: str, min_h: int = 120, gap_s: float = 5.0, per_person: int = 4) -> dict[int, dict]:
    """person -> {"photo": (frame, box), "recorded": [(frame, box), ...]} for clearly seen people."""
    info = dict(line.strip().split("=") for line in open(MOT / seq / "seqinfo.ini") if "=" in line)
    fps = float(info["frameRate"])
    seen = defaultdict(list)
    for line in open(MOT / seq / "gt" / "gt.txt"):
        f = line.split(",")
        if int(f[7]) == 1 and int(f[6]) == 1 and float(f[8]) >= 0.8 and float(f[5]) >= min_h:
            x, y, w, h = map(float, f[2:6])
            seen[int(f[1])].append((int(f[0]), (x, y, x + w, y + h)))
    out = {}
    for pid, rows in seen.items():
        rows.sort()
        first = rows[0]
        later = [r for r in rows if (r[0] - first[0]) / fps >= gap_s]
        if later:
            step = max(1, len(later) // per_person)
            out[pid] = {"photo": first, "recorded": later[::step][:per_person]}
    return out


def crop(seq: str, frame: int, box: tuple, cache: dict) -> np.ndarray:
    if frame not in cache:
        cache.clear()
        cache[frame] = cv2.imread(str(MOT / seq / "img1" / f"{frame:06d}.jpg"))
    img = cache[frame]
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    return img[y1:y2, x1:x2]


def similarities(seq: str, embedder) -> tuple[list[int], np.ndarray]:
    """(people, matrix photo x recorded person) for one video."""
    ppl = people(seq)
    ids = sorted(ppl)
    cache: dict = {}
    photos = embedder.images([crop(seq, *ppl[i]["photo"], cache) for i in ids])
    recorded = np.vstack([embedder.track_vector([crop(seq, *r, cache) for r in ppl[i]["recorded"]]) for i in ids])
    return ids, photos @ recorded.T


def score(mats: list[np.ndarray], limit: float) -> dict:
    found = false = first = photos = others = 0
    for m in mats:
        for i in range(len(m)):
            photos += 1
            others += len(m) - 1
            found += m[i, i] >= limit
            false += int((np.delete(m[i], i) >= limit).sum())
            first += int(np.argmax(m[i]) == i)
    return {"limit": limit, "photos": int(photos), "found": int(found), "false": int(false), "first": int(first),
            "others": int(others)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    load_dotenv()
    from labeler.embed import Embedder
    embedder = Embedder(load_config(args.config)["embeddings"])
    mats = {seq: similarities(seq, embedder)[1] for seq in TUNE + TEST}
    out = {}
    for half, seqs in (("tune", TUNE), ("test", TEST)):
        rows = [score([mats[s] for s in seqs], lim) for lim in LIMITS]
        out[half] = rows
        n = rows[0]["photos"]
        print(f"\n{half}: {', '.join(seqs)} ({n} people; right person first for {rows[0]['first']})")
        print(f"  {'limit':>6s} {'found':>12s} {'others kept':>22s}")
        for r in rows:
            print(f"  {r['limit']:>6.2f} {r['found']:>5d} of {n:<4d} {r['false']:>7d} of {r['others']:<6d}"
                  f" ({100 * r['false'] / max(1, r['others']):.0f}%)")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

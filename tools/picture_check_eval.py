"""Measure the camera picture check (Phase 30, labeler/picture_check.py) on pictures made worse on purpose.

    python tools/picture_check_eval.py [--coco 400] [--mot 20] [--tune] [--json <file>]

Pictures: COCO val2017 photos with a person at least a tenth of the picture high, and MOT16 street frames (both with
their marked person boxes). Each is checked as it is and made worse:
  blur      Gaussian blur, sigma 2.5 px at 640 px wide          -> should say "blurry"
  dark      brightness x 0.2, with sensor noise                 -> "dark"
  bright    brightness x 1.6 + 90 (washed out)                  -> "bright"
  backlight people x 0.35, the rest x 1.3 + 70                  -> "backlight"
  covered   nearly black with noise (a lens cover)              -> "covered"
and made only a little worse, which should still pass (no finding): slight blur (sigma 1), dim (x 0.55).
The picture as it is should pass too (some COCO photos really are dark or blurry: such "false" findings are partly
right; shown apart).

--tune picks each limit on the even-numbered pictures (the most caught with at most --max-false, 1%, of the passing
pictures flagged) and prints them; without it the limits in picture_check.LIMITS are used. Results are always reported on the
odd-numbered pictures only (never used for tuning).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeler import picture_check as pc  # noqa: E402
from labeler.sources import is_blank  # noqa: E402

COCO = ROOT / "data" / "datasets" / "coco"
MOT = ROOT / "data" / "datasets" / "mot16" / "MOT16" / "train"
BAD = {"blur": "blurry", "dark": "dark", "bright": "bright", "backlight": "backlight", "covered": "covered"}
MILD = ["original", "slight_blur", "dim"]


def pictures(n_coco: int, n_mot: int, seed: int = 30) -> list[dict]:
    out = []
    names = sorted(p.stem for p in (COCO / "labels" / "val2017").glob("*.txt"))
    random.Random(seed).shuffle(names)
    for name in names:
        rows = [line.split() for line in open(COCO / "labels" / "val2017" / f"{name}.txt")]
        people = [tuple(map(float, r[1:])) for r in rows if r[0] == "0"]
        if not any(h >= 0.1 for _, _, _, h in people):
            continue
        out.append({"src": "coco", "path": str(COCO / "val2017" / f"{name}.jpg"), "norm": people})
        if len(out) >= n_coco:
            break
    for seq in sorted(p.name for p in MOT.iterdir()):
        info = dict(line.strip().split("=") for line in open(MOT / seq / "seqinfo.ini") if "=" in line)
        n, w, h = int(info["seqLength"]), float(info["imWidth"]), float(info["imHeight"])
        boxes: dict[int, list] = {}
        for line in open(MOT / seq / "gt" / "gt.txt"):
            f = line.split(",")
            if int(f[7]) == 1 and float(f[8]) >= 0.5:
                x, y, bw, bh = map(float, f[2:6])
                boxes.setdefault(int(f[0]), []).append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
        for frame in np.linspace(1, n, n_mot, dtype=int):
            out.append({"src": "mot16", "path": str(MOT / seq / "img1" / f"{frame:06d}.jpg"),
                        "norm": boxes.get(int(frame), [])})
    return out


def worse(img: np.ndarray, kind: str, boxes: list, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    s = img.shape[1] / pc.WIDTH
    noise = lambda sd: rng.normal(0, sd, img.shape).astype(np.float32)   # noqa: E731
    if kind == "blur":
        f = cv2.GaussianBlur(f, (0, 0), 2.5 * s)
    elif kind == "slight_blur":
        f = cv2.GaussianBlur(f, (0, 0), 1.0 * s)
    elif kind == "dark":
        f = f * 0.2 + noise(3)
    elif kind == "dim":
        f = f * 0.55 + noise(2)
    elif kind == "bright":
        f = f * 1.6 + 90
    elif kind == "backlight":
        mask = np.zeros(img.shape[:2], bool)
        for x1, y1, x2, y2 in boxes:
            mask[int(y1):int(y2), int(x1):int(x2)] = True
        f = np.where(mask[..., None], f * 0.35, f * 1.3 + 70)
    elif kind == "covered":
        f = np.full_like(f, 12) + noise(2)
    return np.clip(f, 0, 255).astype(np.uint8)


def measure(items: list[dict]) -> list[dict]:
    rows = []
    rng = np.random.default_rng(30)
    for k, it in enumerate(items):
        img = cv2.imread(it["path"])
        h, w = img.shape[:2]
        boxes = [((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h) for cx, cy, bw, bh in it["norm"]]
        for kind in MILD + list(BAD):
            pic = img if kind == "original" else worse(img, kind, boxes, rng)
            rows.append({"i": k, "src": it["src"], "kind": kind, "blank": bool(is_blank(pic)), **pc.stats(pic, boxes)})
    return rows


def tune(rows: list[dict], max_false: float = 0.01) -> dict:
    """Each limit from the even-numbered pictures: the most of its condition caught, at most max_false of the
    passing pictures (as they are, slightly blurred, dim) flagged."""
    ok = [r for r in rows if r["kind"] in MILD]
    limits = dict(pc.LIMITS)

    def best(key, kind, below=True, cands=None, extra=lambda r: True):
        bad = [r for r in rows if r["kind"] == kind and extra(r)]
        good = [r for r in ok if extra(r)]
        top = None
        for c in cands:
            hit = (lambda v: v < c) if below else (lambda v: v > c)
            false = sum(hit(r[key]) for r in good) / max(1, len(good))
            caught = sum(hit(r[key]) for r in bad) / max(1, len(bad))
            if false <= max_false and (top is None or caught > top[1]):
                top = (c, caught)
        return top[0] if top else limits[key]

    limits["dark_mean"] = best("mean", "dark", True, np.arange(10, 90, 1.0))
    limits["dark_share"] = best("dark_share", "dark", False, np.arange(0.3, 1.0, 0.02))
    limits["bright_share"] = best("bright_share", "bright", False, np.arange(0.05, 0.9, 0.01))
    limits["blur_ratio"] = best("blur_ratio", "blur", True, np.round(np.arange(0.002, 0.2, 0.002), 3))
    with_people = lambda r: "people_mean" in r   # noqa: E731
    ratio = lambda r: r["people_mean"] / max(1.0, r["background_mean"])   # noqa: E731
    for r in rows:
        if with_people(r):
            r["backlight_ratio"] = ratio(r)
    limits["backlight_ratio"] = best("backlight_ratio", "backlight", True, np.arange(0.2, 0.9, 0.01), with_people)
    return {k: float(round(v, 3)) for k, v in limits.items()}


def report(rows: list[dict], limits: dict) -> list[dict]:
    out = []
    for kind in MILD + list(BAD):
        for src in ("coco", "mot16"):
            sel = [r for r in rows if r["kind"] == kind and r["src"] == src]
            found = [pc.findings(r, r["blank"], limits) for r in sel]
            want = BAD.get(kind)
            row = {"kind": kind, "src": src, "pictures": len(sel)}
            if want:
                row["caught"] = sum(want in f for f in found)
                row["other_findings"] = sum(bool(set(f) - {want}) for f in found)
            else:
                row["flagged"] = sum(bool(f) for f in found)
                for code in ("dark", "bright", "blurry", "backlight", "covered"):
                    row[code] = sum(code in f for f in found)
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", type=int, default=400)
    ap.add_argument("--mot", type=int, default=20, help="frames per MOT16 video")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--max-false", type=float, default=0.01, help="tuning: share of passing pictures that may be flagged")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    items = pictures(args.coco, args.mot)
    print(f"{len(items)} pictures", flush=True)
    rows = measure(items)
    limits = tune([r for r in rows if r["i"] % 2 == 0], args.max_false) if args.tune else dict(pc.LIMITS)
    if args.tune:
        print("limits from the even-numbered pictures:", json.dumps(limits))
    test = [r for r in rows if r["i"] % 2 == 1]
    result = report(test, limits)
    print("\nodd-numbered pictures (not used for tuning):")
    for r in result:
        if "caught" in r:
            print(f"  {r['kind']:12s} {r['src']:6s} caught {r['caught']}/{r['pictures']}"
                  f"  (other findings too: {r['other_findings']})")
        else:
            detail = ", ".join(f"{c} {r[c]}" for c in ("dark", "bright", "blurry", "backlight", "covered") if r[c])
            print(f"  {r['kind']:12s} {r['src']:6s} flagged {r['flagged']}/{r['pictures']}" + (f" ({detail})" if detail else ""))
    if args.json:
        args.json.write_text(json.dumps({"limits": limits, "report": result}, indent=1))


if __name__ == "__main__":
    main()

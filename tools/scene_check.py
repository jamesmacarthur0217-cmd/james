"""Measure scene questions (Phase 22): the vision model's yes/no answers on pictures whose answer is known.

    python tools/scene_check.py --config <test_config.yaml> [--every-s 1] [--json <file>]

Two questions, asked exactly as the app asks them (labeler/vlm.py PROMPT_SCENE, the whole picture at 1024 px):

  "Is someone lying on the floor?" on the UR Fall videos (data/datasets/urfd, the color right half, 320x240): the
      falls' frames marked lying (urfall-cam0-falls.csv: 1) should be "yes", their frames marked not lying (-1) "no";
      the everyday activities' frames marked not lying "no". Their frames marked lying are someone lying down on a
      bed or the floor on purpose: shown apart (not counted), since "on the floor" or not is not in the marks.
  "Is there a car in the picture?" on MOT16 street videos: 04 and 13 frames with a marked car (class 3, at least half
      visible, at least 5% of the picture high) should be "yes"; 02, 09 and 11 (a pedestrian street, a shop front, a
      shopping mall: checked by eye, no cars) "no". Frames of 04/13 with only small or hidden cars are left out.

One picture every --every-s seconds of video. Printed per question: right, wrong, "unsure", seconds per answer, and
"twice in a row": how often two following pictures of the same video were both answered wrong the same way, which
is what would make a false alert (or a missed one) with the app's two answers in a row. Following pictures here are
seconds apart; the app asks minutes apart, so this overstates how alike two answers are.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeler.config import load_config  # noqa: E402

URFD = ROOT / "data" / "datasets" / "urfd"
MOT = ROOT / "data" / "datasets" / "mot16" / "MOT16" / "train"
LYING, CAR = "Is someone lying on the floor?", "Is there a car in the picture?"


def urfd_marks() -> dict[str, dict[int, int]]:
    """video -> frame -> -1 not lying, 0 falling, 1 lying."""
    out: dict[str, dict[int, int]] = defaultdict(dict)
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        for row in csv.reader(open(URFD / name)):
            out[row[0]][int(row[1])] = int(row[2])
    return out


def urfd_items(every_s: float) -> list[dict]:
    """(video, frame, truth) for the lying question; truth None = shown apart."""
    items = []
    step = max(1, round(30 * every_s))
    for video, marks in sorted(urfd_marks().items()):
        for frame in sorted(marks)[::step]:
            mark = marks[frame]
            if mark == 0:
                continue   # falling: neither
            fall = video.startswith("fall")
            truth = ("yes" if mark == 1 else "no") if fall else ("no" if mark == -1 else None)
            items.append({"question": LYING, "video": video, "frame": frame, "truth": truth})
    return items


def mot_items(every_s: float) -> list[dict]:
    items = []
    step = max(1, round(30 * every_s))
    for seq, with_cars in (("MOT16-04", True), ("MOT16-13", True), ("MOT16-02", False), ("MOT16-09", False),
                           ("MOT16-11", False)):
        h = float(next(line.split("=")[1] for line in open(MOT / seq / "seqinfo.ini") if line.startswith("imHeight")))
        n = int(next(line.split("=")[1] for line in open(MOT / seq / "seqinfo.ini") if line.startswith("seqLength")))
        big = set()
        for line in open(MOT / seq / "gt" / "gt.txt"):
            f = line.split(",")
            if int(f[7]) == 3 and float(f[8]) >= 0.5 and float(f[5]) >= 0.05 * h:
                big.add(int(f[0]))
        for frame in range(1, n + 1, step):
            if with_cars and frame not in big:
                continue
            items.append({"question": CAR, "video": seq, "frame": frame, "truth": "yes" if with_cars else "no"})
    return items


def picture(item: dict, caps: dict):
    if item["question"] == CAR:
        return cv2.imread(str(MOT / item["video"] / "img1" / f"{item['frame']:06d}.jpg"))
    cap = caps.get(item["video"])
    if cap is None:
        cap = caps[item["video"]] = cv2.VideoCapture(str(URFD / f"{item['video']}-cam0.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame"] - 1)
    ok, frame = cap.read()
    return frame[:, frame.shape[1] // 2:] if ok else None   # the color picture; the left half is a depth map


def report(items: list[dict]) -> list[dict]:
    rows = []
    for question in (LYING, CAR):
        mine = [i for i in items if i["question"] == question and "answer" in i]
        for truth in ("yes", "no", None):
            sel = [i for i in mine if i["truth"] == truth]
            if not sel:
                continue
            c = Counter(i["answer"] for i in sel)
            wrong = "no" if truth == "yes" else "yes"
            twice = pairs = 0
            by_video = defaultdict(list)
            for i in sel:
                by_video[i["video"]].append(i)
            for seq in by_video.values():
                seq.sort(key=lambda i: i["frame"])
                for a, b in zip(seq, seq[1:]):
                    pairs += 1
                    twice += a["answer"] == b["answer"] == wrong
            rows.append({"question": question, "truth": truth or "lying on purpose (apart)", "pictures": len(sel),
                         "yes": c["yes"], "no": c["no"], "unsure": c["unsure"], "wrong_twice": twice, "pairs": pairs,
                         "seconds_median": round(statistics.median(i["seconds"] for i in sel), 2)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a test config (tools/make_test_config.py)")
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    from labeler.vlm import VLMClient
    cfg = load_config(args.config)
    client = VLMClient(cfg["vlm"], cfg["mode"], cfg["labels"])
    ok, why = client.available()
    if not ok:
        sys.exit(f"the vision model is not available: {why}")
    items = urfd_items(args.every_s) + mot_items(args.every_s)
    caps: dict = {}
    print(f"{len(items)} pictures", flush=True)
    for k, item in enumerate(items):
        img = picture(item, caps)
        if img is None:
            continue
        started = time.monotonic()
        item["answer"] = client.label(img, None, ("scene", item["question"]))["answer"]
        item["seconds"] = time.monotonic() - started
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(items)}", flush=True)
    rows = report(items)
    print(f"\n{'question':32s} {'truth':26s} {'pics':>5s} {'yes':>5s} {'no':>5s} {'unsure':>6s} {'wrong twice':>12s} {'s':>5s}")
    for r in rows:
        print(f"{r['question']:32s} {r['truth']:26s} {r['pictures']:>5d} {r['yes']:>5d} {r['no']:>5d} {r['unsure']:>6d}"
              f" {r['wrong_twice']:>5d} of {r['pairs']:<4d} {r['seconds_median']:>5.2f}")
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "items": items}, indent=1))


if __name__ == "__main__":
    main()

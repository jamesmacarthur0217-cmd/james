"""Crowding (Phase 45 step 3, labeler/safety.py): does the app count people right in a crowd, and how do "too many
people" and "running together" behave? Runs the app's own Pipeline (security camera, safety on, a scratch database,
the video's own clock) on MOTChallenge videos with true markings, with one area over the whole picture.

    python tools/crowd_check.py --config <test_config.yaml> [--videos MOT20-01,MOT20-02,MOT16-04] [--fps 10]
           [--crowd-s 5] [--limit-share 0.75] [--json <file>]

For each video:
  true       people the markings have in the picture (median over the frames; MOT20/MOT16 gt.txt: pedestrians and
             people standing still, at least --min-visibility visible)
  counted    people the app has in the area at the same frames (median), and counted / true
  limit      the area's limit for the test: the --limit-share quantile of the true counts (so the true count is over
             it about a quarter of the time)
  over       share of frames over the limit by the true count and by the app's count, and how many of the true "over"
             frames the app also saw over (recall) and how many of its "over" frames were true (precision)
  crowd      "too many" suspects raised (with --crowd-s instead of the app's 30 s: the clips are short), and true
             stretches over the limit at least that long
  rush       "running together" suspects (none expected: nobody runs in these videos)
Videos: data/datasets/mot16/train_videos + gt (MOT16), data/datasets/mot20/train_videos + gt (MOT20, made from the
download by --make-videos). Nothing is written to the user's data: use a test config from tools/make_test_config.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from labeler.config import load_config  # noqa: E402

WHOLE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def paths(name: str) -> tuple[Path, Path]:
    """The video and its true markings for MOT16-xx / MOT20-xx."""
    if name.startswith("MOT20"):
        folder = ROOT / "data" / "datasets" / "mot20"
        return folder / "train_videos" / f"{name}.mp4", folder / "gt" / f"{name}.txt"
    folder = ROOT / "data" / "datasets" / "mot16"   # the full markings (standing people too), not gt/ (walking only)
    return folder / "train_videos" / f"{name}.mp4", folder / "MOT16" / "train" / name / "gt" / "gt.txt"


PEOPLE = (1, 7)   # MOTChallenge classes: pedestrian, static person (the app counts everyone it sees)


def true_counts(gt: Path, min_visibility: float) -> dict[int, int]:
    """frame -> people in the markings (pedestrians and people standing still, visible enough)."""
    out: dict[int, int] = defaultdict(int)
    for line in gt.read_text().splitlines():
        v = line.split(",")
        frame = int(float(v[0]))
        if len(v) >= 9 and (int(float(v[7])) not in PEOPLE or float(v[8]) < min_visibility):
            continue
        out[frame] += 1
    return dict(out)


def make_videos() -> int:
    """MOT20 train image folders -> data/datasets/mot20/train_videos/<name>.mp4 and gt/<name>.txt."""
    import cv2
    base = ROOT / "data" / "datasets" / "mot20"
    src = next(base.glob("download/**/MOT20/train"), None) or next(base.glob("**/MOT20/train"), None)
    if src is None:
        sys.exit("unpack data/datasets/mot20/download/MOT20.zip first")
    (base / "train_videos").mkdir(exist_ok=True)
    (base / "gt").mkdir(exist_ok=True)
    n = 0
    for seq in sorted(p for p in src.iterdir() if p.is_dir()):
        info = dict(line.split("=", 1) for line in (seq / "seqinfo.ini").read_text().splitlines() if "=" in line)
        fps = float(info["frameRate"])
        images = sorted((seq / "img1").glob("*.jpg"))
        first = cv2.imread(str(images[0]))
        out = cv2.VideoWriter(str(base / "train_videos" / f"{seq.name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                              (first.shape[1], first.shape[0]))
        for img in images:
            out.write(cv2.imread(str(img)))
        out.release()
        (base / "gt" / f"{seq.name}.txt").write_text((seq / "gt" / "gt.txt").read_text())
        print(f"{seq.name}: {len(images)} frames at {fps:g} fps")
        n += 1
    return n


def run(video: Path, cfg: dict, embedder, fps: float, limit: int, crowd_s: float, imgsz: int | None = None) -> dict:
    """The Pipeline with one whole-picture area limited to `limit`; per processed frame (number, people counted);
    the crowd suspects (seconds into the video, area or None, reason)."""
    from incident_check import Yes, frames
    from labeler.db import Database
    from labeler.incidents import ConfirmQueue
    from labeler.pipeline import Pipeline, VideoClock
    from labeler.sources import probe_video
    from labeler.zones import inside

    db = Database("sqlite:///" + (Path(tempfile.mkdtemp(prefix="crowd_check_")) / "check.db").as_posix())
    db.add_zone("check", "everywhere", "area", WHOLE, max_people=limit)
    cam = {"name": "check", "source": "imported videos", "mode": "security", "loop_files": False, "ignore_zones": [],
           "vehicles": False, "animals": False, "safety": True, "fall_alert": False}
    if imgsz:
        cam["imgsz"] = imgsz   # the detector's picture size (cameras[].imgsz): far or small people need more
    cfg = copy.deepcopy(cfg)
    cfg["clips"]["enabled"] = False
    cfg["person_videos"]["enabled"] = False
    cfg["safety"]["crowd_s"] = crowd_s
    p = Pipeline(cfg, db, None, embedder, None, camera=cam)
    p.clock = VideoClock(datetime(2026, 1, 1))
    worker = Yes()
    q = ConfirmQueue(cfg, worker, {"check": p.memory}, clock=p.clock.mono)
    worker.queue = q
    suspects = []
    plain = q.suspect

    def suspect(camera, category, reason, *a, **k):
        if category == "crowd":
            suspects.append((round(p.clock.t, 2), (k.get("area") or {}).get("name"), reason))
        return plain(camera, category, reason, *a, **k)

    q.suspect = suspect
    p.incidents = q
    rate = probe_video(video)["fps"]
    counts = []
    for frame, t in frames(video, fps, "", 0):
        p.clock.t = t
        p.process(frame)
        p.frames += 1
        q.tick()
        if p._crowd_counts:   # the camera's own count (the plain detector, once a second)
            n = next(iter(p._crowd_counts.values()))
        else:                             # without it: the people tracked
            n = sum(1 for x in p.safety_watch.people.values() if x.hist and t - x.last <= 1e-6
                    and x.anchor is not None and inside(x.anchor, WHOLE))
        counts.append((int(round(t * rate)) + 1, n))
    for tid in list(p.tracks):
        p._finish(tid)
    return {"counts": counts, "suspects": suspects}


def stretches(over: list[tuple[float, bool]], min_s: float) -> list[tuple[float, float]]:
    """(start, end) of runs of True at least min_s long, from (time, flag) in time order."""
    out, start, last = [], None, None
    for t, flag in over:
        if flag and start is None:
            start = t
        if not flag and start is not None:
            if last - start >= min_s:
                out.append((start, last))
            start = None
        last = t
    if start is not None and last - start >= min_s:
        out.append((start, last))
    return out


def score(true: dict[int, int], res: dict, fps_file: float, limit: int, crowd_s: float) -> dict:
    rows = [(f, true.get(f, 0), n) for f, n in res["counts"]]
    t_over = [(f / fps_file, tc > limit) for f, tc, _ in rows]
    c_over = [n > limit for _, _, n in rows]
    both = sum(a[1] and b for a, b in zip(t_over, c_over))
    true_over, counted_over = sum(a[1] for a in t_over), sum(c_over)
    ratios = [n / tc for _, tc, n in rows if tc]
    return {"frames": len(rows), "true_median": statistics.median(tc for _, tc, _ in rows),
            "counted_median": statistics.median(n for _, _, n in rows),
            "counted_share": round(statistics.median(ratios), 2) if ratios else None,
            "limit": limit, "true_over": round(true_over / len(rows), 2), "counted_over": round(counted_over / len(rows), 2),
            "recall": round(both / true_over, 2) if true_over else None,
            "precision": round(both / counted_over, 2) if counted_over else None,
            "true_stretches": len(stretches(t_over, crowd_s)),
            "crowd_suspects": sum(1 for _, area, _ in res["suspects"] if area),
            "rush_suspects": sum(1 for _, area, _ in res["suspects"] if area is None),
            "suspects": res["suspects"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--videos", default="MOT20-01,MOT20-02,MOT20-03,MOT20-05,MOT16-02,MOT16-04,MOT16-09")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--crowd-s", type=float, default=5.0)
    ap.add_argument("--limit-share", type=float, default=0.75)
    ap.add_argument("--min-visibility", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, help="the detector's picture size (default: the config's)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--make-videos", action="store_true", help="MOT20 image folders -> videos (after unpacking)")
    args = ap.parse_args()
    if args.make_videos:
        print(f"{make_videos()} videos")
        return 0
    if not args.config:
        ap.error("--config is needed")
    logging.basicConfig(level=logging.ERROR)
    cfg = load_config(args.config)
    from labeler.embed import Embedder
    from labeler.sources import probe_video
    embedder = Embedder(cfg["embeddings"])
    out = {}
    print(f"{'video':>9} {'true':>5} {'counted':>8} {'share':>6} {'limit':>6} {'true over':>10} {'app over':>9} "
          f"{'recall':>7} {'precision':>10} {'stretches':>10} {'crowd':>6} {'rush':>5}")
    for name in args.videos.split(","):
        video, gt = paths(name)
        true = true_counts(gt, args.min_visibility)
        limit = int(np.quantile(list(true.values()), args.limit_share))
        res = run(video, cfg, embedder, args.fps, limit, args.crowd_s, args.imgsz)
        r = score(true, res, probe_video(video)["fps"], limit, args.crowd_s)
        out[name] = r
        print(f"{name:>9} {r['true_median']:>5} {r['counted_median']:>8} {r['counted_share']!s:>6} {limit:>6} "
              f"{r['true_over']:>10} {r['counted_over']:>9} {r['recall']!s:>7} {r['precision']!s:>10} "
              f"{r['true_stretches']:>10} {r['crowd_suspects']:>6} {r['rush_suspects']:>5}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

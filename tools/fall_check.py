"""Measure fall detection (labeler/falls.py) on the UR Fall Detection videos and on walking people (MOT16).

    python tools/fall_check.py                       # falls caught and false alarms, both halves
    python tools/fall_check.py --grid                # try settings on the tuning half only
    python tools/fall_check.py --after-s 10 --min-drop 0.4 ...

UR Fall Detection (University of Rzeszow, CC BY-NC-SA 4.0; downloaded with the user's OK, 2026-10-13) in
data/datasets/urfd: fall-NN-cam0.mp4 (side view) and fall-NN-cam1.mp4 (from the ceiling) for 30 falls, adl-NN-cam0.mp4
for 40 everyday activities (sitting, lying on a bed or the floor slowly, picking things up); the color picture is
the right half (320x240, 30 fps). urfall-cam0-falls.csv marks each camera-0 frame: -1 not lying, 0 falling,
1 lying. A clip ends a few seconds after the fall, so its last picture is held for --hold-s seconds, as if the
person stays down (and the everyday clips the same: whoever lies down at the end stays lying).

Halves: falls 1-15 and activities 1-20 to tune on, falls 16-30 and activities 21-40 to report; camera 1 is
reported with the same settings, never tuned. The detector and tracker run once per video (cached in
data/datasets/urfd/cache), so trying settings is fast.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labeler.config import DEFAULTS  # noqa: E402
from labeler.falls import FallWatch  # noqa: E402

URFD = ROOT / "data" / "datasets" / "urfd"
MOT = ROOT / "data" / "datasets" / "mot16" / "train_videos"
CACHE = URFD / "cache"
FPS = 30.0


def videos() -> list[tuple[str, Path, str]]:
    """(name, path, kind) with kind fall0 / fall1 / adl / walk."""
    out = [(f"fall-{i:02d}-cam0", URFD / f"fall-{i:02d}-cam0.mp4", "fall0") for i in range(1, 31)]
    out += [(f"fall-{i:02d}-cam1", URFD / f"fall-{i:02d}-cam1.mp4", "fall1") for i in range(1, 31)]
    out += [(f"adl-{i:02d}-cam0", URFD / f"adl-{i:02d}-cam0.mp4", "adl") for i in range(1, 41)]
    out += [(p.stem, p, "walk") for p in sorted(MOT.glob("*.mp4"))] if MOT.exists() else []
    return out


def half(name: str, kind: str) -> str:
    if kind == "walk":
        return "test"
    n = int(name.split("-")[1])
    return "tune" if (kind == "fall0" and n <= 15) or (kind == "adl" and n <= 20) else "test"


def fall_starts() -> dict[str, float]:
    """Seconds into each camera-0 fall clip where the fall begins (first 'falling' or 'lying' frame)."""
    first: dict[str, int] = {}
    for row in csv.reader(open(URFD / "urfall-cam0-falls.csv")):
        if int(row[2]) >= 0:
            first[row[0]] = min(first.get(row[0], 10 ** 9), int(row[1]))
    return {k: (v - 1) / FPS for k, v in first.items()}


def detections(name: str, path: Path, kind: str, hold_s: float) -> list[tuple[float, list]]:
    """Per frame: (time, [(track id, box, keypoints)]), from the app's detector and tracker settings."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{name}-hold{hold_s:g}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from ultralytics import YOLO
    det = DEFAULTS["detector"]
    model = YOLO(str(ROOT / det["model"]))
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frames, last, i = [], None, 0
    step = max(1, round(fps / 15)) if kind == "walk" else 1   # MOT16 at about 15 fps is plenty
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % step:
            continue
        frames.append(((i - 1) / fps, frame[:, frame.shape[1] // 2:] if kind != "walk" else frame))
    if frames and kind != "walk":
        t0, last = frames[-1]
        frames += [(t0 + k / FPS, last) for k in range(1, int(hold_s * FPS) + 1)]
    out = []
    for t, frame in frames:
        r = model.track(frame, persist=True, tracker=str(ROOT / det["tracker"]), classes=[0], conf=det["conf"],
                        imgsz=det["imgsz"], verbose=False)[0]
        people = []
        if r.boxes is not None and r.boxes.id is not None:
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for j, (tid, box) in enumerate(zip(r.boxes.id.int().tolist(), r.boxes.xyxy.cpu().numpy())):
                people.append((tid, tuple(float(v) for v in box), None if kp is None else kp[j]))
        out.append((t, people))
    cache.write_bytes(pickle.dumps(out))
    return out


def run(dets: list, cfg: dict, frame_h: float) -> list[tuple[float, float]]:
    """(confirmed at, fall began at) for every fall the watcher confirms in one video."""
    watch, events = FallWatch(cfg), []
    for t, people in dets:
        for tid, box, kpts in people:
            e = watch.update(tid, t, box, kpts, frame_h=frame_h)
            if e:
                events.append((t, e["since"]))
        events += [(t, e["since"]) for _, e in watch.tick(t)]
    return events


def frame_height(path: Path, kind: str) -> float:
    if kind != "walk":
        return 240.0
    cap = cv2.VideoCapture(str(path))
    return float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def evaluate(all_dets: dict, cfg: dict, which: str, verbose: bool = False) -> dict:
    starts = fall_starts()
    heights = {(n, k): frame_height(p, k) for n, p, k in videos()}
    res = defaultdict(lambda: [0, 0])   # kind -> [videos with an event, videos]
    delays, notes, extra = [], [], 0
    for (name, kind), dets in all_dets.items():
        if which != "all" and half(name, kind) != which:
            continue
        events = run(dets, cfg, heights[(name, kind)])
        if len(events) > 1:
            extra += len(events) - 1
            notes.append(f"{len(events)} alerts in {name}")
        if kind in ("fall0", "fall1"):
            start = starts.get(name.rsplit("-", 1)[0]) if kind == "fall0" else None
            # Camera 0: only an event after the real fall began counts; camera 1 has no frame marks.
            good = [e for e in events if start is None or e[1] >= start - 1.0]
            res[kind][0] += bool(good)
            if good and start is not None:
                delays.append(good[0][1] - start)
            if not good:
                notes.append(f"missed {name}" + (f" (early event at {events[0][1]:.1f} s)" if events else ""))
        else:
            res[kind][0] += bool(events)
            if events:
                notes.append(f"false alarm {name} (down at {events[0][1]:.1f} s)")
        res[kind][1] += 1
    if verbose:
        for n in notes:
            print("  " + n)
    return {"res": dict(res), "delays": delays, "extra": extra}


def show(label: str, out: dict) -> None:
    r = out["res"]
    parts = [f"falls cam0 {r['fall0'][0]}/{r['fall0'][1]}" if "fall0" in r else "",
             f"falls cam1 {r['fall1'][0]}/{r['fall1'][1]}" if "fall1" in r else "",
             f"false alarms: activities {r['adl'][0]}/{r['adl'][1]}" if "adl" in r else "",
             f"walking videos {r['walk'][0]}/{r['walk'][1]}" if "walk" in r else ""]
    d = sorted(out["delays"])
    med = f", fall seen {d[len(d) // 2]:.1f} s after it began (median)" if d else ""
    med += f", extra alerts for one fall: {out['extra']}"
    print(f"{label}: " + ", ".join(p for p in parts if p) + med)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg = copy.deepcopy(DEFAULTS["falls"])
    for key, value in cfg.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ap.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    ap.add_argument("--hold-s", type=float, default=15.0, help="seconds the last picture is held")
    ap.add_argument("--grid", action="store_true", help="try settings on the tuning half")
    ap.add_argument("--no-walk", action="store_true", help="leave out the MOT16 walking videos")
    args = ap.parse_args()
    for key in cfg:
        if hasattr(args, key):
            cfg[key] = getattr(args, key)
    all_dets = {}
    for name, path, kind in videos():
        if kind == "walk" and (args.no_walk or args.grid):
            continue
        all_dets[(name, kind)] = detections(name, path, kind, args.hold_s)
    if args.grid:
        for drop_s, min_drop, angle in itertools.product([1.0, 1.5, 2.0, 3.0], [0.2, 0.3, 0.4, 0.5], [50, 60, 70]):
            c = {**cfg, "drop_s": drop_s, "min_drop": min_drop, "lying_angle": angle}
            show(f"drop_s {drop_s} min_drop {min_drop} angle {angle}", evaluate(all_dets, c, "tune"))
        return
    print("settings:", cfg)
    for which in ("tune", "test"):
        out = evaluate(all_dets, cfg, which, verbose=True)
        show(which, out)


if __name__ == "__main__":
    main()

"""Made-up night videos (Phase 53): the safety checks' measuring videos as a camera sees them at night.

    python tools/make_night_videos.py [--kinds ir,dark] [--only <part of a path>] [--preview]

Two kinds of night (the user's choice, 2026-10-26):
  ir    black and white like a security camera's infrared mode: flat contrast, grain, slightly soft, brighter in the
        middle (the camera's own IR lamp lights the near middle best);
  dark  colour at about a quarter of the brightness with heavy grain and paler colours: a camera without IR under
        street lights.
For every truth file below, each video is written to data/datasets/night_made/<kind>/ under its own path (relative to
data/datasets), with the truth file and the areas file next to it, paths changed, so the same measuring tools run on
them (tools/incident_check.py --truth data/datasets/night_made/ir/urfd/incidents_truth.csv ...). Same frame rate and
size; the grain is the same for every run (seeded per video). A video already made is skipped (the run can be
stopped and started again). --preview writes one picture per video (day | ir | dark, from the middle) to
data/datasets/night_made/preview/ instead, to check the look. Nothing is downloaded; the folder is git-ignored.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "datasets"
OUT = DATA / "night_made"
# (truth file, areas file or None): the measuring sets of Phases 45-49.
SETS = [("urfd/incidents_truth.csv", None),                       # falls
        ("survfight/incidents_truth.csv", None),                  # fights (and the clips without)
        ("dfire/videos_truth.csv", None),                         # fire and smoke
        ("floor_made/truth.csv", "floor_made/areas.json"),        # floor hazards, blocked exits
        ("behaviour_made/truth.csv", "behaviour_made/areas.json"),  # pacing, camera covered / turned
        ("behaviour_made/everyday_truth.csv", None)]              # everyday videos: false alarms
KINDS = ("ir", "dark")


def _vignette(shape: tuple, edge: float = 0.45) -> np.ndarray:
    """1.0 in the middle falling to `edge` in the corners (an IR lamp lights the near middle best)."""
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    return (1.0 - (1.0 - edge) * r ** 1.5).astype(np.float32)


def ir(frame: np.ndarray, rng: np.random.Generator, vignette: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = 0.6 * gray + 0.4 * float(gray.mean()) + 15          # flat contrast, a little lifted
    gray = cv2.GaussianBlur(gray * vignette, (3, 3), 0.8)       # soft
    gray += rng.normal(0, 8, gray.shape).astype(np.float32)     # grain
    g = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.merge([g, g, g])


def dark(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = frame.astype(np.float32) / 255.0
    f = np.power(f, 1.6) * 0.45                                  # about a quarter of the brightness, deep shadows
    mean = f.mean(axis=2, keepdims=True)
    f = mean + 0.7 * (f - mean)                                  # paler colours
    f = f * 255 + rng.normal(0, 6, f.shape).astype(np.float32)  # colour grain
    f += rng.normal(0, 4, f.shape[:2]).astype(np.float32)[..., None]   # and brightness grain
    return np.clip(f, 0, 255).astype(np.uint8)


def night(kind: str, frame: np.ndarray, rng: np.random.Generator, vignette: np.ndarray | None = None) -> np.ndarray:
    if kind == "ir":
        return ir(frame, rng, vignette if vignette is not None else _vignette(frame.shape))
    if kind == "dark":
        return dark(frame, rng)
    raise ValueError(f"unknown kind of night {kind!r}")


def out_path(kind: str, video: str) -> str:
    """data/datasets/x/y.mp4 -> data/datasets/night_made/<kind>/x/y.mp4 (as the truth files write paths)."""
    rel = Path(video).as_posix()
    prefix = "data/datasets/"
    if not rel.startswith(prefix):
        raise ValueError(f"{video} is not under {prefix}")
    return f"{prefix}night_made/{kind}/{rel[len(prefix):]}"


def _rng(video: str, kind: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{kind}:{video}".encode()))


def make_video(src: Path, dst: Path, kind: str, video: str) -> int:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng, vig, n = _rng(video, kind), _vignette((h, w)), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(night(kind, frame, rng, vig))
        n += 1
    cap.release()
    writer.release()
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"no frames read from {src}")
    tmp.replace(dst)   # only a whole video gets its name: a stopped run makes it again
    return n


def write_set(kind: str, truth: str, areas: str | None) -> list[str]:
    """The night truth file (and areas file) for one set; returns the videos in it."""
    rows = list(csv.DictReader((DATA / truth).open(encoding="utf-8")))
    target = OUT / kind / truth
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        for r in rows:
            out.writerow({**r, "video": out_path(kind, r["video"])})
    if areas:
        drawn = json.loads((DATA / areas).read_text(encoding="utf-8"))
        (OUT / kind / areas).write_text(json.dumps({out_path(kind, v): a for v, a in drawn.items()}, indent=1),
                                        encoding="utf-8")
    return list(dict.fromkeys(r["video"] for r in rows))


def preview(videos: list[str]) -> None:
    folder = OUT / "preview"
    folder.mkdir(parents=True, exist_ok=True)
    for video in videos:
        cap = cv2.VideoCapture(str(ROOT / video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        frame = cv2.resize(frame, (480, round(480 * frame.shape[0] / frame.shape[1])))
        row = np.hstack([frame] + [night(k, frame, _rng(video, k)) for k in KINDS])
        cv2.imwrite(str(folder / (Path(video).stem + ".jpg")), row, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--only", default="", help="only videos whose path holds this")
    ap.add_argument("--preview", action="store_true", help="one picture per video (day | ir | dark) instead")
    args = ap.parse_args()
    kinds = [k for k in args.kinds.split(",") if k]
    for k in kinds:
        if k not in KINDS:
            sys.exit(f"unknown kind {k!r} (ir, dark)")
    videos: list[str] = []
    for truth, areas in SETS:
        for kind in kinds:
            got = write_set(kind, truth, areas)
        videos += [v for v in got if v not in videos]
    videos = [v for v in videos if args.only in v]
    if args.preview:
        preview(videos)
        print(f"{len(videos)} previews in {OUT / 'preview'}")
        return
    for kind in kinds:
        for i, video in enumerate(videos, 1):
            dst = ROOT / out_path(kind, video)
            if dst.exists():
                continue
            n = make_video(ROOT / video, dst, kind, video)
            print(f"{kind} {i}/{len(videos)} {video} ({n} frames)", flush=True)
    size = sum(p.stat().st_size for p in OUT.rglob("*.mp4")) / 1e9
    print(f"done: {len(videos)} videos x {len(kinds)} kinds, {size:.1f} GB in {OUT}")


if __name__ == "__main__":
    main()

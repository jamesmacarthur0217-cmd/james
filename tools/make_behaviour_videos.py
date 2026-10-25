"""Made-up videos for the suspicious-behaviour signs (Phase 49, labeler/behaviour.py; the user's choice: no download),
from the still MOT16 videos here, and the areas for measuring false alarms on the everyday videos.

    python tools/make_behaviour_videos.py [--out data/datasets/behaviour_made]
    python tools/incident_check.py --config <test_config.yaml> --truth data/datasets/behaviour_made/truth.csv
           --areas data/datasets/behaviour_made/areas.json --kinds suspicious

Videos (15 frames a second, at most 1280 px wide, H.264 through the recorder's FFmpeg):
- pacing_<scene>: a stretch of the clip played forwards and backwards for 4 minutes, so the people in it walk back
  and forth. The stretch is where one person (MOT16's true identity marks, data/datasets/mot16/gt) stays fully in view
  while walking at least two body heights across; playing the whole clip back and forth did not work, as people walk
  out of the picture and are followed again as someone new. One area over the whole picture. A suspicious span over
  the whole video (pacing is looked for at night: incident_check.py's video clock starts at midnight).
- covered_MOT16-09: 45 s as it is, then 15 s black (a hand or a cloth over the lens); people were in view just before.
- turned_MOT16-09: 45 s, then 15 s of another view (the right part of the picture enlarged: the camera turned).
- dimmed_MOT16-09: 45 s, then 15 s at 40% brightness (the light switched off, the street still seen): no incident.
- the everyday videos (MOT16, MOT20, UR Fall, the floor videos with nothing put down) get one area over the whole
  picture with the role "fence", the worst case for the climbing sign (everyone is "at a fence"), and for pacing.
Climbing itself has no video here to make it from: only its false alarms are measured.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from make_floor_videos import FPS, read_frames, write  # noqa: E402

MOT16 = "data/datasets/mot16/train_videos"
WHOLE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
CHANGE_AT, CHANGE_S = 45, 15


def walk_window(scene: str, fps: float) -> tuple[int, int, int]:
    """(first frame, last frame, person) of the longest stretch where one person is fully in view (visibility 0.8 or
    more) and walks at least two body heights across, from MOT16's true marks."""
    rows: dict = {}
    for line in (ROOT / "data/datasets/mot16/gt" / f"{scene}.txt").read_text().splitlines():
        f, pid, x, y, w, h, _, _, vis = (float(v) for v in line.split(",")[:9])
        rows.setdefault(int(pid), []).append((int(f), x + w / 2, h, vis))
    best = (0, 0, 0, -1)
    for pid, marks in rows.items():
        marks.sort()
        run = []
        for m in marks + [(10 ** 9, 0, 1, 0)]:
            if m[3] >= 0.8 and (not run or m[0] == run[-1][0] + 1):
                run.append(m)
                continue
            if len(run) >= 4 * fps:
                travel = max(r[1] for r in run) - min(r[1] for r in run)
                if travel >= 2 * float(np.median([r[2] for r in run])) and len(run) > best[3]:
                    best = (run[0][0], run[-1][0], pid, len(run))
            run = [m] if m[3] >= 0.8 else []
    if best[3] < 0:
        raise RuntimeError(f"{scene}: nobody walks across in full view")
    return best[0], best[1], best[2]


def write_frames(path: Path, frames: list[np.ndarray]) -> None:
    """Writes exactly these frames (make_floor_videos.write loops a clip with an optional layer)."""
    write(path, frames, len(frames) // FPS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data/datasets/behaviour_made")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    truth, areas = [], {}

    def add(name, frames, kind, start, end, video_areas=None):
        path = out / name
        if not path.is_file():
            write_frames(path, frames)
        rel = path.relative_to(ROOT).as_posix()
        truth.append([rel, kind, start if kind else "", end if kind else "", ""])
        if video_areas:
            areas[rel] = video_areas
        print(name, len(frames) / FPS, "s", kind or "no incident")

    for scene in ("MOT16-02", "MOT16-09"):
        frames, fps = read_frames(ROOT / MOT16 / f"{scene}.mp4")
        first, last, pid = walk_window(scene, fps)
        part = frames[int((first - 1) * FPS / fps):int((last - 1) * FPS / fps) + 1]
        print(f"{scene}: person {pid} walks across in full view in frames {first}-{last} ({len(part) / FPS:.1f} s)")
        bounce = part + part[::-1]
        n = 240 * FPS
        add(f"pacing_{scene}.mp4", [bounce[i % len(bounce)] for i in range(n)], "suspicious", 0, 240,
            [{"name": "yard", "role": "", "points": WHOLE}])
    frames, _ = read_frames(ROOT / MOT16 / "MOT16-09.mp4")
    before = [frames[i % len(frames)] for i in range(CHANGE_AT * FPS)]
    h, w = frames[0].shape[:2]
    black = [np.full_like(frames[0], 8)] * (CHANGE_S * FPS)
    add("covered_MOT16-09.mp4", before + black, "suspicious", CHANGE_AT, CHANGE_AT + CHANGE_S)
    turned = [cv2.resize(f[h // 4:, w // 2:], (w, h)) for f in
              (frames[i % len(frames)] for i in range(CHANGE_AT * FPS, (CHANGE_AT + CHANGE_S) * FPS))]
    add("turned_MOT16-09.mp4", before + turned, "suspicious", CHANGE_AT, CHANGE_AT + CHANGE_S)
    dimmed = [(f * 0.4).astype(np.uint8) for f in
              (frames[i % len(frames)] for i in range(CHANGE_AT * FPS, (CHANGE_AT + CHANGE_S) * FPS))]
    add("dimmed_MOT16-09.mp4", before + dimmed, "", 0, 0)
    with open(out / "truth.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["video", "kind", "start_s", "end_s", "half"])
        wr.writerows(truth)
    # The everyday videos: the whole picture as a "fence" (climbing's worst case) and for pacing.
    everyday = sorted((ROOT / MOT16).glob("*.mp4")) + sorted((ROOT / "data/datasets/mot20/train_videos").glob("*.mp4"))
    everyday += sorted((ROOT / "data/datasets/urfd").glob("*-cam0.mp4"))
    everyday += sorted((ROOT / "data/datasets/floor_made").glob("*_none.mp4"))
    rows = []
    for v in everyday:
        rel = v.relative_to(ROOT).as_posix()
        areas[rel] = [{"name": "everywhere", "role": "fence", "points": WHOLE}]
        rows.append([rel, "", "", "", "right" if "urfd" in rel else ""])
    with open(out / "everyday_truth.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["video", "kind", "start_s", "end_s", "half"])
        wr.writerows(rows)
    (out / "areas.json").write_text(json.dumps(areas, indent=1), encoding="utf-8")
    print(len(truth), "made-up videos,", len(rows), "everyday videos;", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

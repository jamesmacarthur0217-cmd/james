"""Unsafe equipment (Phase 47 step 3, labeler/equipment.py): what does the vision model answer to the app's equipment
question about an area, on ordinary places (every "yes" is a false alarm) and on made-up unsafe ones?

    python tools/equipment_check.py [--half tune|report|all] [--coco 150]

Ordinary: the still scenes of tools/make_floor_videos.py (MOT16-02/-04/-09, D-Fire's guard booth FP9/FP10) with
their walkway and exit areas cut out as the app does (scene.area_picture "part") and whole; UR Fall's rooms (offices
with desks and chairs, the color half, first frame of the everyday clips); COCO val2017 photos of rooms (a chair,
couch, bed or dining table), whole. Made-up unsafe: the same scenes' walkway with a chair (cut out of COCO, as in
make_floor_videos.py) put down on its side or upside down, or a cable (a dark curve with a highlight) drawn across
it, 3 of each per scene. Halves (pitfall 21): the scenes' own (tune MOT16-02 and FP9), the other photos by order (even
= tune, odd = report). One picture per check, as the rounds send.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from labeler.config import load_config  # noqa: E402
from labeler.incidents import CATEGORIES  # noqa: E402
from labeler.scene import area_picture  # noqa: E402

COCO = ROOT / "data" / "datasets" / "coco"
ROOMS = {56: "chair", 57: "couch", 59: "bed", 60: "dining table"}


def scene_frames() -> list[tuple[str, str, np.ndarray, list, list]]:
    """(scene, its half, frame 10 s in, walkway points, exit points or None) of the still scenes."""
    import make_floor_videos as mf
    out = []
    for name, (video, half, walkway, exit_area, _) in mf.SCENES.items():
        cap = cv2.VideoCapture(str(ROOT / video))
        cap.set(cv2.CAP_PROP_POS_MSEC, 10_000)   # every scene is at least 17 s long
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read {video}")
        if f.shape[1] > mf.MAX_W:
            f = cv2.resize(f, (mf.MAX_W, round(f.shape[0] * mf.MAX_W / f.shape[1])), interpolation=cv2.INTER_AREA)
        out.append((name, half, f, walkway, exit_area))
    return out


def cable(frame: np.ndarray, area: list, rng) -> np.ndarray:
    """A dark cable across the area: a smooth curve from one side to the other, with a thin highlight."""
    h, w = frame.shape[:2]
    xs, ys = zip(*area)
    y0, y1 = rng.uniform(min(ys) + 0.05, max(ys) - 0.02, 2)
    pts = []
    for i in range(30):
        u = i / 29
        x = min(xs) + (max(xs) - min(xs)) * u
        y = y0 + (y1 - y0) * u + 0.015 * np.sin(u * 7 + rng.uniform(0, 6))
        pts.append((int(x * w), int(y * h)))
    out = frame.copy()
    thick = max(3, h // 120)
    cv2.polylines(out, [np.array(pts, np.int32)], False, (25, 25, 25), thick, cv2.LINE_AA)
    cv2.polylines(out, [np.array(pts, np.int32) - [0, thick // 3]], False, (90, 90, 90), max(1, thick // 3), cv2.LINE_AA)
    return out


def tipped_chair(frame: np.ndarray, area: list, rng) -> np.ndarray:
    """A COCO chair put down on its side or upside down in the area, sized to the people there."""
    import make_floor_videos as mf
    objects = mf.cut_objects(ROOT / "data" / "datasets" / "floor_made")
    chair = cv2.imread(str(objects["chair"][rng.integers(len(objects["chair"]))]), cv2.IMREAD_UNCHANGED)
    chair = cv2.rotate(chair, [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180][rng.integers(3)])
    box = mf.place(frame.shape, area, np.poly1d([0.3]), rng, 0.45, chair.shape[1] / chair.shape[0])
    times, plus = mf.object_layer(chair, box, frame, rng)
    return (frame * times + plus).clip(0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--half", choices=["tune", "report", "all"], default="tune")
    ap.add_argument("--coco", type=int, default=150, help="COCO room photos per half")
    args = ap.parse_args()
    keep = (lambda i: True) if args.half == "all" else (lambda i: i % 2 == (0 if args.half == "tune" else 1))
    cfg = load_config(Path(args.config))
    from labeler.vlm import VLMClient, incident_prompt
    client = VLMClient(cfg["vlm"], "security")
    question = CATEGORIES["equipment"][2]
    rng = np.random.default_rng(47)
    cases = []   # (group, unsafe?, picture)
    for name, half, f, walkway, exit_area in scene_frames():
        if args.half not in ("all", half):   # the scenes' own halves (make_floor_videos.py)
            continue
        cases.append(("scene whole", False, f))
        for poly in (walkway, exit_area):
            if poly:
                cases.append(("scene area", False, area_picture(f, poly, "part")))
        for _ in range(3):
            cases.append(("chair on its side", True, area_picture(tipped_chair(f, walkway, rng), walkway, "part")))
            cases.append(("cable across", True, area_picture(cable(f, walkway, rng), walkway, "part")))
    urfd = sorted((ROOT / "data" / "datasets" / "urfd").glob("adl-*-cam0.mp4"))
    for i, v in enumerate(urfd):
        if keep(i):
            ok, f = cv2.VideoCapture(str(v)).read()
            if ok:
                cases.append(("UR Fall room", False, f[:, f.shape[1] // 2:]))
    rooms = []
    for txt in sorted((COCO / "labels" / "val2017").glob("*.txt")):
        if any(int(r.split()[0]) in ROOMS for r in txt.read_text().splitlines() if r.strip()):
            rooms.append(txt.stem)
    for i, stem in enumerate(rooms[:args.coco * 2]):
        if keep(i):
            cases.append(("COCO room", False, cv2.imread(str(COCO / "val2017" / f"{stem}.jpg"))))
    answers: dict = {}
    for group, unsafe, pic in cases:
        got = client.label([pic], None, ("incident", incident_prompt(1, 0, "", question)))
        answers.setdefault(group, Counter())[got.get("answer")] += 1
        if (got.get("answer") == "yes") != unsafe:
            print(f"  {group}: {got.get('answer')} ({got.get('what', '')})")
    print(f"half {args.half}:")
    for group, c in answers.items():
        print(f"  {group}: {dict(c)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

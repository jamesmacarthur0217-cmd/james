"""Made-up videos for floor hazards and blocked exits (Phase 47 step 2, labeler/floor.py; the user's choice: no
download). Still-camera videos from this PC (MOT16, D-Fire) are played in a loop for a few minutes, and at a known
second something is put on the floor and stays: a bag on the walkway, a spill (dark like coffee or light like milk)
on the walkway, or a chair or suitcase in the exit area. A copy with nothing put down counts false alarms.

    python tools/make_floor_videos.py [--out data/datasets/floor_made] [--preview]
    python tools/incident_check.py --config <test_config.yaml> --truth data/datasets/floor_made/truth.csv
           --areas data/datasets/floor_made/areas.json --kinds floor,exit --match tune_

Objects are cut out of COCO val2017 photos (backpacks, handbags, suitcases, chairs; boxes from the YOLO labels here)
with the YOLOE segmentation model (data/models/yoloe), sized to the people near that spot (the pose detector on the
same video: about a third of a person's height for a bag), matched to the floor's brightness and given a soft
shadow. They are drawn on top of everything: someone walking past goes "behind" them, which a real bag would not do.
Spills are soft blobs that darken or lighten the floor. Real spills (water, glass) are harder than these.

Only still cameras: the MOT20 cameras pan and zoom during their videos (an empty picture cannot be learned there; the
app learns again each time, as it should for a moved camera), and most of D-Fire's sample videos without fire (CC0)
come from turning cameras that visit several preset views. Two of D-Fire's are still and have paths: a guard booth
(FP9 in the day, FP10 at dawn after rain; 2 frames a second). Scenes are split in halves (pitfall 21): "tune" (MOT16-02,
FP9) for choosing the settings, "report" (MOT16-04, MOT16-09, FP10: the same booth as FP9 on another day) for the
numbers. Writes <out>/<half>_<scene>_<kind>.mp4 (H.264
through the recorder's FFmpeg, 15 frames a second, at most 1280 px wide), truth.csv, areas.json, objects/ and with
--preview the areas drawn on each scene (preview_<scene>.jpg). Seeded: the same files every time. MOT16: CC BY-NC-SA
3.0, measuring only.
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

FPS, MAX_W = 15, 1280
PUT_AT = 60            # seconds into the video when the object appears
SECONDS = {"none": 240, "bag": 240, "spill": 240, "exit": 300}
MOT16, DFIRE = "data/datasets/mot16/train_videos", "data/datasets/dfire/videos_download/extract/Videos"
ALL = ("none", "bag", "spill", "exit")
# D-Fire's guard booth: the red path left of the booth, and the path on its right as the "exit"
BOOTH_PATH = [(0.27, 0.72), (0.41, 0.62), (0.41, 0.98), (0.26, 1.0)]
BOOTH_EXIT = [(0.79, 0.63), (0.88, 0.63), (0.87, 0.82), (0.77, 0.85)]
# scene -> (video, half, walkway area, exit area or None, kinds); points are fractions of the picture
SCENES = {
    "MOT16-02": (f"{MOT16}/MOT16-02.mp4", "tune", [(0.3, 0.6), (1.0, 0.6), (1.0, 1.0), (0.3, 1.0)],
                 [(0.0, 0.56), (0.25, 0.56), (0.25, 0.75), (0.0, 0.75)], ALL),
    "FP9": (f"{DFIRE}/FP9.mp4", "tune", BOOTH_PATH, BOOTH_EXIT, ALL),
    "MOT16-04": (f"{MOT16}/MOT16-04.mp4", "report", [(0.06, 0.4), (0.32, 0.4), (0.34, 1.0), (0.06, 1.0)],
                 [(0.79, 0.62), (0.92, 0.62), (0.92, 0.8), (0.8, 0.8)], ALL),
    "MOT16-09": (f"{MOT16}/MOT16-09.mp4", "report", [(0.0, 0.78), (1.0, 0.7), (1.0, 1.0), (0.0, 1.0)],
                 [(0.14, 0.69), (0.55, 0.66), (0.57, 0.72), (0.13, 0.77)], ALL),
    "FP10": (f"{DFIRE}/FP10.mp4", "report", BOOTH_PATH, BOOTH_EXIT, ALL),
}
COCO_OBJECTS = {24: "backpack", 26: "handbag", 28: "suitcase", 56: "chair"}
SIZE = {"backpack": 0.33, "handbag": 0.25, "suitcase": 0.45, "chair": 0.55}   # height, in a nearby person's heights


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    """The whole video at FPS frames a second, at most MAX_W wide."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out, i, next_t = [], 0, 0.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i / fps + 1e-6 >= next_t:
            if f.shape[1] > MAX_W:
                f = cv2.resize(f, (MAX_W, round(f.shape[0] * MAX_W / f.shape[1])), interpolation=cv2.INTER_AREA)
            out.append(f)
            next_t += 1 / FPS
        i += 1
    cap.release()
    return out, fps


def person_height(frames: list[np.ndarray]) -> np.poly1d:
    """A person's height (fraction of the picture) by where their feet are (fraction), from the pose detector."""
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "yolo11s-pose.pt"))
    pts = []
    for f in frames[::10]:
        r = model.predict(f, classes=[0], conf=0.5, verbose=False)[0]
        h = f.shape[0]
        pts += [(b[3] / h, (b[3] - b[1]) / h) for b in r.boxes.xyxy.tolist() if (b[3] - b[1]) / h > 0.04]
    if len(pts) < 5:
        return np.poly1d([0.25])
    ys, hs = np.array(pts).T
    return np.poly1d(np.polyfit(ys, hs, 1))


def inside(pt, poly) -> bool:
    return cv2.pointPolygonTest(np.array(poly, np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def cut_objects(out: Path, per_class: int = 4) -> dict[str, list[Path]]:
    """Objects cut out of COCO photos (BGRA PNGs), the largest whole ones of each kind: nobody overlapping them (no
    worn bags, no one sitting on the chair), the mask's largest piece filling the photo's box for it."""
    have = {name: sorted((out / "objects").glob(f"{name}_*.png")) for name in COCO_OBJECTS.values()}
    if all(have.values()):   # cut before (and looked at by eye: a bad one can be deleted)
        return have
    from ultralytics import YOLOE
    model = YOLOE(str(ROOT / "data/models/yoloe/yoloe-11s-seg-pf.pt"))
    (out / "objects").mkdir(parents=True, exist_ok=True)
    labels = ROOT / "data/datasets/coco/labels/val2017"
    found: dict[int, list] = {c: [] for c in COCO_OBJECTS}
    for txt in sorted(labels.glob("*.txt")):
        rows = [r.split() for r in txt.read_text().splitlines() if r.strip()]
        for c in COCO_OBJECTS:
            mine = [list(map(float, r[1:5])) for r in rows if int(r[0]) == c]
            if len(mine) != 1:
                continue
            cx, cy, w, h = mine[0]
            people = [list(map(float, r[1:5])) for r in rows if int(r[0]) == 0]
            if any(_overlap((cx, cy, w, h), q) > 0.05 for q in people):
                continue
            if 0.04 < w * h < 0.5 and cx - w / 2 > 0.02 and cx + w / 2 < 0.98 and cy - h / 2 > 0.02 and cy + h / 2 < 0.98:
                found[c].append((w * h, txt.stem, (cx, cy, w, h)))
    result = {}
    for c, name in COCO_OBJECTS.items():
        paths = []
        for _, stem, (cx, cy, w, h) in sorted(found[c], reverse=True):
            if len(paths) >= per_class:
                break
            img = cv2.imread(str(ROOT / "data/datasets/coco/val2017" / f"{stem}.jpg"))
            H, W = img.shape[:2]
            box = np.array([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            r = model.predict(img, conf=0.1, retina_masks=True, verbose=False)[0]
            if r.masks is None:
                continue
            best, best_iou = None, 0.6
            for m, b in zip(r.masks.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                iou = _iou(b, box)
                if iou > best_iou:
                    best, best_iou = m, iou
            if best is None:
                continue
            mask = (cv2.resize(best.astype(np.float32), (W, H)) > 0.5).astype(np.uint8)
            n, comp, stats, _ = cv2.connectedComponentsWithStats(mask)
            if n < 2:
                continue
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (comp == biggest).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            ys, xs = np.nonzero(mask)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            fit = _iou((x1, y1, x2, y2), box)
            if fit < 0.75 or mask.sum() < 0.45 * (box[2] - box[0]) * (box[3] - box[1]):
                continue   # a piece of it, or something else
            rgba = np.dstack([img[y1:y2, x1:x2], mask[y1:y2, x1:x2] * 255])
            path = out / "objects" / f"{name}_{len(paths)}.png"
            cv2.imwrite(str(path), rgba)
            paths.append(path)
        result[name] = paths
    return result


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix)


def _overlap(a, b) -> float:
    """How much of box a (YOLO cx, cy, w, h) box b covers."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return ix / (a[2] * a[3])


def place(frame_shape, area, height_at, rng, size_of_person: float, aspect: float) -> tuple[int, int, int, int]:
    """A box (pixels) for an object standing on the floor inside the area: its bottom middle inside, sized to the
    people there."""
    h, w = frame_shape[:2]
    xs, ys = zip(*area)
    for _ in range(500):
        fx, fy = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
        oh = max(8, int(max(0.02, float(height_at(fy))) * size_of_person * h))
        ow = max(8, int(oh * aspect))
        x1, y2 = int(fx * w - ow / 2), int(fy * h)
        corners = [(x1 / w, fy), ((x1 + ow) / w, fy)]   # it stands on the floor; its top may rise out of the area
        if all(inside(c, area) for c in corners) and y2 - oh > 0 and x1 > 0 and x1 + ow < w:
            return x1, y2 - oh, x1 + ow, y2
    raise RuntimeError("no place for the object in the area")


def object_layer(obj: np.ndarray, box, frame: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """(times, plus) over the whole frame (each frame becomes frame * times + plus): the object sized into box,
    matched to the floor's brightness, with a soft shadow."""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    pic = cv2.resize(obj[..., :3], (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA).astype(np.float32)
    a = cv2.resize(obj[..., 3], (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    floor = frame[max(0, y2 - (y2 - y1) // 3):min(h, y2 + 4), max(0, x1):min(w, x2)].astype(np.float32)
    k = float(np.clip(floor.mean() / max(1.0, (pic * a[..., None]).sum() / max(1.0, a.sum() * 3))
                      * rng.uniform(0.6, 1.0), 0.35, 1.2))
    shadow = np.zeros((h, w), np.float32)   # a soft shadow on the floor under it
    cv2.ellipse(shadow, ((x1 + x2) // 2, y2), (max(2, int((x2 - x1) * 0.6)), max(2, int((y2 - y1) * 0.12))),
                0, 0, 360, 0.45, -1)
    times = (1 - cv2.GaussianBlur(shadow, (0, 0), max(1.0, (x2 - x1) * 0.1)))[..., None].repeat(3, 2)
    plus = np.zeros((h, w, 3), np.float32)
    times[y1:y2, x1:x2] *= 1 - a[..., None]
    plus[y1:y2, x1:x2] = pic * k * a[..., None]
    return times, plus


def spill_layer(frame: np.ndarray, area, height_at, rng, light: bool) -> tuple[np.ndarray, np.ndarray, str]:
    """(times, plus, words), as object_layer."""
    # A spill: a soft blob that darkens (coffee, water on a light floor) or lightens (milk) the floor.
    h, w = frame.shape[:2]
    box = place(frame.shape, area, height_at, rng, 0.25, 3.0)   # flat: wide and low
    cx, cy = (box[0] + box[2]) / 2, box[3] - (box[3] - box[1]) / 2
    rx, ry = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    radius = 1 + 0.25 * np.convolve(rng.normal(0, 1, 24 + 4), np.ones(5) / 5, "valid")
    pts = np.stack([cx + rx * radius * np.cos(angles), cy + ry * radius * np.sin(angles)], 1).astype(np.int32)
    m = np.zeros((h, w), np.float32)
    cv2.fillPoly(m, [pts], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, ry * 0.15))[..., None].repeat(3, 2)
    if light:
        return 1 - 0.6 * m, 141 * m, "light spill"
    return 1 - 0.5 * m, np.zeros_like(m), "dark spill"


def write(path: Path, frames: list[np.ndarray], seconds: int, times=None, plus=None) -> None:
    from labeler.config import load_config
    from labeler.recorder import FFmpegWriter, ffmpeg_path, pick_encoder
    cfg = load_config()
    ffmpeg = ffmpeg_path(cfg)
    h, w = frames[0].shape[:2]
    writer = FFmpegWriter(ffmpeg, path, (w, h), FPS, pick_encoder(ffmpeg) or "libx264", 28)
    for i in range(seconds * FPS):
        f = frames[i % len(frames)]
        if times is not None and i >= PUT_AT * FPS:
            f = (f * times + plus).clip(0, 255).astype(np.uint8)
        writer.write(f)
    ok, message = writer.close()
    if not ok:
        raise RuntimeError(f"writing {path} failed: {message}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data/datasets/floor_made")
    ap.add_argument("--preview", action="store_true", help="also draw the areas on each scene")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    objects = cut_objects(out)
    print({k: len(v) for k, v in objects.items()}, "objects cut out")
    truth, areas = [], {}
    for n, (scene, (video, half, walkway, exit_area, kinds)) in enumerate(SCENES.items()):
        rng = np.random.default_rng(47 + n)
        frames, _ = read_frames(ROOT / video)
        first = frames[0]
        height_at = person_height(frames)
        if args.preview:
            show = first.copy()
            for poly, color in ((walkway, (0, 255, 0)), (exit_area, (0, 0, 255))):
                if poly:
                    pts = (np.array(poly) * [first.shape[1], first.shape[0]]).astype(np.int32)
                    cv2.polylines(show, [pts], True, color, 2)
            cv2.imwrite(str(out / f"preview_{scene}.jpg"), show)
        for kind in kinds:
            name = f"{half}_{scene}_{kind}.mp4"
            rel = (out / name).relative_to(ROOT).as_posix()
            times = plus = None
            what = ""
            if kind in ("bag", "exit"):
                choice = ["backpack", "handbag", "suitcase"] if kind == "bag" else ["chair", "suitcase"]
                obj_name = choice[rng.integers(len(choice))]
                obj = cv2.imread(str(objects[obj_name][rng.integers(len(objects[obj_name]))]), cv2.IMREAD_UNCHANGED)
                box = place(first.shape, walkway if kind == "bag" else exit_area, height_at, rng,
                            SIZE[obj_name] * (1.3 if kind == "exit" else 1.0), obj.shape[1] / obj.shape[0])
                times, plus = object_layer(obj, box, first, rng)
                what = obj_name
            elif kind == "spill":
                times, plus, what = spill_layer(first, walkway, height_at, rng, light=bool(n % 2))
            if not (out / name).is_file():
                write(out / name, frames, SECONDS[kind], times, plus)
            if times is not None and args.preview:
                cv2.imwrite(str(out / f"preview_{half}_{scene}_{kind}.jpg"),
                            (first * times + plus).clip(0, 255).astype(np.uint8))
            category = {"none": "", "bag": "floor", "spill": "floor", "exit": "exit"}[kind]
            truth.append([rel, category, PUT_AT if category else "", SECONDS[kind] if category else "", ""])
            areas[rel] = [{"name": "walkway", "role": "walkway", "points": walkway}]
            if exit_area:
                areas[rel].append({"name": "exit", "role": "exit", "points": exit_area})
            print(f"{name}: {what or 'nothing put down'}")
    with open(out / "truth.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["video", "kind", "start_s", "end_s", "half"])
        wr.writerows(truth)
    (out / "areas.json").write_text(json.dumps(areas, indent=1), encoding="utf-8")
    print(len(truth), "videos;", out / "truth.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

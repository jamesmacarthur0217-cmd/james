"""What safety checks cost on this PC (Phase 44 step 2): the vision model's time per check, and whether people's
labels stay on time while checks jump ahead of them.

    python tools/incident_bench.py [--video data/datasets/mot16/train_videos/MOT16-09.mp4] [--n 8] [--minutes 2]

1. Time per check with 1, 3 and 4 pictures (labeler/incidents.py sends `safety.frames`, 3 by default), each request
   with pictures not sent before: Ollama reuses its work on a repeated picture (HANDOFF pitfall 20).
2. People's labels on time: people from the video's true boxes (MOT16 gt), one every --person-every-s, go through the
   shared vision model queue (labeler/vlm.py VLMWorker) for --minutes, first alone, then with a critical check
   (3 pictures) every 60 / --per-min seconds. Reported: how long each label waited (sent -> answered), median and 90th
   percentile, and labels still waiting at the end.
Needs Ollama running with the configured model, and the MOT16 videos (data/datasets/mot16, not in git).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeler.config import load_config  # noqa: E402
from labeler.incidents import CATEGORIES  # noqa: E402
from labeler.vlm import CRITICAL, VLMClient, VLMWorker, incident_prompt  # noqa: E402

GT = ROOT / "data" / "datasets" / "mot16" / "gt"


def read(video: Path, every: int = 4, limit: int = 200):
    """Frames (every `every`-th, up to `limit`) and the people in them (crops of true boxes at least 120 px tall)."""
    boxes: dict[int, list] = {}
    for line in (GT / f"{video.stem}.txt").read_text().splitlines():
        f, _, x, y, w, h = (float(v) for v in line.split(",")[:6])
        if h >= 120:
            boxes.setdefault(int(f), []).append((int(x), int(y), int(w), int(h)))
    cap = cv2.VideoCapture(str(video))
    frames, people, i = [], [], 0
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % every:
            continue
        frames.append(frame)
        for x, y, w, h in boxes.get(i, [])[:2]:
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size:
                people.append(crop)
    cap.release()
    return frames, people


def pq(values: list[float]) -> str:
    if not values:
        return "-"
    v = sorted(values)
    return f"median {statistics.median(v):.1f} s, 90% {v[min(len(v) - 1, int(len(v) * 0.9))]:.1f} s (n={len(v)})"


def check_prompt(n: int) -> str:
    return incident_prompt(n, 2, "", CATEGORIES["person_down"][2])


def times_per_check(client: VLMClient, frames: list, n: int) -> None:
    pos = 0
    took: dict[int, list[float]] = {1: [], 3: [], 4: []}
    if len(frames) < n * sum(took):
        sys.exit(f"only {len(frames)} pictures for {n * sum(took)} new ones: use a smaller --n")
    for _ in range(n):
        for k in took:   # alternated, each with new pictures
            pics = frames[pos:pos + k]
            pos += k
            t = time.monotonic()
            try:
                client.label(pics, None, ("incident", check_prompt(k)))
            except requests.HTTPError as e:   # say what Ollama did not like, and go on
                print(f"  {k} pictures: Ollama said {e.response.status_code}: {e.response.text[:200]}")
                continue
            took[k].append(time.monotonic() - t)
    for k, v in took.items():
        print(f"  {k} picture{'s' if k > 1 else ''}: {pq(v)}")


def run(client: VLMClient, cfg: dict, frames: list, people: list, minutes: float, person_every: float,
        per_min: int) -> None:
    sent: dict[int, float] = {}
    waits: list[float] = []
    checks: list[float] = []
    lock = threading.Lock()

    def done(job: int, results: list) -> None:
        with lock:
            t0 = sent.pop(job, None)
            if t0 is not None:
                (checks if job < 0 else waits).append(time.monotonic() - t0)

    worker = VLMWorker(client, done, cfg["vlm"]["max_queue"], cfg["vlm"]["workers"])
    per_track = int(cfg["vlm"]["frames_per_track"])
    start = time.monotonic()
    next_person, next_check, person, check, pic = start, start + 5, 0, 0, 0
    while time.monotonic() - start < minutes * 60:
        now = time.monotonic()
        if now >= next_person:
            person += 1
            crops = [(people[(person * per_track + j) % len(people)], None) for j in range(per_track)]
            with lock:
                sent[person] = now
            worker.submit(person, crops, "security")
            next_person += person_every
        if per_min and now >= next_check:
            check += 1
            pics = [frames[(pic + j) % len(frames)] for j in range(3)]
            pic += 3
            with lock:
                sent[-check] = now
            worker.submit(-check, [(pics, None)], ("incident", check_prompt(3)), priority=CRITICAL)
            next_check += 60 / per_min
        time.sleep(0.05)
    left = len([j for j in sent if j > 0])
    worker.close(120)
    print(f"  people's labels waited {pq(waits)}; still waiting at the end: {left}")
    if per_min:
        print(f"  checks waited {pq(checks)} ({check} sent)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", type=Path, default=ROOT / "data/datasets/mot16/train_videos/MOT16-09.mp4")
    ap.add_argument("--n", type=int, default=8, help="checks per picture count in part 1")
    ap.add_argument("--minutes", type=float, default=2.0, help="length of each run in part 2")
    ap.add_argument("--person-every-s", type=float, default=4.0, help="a new person this often (a busy camera)")
    ap.add_argument("--per-min", type=int, default=6, help="checks a minute in the second run (safety.confirm_per_min)")
    ap.add_argument("--skip-times", action="store_true")
    args = ap.parse_args()
    cfg = load_config(use_settings=False)
    client = VLMClient(cfg["vlm"], "security", cfg["labels"])
    ok, why = client.available()
    if not ok:
        sys.exit(why)
    frames, people = read(args.video)
    print(f"{args.video.name}: {len(frames)} pictures, {len(people)} people; model {cfg['vlm']['model']}, "
          f"{cfg['vlm']['workers']} at once, {cfg['vlm']['frames_per_track']} pictures per person")
    client.label(people[0], None, "security")   # load the model first
    if not args.skip_times:
        print("1. time per check")
        times_per_check(client, frames, args.n)
    print(f"2. people's labels, one person every {args.person_every_s:g} s for {args.minutes:g} min")
    print("  without checks:")
    run(client, cfg, frames, people, args.minutes, args.person_every_s, 0)
    print(f"  with {args.per_min} critical checks a minute:")
    run(client, cfg, frames, people, args.minutes, args.person_every_s, args.per_min)


if __name__ == "__main__":
    main()

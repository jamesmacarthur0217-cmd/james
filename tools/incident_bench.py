"""What safety checks cost on this PC (Phase 44 step 2): the vision model's time per check, and whether people's
labels stay on time while checks jump ahead of them.

    python tools/incident_bench.py [--video data/datasets/mot16/train_videos/MOT16-09.mp4] [--n 8] [--minutes 2]
    python tools/incident_bench.py --busy [--share 1.0] [--minutes 3] [--workers 2] [--stay-s 2] [--rates 1,6,2,1]

1. Time per check with 1, 3 and 4 pictures (labeler/incidents.py sends `safety.frames`, 3 by default), each request
   with pictures not sent before: Ollama reuses its work on a repeated picture (HANDOFF pitfall 20).
2. People's labels on time: people from the video's true boxes (MOT16 gt), one every --person-every-s, go through the
   shared vision model queue (labeler/vlm.py VLMWorker) for --minutes, first alone, then with a critical check
   (3 pictures) every 60 / --per-min seconds. Reported: how long each label waited (sent -> answered), median and 90th
   percentile, and labels still waiting at the end.
3. --busy (Phase 50): a busy camera in real time. People come and go as in the video's true marks (MOT16 gt, looped
   to --minutes; --share of them), each labelled like the camera does (vlm.frames_per_track pictures,
   tracking.min_frames_for_label frames after they appear, or with --stay-s only people who stay that long) and
   labelled again every vlm.relabel_every_s while they stay, when nothing waits (as Pipeline._due_for_relabel).
   Safety checks go through the real confirm queue (labeler/incidents.py ConfirmQueue, its tick every 0.5 s as in
   run.py) at --rates a minute: critical, warning, slow round, and the question after a direct alert. Reported per kind:
   how many, how long they waited (raised -> answered), how many were dropped (waited over safety.max_wait_s) or
   refused (people: the queue full); and the vision model's answers a minute and how busy it was.
Needs Ollama running with the configured model, and the MOT16 videos (data/datasets/mot16, not in git).
"""
from __future__ import annotations

import argparse
import math
import random
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


def arrivals(video: Path, minutes: float, share: float, seed: int = 47) -> list[tuple[float, float, object]]:
    """(first seen, stay, a crop) of each person in the video's true marks, the video looped to `minutes`, `share` of
    the people (a fixed random choice)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    marks: dict[int, list] = {}
    for line in (GT / f"{video.stem}.txt").read_text().splitlines():
        f, pid, x, y, w, h = (float(v) for v in line.split(",")[:6])
        marks.setdefault(int(pid), []).append((int(f), int(x), int(y), int(w), int(h)))
    want: dict[int, list] = {}   # frame -> people to cut out there (where each is tallest)
    for pid, m in marks.items():
        f, x, y, w, h = max(m, key=lambda r: r[4])
        want.setdefault(f, []).append((pid, x, y, w, h))
    crops, i = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        for pid, x, y, w, h in want.get(i, []):
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size:
                crops[pid] = crop
    cap.release()
    length = i / fps
    rng = random.Random(seed)
    people = [(min(r[0] for r in m) / fps, (max(r[0] for r in m) - min(r[0] for r in m)) / fps, crops[pid])
              for pid, m in sorted(marks.items()) if pid in crops and rng.random() < share]
    loops = math.ceil(minutes * 60 / length)
    return sorted(((k * length + t, stay, crop) for k in range(loops) for t, stay, crop in people
                   if k * length + t < minutes * 60), key=lambda a: a[0])


def busy(client: VLMClient, cfg: dict, frames: list, people: list, minutes: float, workers: int,
         stay_s: float | None, rates: dict[str, float]) -> None:
    """Part 3 (see the module text)."""
    from datetime import datetime

    from labeler.incidents import ConfirmQueue
    from labeler.vlm import PEOPLE
    lock = threading.Lock()
    busy_s = [0.0]
    answers = [0]
    label = client.label

    def timed(*a, **k):   # how long the vision model works (summed over requests at once)
        t = time.monotonic()
        try:
            return label(*a, **k)
        finally:
            with lock:
                busy_s[0] += time.monotonic() - t
                answers[0] += 1

    client.label = timed
    labels = {"sent": {}, "waits": [], "refused": 0, "relabels": 0}
    checks = {k: {"waits": [], "dropped": 0, "raised": 0, "refused": 0} for k in rates}
    last_label: dict[int, float] = {}

    def route(job, results):
        if q.owns(job):
            q.on_result(job, results)
            return
        with lock:
            t0 = labels["sent"].pop(job, None)
            if t0 is not None:
                labels["waits"].append(time.monotonic() - t0)
                last_label[job % 100000] = time.monotonic()

    worker = VLMWorker(client, route, cfg["vlm"]["max_queue"], workers)
    q = ConfirmQueue(cfg, worker, {})
    stop = threading.Event()
    threading.Thread(target=q.loop, args=(stop, 0.5), daemon=True).start()
    per_track = int(cfg["vlm"]["frames_per_track"])
    first_after = float(cfg["tracking"]["min_frames_for_label"]) / 30   # frames at about 30 a second
    every = float(cfg["vlm"]["relabel_every_s"])
    kinds = {"critical": ("person_down", None, None), "warning": ("floor", None, None),
             "round": ("equipment", "round", None), "words": ("suspicious", None, "What is the person doing?")}
    start = time.monotonic()
    due = [(t + (stay_s if stay_s is not None else first_after), stay, crop, n)
           for n, (t, stay, crop) in enumerate(people) if stay_s is None or stay >= stay_s]
    skipped = len(people) - len(due)
    nxt = {k: start + 3 + i for i, k in enumerate(rates)}
    count, pic, k = 0, 0, 0
    while time.monotonic() - start < minutes * 60:
        now = time.monotonic()
        while k < len(due) and now - start >= due[k][0]:
            _, stay, crop, n = due[k]
            k += 1
            job = n + 1
            with lock:
                labels["sent"][job] = now
            if not worker.submit(job, [(crop, None)] * per_track, "security", priority=PEOPLE):
                with lock:
                    labels["sent"].pop(job, None)
                    labels["refused"] += 1
        if every > 0 and worker.pending == 0:   # labelled again while they stay, when nothing waits
            for n, (t, stay, crop) in enumerate(people):
                done_at = last_label.get(n + 1)
                if done_at is not None and t <= now - start <= t + stay and now - done_at >= every \
                        and (n + 1) + 100000 * 1 not in labels["sent"]:
                    job = (n + 1) + 100000
                    with lock:
                        labels["sent"][job] = now
                        labels["relabels"] += 1
                        last_label[n + 1] = now
                    worker.submit(job, [(crop, None)], "security", priority=PEOPLE)
                    break
        for kind, per_min in rates.items():
            if per_min and now >= nxt[kind]:
                nxt[kind] += 60 / per_min
                count += 1
                category, level, question = kinds[kind]
                pics = [(datetime.now(), frames[(pic + j) % len(frames)], []) for j in range(3)]
                pic += 3
                c = checks[kind]
                raised = time.monotonic()

                def done(answer, c=c, raised=raised):
                    with lock:
                        if answer is None:
                            c["dropped"] += 1
                        else:
                            c["waits"].append(time.monotonic() - raised)
                got = q.suspect("cam", category, f"bench {kind}", people=[count], pictures=pics, level=level,
                                question=question, on_done=done, alert_if_yes=False if question else None,
                                cooldown=not question)
                c["raised"] += 1
                if got != "queued":
                    c["refused"] += 1
        time.sleep(0.05)
    seconds = time.monotonic() - start
    with lock:   # at the end of the time, not after the queue has emptied
        worked, answered = busy_s[0], answers[0]
    left_labels = len([j for j in labels["sent"] if j < 100000])
    stop.set()
    worker.close(60)
    client.label = label
    print(f"  {len(people)} people arrived ({len(people) / seconds * 60:.0f} a minute); labelled: {len(due)}"
          + (f" (passing in under {stay_s:g} s: {skipped} colours only)" if stay_s is not None else ""))
    print(f"  labels waited {pq(labels['waits'])}; refused (queue full) {labels['refused']}; still waiting at the end "
          f"{left_labels}; labelled again {labels['relabels']}")
    for kind, c in checks.items():
        if rates[kind]:
            print(f"  {kind:8s} {c['raised']} raised: waited {pq(c['waits'])}; dropped {c['dropped']}"
                  + (f", not sent {c['refused']}" if c["refused"] else ""))
    print(f"  the vision model: {answered / seconds * 60:.0f} answers a minute, busy "
          f"{worked / seconds / max(1, workers):.0%} of the time ({workers} at once)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", type=Path, default=ROOT / "data/datasets/mot16/train_videos/MOT16-09.mp4")
    ap.add_argument("--n", type=int, default=8, help="checks per picture count in part 1")
    ap.add_argument("--minutes", type=float, default=2.0, help="length of each run in part 2")
    ap.add_argument("--person-every-s", type=float, default=4.0, help="a new person this often (a busy camera)")
    ap.add_argument("--per-min", type=int, default=6, help="checks a minute in the second run (safety.confirm_per_min)")
    ap.add_argument("--skip-times", action="store_true")
    ap.add_argument("--busy", action="store_true", help="part 3 only: a busy camera (Phase 50)")
    ap.add_argument("--share", type=float, default=1.0, help="--busy: this share of the video's people")
    ap.add_argument("--workers", type=int, help="--busy: requests at once (default: vlm.workers)")
    ap.add_argument("--stay-s", type=float, help="--busy: label only people who stay this long")
    ap.add_argument("--rates", default="1,6,2,1", help="--busy: critical,warning,round,words checks a minute")
    args = ap.parse_args()
    cfg = load_config(use_settings=False)
    client = VLMClient(cfg["vlm"], "security", cfg["labels"])
    ok, why = client.available()
    if not ok:
        sys.exit(why)
    frames, people = read(args.video)
    if args.busy:
        rates = dict(zip(("critical", "warning", "round", "words"), (float(r) for r in args.rates.split(","))))
        coming = arrivals(args.video, args.minutes, args.share)
        workers = args.workers or int(cfg["vlm"]["workers"])
        client.label(people[0], None, "security")   # load the model first
        print(f"3. a busy camera: {args.video.name}, {args.share:g} of its people, {args.minutes:g} min, "
              f"checks a minute {rates}")
        busy(client, cfg, frames, coming, args.minutes, workers, args.stay_s, rates)
        return
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

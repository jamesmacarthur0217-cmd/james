"""How well do the safety checks work? (Phases 45-49; labeler/incidents.py.) Runs the real camera code on videos with
true time spans per kind of incident, and compares what the quick checks raised (suspects) and what the vision model
confirmed (alerts) with the truth.

    python tools/incident_check.py --config <test_config.yaml> --truth <truth.csv> [--kinds person_down,fall]
           [--fps 10] [--hold-s 0] [--no-vlm] [--json <file>]
    python tools/incident_check.py --make-truth urfd      # writes data/datasets/urfd/incidents_truth.csv
    python tools/incident_check.py --make-truth survfight # data/datasets/survfight/incidents_truth.csv (Phase 48)

The truth file (CSV, header video,kind,start_s,end_s,half): one row per incident span, seconds into the video; a video
with no incident has one row with an empty kind (it only counts for false alarms). half: "right" for the UR Fall
videos (the color picture; the left half is a depth map), else empty. Paths are relative to the project folder.

Each video runs through the app's own Pipeline (security camera, safety on, a scratch database) on the video's own
clock (labeler/pipeline.py VideoClock), every --fps frames a second. The confirm queue (labeler/incidents.py) also runs
on the video's clock (its per-minute limit, cooldown and waiting limit in video seconds), and the video pauses while
the vision model answers: this measures what is found, not how fast the PC is (tools/incident_bench.py measures that).
--hold-s repeats each video's last picture that long (UR Fall clips end a few seconds after the fall). --no-vlm answers
every check "yes": the quick checks alone. --labels: people also get their labels from the vision model, as in the app
(the age group decides who is a child; the video waits for them too). --areas: a JSON file {video: [{"name", "role",
"points"}]} with areas drawn on that video's picture (fractions), e.g. a cooking area; "kind": "line" for a counting
line (2 points), e.g. a door (Phase 54).

Per kind it prints:
  spans      true incident spans in the videos (kind given)
  raised     of these, spans with a suspect of that kind from start - 2 s to end + --late-s
  found      of these, spans with a confirmed alert ("yes")
  false/h    suspects (raised) and alerts (found) outside every true span of that kind, per hour of video
  late       seconds from the span's start to the first alert in it (median, worst)
  answers    the vision model's yes / no / unsure for that kind
Nothing is written to the user's data: use a test config from tools/make_test_config.py.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import re
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeler.config import load_config  # noqa: E402

SLACK_S = 2.0   # a suspect or alert up to this long before a span's start still belongs to it


def read_truth(path: Path) -> dict[str, dict]:
    """video -> {"half": "", "spans": [(kind, start, end)]}, in file order."""
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = out.setdefault(row["video"].strip(), {"half": "", "spans": []})
            v["half"] = v["half"] or (row.get("half") or "").strip()
            if (row.get("kind") or "").strip():
                v["spans"].append((row["kind"].strip(), float(row["start_s"]), float(row["end_s"])))
    return out


def score(truth: dict[str, dict], results: dict[str, dict], kinds: list[str], late_s: float) -> list[dict]:
    """results: video -> {"seconds", "suspects": [(t, kind)], "answers": [(t, kind, answer)]}. One row per kind."""
    rows = []
    hours = sum(r["seconds"] for r in results.values()) / 3600 or 1e-9
    for kind in kinds:
        spans = raised = found = 0
        false_s = false_a = 0
        late: list[float] = []
        answers = defaultdict(int)
        for video, res in results.items():
            mine = [(a, b) for k, a, b in truth.get(video, {"spans": []})["spans"] if k == kind]
            sus = [t for t, k in res["suspects"] if k == kind]
            yes = [t for t, k, ans, *_ in res["answers"] if k == kind and ans == "yes"]
            for _, k, ans, *_ in res["answers"]:
                if k == kind:
                    answers[ans] = answers.get(ans, 0) + 1
            inside = lambda t: any(a - SLACK_S <= t <= b + late_s for a, b in mine)   # noqa: E731
            false_s += sum(not inside(t) for t in sus)
            false_a += sum(not inside(t) for t in yes)
            for a, b in mine:
                spans += 1
                raised += any(a - SLACK_S <= t <= b + late_s for t in sus)
                hits = [t for t in yes if a - SLACK_S <= t <= b + late_s]
                if hits:
                    found += 1
                    late.append(max(0.0, min(hits) - a))
        rows.append({"kind": kind, "spans": spans, "raised": raised, "found": found,
                     "false_raised_per_h": round(false_s / hours, 1), "false_found_per_h": round(false_a / hours, 1),
                     "false_raised": false_s, "false_found": false_a,
                     "late_median_s": round(statistics.median(late), 1) if late else None,
                     "late_worst_s": round(max(late), 1) if late else None,
                     "answers": dict(answers), "hours": round(hours, 3)})
    return rows


def frames(video: Path, fps: float, half: str, hold_s: float):
    """(frame, seconds) of the video at about fps, the right or left half if asked, the last picture held hold_s."""
    from labeler.importer import stride_for
    from labeler.sources import RecordedVideo, probe_video
    stride = stride_for(probe_video(video)["fps"], fps)
    last, t_last = None, 0.0
    for frame, t in RecordedVideo(video, stride).frames():
        if half == "right":
            frame = frame[:, frame.shape[1] // 2:]
        elif half == "left":
            frame = frame[:, :frame.shape[1] // 2]
        last, t_last = frame, t
        yield frame, t
    if last is not None:
        step = 1 / max(fps, 0.1)
        for i in range(1, int(hold_s * fps) + 1):
            yield last, t_last + i * step


class Yes:
    """--no-vlm: every check answered "yes" at once."""

    def __init__(self):
        self.queue = None

    def submit(self, job, pictures, mode=None, priority=0):
        self.queue.on_result(job, [{"answer": "yes", "what": "(no vision model)", "seconds": "0"}])
        return True


def run_video(video: Path, cfg: dict, embedder, fps: float, half: str, hold_s: float, client, labels: bool = False,
              areas: list | None = None) -> dict:
    """One video through the Pipeline and the confirm queue; returns its seconds, suspects and answers."""
    from labeler.db import Database
    from labeler.incidents import ConfirmQueue
    from labeler.pipeline import Pipeline, VideoClock
    from labeler.vlm import VLMWorker

    db = Database("sqlite:///" + (Path(tempfile.mkdtemp(prefix="incident_check_")) / "check.db").as_posix())
    for a in areas or []:
        db.add_zone("check", a["name"], a.get("kind", "area"), a["points"], role=a.get("role", ""))
    cam = {"name": "check", "source": "imported videos", "mode": "security", "loop_files": False, "ignore_zones": [],
           "vehicles": False, "animals": False, "safety": True}
    cfg = copy.deepcopy(cfg)
    cfg["clips"]["enabled"] = False
    cfg["person_videos"]["enabled"] = False
    p = Pipeline(cfg, db, None, embedder, None, camera=cam)
    p.clock = VideoClock(datetime(2026, 1, 1))
    if p.floor_watch is not None:
        p.floor_watch.path = None   # each video learns its own empty picture, and nothing is saved
    suspects, answers = [], []
    worker = Yes() if client is None else None
    q = ConfirmQueue(cfg, worker, {"check": p.memory}, clock=p.clock.mono)
    if client is None:
        worker.queue = q
    else:
        def route(job, results):   # as run.py: the checks' answers to the queue, people's labels to the camera
            (q.on_result if q.owns(job) else p.on_vlm_result)(job, results)
        q.worker = VLMWorker(client, route)
        if labels:
            p.vlm = q.worker
    plain = q.suspect

    def suspect(camera, category, reason, *a, **k):
        got = plain(camera, category, reason, *a, **k)
        suspects.append((round(p.clock.t, 2), category))
        return got

    q.suspect = suspect
    plain_direct = q.direct

    def direct(camera, category, reason, *a, **k):   # Phase 49: an alert without the vision model's yes
        got = plain_direct(camera, category, reason, *a, **k)
        suspects.append((round(p.clock.t, 2), category))   # its "yes" comes through the listeners
        return got

    q.direct = direct
    # A first look (floor hazards: two "yes" alert) is listed, but its "yes" is no alert.
    q.listeners.append(lambda s, a: answers.append((round(p.clock.t, 2), s.category,
                                                    a["answer"] + (" (first look)" if s.alert_if_yes is False else ""),
                                                    s.reason, a.get("what", ""))))
    p.incidents = q   # the quick checks (Phase 45+) raise suspects through this
    t = 0.0
    for frame, t in frames(video, fps, half, hold_s):
        p.clock.t = t
        p.process(frame)
        p.frames += 1
        if q.tick() is not None:
            while q.sent:   # the video waits for the answer (see the module text)
                time.sleep(0.05)
        if labels and client is not None:
            while not q.worker.idle:   # and for people's labels
                time.sleep(0.05)
    for tid in list(p.tracks):
        p._finish(tid)
    if client is not None:
        q.worker.close(5)
    return {"seconds": t, "suspects": suspects, "answers": answers}


def make_truth_urfd(out: Path) -> int:
    """UR Fall (camera 0, the color half): a "fall" span over each fall's falling frames (marked 0) and a
    "person_down" span from the first lying frame (marked 1) to the end of the clip plus 60 s (held); the everyday
    clips have no incident (their slow lying down on a bed or the floor is the false-alarm test)."""
    urfd = ROOT / "data" / "datasets" / "urfd"
    marks: dict[str, dict[int, int]] = defaultdict(dict)
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        for row in csv.reader(open(urfd / name)):
            marks[row[0]][int(row[1])] = int(row[2])
    rows = []
    for video, m in sorted(marks.items()):
        path = f"data/datasets/urfd/{video}-cam0.mp4"
        if not (ROOT / path).is_file():
            continue
        falling = [f for f, v in m.items() if v == 0]
        lying = [f for f, v in m.items() if v == 1]
        if video.startswith("fall") and falling and lying:
            rows.append([path, "fall", round((min(falling) - 1) / 30, 2), round(max(falling) / 30, 2), "right"])
            rows.append([path, "person_down", round((min(lying) - 1) / 30, 2), round(max(m) / 30 + 60, 2), "right"])
        else:
            rows.append([path, "", "", "", "right"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "kind", "start_s", "end_s", "half"])
        w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="a test config (tools/make_test_config.py)")
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--kinds", help="kinds to report (default: those in the truth file)")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--hold-s", type=float, default=0.0)
    ap.add_argument("--late-s", type=float, default=30.0, help="an alert this long after a span still finds it")
    ap.add_argument("--no-vlm", action="store_true", help="answer every check yes (the quick checks alone)")
    ap.add_argument("--labels", action="store_true", help="people get labels from the vision model (their age)")
    ap.add_argument("--areas", type=Path, help="JSON {video: [{name, role, points}]}")
    ap.add_argument("--limit", type=int, help="only the first N videos")
    ap.add_argument("--match", help="only videos whose path matches this regular expression (a tuning half)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--make-truth", choices=["urfd", "mot16", "survfight"])
    args = ap.parse_args()
    if args.make_truth == "survfight":   # Surveillance Camera Fight Dataset (Phase 48): a fight over each whole fight clip
        import cv2
        folder = ROOT / "data" / "datasets" / "survfight"
        out = folder / "incidents_truth.csv"
        rows = []
        for kind, sub in (("fight", "fight"), ("", "noFight")):
            for v in sorted((folder / sub).glob("*.mp4")):
                cap = cv2.VideoCapture(str(v))
                length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25)
                rows.append([v.relative_to(ROOT).as_posix(), kind, 0 if kind else "",
                             round(length, 2) if kind else "", ""])
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "kind", "start_s", "end_s", "half"])
            w.writerows(rows)
        print(f"{len(rows)} rows -> {out}  (tune: --match '[02468]\\.mp4$', report: --match '[13579]\\.mp4$')")
        return 0
    if args.make_truth == "urfd":
        out = ROOT / "data" / "datasets" / "urfd" / "incidents_truth.csv"
        print(f"{make_truth_urfd(out)} rows -> {out}")
        return 0
    if args.make_truth == "mot16":   # people walking and standing in streets and a mall: no incident
        out = ROOT / "data" / "datasets" / "mot16" / "incidents_truth.csv"
        videos = sorted((ROOT / "data" / "datasets" / "mot16" / "train_videos").glob("*.mp4"))
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "kind", "start_s", "end_s", "half"])
            w.writerows([[v.relative_to(ROOT).as_posix(), "", "", "", ""] for v in videos])
        print(f"{len(videos)} rows -> {out}")
        return 0
    if not args.config or not args.truth:
        ap.error("--config and --truth are needed (or --make-truth)")
    logging.basicConfig(level=logging.ERROR)
    cfg = load_config(args.config)
    truth = read_truth(args.truth)
    from labeler.incidents import CATEGORIES
    kinds = args.kinds.split(",") if args.kinds else sorted({k for v in truth.values() for k, _, _ in v["spans"]})
    unknown = [k for k in kinds if k not in CATEGORIES]
    if unknown:
        ap.error(f"unknown kinds {unknown} (one of: {', '.join(CATEGORIES)})")
    client = None
    if not args.no_vlm:
        from labeler.vlm import VLMClient
        client = VLMClient(cfg["vlm"], "security", cfg["labels"])
        ok, why = client.available()
        if not ok:
            sys.exit(f"{why} (or use --no-vlm)")
    from labeler.embed import Embedder
    embedder = Embedder(cfg["embeddings"])
    results = {}
    areas = json.loads(args.areas.read_text(encoding="utf-8")) if args.areas else {}
    if args.labels and args.no_vlm:
        ap.error("--labels needs the vision model")
    videos = [v for v in truth if not args.match or re.search(args.match, v)]
    videos = videos[: args.limit] if args.limit else videos
    for i, video in enumerate(videos, 1):
        started = time.monotonic()
        results[video] = run_video(ROOT / video, cfg, embedder, args.fps, truth[video]["half"], args.hold_s, client,
                                   args.labels, areas.get(video))
        r = results[video]
        print(f"[{i}/{len(videos)}] {Path(video).name}: {r['seconds']:.0f} s, {len(r['suspects'])} suspects, "
              f"{sum(a[2] == 'yes' for a in r['answers'])} confirmed ({time.monotonic() - started:.0f} s)", flush=True)
    rows = score(truth, results, kinds, args.late_s)
    print(f"\n{len(videos)} videos, {rows[0]['hours'] if rows else 0} h" + (" (no vision model: every check yes)"
                                                                          if args.no_vlm else ""))
    print(f"{'kind':>12} {'spans':>6} {'raised':>7} {'found':>6} {'false/h raised':>15} {'false/h found':>14} "
          f"{'late':>11}  answers")
    for r in rows:
        late = f"{r['late_median_s']}/{r['late_worst_s']} s" if r["late_median_s"] is not None else "-"
        print(f"{r['kind']:>12} {r['spans']:>6} {r['raised']:>7} {r['found']:>6} {r['false_raised_per_h']:>15} "
              f"{r['false_found_per_h']:>14} {late:>11}  {r['answers']}")
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "results": results}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

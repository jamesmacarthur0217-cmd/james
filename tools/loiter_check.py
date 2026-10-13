"""How well do loitering alerts work? Runs the real pipeline on fixed-camera MOTChallenge videos that have true
per-person markings (gt.txt) with areas drawn on the picture and "stayed" rules, and compares the alerts with how
long each true person really stayed inside each area.

    python tools/loiter_check.py --config <test_config.yaml> [--clips MOT16-02,MOT16-04,MOT16-09]
           [--stays 5,10,15] [--fps 10] [--imgsz 1280] [--json <file>]

Areas (fractions of the picture): "all" = the whole picture (staying in view), "middle" = the middle third from
left to right. A person is inside where their feet are (the bottom middle of the box), as on a security camera.

For each rule time it prints:
  stayed    true people who stayed inside that long in one go (from the markings; feet inside), counting only
            people at least --min-visibility visible on average (0.5: MOT16 also marks people hidden behind others)
  seen      of these, people the system recorded at all (a record follows them 5+ frames)
  held      of these, people one record followed at least that long (a broken track starts the time again)
  found     of these, people with an alert of that rule
  false     alerts on someone who did not stay that long (1 s of slack: boxes and feet differ a little)
  late      seconds from the moment the true person reached the time to the alert (median, worst)

An alert belongs to the true person its record follows most (IoU >= 0.5, tools/relink_check.py). Videos and
markings: data/datasets/mot16/train_videos and data/datasets/mot16/gt (walking people only). Nothing is written to
the user's data: use a test config from tools/make_test_config.py.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from labeler.config import load_config  # noqa: E402
from labeler.zones import inside  # noqa: E402
from relink_check import match, read_gt  # noqa: E402

DATA = ROOT / "data" / "datasets" / "mot16"
AREAS = {"all": [(0.01, 0.01), (0.99, 0.01), (0.99, 0.99), (0.01, 0.99)],
         "middle": [(1 / 3, 0.0), (2 / 3, 0.0), (2 / 3, 1.0), (1 / 3, 1.0)]}
SLACK_S = 1.0
GT_FPS = 30.0


def visibility(path: Path) -> dict[int, float]:
    """true person -> their mean visibility in the markings (MOTChallenge column 9: 1 = fully visible)."""
    vis: dict[int, list[float]] = defaultdict(list)
    for line in path.read_text().split():
        f = line.split(",")
        vis[int(float(f[1]))].append(float(f[8]) if len(f) > 8 else 1.0)
    return {pid: sum(v) / len(v) for pid, v in vis.items()}


def true_stays(gt: dict, size: tuple[int, int], area: list) -> dict[int, list[tuple[float, float]]]:
    """true person -> [(start s, end s)] of each stretch with their feet inside the area (one frame apart)."""
    w, h = size
    frames: dict[int, list[int]] = defaultdict(list)
    for frame, people in gt.items():
        for pid, (x1, y1, x2, y2) in people:
            if inside(((x1 + x2) / 2 / w, y2 / h), area):
                frames[pid].append(frame)
    out = {}
    for pid, fs in frames.items():
        fs.sort()
        runs, start, prev = [], fs[0], fs[0]
        for f in fs[1:]:
            if f != prev + 1:
                runs.append(((start - 1) / GT_FPS, prev / GT_FPS))
                start = f
            prev = f
        runs.append(((start - 1) / GT_FPS, prev / GT_FPS))
        out[pid] = runs
    return out


def reached(runs: list[tuple[float, float]], secs: float) -> float | None:
    """When the person first had been inside `secs` in one go, or None."""
    for a, b in runs:
        if b - a >= secs:
            return a + secs
    return None


def run(video: Path, cfg: dict, imgsz: int, fps: float, stays: list[int], embedder) -> tuple[dict, list[dict]]:
    """The pipeline with the areas and a rule per area and time; returns record -> [(gt frame, box)] and the
    alerts as (rule name, record, seconds into the video)."""
    from labeler.alerts import AlertEngine
    from labeler.db import Database
    from labeler.importer import stride_for
    from labeler.pipeline import Pipeline, VideoClock
    from labeler.sources import RecordedVideo, probe_video

    db = Database("sqlite:///" + (Path(tempfile.mkdtemp(prefix="loiter_check_")) / "check.db").as_posix())
    for name, points in AREAS.items():
        db.add_zone("check", name, "area", points)
        for s in stays:
            db.add_rule(f"{name}:{s}", [["stayed", f"{name}:{s}"]])
    cam = {"name": "check", "source": "imported videos", "mode": "security", "loop_files": False,
           "ignore_zones": [], "imgsz": imgsz, "vehicles": False, "fall_alert": False}
    cfg = copy.deepcopy(cfg)
    cfg["clips"]["enabled"] = False
    cfg["person_videos"]["enabled"] = False
    p = Pipeline(cfg, db, None, embedder, None, camera=cam)
    p.clock = VideoClock(datetime(2026, 1, 1))
    engine = AlertEngine(db)
    rule_name = {r["id"]: r["name"] for r in db.list_rules()}
    alerts = []
    plain = engine.check

    def check(track_id, camera, fallen=False, stayed=None):
        fired = plain(track_id, camera, fallen=fallen, stayed=stayed)
        alerts.extend({"rule": rule_name[e["rule_id"]], "record": track_id, "t": round(p.clock.t, 2)} for e in fired)
        return fired

    engine.check = check
    p.alerts = engine
    seen: dict[int, list] = defaultdict(list)
    states = {}
    stride = stride_for(probe_video(video)["fps"], fps)
    for i, (frame, t) in enumerate(RecordedVideo(video, stride).frames()):
        p.clock.t = t
        p.process(frame)
        p.frames += 1
        for st in p.tracks.values():
            if st.last_seen == p.clock.mono() and not st.deleted:
                states[id(st)] = st
                seen[id(st)].append((i * stride + 1, tuple(float(v) for v in st.box)))
    for tid in list(p.tracks):
        p._finish(tid)
    records: dict[int, list] = defaultdict(list)
    for key, s in seen.items():
        if states[key].db_id is not None:
            records[states[key].db_id] += s
    return records, alerts


def follows(records: dict, gt: dict) -> dict[int, float]:
    """true person -> the longest time one record followed them (first to last matched frame, seconds)."""
    from labeler.geometry import box_iou
    spans: dict[tuple, list[int]] = defaultdict(list)
    for rec, seen in records.items():
        for frame, box in seen:
            best = max(((box_iou(box, g), pid) for pid, g in gt.get(frame, [])), default=(0.0, None))
            if best[0] >= 0.5:
                spans[(rec, best[1])].append(frame)
    out: dict[int, float] = {}
    for (rec, pid), frames in spans.items():
        if len(frames) >= 5:
            out[pid] = max(out.get(pid, 0.0), (max(frames) - min(frames)) / GT_FPS)
    return out


def score(clip: str, gt: dict, size: tuple, records: dict, alerts: list[dict], stays: list[int],
          vis: dict[int, float], min_vis: float) -> list[dict]:
    owner = {rec: c.most_common(1)[0][0] for rec, c in match(records, gt).items() if c}
    longest = follows(records, gt)
    rows = []
    for name, area in AREAS.items():
        truth = true_stays(gt, size, area)
        for s in stays:
            rule = f"{name}:{s}"
            # Counted: people mostly visible (min_vis). An alert on a hidden person who did stay is still right.
            stayed = {pid: reached(runs, s) for pid, runs in truth.items()
                      if reached(runs, s) is not None and vis.get(pid, 1.0) >= min_vis}
            got = [a for a in alerts if a["rule"] == rule]
            found, false, late = set(), 0, []
            for a in got:
                pid = owner.get(a["record"])
                if pid is not None and reached(truth.get(pid, []), s - SLACK_S) is not None:
                    if pid in stayed and pid not in found:
                        found.add(pid)
                        late.append(a["t"] - stayed[pid])
                else:
                    false += 1
            seen = [pid for pid in stayed if pid in longest]
            held = [pid for pid in stayed if longest.get(pid, 0.0) >= s]
            rows.append({"clip": clip, "area": name, "stay_s": s, "stayed": len(stayed), "seen": len(seen),
                         "held": len(held), "found": len(found),
                         "alerts": len(got), "false": false,
                         "late_median_s": round(statistics.median(late), 1) if late else None,
                         "late_max_s": round(max(late), 1) if late else None})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a test config (tools/make_test_config.py)")
    ap.add_argument("--clips", default="MOT16-02,MOT16-04,MOT16-09")
    ap.add_argument("--stays", default="5,10,15")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--min-visibility", type=float, default=0.5,
                    help="count only people at least this visible on average in the markings (0 = everyone)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    cfg = load_config(args.config)
    from labeler.embed import Embedder
    from labeler.sources import probe_video
    embedder = Embedder(cfg["embeddings"])
    stays = [int(s) for s in args.stays.split(",")]
    rows = []
    print(f"{'clip':>9} {'area':>7} {'stay':>5} {'stayed':>7} {'seen':>5} {'held':>5} {'found':>6} {'alerts':>7} "
          f"{'false':>6} {'late':>11}")
    for clip in args.clips.split(","):
        video, gt_file = DATA / "train_videos" / f"{clip}.mp4", DATA / "gt" / f"{clip}.txt"
        info = probe_video(video)
        gt = read_gt(gt_file)
        records, alerts = run(video, cfg, args.imgsz, args.fps, stays, embedder)
        for r in score(clip, gt, (info["width"], info["height"]), records, alerts, stays, visibility(gt_file),
                       args.min_visibility):
            rows.append(r)
            late = "-" if r["late_median_s"] is None else f"{r['late_median_s']}/{r['late_max_s']} s"
            print(f"{clip:>9} {r['area']:>7} {r['stay_s']:>4}s {r['stayed']:>7} {r['seen']:>5} {r['held']:>5} "
                  f"{r['found']:>6} {r['alerts']:>7} "
                  f"{r['false']:>6} {late:>11}", flush=True)
    print("total:")
    for name in AREAS:
        for s in stays:
            sel = [r for r in rows if r["area"] == name and r["stay_s"] == s]
            print(f"  {name:>7} {s:>3}s: found {sum(r['found'] for r in sel)} of {sum(r['stayed'] for r in sel)} "
                  f"(seen {sum(r['seen'] for r in sel)}, held {sum(r['held'] for r in sel)}), "
                  f"false {sum(r['false'] for r in sel)} of {sum(r['alerts'] for r in sel)} alerts")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

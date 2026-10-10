"""How well does the system count people? Runs the real pipeline on a video that has true per-person markings
(MOTChallenge gt.txt: frame, person, left, top, width, height, ...) and compares.

    python tools/relink_check.py <video> <gt.txt> --config <test_config.yaml> [--imgsz 1280] [--fps 10]
           [--long-gap 0.90,0.95,off] [--sheet <folder>]

For each setting of tracking.relink.long_gap_min_similarity ("off" = re-linking turned off) it prints:
  true      people in the markings
  counted   people the system recorded
  found     true people the system recorded at all
  merged    records that contain two or more true people (wrong joins: different people made one)
  split     true people recorded as two or more records (missed joins)
  joins     re-link decisions; right = the same true person came back, wrong = a different one

A record follows a true person in a frame when their boxes overlap by IoU >= 0.5. Nothing is written to the
user's data: use a test config from tools/make_test_config.py. With --sheet, pictures of the wrong joins.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labeler.config import load_config  # noqa: E402
from labeler.geometry import box_iou  # noqa: E402

MIN_SHARE_FRAMES = 5   # a true person counts as part of a record from this many matched frames on


def read_gt(path: Path) -> dict[int, list[tuple[int, tuple]]]:
    """frame (1-based) -> [(true person, (x1, y1, x2, y2))]"""
    out: dict[int, list] = defaultdict(list)
    for line in path.read_text().split():
        f = line.split(",")
        frame, pid, x, y, w, h = int(float(f[0])), int(float(f[1])), *map(float, f[2:6])
        out[frame].append((pid, (x, y, x + w, y + h)))
    return out


def match(records: dict[int, list[tuple[int, tuple]]], gt: dict) -> dict[int, Counter]:
    """record -> Counter(true person -> frames where they overlap)."""
    out: dict[int, Counter] = defaultdict(Counter)
    for rec, seen in records.items():
        for frame, box in seen:
            best = max(((box_iou(box, g), pid) for pid, g in gt.get(frame, [])), default=(0.0, None))
            if best[0] >= 0.5:
                out[rec][best[1]] += 1
    return out


def main_person(seen: list, gt: dict) -> int | None:
    """The true person a piece of track follows most (None if it follows nobody, e.g. a false detection)."""
    c = match({0: seen}, gt).get(0)
    return c.most_common(1)[0][0] if c else None


def judge(joins: list[dict], gt: dict) -> None:
    """Each join is right when the returning piece follows the same true person as the record so far."""
    for j in joins:
        old, new = main_person(j["old_seen"], gt), main_person(j["new_seen"], gt)
        j["right"] = old is not None and old == new
        j["old_person"], j["new_person"] = old, new
        # How fast the person would have walked: from where the record was last seen to where the new piece
        # appeared, in body heights per second (walking is about 0.8: 1.4 m/s for a 1.7 m person).
        if j["old_seen"] and j["new_seen"]:
            (f0, a), (f1, b) = j["old_seen"][-1], j["new_seen"][0]
            height = max(1.0, ((a[3] - a[1]) + (b[3] - b[1])) / 2)
            dist = (((a[0] + a[2]) - (b[0] + b[2])) ** 2 + ((a[1] + a[3]) - (b[1] + b[3])) ** 2) ** 0.5 / 2
            j["speed_heights_s"] = round(dist / height / max(j["gap_s"], 0.1), 2)


def score(records: dict, gt: dict, joins: list[dict]) -> dict:
    m = match(records, gt)
    people = {rec: {p for p, n in c.items() if n >= MIN_SHARE_FRAMES} for rec, c in m.items()}
    true_ids = {pid for rows in gt.values() for pid, _ in rows}
    by_person = defaultdict(set)
    for rec, ps in people.items():
        for p in ps:
            by_person[p].add(rec)
    wrong = [j for j in joins if j["record"] in people and len(people[j["record"]]) > 1]
    return {"true": len(true_ids), "counted": len(records), "found": len(by_person),
            "merged": sum(1 for ps in people.values() if len(ps) > 1),
            "split": sum(1 for recs in by_person.values() if len(recs) > 1),
            "joins": len(joins), "right": sum(1 for j in joins if j.get("right")),
            "wrong": sum(1 for j in joins if j.get("right") is False),
            "records_with_wrong_join": len({j["record"] for j in wrong}),
            "_people": {str(k): sorted(v) for k, v in people.items()}}


class JoinLog(logging.Handler):
    PATTERN = re.compile(r"person #(\d+) is back after ([\d.]+) s \(similarity ([\d.]+), box overlap ([\d.]+)\)")

    def __init__(self):
        super().__init__()
        self.joins: list[dict] = []

    def emit(self, record):
        m = self.PATTERN.search(record.getMessage())
        if m:
            self.joins.append({"record": int(m.group(1)), "gap_s": float(m.group(2)),
                               "similarity": float(m.group(3)), "overlap": float(m.group(4))})


vectors: dict = {}   # record -> appearance vector, from the last run (for the look-alike check)


def lookalikes(records: dict, gt: dict, thresholds=(0.8, 0.85, 0.9, 0.95)) -> list[dict]:
    """Look-alike search and the timeline rank people by the same appearance similarity: of the pairs of records
    at or above a threshold, how many are really the same person?"""
    who = {rec: main_person(seen, gt) for rec, seen in records.items()}
    ids = [r for r in records if r in vectors and who[r] is not None]
    pairs = [(float(vectors[a] @ vectors[b]), who[a] == who[b]) for i, a in enumerate(ids) for b in ids[i + 1:]]
    same_total = sum(1 for _, same in pairs if same)
    out = []
    for th in thresholds:
        hits = [same for sim, same in pairs if sim >= th]
        out.append({"threshold": th, "pairs": len(hits), "same_person": sum(hits),
                    "found_of_same": f"{sum(hits)}/{same_total}"})
    # Look-alikes are shown closest first: for records whose person has another record, is the closest one theirs?
    top1 = top3 = n = 0
    for a in ids:
        others = sorted((b for b in ids if b != a), key=lambda b: -float(vectors[a] @ vectors[b]))
        if not any(who[b] == who[a] for b in others):
            continue
        n += 1
        top1 += who[others[0]] == who[a]
        top3 += any(who[b] == who[a] for b in others[:3])
    out.append({"ranking": f"closest is the same person {top1}/{n}, among the closest 3: {top3}/{n}"})
    return out


def run(video: Path, cfg: dict, imgsz: int, fps: float, shared: dict) -> tuple[dict, list[dict]]:
    """The pipeline on every stride-th frame; returns record -> [(gt frame number, box)] and the joins."""
    from labeler.db import Database
    from labeler.importer import stride_for
    from labeler.pipeline import Pipeline, VideoClock
    from labeler.sources import RecordedVideo, probe_video

    import tempfile
    db = Database("sqlite:///" + (Path(tempfile.mkdtemp(prefix="relink_check_")) / "check.db").as_posix())
    cam = {"name": "check", "source": "imported videos", "mode": "security", "loop_files": False,
           "ignore_zones": [], "imgsz": imgsz}
    cfg = copy.deepcopy(cfg)
    cfg["clips"]["enabled"] = False
    p = Pipeline(cfg, db, None, shared["embedder"], shared["student"], camera=cam)
    p.age = shared["age"]
    p.clock = VideoClock(datetime(2026, 1, 1))
    log = JoinLog()
    logging.getLogger("labeler.pipeline").addHandler(log)
    seen_by_state: dict[int, list] = defaultdict(list)
    states = {}
    pairs = []   # (old state, new piece) at each join, with what each had seen until then
    original = p._try_relink

    def try_relink(tid, new, now):
        result = original(tid, new, now)
        if result is not new:
            pairs.append({"old_seen": list(seen_by_state[id(result)]), "new_seen": list(seen_by_state[id(new)])})
        return result

    p._try_relink = try_relink
    info = probe_video(video)
    stride = stride_for(info["fps"], fps)
    try:
        for i, (frame, t) in enumerate(RecordedVideo(video, stride).frames()):
            p.clock.t = t
            p.process(frame)
            p.frames += 1
            for st in p.tracks.values():
                if st.last_seen == p.clock.mono() and not st.deleted:   # seen in this very frame
                    states[id(st)] = st
                    seen_by_state[id(st)].append((i * stride + 1, tuple(float(v) for v in st.box)))
        for tid in list(p.tracks):
            p._finish(tid)
    finally:
        logging.getLogger("labeler.pipeline").removeHandler(log)
    records: dict[int, list] = defaultdict(list)
    vectors.clear()
    for key, seen in seen_by_state.items():
        st = states[key]
        if st.db_id is not None and st.n_frames >= cfg["tracking"]["min_frames_keep"]:
            records[st.db_id] += seen
            if st.vector is not None:
                vectors[st.db_id] = st.vector
    for j, pair in zip(log.joins, pairs):   # the log line and the pair come from the same join, in order
        j.update(pair)
    return records, log.joins


def _off(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["tracking"]["relink"]["enabled"] = False
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("--config", required=True, help="a test config (tools/make_test_config.py)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--long-gap", default="0.90,0.93,0.95,0.98,off")
    ap.add_argument("--json", type=Path, help="also write the results here")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("labeler.pipeline").setLevel(logging.INFO)

    cfg = load_config(args.config)
    from labeler.age import AgeEstimator
    from labeler.embed import Embedder
    embedder = Embedder(cfg["embeddings"])
    shared = {"embedder": embedder, "student": None, "age": None}
    _ = AgeEstimator   # the age model does not change who is counted; left out for speed
    gt = read_gt(args.gt)
    results = []
    print(f"{'setting':>10} {'true':>5} {'counted':>8} {'found':>6} {'merged':>7} {'split':>6} {'joins':>6} "
          f"{'right':>6} {'wrong':>6}")
    for value in args.long_gap.split(","):
        c = copy.deepcopy(cfg)
        if value == "off":
            c["tracking"]["relink"]["enabled"] = False
        elif value == "default":
            pass   # the config's own value (per mode)
        else:
            c["tracking"]["relink"]["long_gap_min_similarity"] = float(value)
        records, joins = run(args.video, c, args.imgsz, args.fps, shared)
        judge(joins, gt)
        s = score(records, gt, joins)
        s["join_list"] = [{k: v for k, v in j.items() if not k.endswith("_seen")} for j in joins]
        s["setting"] = value
        results.append(s)
        print(f"{value:>10} {s['true']:>5} {s['counted']:>8} {s['found']:>6} {s['merged']:>7} {s['split']:>6} "
              f"{s['joins']:>6} {s['right']:>6} {s['wrong']:>6}", flush=True)
    if "off" in args.long_gap.split(","):
        records, _ = run(args.video, _off(cfg), args.imgsz, args.fps, shared)
        print("look-alikes (re-linking off): threshold, pairs at or above it, of which the same person")
        for row in lookalikes(records, gt):
            if "ranking" in row:
                print("  " + row["ranking"])
            else:
                print(f"  {row['threshold']:.2f}  {row['pairs']:>4}  {row['same_person']:>4}  (same-person pairs "
                      f"found {row['found_of_same']})")
            results.append({"lookalike": row})
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

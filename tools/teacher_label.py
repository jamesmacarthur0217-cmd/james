"""Label extracted people with the VLM (the "teacher"). Resumable: already-labeled people are skipped.

    python tools/teacher_label.py data/datasets/coco_people --workers 3
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labeler.config import load_config  # noqa: E402
from labeler.vlm import VLMClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("people_dir", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    client = VLMClient(cfg["vlm"], "kiosk")  # kiosk schema: the fullest set of attributes
    rows = [json.loads(line) for line in open(args.people_dir / "people.jsonl", encoding="utf-8")]
    out_path = args.people_dir / "teacher.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in open(out_path, encoding="utf-8")}
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} already labeled, {len(todo)} to go, model {client.model}, workers {args.workers}")

    lock = threading.Lock()
    count = [0]
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        def work(row: dict) -> None:
            person = cv2.imread(str(args.people_dir / "person" / f"{row['id']}.jpg"))
            face = cv2.imread(str(args.people_dir / "face" / f"{row['id']}.jpg")) if row["face"] else None
            try:
                labels = client.label(person, face)
            except Exception as e:  # keep going; a failed person is retried on the next run
                print(f"failed {row['id']}: {e}", flush=True)
                return
            with lock:
                out.write(json.dumps({"id": row["id"], "labels": labels, "model": client.model}) + "\n")
                out.flush()
                count[0] += 1
                if count[0] % 100 == 0:
                    rate = count[0] / (time.time() - t0)
                    print(f"{count[0]}/{len(todo)} labeled, {rate * 60:.0f}/min, "
                          f"~{(len(todo) - count[0]) / rate / 60:.0f} min left", flush=True)

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(work, todo))
    print(f"done: {count[0]} labeled in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

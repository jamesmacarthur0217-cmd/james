"""Benchmark VLM labeling throughput.

    python tools/bench_vlm.py parallel <people_dir> --workers 3 --n 24
    python tools/bench_vlm.py batch    <people_dir> --batch 4  --n 24

parallel: N people labeled by W concurrent requests (Ollama must run with OLLAMA_NUM_PARALLEL >= W).
batch:    the same people labeled K per request, compared with one-per-request answers for agreement.
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labeler import schema  # noqa: E402
from labeler.vlm import PROMPT_KIOSK, VLMClient, _b64  # noqa: E402

CFG = {"host": "http://127.0.0.1:11434", "model": "qwen3-vl:4b-instruct", "timeout_s": 300}


def load_people(people_dir: Path, n: int, offset: int = 0) -> list[tuple[str, object]]:
    """n people spread over the dataset; a different offset gives a different set, so runs never
    repeat a request Ollama could answer from its cache."""
    rows = [json.loads(line) for line in open(people_dir / "people.jsonl", encoding="utf-8")]
    out = []
    step = max(1, len(rows) // n)
    for row in rows[offset % step::step][:n]:
        out.append((row["id"], cv2.imread(str(people_dir / "person" / f"{row['id']}.jpg"))))
    return out


def vram_mb() -> int:
    import subprocess
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return int(q.stdout.strip().splitlines()[0])


def label_single(client: VLMClient, crop) -> tuple[dict, float]:
    t = time.perf_counter()
    out = client.label(crop, None)
    return out, time.perf_counter() - t


def label_batch(crops: list) -> tuple[list[dict], float]:
    """Several people in one request: image k is person k; answer is a list in the same order."""
    k = len(crops)
    person_schema = schema.json_schema("kiosk")
    body = {
        "model": CFG["model"],
        "messages": [{"role": "user", "images": [_b64(c, 512) for c in crops], "content":
                      f"There are {k} images. Each image shows a different person.\n"
                      f"Return a list with exactly {k} entries, entry 1 for image 1, entry 2 for image 2, "
                      "and so on. Describe each person only from their own image.\n\n" + PROMPT_KIOSK}],
        "format": {"type": "object", "properties": {"people": {
            "type": "array", "items": person_schema, "minItems": k, "maxItems": k}}, "required": ["people"]},
        "stream": False, "think": False,
        "options": {"temperature": 0, "num_predict": 200 * k, "num_ctx": 1400 * k + 512}, "keep_alive": "30m",
    }
    t = time.perf_counter()
    r = requests.post(f"{CFG['host']}/api/chat", json=body, timeout=CFG["timeout_s"])
    r.raise_for_status()
    people = json.loads(r.json()["message"]["content"])["people"]
    return [schema.clean(p, "kiosk") for p in people], time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parallel", "batch"])
    ap.add_argument("people_dir", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    people = load_people(args.people_dir, args.n, args.offset)
    client = VLMClient(CFG, "kiosk")
    client.label(people[0][1], None)  # warm up: load the model before timing

    if args.mode == "parallel":
        vram_before = vram_mb()
        t = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda p: label_single(client, p[1]), people))
        wall = time.perf_counter() - t
        lat = [d for _, d in results]
        print(f"PARALLEL workers={args.workers}: {len(people)} people in {wall:.1f}s = "
              f"{wall / len(people):.2f}s per person ({len(people) / wall * 60:.0f} people/min); "
              f"latency median {statistics.median(lat):.1f}s max {max(lat):.1f}s; "
              f"VRAM {vram_before} -> {vram_mb()} MB")
        return

    # Batch: one-per-request reference answers first, then K per request.
    singles, t_single = [], 0.0
    for _, crop in people:
        out, d = label_single(client, crop)
        singles.append(out)
        t_single += d
    batched, t_batch = [], 0.0
    for i in range(0, len(people), args.batch):
        outs, d = label_batch([c for _, c in people[i:i + args.batch]])
        batched += outs
        t_batch += d
    fields = schema.MODE_FIELDS["kiosk"]
    agree = {f: sum(s[f] == b[f] for s, b in zip(singles, batched)) / len(people) for f in fields}
    print(f"BATCH k={args.batch}: one-per-request {t_single / len(people):.2f}s per person, "
          f"batched {t_batch / len(people):.2f}s per person ({t_single / t_batch:.1f}x faster)")
    print("  agreement with one-per-request answers: "
          + ", ".join(f"{f}={a:.0%}" for f, a in agree.items())
          + f"  | mean {statistics.mean(agree.values()):.0%}")


if __name__ == "__main__":
    main()

"""Benchmark age and gender estimation on FairFace (faces labeled with age group and gender by people).

    python tools/bench_age.py --n 300 --methods qwen3-vl:4b-instruct qwen3-vl:8b-instruct classifier

Scores: exact age group, within one group, gender, and bias (+ = guesses too old). Results are cached per
method in data/datasets/fairface/results-<method>.jsonl so methods can be added later.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "datasets" / "fairface"
BINS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDERS = ["man", "woman"]
OLLAMA = "http://127.0.0.1:11434"
PROMPT = ("Look at this person's face. Estimate their apparent age group and apparent gender. "
          "Choose the single most likely age group.")
CLASSIFIER = "dima806/fairface_age_image_detection"


def sample(n: int, seed: int = 0) -> list[dict]:
    t = pq.read_table(DATA / "validation.parquet")
    idx = random.Random(seed).sample(range(t.num_rows), n)
    rows = t.take(idx).to_pylist()
    return [{"i": i, "image": r["image"]["bytes"], "age": BINS[r["age"]], "gender": GENDERS[r["gender"]]}
            for i, r in zip(idx, rows)]


def vlm(model: str):
    schema = {"type": "object", "properties": {"age": {"type": "string", "enum": BINS},
                                               "gender": {"type": "string", "enum": GENDERS}},
              "required": ["age", "gender"]}

    def predict(img: bytes) -> dict:
        body = {"model": model, "stream": False, "think": False, "format": schema, "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 50},
                "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(img).decode()]}]}
        r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=120)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
    return predict


def classifier():
    import torch
    from transformers import pipeline
    pipe = pipeline("image-classification", model=CLASSIFIER, device=0 if torch.cuda.is_available() else -1)
    rename = {"more than 70": "70+"}

    def predict(img: bytes) -> dict:
        top = pipe(Image.open(io.BytesIO(img)).convert("RGB"), top_k=1)[0]["label"]
        return {"age": rename.get(top, top), "gender": None}
    return predict


def run(method: str, faces: list[dict]) -> list[dict]:
    out_path = DATA / f"results-{method.replace(':', '_').replace('/', '_')}.jsonl"
    done = {}
    if out_path.exists():
        done = {r["i"]: r for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines())}
    todo = [f for f in faces if f["i"] not in done]
    if todo:
        predict = classifier() if method == "classifier" else vlm(method)
        predict(todo[0]["image"])  # warm up before timing
        with open(out_path, "a", encoding="utf-8") as f:
            for face in todo:
                t = time.perf_counter()
                p = predict(face["image"])
                row = {"i": face["i"], "age": p["age"], "gender": p["gender"], "s": time.perf_counter() - t}
                f.write(json.dumps(row) + "\n")
                done[face["i"]] = row
    return [done[f["i"]] for f in faces]


def score(faces: list[dict], preds: list[dict]) -> dict:
    truth = np.array([BINS.index(f["age"]) for f in faces])
    guess = np.array([BINS.index(p["age"]) if p["age"] in BINS else -99 for p in preds])
    g_ok = [p["gender"] == f["gender"] for f, p in zip(faces, preds) if p["gender"] is not None]
    return {"exact": float((truth == guess).mean()), "within1": float((abs(truth - guess) <= 1).mean()),
            "gender": float(np.mean(g_ok)) if g_ok else None, "bias": float((guess - truth).mean()),
            "seconds": float(np.median([p["s"] for p in preds]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--methods", nargs="+", default=["qwen3-vl:4b-instruct", "classifier"])
    args = ap.parse_args()
    faces = sample(args.n)
    print(f"{len(faces)} FairFace faces; true age groups: "
          + ", ".join(f"{b} {sum(f['age'] == b for f in faces)}" for b in BINS))
    print(f"{'method':26s} {'exact age':>9s} {'±1 group':>9s} {'gender':>7s} {'bias':>6s} {'s/face':>7s}")
    for m in args.methods:
        s = score(faces, run(m, faces))
        gender = f"{s['gender']:7.0%}" if s["gender"] is not None else "      -"
        print(f"{m:26s} {s['exact']:9.0%} {s['within1']:9.0%} {gender} {s['bias']:+6.2f} {s['seconds']:7.2f}",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())

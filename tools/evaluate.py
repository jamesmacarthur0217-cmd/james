"""Score labeling sources against the hand-made answer key (data/datasets/coco_people/answer_key.jsonl).

    python tools/evaluate.py teacher student vlm:qwen3-vl:4b-instruct:noface vlm:qwen3-vl:8b-instruct

Sources:
  teacher        the VLM labels already stored in teacher.jsonl (4B model, body + face close-up)
  student        the fast classifiers, retrained here WITHOUT the answer-key people (no peeking)
  vlm:<model>    run that Ollama model now (add ":noface" to send only the body crop); cached per source

For each attribute: answered = share of key labels the source gave an answer for (not "unknown"),
right = accuracy of those answers, overall = right answers / all key labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labeler import schema  # noqa: E402
from labeler.vlm import PROMPT_KIOSK, PROMPT_MOBILITY, VLMClient  # noqa: E402

PEOPLE = ROOT / "data" / "datasets" / "coco_people"
FIELDS = ["apparent_gender", "age_group", "upper_type", "upper_color", "lower_type", "lower_color",
          "headwear", "eyewear", "carrying"]
AGE_GROUP = {"0-12": "child", "13-17": "teen", "18-24": "adult", "25-34": "adult", "35-44": "adult",   # old ranges
             "45-54": "adult", "55-64": "adult", "65+": "senior",
             "0-9": "child", "10-19": "teen", "20-29": "adult", "30-39": "adult", "40-49": "adult",       # decades
             "50-59": "adult", "60-69": "senior", "70+": "senior", "unknown": "unknown"}


def load_key() -> dict[str, dict]:
    return {r["id"]: r["labels"] for r in map(json.loads, (PEOPLE / "answer_key.jsonl").read_text("utf-8").splitlines())}


def with_age_group(labels: dict) -> dict:
    out = dict(labels)
    if "age_range" in out and "age_group" not in out:
        out["age_group"] = AGE_GROUP.get(out["age_range"], "unknown")
    return out


def teacher(ids: list[str]) -> tuple[dict, float | None]:
    rows = {r["id"]: r["labels"] for r in map(json.loads, (PEOPLE / "teacher.jsonl").read_text("utf-8").splitlines())}
    return {i: with_age_group(rows[i]) for i in ids if i in rows}, None


def student(ids: list[str]) -> tuple[dict, float | None]:
    sys.path.insert(0, str(ROOT / "tools"))
    from train_attributes import embeddings, fit, targets
    rows = [r for r in map(json.loads, (PEOPLE / "teacher.jsonl").read_text("utf-8").splitlines())]
    train = [r for r in rows if r["id"] not in set(ids)]
    from labeler.config import load_config
    model_name = load_config()["embeddings"]["model"]
    X_train = embeddings(PEOPLE, [r["id"] for r in train], model_name)
    X_key = embeddings(PEOPLE, ids, model_name)
    Y = targets([r["labels"] for r in train])
    preds: dict[str, dict] = {i: {} for i in ids}
    for name, ys in Y.items():
        y = np.array(ys)
        classes = {c for c in set(y) if (y == c).sum() >= 8}
        if len(classes) < 2:
            continue
        keep = np.isin(y, list(classes))
        clf = fit(X_train[keep], y[keep])
        for i, v in zip(ids, clf.predict(X_key)):
            preds[i][name] = str(v)
    for i in ids:   # carried items -> one comma list, like the app
        items = sorted(item for item in schema.LIST_FIELDS["carrying"] if preds[i].get(f"carrying:{item}") == "yes")
        preds[i]["carrying"] = ",".join(items) or "none"
    return preds, None


def vlm(spec: str, ids: list[str], labels: dict | None = None) -> tuple[dict, float]:
    model = spec[len("vlm:"):]
    noface = model.endswith(":noface")
    model = model.removesuffix(":noface")
    import hashlib
    # A changed prompt gets fresh answers. Without mobility aids the id is the one used before they existed.
    asked = PROMPT_KIOSK
    if "mobility_aid" in schema.mode_fields("kiosk", labels):
        asked += PROMPT_MOBILITY + json.dumps(schema.json_schema("kiosk", labels), sort_keys=True)
    prompt_id = hashlib.sha1(asked.encode()).hexdigest()[:6]
    cache = PEOPLE / f"eval-{spec.replace(':', '_').replace('/', '_')}-{prompt_id}.jsonl"
    done = {}
    if cache.exists():
        done = {r["id"]: r for r in map(json.loads, cache.read_text("utf-8").splitlines())}
    from labeler.config import load_config
    cfg = load_config()["vlm"]
    client = VLMClient({**cfg, "model": model}, "kiosk", labels)
    meta = {r["id"]: r for r in map(json.loads, (PEOPLE / "people.jsonl").read_text("utf-8").splitlines())}
    with open(cache, "a", encoding="utf-8") as f:
        for i in ids:
            if i in done:
                continue
            person = cv2.imread(str(PEOPLE / "person" / f"{i}.jpg"))
            face = None if noface or not meta[i]["face"] else cv2.imread(str(PEOPLE / "face" / f"{i}.jpg"))
            t = time.perf_counter()
            labels = client.label(person, face)
            done[i] = {"id": i, "labels": labels, "s": time.perf_counter() - t}
            f.write(json.dumps(done[i]) + "\n")
    secs = float(np.median([done[i]["s"] for i in ids]))
    return {i: with_age_group(done[i]["labels"]) for i in ids}, secs


def score(key: dict, preds: dict) -> dict[str, dict]:
    out = {}
    for name in FIELDS:
        total = answered = right = 0
        for pid, labels in key.items():
            if name not in labels or pid not in preds:
                continue
            total += 1
            guess = preds[pid].get(name, "unknown")
            if guess == "unknown":
                continue
            answered += 1
            right += guess == labels[name]
        if total:
            out[name] = {"n": total, "answered": answered / total, "right": right / answered if answered else 0,
                         "overall": right / total}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--no-mobility-aids", action="store_true", help="ask the VLM without the mobility-aid question")
    args = ap.parse_args()
    labels = {"mobility_aids": not args.no_mobility_aids}
    key = load_key()
    ids = list(key)
    print(f"answer key: {len(ids)} people, {sum(len(v) for v in key.values())} labels\n")
    results = {}
    for src in args.sources:
        if src == "teacher":
            preds, secs = teacher(ids)
        elif src == "student":
            preds, secs = student(ids)
        elif src.startswith("vlm:"):
            preds, secs = vlm(src, ids, labels)
        else:
            raise SystemExit(f"unknown source {src}")
        results[src] = (score(key, preds), secs)

    print(f"{'attribute':16s} {'n':>3s} " + " ".join(f"{s[:26]:>26s}" for s in results))
    print(f"{'':16s} {'':>3s} " + " ".join(f"{'answered  right  overall':>26s}" for _ in results))
    for name in FIELDS:
        row = [results[s][0].get(name) for s in results]
        if not any(row):
            continue
        n = next(r["n"] for r in row if r)
        cells = [f"{r['answered']:8.0%} {r['right']:6.0%} {r['overall']:8.0%}" if r else f"{'-':>26s}" for r in row]
        print(f"{name:16s} {n:3d} " + " ".join(f"{c:>26s}" for c in cells))
    means = []
    for s, (sc, secs) in results.items():
        overall = np.mean([v["overall"] for v in sc.values()])
        means.append(f"{s}: mean overall {overall:.0%}" + (f", {secs:.1f} s/person" if secs else ""))
    print("\n" + "\n".join(means))


if __name__ == "__main__":
    main()

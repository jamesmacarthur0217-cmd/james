"""Train fast attribute classifiers (the "student") on SigLIP embeddings, from VLM ("teacher") labels.

    python tools/train_attributes.py data/datasets/coco_people [--target 0.9]

For each attribute it reports, on people from photos the student never saw:
  - accuracy vs the teacher, and the majority-class baseline;
  - the confidence threshold above which the student agrees with the teacher at least --target of the
    time, and how many people clear it (those can skip the VLM).
Writes models/attributes.joblib, which the live pipeline loads.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labeler import schema  # noqa: E402
from labeler.config import load_config  # noqa: E402

AGE_GROUP = {"0-12": "child", "13-17": "teen", "18-24": "adult", "25-34": "adult", "35-44": "adult",   # old ranges
             "45-54": "adult", "55-64": "adult", "65+": "senior",
             "0-9": "child", "10-19": "teen", "20-29": "adult", "30-39": "adult", "40-49": "adult",       # decades
             "50-59": "adult", "60-69": "senior", "70+": "senior", "unknown": "unknown"}
MIN_CLASS = 8          # classes rarer than this in training are left to the VLM
MIN_TEST_CLASS = 5     # each answer needs this many test examples before the student can be trusted
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def embeddings(people_dir: Path, ids: list[str], model_name: str) -> np.ndarray:
    """SigLIP embeddings of the person crops, cached in embeddings.npz."""
    cache = people_dir / "embeddings.npz"
    have: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["model"]) == model_name:
            have = dict(zip(z["ids"].tolist(), z["vectors"]))
    missing = [i for i in ids if i not in have]
    if missing:
        from labeler.embed import Embedder
        emb = Embedder({"model": model_name, "device": "cuda"})
        t = time.time()
        for s in range(0, len(missing), 32):
            chunk = missing[s:s + 32]
            crops = [cv2.imread(str(people_dir / "person" / f"{i}.jpg")) for i in chunk]
            for i, v in zip(chunk, emb.images(crops)):
                have[i] = v
        print(f"embedded {len(missing)} people in {time.time() - t:.0f}s")
        np.savez(cache, model=model_name, ids=np.array(list(have)), vectors=np.stack(list(have.values())))
    return np.stack([have[i] for i in ids])


def targets(labels: list[dict]) -> dict[str, list[str]]:
    """One target per attribute; carried items become one yes/no target each."""
    out: dict[str, list[str]] = {}
    for name in schema.MODE_FIELDS["kiosk"]:
        if name in schema.LIST_FIELDS:
            for item in schema.LIST_FIELDS[name]:
                out[f"{name}:{item}"] = ["yes" if item in l[name].split(",") else "no" for l in labels]
        else:
            out[name] = [l[name] for l in labels]
    out["age_group"] = [AGE_GROUP[l["age_range"]] for l in labels]
    return out


BALANCED_WEIGHTS = True   # set by --no-balance


def fit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    # Weighting rare answers up keeps the student from always guessing the common one ("no bag").
    return LogisticRegression(C=2.0, max_iter=3000,
                              class_weight="balanced" if BALANCED_WEIGHTS else None).fit(X, y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("people_dir", type=Path)
    ap.add_argument("--target", type=float, default=0.9, help="agreement needed to trust the student")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "attributes.joblib")
    ap.add_argument("--no-balance", action="store_true", help="do not weight rare answers up")
    args = ap.parse_args()
    global BALANCED_WEIGHTS
    BALANCED_WEIGHTS = not args.no_balance

    cfg = load_config()
    model_name = cfg["embeddings"]["model"]
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(args.people_dir / "people.jsonl", encoding="utf-8")}
    teacher = [json.loads(l) for l in open(args.people_dir / "teacher.jsonl", encoding="utf-8")]
    ids = [t["id"] for t in teacher]
    groups = np.array([meta[i]["image"] for i in ids])   # people from one photo stay on one side
    X = embeddings(args.people_dir, ids, model_name)
    Y = targets([t["labels"] for t in teacher])
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, groups=groups))
    print(f"{len(ids)} people from {len(set(groups))} photos: {len(train)} train, {len(test)} test "
          f"(split by photo)\n")

    print(f"{'attribute':24s} {'baseline':>8s} {'accuracy':>8s} {'covered':>8s} {'agree':>6s}   trusted answers")
    models = {}
    total_cov = []
    trusted_by_attr: dict[str, np.ndarray] = {}
    for name, ys in Y.items():
        y = np.array(ys)
        counts = Counter(y[train])
        keep_classes = {c for c, n in counts.items() if n >= MIN_CLASS}
        if len(keep_classes) < 2:
            print(f"{name:24s} skipped (only {len(keep_classes)} common answer)")
            continue
        tr = np.array([i for i in train if y[i] in keep_classes])
        clf = fit(X[tr], y[tr])
        proba = clf.predict_proba(X[test])
        pred = clf.classes_[proba.argmax(1)]
        conf = proba.max(1)
        yt = y[test]
        acc = float((pred == yt).mean())
        baseline = max(Counter(yt).values()) / len(test)

        # Trust each answer separately: the lowest confidence above which, when the student gives this
        # answer, the teacher agrees at least --target of the time AND the student halves the error of
        # always guessing this answer (so "no backpack", right 94% of the time by guessing, needs 97%).
        thresholds: dict[str, float] = {}
        for c in sorted(keep_classes):
            prior = float((yt == c).mean())
            if (yt == c).sum() < MIN_TEST_CLASS:
                continue
            need = max(args.target, 1 - (1 - prior) / 2)
            for t in THRESHOLDS:
                m = (pred == c) & (conf >= t)
                if m.sum() >= MIN_TEST_CLASS and (yt[m] == c).mean() >= need:
                    thresholds[c] = t
                    break
        trusted = np.array([p in thresholds and cf >= thresholds[p] for p, cf in zip(pred, conf)])
        trusted_by_attr[name] = trusted
        covered = float(trusted.mean())
        agree = float((pred[trusted] == yt[trusted]).mean()) if trusted.any() else float("nan")
        total_cov.append(covered)
        shown = ", ".join(f"{c}>={t:.2f}" for c, t in thresholds.items()) or "none (VLM decides)"
        agree_s = f"{agree:6.0%}" if trusted.any() else "     -"
        print(f"{name:24s} {baseline:8.0%} {acc:8.0%} {covered:8.0%} {agree_s}   {shown}")
        # Final model on all data, used live.
        all_idx = np.array([i for i in range(len(y)) if y[i] in keep_classes])
        models[name] = {"model": fit(X[all_idx], y[all_idx]), "thresholds": thresholds, "test_accuracy": acc,
                        "baseline": baseline, "coverage": covered, "agreement_when_trusted": agree}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"embedding_model": model_name, "teacher_model": teacher[0]["model"], "n_people": len(ids),
                 "target_agreement": args.target, "attributes": models, "trained_at": time.strftime("%Y-%m-%d %H:%M")},
                args.out)
    # How often the student is confident about every attribute of a mode, so the VLM can be skipped.
    for mode in ("kiosk", "security"):
        need = []
        for f in schema.MODE_FIELDS[mode]:
            parts = [f"{f}:{i}" for i in schema.LIST_FIELDS[f]] if f in schema.LIST_FIELDS else [f]
            need += [p for p in parts if p in trusted_by_attr]
        allc = np.all([trusted_by_attr[p] for p in need], axis=0)
        print(f"{mode}: student confident about every attribute for {allc.mean():.0%} of test people "
              f"(those skip the VLM)")
    usable = sum(bool(m["thresholds"]) for m in models.values())
    print(f"\nsaved {len(models)} classifiers ({usable} with trusted answers; average coverage "
          f"{np.mean(total_cov):.0%}) to {args.out}")


if __name__ == "__main__":
    main()

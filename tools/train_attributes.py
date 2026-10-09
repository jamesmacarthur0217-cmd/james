"""Train fast attribute classifiers (the "student") on SigLIP embeddings, from VLM ("teacher") labels.

    python tools/train_attributes.py data/datasets/coco_people [--target 0.9] [--no-balance]

For each attribute it reports, on people from photos the student never saw:
  - accuracy vs the teacher, and the majority-class baseline;
  - the confidence threshold above which the student agrees with the teacher at least --target of the
    time, and how many people clear it (those can skip the VLM).
Writes models/attributes.joblib, the shipped baseline the live pipeline loads (the weekly retraining,
labeler/retrain.py, writes data/models/attributes.joblib, which is preferred when present).
The training itself is in labeler/training.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labeler import schema, training  # noqa: E402
from labeler.config import load_config  # noqa: E402

AGE_GROUP = training.AGE_GROUP
targets = training.targets
BALANCED_WEIGHTS = True   # set by --no-balance; tools/evaluate.py uses fit() with this default


def fit(X: np.ndarray, y: np.ndarray):
    return training.fit(X, y, BALANCED_WEIGHTS)


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


def load_teacher(people_dir: Path) -> tuple[list[str], list[dict], np.ndarray, str]:
    """(ids, labels, groups by photo, teacher model name) of the COCO people labeled by the VLM."""
    meta = {r["id"]: r for r in map(json.loads, open(people_dir / "people.jsonl", encoding="utf-8"))}
    teacher = [json.loads(line) for line in open(people_dir / "teacher.jsonl", encoding="utf-8")]
    ids = [t["id"] for t in teacher]
    groups = np.array([meta[i]["image"] for i in ids])   # people from one photo stay on one side
    return ids, [t["labels"] for t in teacher], groups, teacher[0]["model"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("people_dir", type=Path)
    ap.add_argument("--target", type=float, default=0.9, help="agreement needed to trust the student")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "attributes.joblib")
    ap.add_argument("--no-balance", action="store_true", help="do not weight rare answers up")
    args = ap.parse_args()

    model_name = load_config()["embeddings"]["model"]
    ids, labels, groups, teacher_model = load_teacher(args.people_dir)
    X = embeddings(args.people_dir, ids, model_name)
    models, trusted_by_attr = training.train(X, labels, groups, args.target, balanced=not args.no_balance)
    training.save(args.out, models, model_name, teacher_model, len(ids), args.target)
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
          f"{np.mean([m['coverage'] for m in models.values()]):.0%}) to {args.out}")


if __name__ == "__main__":
    main()

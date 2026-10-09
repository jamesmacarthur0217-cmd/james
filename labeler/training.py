"""Training the fast attribute classifiers (the "student", labeler/student.py) on SigLIP embeddings.

Used by tools/train_attributes.py (COCO teacher labels) and by the weekly retraining (labeler/retrain.py), which
adds the user's own camera people. For each attribute it reports, on held-out people: accuracy vs the labels,
the majority-class baseline, and per answer the confidence above which the student agrees with the labels at
least `target` of the time (those predictions are "trusted").
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from . import schema

AGE_GROUP = {"0-12": "child", "13-17": "teen", "18-24": "adult", "25-34": "adult", "35-44": "adult",   # old ranges
             "45-54": "adult", "55-64": "adult", "65+": "senior",
             "0-9": "child", "10-19": "teen", "20-29": "adult", "30-39": "adult", "40-49": "adult",       # decades
             "50-59": "adult", "60-69": "senior", "70+": "senior", "unknown": "unknown"}
MIN_CLASS = 8          # classes rarer than this in training are left to the VLM
MIN_TEST_CLASS = 5     # each answer needs this many test examples before the student can be trusted
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def targets(labels: list[dict]) -> dict[str, list[str]]:
    """One target per attribute; carried items become one yes/no target each. Missing fields are "unknown"."""
    out: dict[str, list[str]] = {}
    for name in schema.MODE_FIELDS["kiosk"]:
        if name in schema.LIST_FIELDS:
            for item in schema.LIST_FIELDS[name]:
                out[f"{name}:{item}"] = ["yes" if item in l.get(name, "none").split(",") else "no" for l in labels]
        else:
            out[name] = [l.get(name, "unknown") for l in labels]
    out["age_group"] = [l["age_group"] if l.get("age_group") not in (None, "unknown")
                        else AGE_GROUP.get(l.get("age_range", "unknown"), "unknown") for l in labels]
    return out


def fit(X: np.ndarray, y: np.ndarray, balanced: bool = False, weights: np.ndarray | None = None) -> LogisticRegression:
    # Weighting rare answers up keeps the student from always guessing the common one ("no bag").
    return LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced" if balanced else None).fit(
        X, y, sample_weight=weights)


def train(X: np.ndarray, labels: list[dict], groups: np.ndarray, target: float = 0.9, balanced: bool = False,
          weights: np.ndarray | None = None, report=print) -> tuple[dict, dict[str, np.ndarray]]:
    """Train one classifier per attribute. Returns (models as stored in the joblib file, which held-out people
    each attribute trusted). People in one group (one photo, one camera person) stay on one side of the split."""
    Y = targets(labels)
    train_idx, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, groups=groups))
    report(f"{len(labels)} people in {len(set(groups))} groups: {len(train_idx)} train, {len(test)} test\n")
    report(f"{'attribute':24s} {'baseline':>8s} {'accuracy':>8s} {'covered':>8s} {'agree':>6s}   trusted answers")
    w = None if weights is None else np.asarray(weights, dtype=float)
    models, trusted_by_attr = {}, {}
    for name, ys in Y.items():
        y = np.array(ys)
        keep = {c for c, n in Counter(y[train_idx]).items() if n >= MIN_CLASS}
        if len(keep) < 2:
            report(f"{name:24s} skipped (only {len(keep)} common answer)")
            continue
        tr = np.array([i for i in train_idx if y[i] in keep])
        clf = fit(X[tr], y[tr], balanced, None if w is None else w[tr])
        proba = clf.predict_proba(X[test])
        pred, conf, yt = clf.classes_[proba.argmax(1)], proba.max(1), y[test]
        acc = float((pred == yt).mean())
        baseline = max(Counter(yt).values()) / len(test)
        # Trust each answer separately: the lowest confidence above which, when the student gives this answer,
        # the labels agree at least `target` of the time AND the student halves the error of always guessing
        # this answer (so "no backpack", right 94% of the time by guessing, needs 97%).
        thresholds: dict[str, float] = {}
        for c in sorted(keep):
            if (yt == c).sum() < MIN_TEST_CLASS:
                continue
            need = max(target, 1 - (1 - float((yt == c).mean())) / 2)
            for t in THRESHOLDS:
                m = (pred == c) & (conf >= t)
                if m.sum() >= MIN_TEST_CLASS and (yt[m] == c).mean() >= need:
                    thresholds[c] = t
                    break
        trusted = np.array([p in thresholds and cf >= thresholds[p] for p, cf in zip(pred, conf)])
        trusted_by_attr[name] = trusted
        covered = float(trusted.mean())
        agree = float((pred[trusted] == yt[trusted]).mean()) if trusted.any() else float("nan")
        shown = ", ".join(f"{c}>={t:.2f}" for c, t in thresholds.items()) or "none (VLM decides)"
        report(f"{name:24s} {baseline:8.0%} {acc:8.0%} {covered:8.0%} "
               f"{f'{agree:6.0%}' if trusted.any() else '     -'}   {shown}")
        everyone = np.array([i for i in range(len(y)) if y[i] in keep])   # final model on all data, used live
        models[name] = {"model": fit(X[everyone], y[everyone], balanced, None if w is None else w[everyone]),
                        "thresholds": thresholds, "test_accuracy": acc, "baseline": baseline,
                        "coverage": covered, "agreement_when_trusted": agree}
    return models, trusted_by_attr


def save(path: Path, models: dict, embedding_model: str, teacher_model: str, n_people: int, target: float,
         **extra) -> dict:
    import joblib
    data = {"embedding_model": embedding_model, "teacher_model": teacher_model, "n_people": n_people,
            "target_agreement": target, "attributes": models, "trained_at": time.strftime("%Y-%m-%d %H:%M"), **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(data, tmp)
    tmp.replace(path)          # the app never reads a half-written file
    return data

"""Fast attribute classifiers (the "student") that predict labels from a person's SigLIP embedding.

Trained from VLM labels by tools/train_attributes.py. Each answer of each attribute ("gender = woman",
"backpack = no") has its own confidence threshold, chosen on held-out people, above which the student agrees
with the VLM at least the target rate (90%) and clearly beats guessing. Predictions above it are "trusted";
anything else is left for the VLM to decide.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import schema

log = logging.getLogger(__name__)


class Student:
    def __init__(self, path: Path, embedding_model: str):
        import joblib
        data = joblib.load(path)
        if data["embedding_model"] != embedding_model:
            raise ValueError(f"classifiers were trained on {data['embedding_model']} embeddings, "
                             f"but the app uses {embedding_model}; retrain with tools/train_attributes.py")
        self.attributes: dict = data["attributes"]
        self.name = f"student-{data['trained_at']}"
        self.info = {k: data[k] for k in ("teacher_model", "n_people", "target_agreement", "trained_at")}
        log.info("attribute classifiers loaded: %d attributes, trained on %d people (%s)",
                 len(self.attributes), data["n_people"], data["trained_at"])

    def _one(self, name: str, x: np.ndarray) -> tuple[str, float, bool] | None:
        a = self.attributes.get(name)
        if a is None:
            return None
        proba = a["model"].predict_proba(x)[0]
        k = int(proba.argmax())
        value, conf = str(a["model"].classes_[k]), float(proba[k])
        # Each answer has its own threshold, chosen on held-out people (see tools/train_attributes.py).
        threshold = a["thresholds"].get(value)
        return value, conf, threshold is not None and conf >= threshold

    def predict(self, vector: np.ndarray, mode: str) -> dict[str, tuple[str, float, bool]]:
        """name -> (value, confidence, trusted) for the attributes of this mode the student knows."""
        x = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        out: dict[str, tuple[str, float, bool]] = {}
        for name in schema.MODE_FIELDS[mode]:
            if name in schema.LIST_FIELDS:
                all_parts = [(item, self._one(f"{name}:{item}", x)) for item in schema.LIST_FIELDS[name]]
                parts = [(item, p) for item, p in all_parts if p is not None]
                if not parts:
                    continue
                items = sorted(item for item, p in parts if p[0] == "yes")
                conf = min(p[1] for _, p in parts)
                # An item with no classifier (too rare to train, e.g. luggage) can't be ruled out.
                trusted = len(parts) == len(all_parts) and all(p[2] for _, p in parts)
                out[name] = (",".join(items) if items else "none", conf, trusted)
            else:
                p = self._one(name, x)
                if p is not None:
                    out[name] = p
        return out

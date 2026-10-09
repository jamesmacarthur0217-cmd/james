"""Weekly retraining of the fast classifiers on the user's own cameras, with a quality gate.

Training data: the COCO practice people (VLM labels, embeddings cached in data/datasets/coco_people) plus the
user's own finished people (the VLM's labels, or the human's where they were reviewed), the own ones weighted
`own_weight` times, because they look like what the cameras really see. The answer-key people are never trained
on: they are the test. The new classifiers replace the current ones only if they score at least as well on the
answer key; otherwise the current ones stay. Every run is recorded (model_history, shown on the Answer key page).

Files: data/models/attributes.joblib (the current retrained classifiers, preferred over the shipped
models/attributes.joblib), data/models/attributes.prev.joblib (the one before). data/ is not in git.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import schema, training
from .config import resolve
from .student import Student

log = logging.getLogger(__name__)

# What the score counts: what the fast classifiers are used for (age comes from the age model).
SCORE_FIELDS = ["apparent_gender", "upper_type", "upper_color", "lower_type", "lower_color", "headwear",
                "eyewear", "face_mask", "carrying"]


def student_path(cfg: dict) -> Path:
    """The classifiers to use: the retrained ones when there are any, else the shipped baseline."""
    retrained = resolve(cfg["student"]["retrained_path"])
    return retrained if retrained.exists() else resolve(cfg["student"]["model_path"])


def score(student: Student, people: list[tuple[int, np.ndarray, dict]]) -> dict:
    """How often the classifiers' answer (trusted or not) matches the answer key, per attribute and on average."""
    right: dict[str, list[int]] = {}
    for _, vector, key in people:
        preds = student.predict(vector, "kiosk")
        for name in SCORE_FIELDS:
            if name in key and name in preds:
                right.setdefault(name, []).append(int(preds[name][0] == key[name]))
    attrs = {n: round(sum(v) / len(v), 3) for n, v in right.items()}
    return {"people": len(people), "mean": round(float(np.mean(list(attrs.values()))), 4) if attrs else None,
            "attributes": attrs}


def coco_people(people_dir: Path, model_name: str) -> tuple[list[str], np.ndarray, list[dict], np.ndarray, str]:
    """COCO practice people whose embedding is already cached (no GPU needed here)."""
    cache, teacher_file = people_dir / "embeddings.npz", people_dir / "teacher.jsonl"
    if not (cache.exists() and teacher_file.exists()):
        return [], np.zeros((0, 0), np.float32), [], np.array([]), ""
    z = np.load(cache, allow_pickle=False)
    if str(z["model"]) != model_name:
        return [], np.zeros((0, 0), np.float32), [], np.array([]), ""
    vectors = dict(zip(z["ids"].tolist(), z["vectors"]))
    meta = {r["id"]: r for r in map(json.loads, open(people_dir / "people.jsonl", encoding="utf-8"))}
    teacher = [json.loads(line) for line in open(teacher_file, encoding="utf-8")]
    teacher = [t for t in teacher if t["id"] in vectors]
    ids = [t["id"] for t in teacher]
    X = np.stack([vectors[i] for i in ids]) if ids else np.zeros((0, 0), np.float32)
    return ids, X, [t["labels"] for t in teacher], np.array([meta[i]["image"] for i in ids]), \
        teacher[0]["model"] if teacher else ""


def retrain(db, cfg: dict, report=log.info) -> dict:
    """Train new classifiers, compare them with the current ones on the answer key, keep the better.
    Returns the model_history entry."""
    rcfg, model_name = cfg["retrain"], cfg["embeddings"]["model"]
    own = db.training_people(model_name)
    key = db.answer_key_people(model_name)
    coco_ids, Xc, coco_labels, coco_groups, teacher_model = coco_people(resolve(rcfg["coco_dir"]), model_name)
    base = {"own_people": len(own), "coco_people": len(coco_ids), "key_people": len(key)}
    if len(own) < rcfg["min_own_people"]:
        return db.add_model_history(**base, note=f"not enough people from your cameras yet "
                                                 f"({len(own)} of {rcfg['min_own_people']}); nothing changed")
    if len(key) < rcfg["min_answer_key"]:
        return db.add_model_history(**base, note=f"not enough answer-key people to check a new version "
                                                 f"({len(key)} of {rcfg['min_answer_key']}); nothing changed")
    Xo = np.stack([v for _, v, _ in own])
    X = np.vstack([Xc, Xo]) if len(coco_ids) else Xo
    labels = coco_labels + [lab for _, _, lab in own]
    groups = np.concatenate([coco_groups, np.array([f"track-{t}" for t, _, _ in own])])
    weights = np.concatenate([np.ones(len(coco_ids)), np.full(len(own), float(rcfg["own_weight"]))])
    report(f"retraining on {len(own)} people from your cameras (weight {rcfg['own_weight']}) "
           f"and {len(coco_ids)} COCO people")
    models, _ = training.train(X, labels, groups, cfg["retrain"]["target"], weights=weights,
                               report=lambda line: log.debug(line))
    current_path = student_path(cfg)
    final = resolve(cfg["student"]["retrained_path"])
    candidate = final.with_name("attributes.new.joblib")
    training.save(candidate, models, model_name, teacher_model or "vlm", len(labels), rcfg["target"],
                  own_people=len(own))
    before = score(Student(current_path, model_name), key)
    after = score(Student(candidate, model_name), key)
    kept = after["mean"] is not None and (before["mean"] is None or after["mean"] >= before["mean"])
    if kept:
        if final.exists():
            final.replace(final.with_name("attributes.prev.joblib"))
        candidate.replace(final)
        note = "new version kept: at least as good on the answer key"
    else:
        candidate.unlink(missing_ok=True)
        note = "new version scored lower on the answer key; the current one stays"
    report(f"retraining: answer key {before['mean']} -> {after['mean']}: {note}")
    return db.add_model_history(**base, score_before=before["mean"], score_after=after["mean"], kept=kept, note=note)


class Retrainer:
    """Runs the retraining in the background (weekly, or when an admin asks) and gives the cameras the new
    classifiers when they are kept. Also records the answer-key score once a day (accuracy history)."""

    def __init__(self, db, cfg: dict, pipelines):
        self.db, self.cfg, self.pipelines = db, cfg, pipelines
        self.running = threading.Event()
        self._last_accuracy_day = None

    def run_now(self) -> bool:
        """Start a retraining in the background; False if one is already running."""
        if self.running.is_set():
            return False
        self.running.set()
        threading.Thread(target=self._run, name="retrain", daemon=True).start()
        return True

    def _run(self) -> None:
        try:
            entry = retrain(self.db, self.cfg)
            if entry["kept"]:
                path = student_path(self.cfg)
                for p in self.pipelines:
                    if p.student is not None:
                        p.student = Student(path, self.cfg["embeddings"]["model"])   # one assignment: safe
            self.db.record_accuracy(datetime.now())
        except Exception:
            log.exception("retraining failed; the current classifiers stay")
        finally:
            self.running.clear()

    def due(self, now: datetime) -> bool:
        history = self.db.list_model_history(limit=1)
        last = datetime.fromisoformat(history[0]["created_at"]) if history else None
        return now.hour >= self.cfg["retrain"]["hour"] and (
            last is None or now - last >= timedelta(days=self.cfg["retrain"]["every_days"]))

    def tick(self, now: datetime | None = None) -> None:
        now = now or datetime.now()
        if now.date() != self._last_accuracy_day:
            self.db.record_accuracy(now)
            self._last_accuracy_day = now.date()
        if self.cfg["retrain"]["enabled"] and self.due(now):
            self.run_now()

    def loop(self, stop: threading.Event, every_s: float = 3600) -> None:
        while True:
            try:
                self.tick()
            except Exception:
                log.exception("retraining check failed")
            if stop.wait(every_s):
                return

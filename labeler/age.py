"""Apparent age from face crops with a dedicated classifier (dima806/fairface_age_image_detection, Apache-2.0).

Measured on 300 FairFace faces (human-labeled age groups): 60% exact age group and 96% within one group,
at ~0.01 s per face, versus 51% / 90% at ~0.7 s for the Qwen3-VL 4B model. See tools/bench_age.py.
The classifier was trained on FairFace-style photos, which flatters it on that test; the answer-key page
measures it on this camera.
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

MODEL = "dima806/fairface_age_image_detection"
# Classifier labels -> the app's decade age ranges (0-2 and 3-9 are merged).
TO_RANGE = {"0-2": "0-9", "3-9": "0-9", "10-19": "10-19", "20-29": "20-29", "30-39": "30-39",
            "40-49": "40-49", "50-59": "50-59", "60-69": "60-69", "more than 70": "70+"}
TO_GROUP = {"0-9": "child", "10-19": "teen", "20-29": "adult", "30-39": "adult", "40-49": "adult",
            "50-59": "adult", "60-69": "senior", "70+": "senior"}
MIN_FACE_PX = 40


class AgeEstimator:
    def __init__(self, model: str = MODEL, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.device = device if torch.cuda.is_available() else "cpu"
        log.info("loading age model %s", model)
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModelForImageClassification.from_pretrained(model).to(self.device).eval()
        labels = self.model.config.id2label
        self.ranges = [TO_RANGE[labels[i]] for i in range(len(labels))]
        self.name = model
        self._lock = threading.Lock()

    @torch.no_grad()
    def estimate(self, faces: list[np.ndarray]) -> tuple[str, float] | None:
        """(decade age range, confidence) from one or more BGR face crops of the same person."""
        faces = [f for f in faces if f is not None and min(f.shape[:2]) >= MIN_FACE_PX]
        if not faces:
            return None
        pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in faces]
        with self._lock:
            inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
            probs = self.model(**inputs).logits.softmax(-1).mean(0).cpu().numpy()
        # Merge classifier classes that map to the same range (0-2 and 3-9).
        merged: dict[str, float] = {}
        for rng, p in zip(self.ranges, probs):
            merged[rng] = merged.get(rng, 0.0) + float(p)
        best = max(merged, key=merged.get)
        return best, merged[best]

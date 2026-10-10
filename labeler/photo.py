"""Search by photo: find recorded people who look like the person in a picture (e.g. one from a phone).

The picture is handled like a camera frame: the detector finds the people in it, the largest one is cropped as the
pipeline crops a person, and the appearance model turns the crop into the same kind of vector the database ranks
for look-alikes. With nobody found, the whole picture is used (a picture that is already a crop of one person).
The picture itself is not stored; only that a photo search happened is in the Audit log.

The detector here is a separate copy: the cameras' detectors keep tracking state between frames and run on
their own threads.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from . import geometry

MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 4000        # phone photos can be 12 MP+: shrink first, detection does not need more


def pick(boxes: list[tuple], w: int, h: int) -> tuple:
    """The person the photo is of: the one at the centre of the picture (people centre what they photograph),
    else the largest."""
    cx, cy = w / 2, h / 2
    central = [b for b in boxes if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
    return max(central or boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


class PhotoSearch:
    def __init__(self, cfg: dict, embedder, detector=None):
        self.cfg, self.embedder = cfg, embedder
        self._detector = detector          # tests pass a fake; otherwise loaded on first use
        self._lock = threading.Lock()

    def _detect(self, image: np.ndarray) -> list[tuple[float, float, float, float]]:
        with self._lock:
            if self._detector is None:
                from ultralytics import YOLO
                self._detector = YOLO(self.cfg["detector"]["model"])
            result = self._detector.predict(image, classes=[0], conf=0.25, imgsz=960,
                                            device=self.cfg["detector"]["device"], verbose=False)[0]
        if result.boxes is None:
            return []
        return [tuple(float(v) for v in b) for b in result.boxes.xyxy.cpu().numpy()]

    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("no picture was sent")
        if len(data) > MAX_BYTES:
            raise ValueError("the picture is larger than 10 MB")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("this is not a JPEG or PNG picture")
        h, w = image.shape[:2]
        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return image

    def vector(self, data: bytes) -> tuple[np.ndarray, dict]:
        """(appearance vector, what was used: picture size, people found, the box searched for)."""
        image = self.decode(data)
        h, w = image.shape[:2]
        boxes = self._detect(image)
        info = {"width": w, "height": h, "people": len(boxes), "box": None}
        crop = image
        if boxes:
            box = pick(boxes, w, h)
            person = geometry.crop(image, box, pad=0.05)
            if person is not None and person.size:
                crop = person
                info["box"] = [round(v) for v in box]
        return self.embedder.track_vector([crop]), info

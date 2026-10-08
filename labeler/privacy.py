"""Face blurring for privacy.

- web.blur_faces_for_viewers (default on): viewer accounts see blurred faces in the live video, the live
  thumbnails and saved pictures. Admins see faces. Saved pictures get a blurred copy in <crop_dir>/blurred/.
- storage.blur_faces (default off): faces are blurred in the saved pictures themselves, so nobody sees them
  later. Labeling is not affected either way: it uses the frames in memory, before anything is saved.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

BLOCKS = 6     # a face becomes about 6 x 6 coarse squares: not recognizable, clearly intentional
PICTURE_NAME = re.compile(r"[A-Za-z0-9_.-]+\.jpg")


def blur_boxes(image: np.ndarray, boxes: list) -> np.ndarray:
    """A copy of the image with each box pixelated."""
    out = image.copy()
    h, w = out.shape[:2]
    for box in boxes:
        if box is None:
            continue
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = out[y1:y2, x1:x2]
        bw = max(1, min(BLOCKS, x2 - x1))
        bh = max(1, round(bw * (y2 - y1) / (x2 - x1)))
        small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_AREA)
        out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def blur_whole(image: np.ndarray) -> np.ndarray:
    """For a picture whose face position is unknown (saved before blurring existed): blur all of it."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blurred_path(crop_dir: Path, name: str) -> Path:
    return crop_dir / "blurred" / name


def remove_picture(crop_dir: Path, name: str) -> bool:
    """Delete a saved picture and its blurred copy. True if the picture existed."""
    blurred_path(crop_dir, name).unlink(missing_ok=True)
    path = crop_dir / name
    if path.is_file():
        path.unlink()
        return True
    return False


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode image")
    return buf.tobytes()

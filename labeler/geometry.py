"""Body regions and frame-quality scoring from COCO pose keypoints."""
from __future__ import annotations

import cv2
import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14

Box = tuple[float, float, float, float]


def _pt(kpts: np.ndarray | None, i: int, min_conf: float = 0.5) -> tuple[float, float] | None:
    if kpts is None:
        return None
    x, y, c = kpts[i]
    return (float(x), float(y)) if c >= min_conf else None


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop(frame: np.ndarray, box: Box | None, pad: float = 0.0) -> np.ndarray | None:
    if box is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2]


def torso_box(kpts: np.ndarray | None, box: Box) -> Box | None:
    """Chest/belly region between the shoulders, inset to avoid arms and background."""
    ls, rs = _pt(kpts, L_SHOULDER), _pt(kpts, R_SHOULDER)
    if not (ls and rs):
        return None
    left, right = sorted([ls[0], rs[0]])
    width = right - left
    if width < 8:
        return None
    top = max(ls[1], rs[1]) + 0.1 * width
    lh, rh = _pt(kpts, L_HIP), _pt(kpts, R_HIP)
    if lh and rh:
        bottom = min(lh[1], rh[1]) - 0.05 * width
    else:
        # Kiosk close-ups often cut off at the waist.
        bottom = min(box[3], top + 0.9 * width)
    if bottom - top < 6:
        return None
    return (left + 0.2 * width, top, right - 0.2 * width, bottom)


def legs_box(kpts: np.ndarray | None, box: Box) -> Box | None:
    """Upper-thigh region; requires hips and knees to be visible."""
    lh, rh = _pt(kpts, L_HIP), _pt(kpts, R_HIP)
    lk, rk = _pt(kpts, L_KNEE), _pt(kpts, R_KNEE)
    if not (lh and rh and lk and rk):
        return None
    left, right = sorted([lh[0], rh[0]])
    width = max(right - left, 8.0)
    top = max(lh[1], rh[1]) + 0.15 * width
    knee_y = max(lk[1], rk[1])
    bottom = top + 0.6 * (knee_y - top)
    if bottom - top < 6:
        return None
    return (left - 0.1 * width, top, right + 0.1 * width, bottom)


def face_box(kpts: np.ndarray | None) -> Box | None:
    """Square face box estimated from nose, eyes and ears."""
    nose, le, re = _pt(kpts, NOSE), _pt(kpts, L_EYE), _pt(kpts, R_EYE)
    if not (nose and le and re):
        return None
    eye_dist = float(np.hypot(le[0] - re[0], le[1] - re[1]))
    size = eye_dist * 3.2
    lear, rear = _pt(kpts, L_EAR, 0.4), _pt(kpts, R_EAR, 0.4)
    if lear and rear:
        size = max(size, abs(lear[0] - rear[0]) * 1.4)
    if size < 24:
        return None
    cx = (le[0] + re[0] + nose[0]) / 3
    cy = nose[1] - 0.1 * size
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def head_box(kpts: np.ndarray | None, box: Box) -> Box:
    """Where to blur for privacy: larger than face_box so profiles, hair and chin are covered. Always returns
    a box: without head keypoints, the top of the person box (rather blur too much than leave a face)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pts = [p for p in (_pt(kpts, i, 0.3) for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)) if p]
    if pts:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        size = max(1.6 * (max(xs) - min(xs)), 0.3 * bw, 16.0)
        le, re = _pt(kpts, L_EYE, 0.3), _pt(kpts, R_EYE, 0.3)
        if le and re:
            size = max(size, 3.2 * float(np.hypot(le[0] - re[0], le[1] - re[1])))
        return (cx - size / 2, cy - 0.65 * size, cx + size / 2, cy + 0.5 * size)
    top = min(0.2 * bh, bw) if bh > 1.2 * bw else 0.35 * bh    # standing: head at the top; else be generous
    return (x1, y1, x2, y1 + top)


def crop_origin(frame: np.ndarray, box: Box, pad: float = 0.0) -> tuple[int, int]:
    """Top-left corner of what crop() cuts out, to move boxes into the crop's coordinates."""
    x1, y1, x2, y2 = box
    return max(0, int(x1 - (x2 - x1) * pad)), max(0, int(y1 - (y2 - y1) * pad))


def frontality(kpts: np.ndarray | None) -> float:
    """0..1: how clearly the face is visible and facing the camera."""
    if kpts is None:
        return 0.0
    conf = float(np.mean(kpts[[NOSE, L_EYE, R_EYE], 2]))
    nose, le, re = _pt(kpts, NOSE, 0.3), _pt(kpts, L_EYE, 0.3), _pt(kpts, R_EYE, 0.3)
    if not (nose and le and re):
        return conf * 0.3
    dl = np.hypot(nose[0] - le[0], nose[1] - le[1])
    dr = np.hypot(nose[0] - re[0], nose[1] - re[1])
    symmetry = 1.0 - abs(dl - dr) / max(dl, dr, 1e-6)
    return conf * float(symmetry)


def quality(frame: np.ndarray, box: Box, kpts: np.ndarray | None, mode: str) -> float:
    """Score how useful this frame of a person is for labeling."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    size = min(1.0, ((x2 - x1) * (y2 - y1)) / (w * h) / 0.15)
    complete = 0.6 if (x1 < 3 or y1 < 3 or x2 > w - 3 or y2 > h - 3) else 1.0

    region = crop(frame, box)
    sharp = 0.0
    if region is not None:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        sharp = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 300.0)

    face = frontality(kpts)
    if mode == "kiosk":
        return 0.35 * face + 0.25 * sharp + 0.25 * size + 0.15 * complete
    return 0.20 * face + 0.25 * sharp + 0.35 * size + 0.20 * complete

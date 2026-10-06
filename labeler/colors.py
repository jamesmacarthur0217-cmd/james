"""Deterministic clothing-color estimation from pixel regions (no model needed)."""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

_MAX_SIDE = 64


def color_histogram(region: np.ndarray | None) -> Counter:
    """Count pixels per named color in a BGR region."""
    if region is None or region.size == 0 or region.shape[0] * region.shape[1] < 50:
        return Counter()
    h, w = region.shape[:2]
    scale = _MAX_SIDE / max(h, w)
    if scale < 1:
        region = cv2.resize(region, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hue = hsv[:, 0] * 2.0          # degrees 0-360
    sat = hsv[:, 1] / 255.0
    val = hsv[:, 2] / 255.0

    conditions = [
        val < 0.18,
        (sat < 0.20) & (val > 0.78),
        sat < 0.20,
        (hue >= 15) & (hue < 45) & (sat < 0.45) & (val > 0.55),
        (hue >= 10) & (hue < 45) & (val < 0.55),
        (hue < 15) | (hue >= 340),
        hue < 40,
        hue < 70,
        hue < 165,
        hue < 255,
        hue < 290,
    ]
    names = ["black", "white", "gray", "beige", "brown", "red", "orange", "yellow", "green", "blue", "purple"]
    labels = np.select(conditions, names, default="pink")
    return Counter(labels.tolist())


def dominant(hist: Counter, min_pixels: int = 200) -> tuple[str, float] | None:
    """Most common color and the share of pixels it covers."""
    total = sum(hist.values())
    if total < min_pixels:
        return None
    name, count = hist.most_common(1)[0]
    return name, count / total


def is_low_color(frame: np.ndarray) -> bool:
    """True when the frame is effectively grayscale (e.g. an infrared night image),
    where color labels would be meaningless."""
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(sat.mean()) / 255.0 < 0.06

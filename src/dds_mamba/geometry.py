"""Continuous image/crop box operations with explicit half-open coordinates."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

Box = Tuple[float, float, float, float]


def finite_box(box: Box, min_size: float = 1.0) -> bool:
    return all(math.isfinite(x) for x in box) and box[2] >= min_size and box[3] >= min_size


def encode(box: Box, width: float, height: float) -> np.ndarray:
    x, y, w, h = box
    return np.array([x / width, y / height, math.log(max(w, 1.0) / width), math.log(max(h, 1.0) / height)])


def decode(value: np.ndarray, width: float, height: float) -> Box:
    return (float(width * value[0]), float(height * value[1]), float(width * np.exp(value[2])), float(height * np.exp(value[3])))


def crop_to_image(box: Box, crop_x: float, crop_y: float, side: float) -> Box:
    x, y, w, h = box
    return (crop_x - side / 2 + x * side, crop_y - side / 2 + y * side, w * side, h * side)


def image_to_crop(box: Box, crop_x: float, crop_y: float, side: float) -> Box:
    x, y, w, h = box
    return ((x - crop_x + side / 2) / side, (y - crop_y + side / 2) / side, w / side, h / side)


def iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return 0.0 if union <= 0 else inter / union


def roi_ratio(box: Box, width: float, height: float) -> float:
    x0, y0 = box[0] - box[2] / 2, box[1] - box[3] / 2
    x1, y1 = box[0] + box[2] / 2, box[1] + box[3] / 2
    inside = max(0.0, min(x1, width) - max(x0, 0.0)) * max(0.0, min(y1, height) - max(y0, 0.0))
    return inside / max(box[2] * box[3], 1e-6)

"""Признаки освещения и цвета верхней трети кадра (детектор неба).

Вектор: [mean_luminance, std_luminance, sky_r, sky_g, sky_b]. Все компоненты
нормированы в диапазон [0, 1]. Принимает путь либо BGR numpy-массив.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

LIGHTING_FEATURE_DIM = 5
_BYTE_MAX = 255.0
_SKY_FRACTION = 1.0 / 3.0

ImageInput = Union[Path, np.ndarray]


def build_lighting_feature_vector(image: ImageInput) -> np.ndarray:
    bgr = _resolve_bgr(image)
    mean_lum, std_lum = _luminance_stats(bgr)
    sky_r, sky_g, sky_b = _mean_top_third_rgb(bgr)
    return np.array([mean_lum, std_lum, sky_r, sky_g, sky_b], dtype=np.float32)


def _luminance_stats(bgr: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / _BYTE_MAX
    return float(gray.mean()), float(gray.std())


def _mean_top_third_rgb(bgr: np.ndarray) -> tuple[float, float, float]:
    height = bgr.shape[0]
    cutoff = max(1, int(height * _SKY_FRACTION))
    top_strip = bgr[:cutoff]
    mean_bgr = top_strip.reshape(-1, 3).mean(axis=0) / _BYTE_MAX
    blue, green, red = mean_bgr.tolist()
    return red, green, blue


def _resolve_bgr(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    raw = np.fromfile(str(image), dtype=np.uint8)
    if raw.size == 0:
        raise FileNotFoundError(f"Не удалось прочитать файл: {image}")
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV не смог декодировать изображение: {image}")
    return bgr

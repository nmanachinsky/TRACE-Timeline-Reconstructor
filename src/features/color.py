"""Цветовые признаки для определения сезона.

- HSV-гистограмма (нормированная) — общий цветовой профиль кадра.
- Бинарные индикаторы снега и листвы — сильные сезонные сигналы.

Все функции принимают либо путь, либо уже декодированное BGR-изображение, чтобы
можно было считать файл один раз и переиспользовать в нескольких признаках.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from src.config import COLOR

SEASONAL_INDICATOR_DIM = 2  # snow, foliage

_OPENCV_HUE_MAX = 180  # OpenCV кодирует H в диапазоне 0..179
_OPENCV_SAT_MAX = 255
_OPENCV_VAL_MAX = 255

ImageInput = Union[Path, np.ndarray]


def hsv_histogram(image: ImageInput, bins: tuple[int, int, int] = COLOR.hsv_bins) -> np.ndarray:
    """Нормированная HSV-гистограмма, развёрнутая в 1D float32-вектор."""
    bgr = _resolve_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=list(bins),
        ranges=[0, _OPENCV_HUE_MAX, 0, _OPENCV_SAT_MAX + 1, 0, _OPENCV_VAL_MAX + 1],
    )
    flat = hist.flatten().astype(np.float32)
    total = flat.sum()
    if total > 0:
        flat /= total
    return flat


def detect_snow(image: ImageInput) -> float:
    """Доля 'снежных' пикселей > порога → 1.0, иначе 0.0."""
    hsv = cv2.cvtColor(_resolve_bgr(image), cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1].astype(np.float32) / _OPENCV_SAT_MAX
    value = hsv[..., 2].astype(np.float32) / _OPENCV_VAL_MAX

    mask = (value > COLOR.snow_value_threshold) & (saturation < COLOR.snow_saturation_threshold)
    ratio = float(mask.mean())
    return 1.0 if ratio > COLOR.snow_pixel_ratio else 0.0


def detect_foliage(image: ImageInput) -> float:
    """Доля 'лиственных' пикселей (зелёный hue + насыщенность) > 25% → 1.0."""
    hsv = cv2.cvtColor(_resolve_bgr(image), cv2.COLOR_BGR2HSV)
    hue_deg = hsv[..., 0].astype(np.float32) * (360.0 / _OPENCV_HUE_MAX)
    saturation = hsv[..., 1].astype(np.float32) / _OPENCV_SAT_MAX

    mask = (
        (hue_deg >= COLOR.foliage_hue_min_deg)
        & (hue_deg <= COLOR.foliage_hue_max_deg)
        & (saturation > COLOR.foliage_saturation_threshold)
    )
    ratio = float(mask.mean())
    return 1.0 if ratio > 0.25 else 0.0


def build_color_feature_vector(
    image: ImageInput, bins: tuple[int, int, int] = COLOR.hsv_bins
) -> np.ndarray:
    """Полный цветовой признак: HSV-гистограмма + сезонные индикаторы.

    При передаче пути файл декодируется один раз — затем все три признака считаются
    на одном и том же BGR-массиве.
    """
    bgr = _resolve_bgr(image)
    hist = hsv_histogram(bgr, bins=bins)
    indicators = np.array([detect_snow(bgr), detect_foliage(bgr)], dtype=np.float32)
    return np.concatenate([hist, indicators]).astype(np.float32)


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

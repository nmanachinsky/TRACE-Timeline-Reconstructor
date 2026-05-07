"""Тесты цветовых признаков для определения сезона.

- HSV-гистограмма размером H×S×V (по умолчанию 8×8×8 = 512), нормированная.
- Бинарный индикатор снега: V>порога ∧ S<порога на >пороге пикселей.
- Бинарный индикатор листвы: H ∈ [60,150]° ∧ S>порога.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.color import (
    SEASONAL_INDICATOR_DIM,
    build_color_feature_vector,
    detect_foliage,
    detect_snow,
    hsv_histogram,
)


class TestHsvHistogram:
    def test_возвращает_массив_правильной_формы(self, tmp_path: Path, make_jpeg) -> None:
        img = make_jpeg(tmp_path / "a.jpg", color=(128, 128, 128))

        hist = hsv_histogram(img, bins=(8, 8, 8))

        assert hist.shape == (8 * 8 * 8,)
        assert hist.dtype == np.float32

    def test_гистограмма_нормирована_к_единице(self, tmp_path: Path, make_jpeg) -> None:
        img = make_jpeg(tmp_path / "a.jpg", color=(50, 100, 200))

        hist = hsv_histogram(img, bins=(8, 8, 8))

        assert np.isclose(hist.sum(), 1.0, atol=1e-5)

    def test_белое_фото_попадает_в_бин_низкой_насыщенности(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        img = make_jpeg(tmp_path / "white.jpg", color=(255, 255, 255))

        hist = hsv_histogram(img, bins=(4, 4, 4))

        # Реструктурируем: hist[h, s, v] — и проверяем, что s=0 (низкая насыщенность)
        # содержит ~всю массу
        reshaped = hist.reshape(4, 4, 4)
        low_s_mass = reshaped[:, 0, :].sum()
        assert low_s_mass > 0.95

    def test_зелёное_фото_попадает_в_зелёный_оттенок(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        img = make_jpeg(tmp_path / "green.jpg", color=(0, 255, 0))

        hist = hsv_histogram(img, bins=(8, 4, 4))
        reshaped = hist.reshape(8, 4, 4)

        # Зелёный (H≈120° на шкале 0-360, или H=60 в OpenCV 0-179)
        # При 8 бинах H: bin индекс ≈ 60/180 * 8 ≈ 2.7 → бин 2 или 3
        green_bins_mass = reshaped[2:4, :, :].sum()
        assert green_bins_mass > 0.9


class TestDetectSnow:
    def test_белое_фото_детектируется_как_снег(self, tmp_path: Path, make_jpeg) -> None:
        img = make_jpeg(tmp_path / "snow.jpg", color=(245, 248, 250))

        assert detect_snow(img) == 1.0

    def test_зелёное_фото_не_детектируется_как_снег(self, tmp_path: Path, make_jpeg) -> None:
        img = make_jpeg(tmp_path / "summer.jpg", color=(20, 180, 60))

        assert detect_snow(img) == 0.0


class TestDetectFoliage:
    def test_зелёное_фото_детектируется_как_листва(self, tmp_path: Path, make_jpeg) -> None:
        img = make_jpeg(tmp_path / "summer.jpg", color=(20, 180, 60))

        assert detect_foliage(img) == 1.0

    def test_белое_фото_не_детектируется_как_листва(self, tmp_path: Path, make_jpeg) -> None:
        img = make_jpeg(tmp_path / "snow.jpg", color=(250, 250, 250))

        assert detect_foliage(img) == 0.0


class TestBuildColorFeatureVector:
    def test_длина_равна_гистограмма_плюс_индикаторы(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        img = make_jpeg(tmp_path / "a.jpg", color=(128, 128, 128))

        vec = build_color_feature_vector(img, bins=(8, 8, 8))

        assert vec.shape == (8 * 8 * 8 + SEASONAL_INDICATOR_DIM,)
        assert vec.dtype == np.float32

    def test_воспроизводим_для_одного_файла(self, tmp_path: Path, make_jpeg) -> None:
        img = make_jpeg(tmp_path / "a.jpg", color=(50, 100, 150))

        v1 = build_color_feature_vector(img)
        v2 = build_color_feature_vector(img)

        assert np.array_equal(v1, v2)


@pytest.mark.parametrize(
    ("color", "expected_indicator"),
    [
        ((250, 250, 250), "snow"),
        ((20, 180, 60), "foliage"),
        ((100, 50, 30), "neither"),
    ],
)
def test_seasonal_indicators_согласованы(
    tmp_path: Path, make_jpeg, color: tuple[int, int, int], expected_indicator: str
) -> None:
    img = make_jpeg(tmp_path / "a.jpg", color=color)
    snow = detect_snow(img)
    foliage = detect_foliage(img)

    if expected_indicator == "snow":
        assert snow == 1.0 and foliage == 0.0
    elif expected_indicator == "foliage":
        assert snow == 0.0 and foliage == 1.0
    else:
        assert snow == 0.0 and foliage == 0.0

"""Тесты признаков освещения и цвета верхней трети кадра.

Признаки:
- mean_luminance, std_luminance — общая яркость и контраст в grayscale.
- sky_red, sky_green, sky_blue — средний RGB верхней трети кадра (зачастую небо).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.lighting import LIGHTING_FEATURE_DIM, build_lighting_feature_vector


class TestLightingFeatures:
    def test_длина_вектора_фиксирована(self, tmp_path: Path, make_jpeg) -> None:
        img = make_jpeg(tmp_path / "a.jpg")

        vec = build_lighting_feature_vector(img)

        assert vec.shape == (LIGHTING_FEATURE_DIM,)
        assert vec.dtype == np.float32

    def test_яркое_фото_имеет_высокую_среднюю_яркость(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        bright = make_jpeg(tmp_path / "bright.jpg", color=(240, 240, 240))
        dark = make_jpeg(tmp_path / "dark.jpg", color=(20, 20, 20))

        bright_vec = build_lighting_feature_vector(bright)
        dark_vec = build_lighting_feature_vector(dark)

        # mean_luminance — первый признак вектора (нормирован 0..1)
        assert bright_vec[0] > 0.85
        assert dark_vec[0] < 0.15

    def test_однородное_фото_имеет_низкий_контраст(self, tmp_path: Path, make_jpeg) -> None:
        img = make_jpeg(tmp_path / "flat.jpg", color=(128, 128, 128))

        vec = build_lighting_feature_vector(img)

        # std_luminance — второй признак, для однородного фото близок к 0
        assert vec[1] < 0.05

    def test_синее_верхнее_поле_отражается_в_sky_blue(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        from PIL import Image

        path = tmp_path / "sky.jpg"
        img = Image.new("RGB", (60, 60), color=(50, 50, 50))
        # Верхняя треть — синяя
        for y in range(20):
            for x in range(60):
                img.putpixel((x, y), (10, 30, 220))
        img.save(path, format="JPEG")

        vec = build_lighting_feature_vector(path)

        # Структура: [mean_lum, std_lum, sky_r, sky_g, sky_b]
        # sky_b должен быть выше sky_r (нормированы 0..1)
        sky_r, sky_g, sky_b = vec[2], vec[3], vec[4]
        assert sky_b > sky_r
        assert sky_b > 0.7

    def test_воспроизводим(self, tmp_path: Path, make_jpeg) -> None:
        img = make_jpeg(tmp_path / "a.jpg", color=(100, 100, 100))

        v1 = build_lighting_feature_vector(img)
        v2 = build_lighting_feature_vector(img)

        assert np.array_equal(v1, v2)

"""Тесты OCR-признаков.

Чистая логика парсинга годов + обёртка с заглушкой EasyOCR.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.ocr import (
    OCR_FEATURE_DIM,
    OCR_YEAR_DIM,
    SUPPORTED_YEAR_RANGE,
    OcrFeatureExtractor,
    build_ocr_features,
    extract_years_from_texts,
)


class TestExtractYears:
    def test_находит_единственный_год(self) -> None:
        assert extract_years_from_texts(["Сделано в 2023 году"]) == (2023,)

    def test_дедуплицирует_повторяющиеся_годы(self) -> None:
        assert extract_years_from_texts(["2022 hello 2022"]) == (2022,)

    def test_сохраняет_порядок_первого_появления(self) -> None:
        assert extract_years_from_texts(["2024 then 2019"]) == (2024, 2019)

    def test_фильтрует_годы_вне_допустимого_диапазона(self) -> None:
        assert extract_years_from_texts(["1999 was old, 2050 is too far"]) == ()

    def test_годы_внутри_слов_не_извлекаются(self) -> None:
        # "abc2023def" не должно извлечь 2023 (granica слов)
        assert extract_years_from_texts(["abc2023def"]) == ()

    def test_несколько_годов_из_разных_строк(self) -> None:
        assert extract_years_from_texts(["First: 2020", "Second: 2021"]) == (2020, 2021)

    def test_пустой_список_текста(self) -> None:
        assert extract_years_from_texts([]) == ()


class TestBuildOcrFeatures:
    def test_пустой_текст_всё_нули_и_has_text_ноль(self) -> None:
        feats = build_ocr_features([])

        assert feats.has_text == 0.0
        assert feats.year_one_hot.shape == (OCR_YEAR_DIM,)
        assert feats.year_one_hot.dtype == np.float32
        assert np.array_equal(feats.year_one_hot, np.zeros(OCR_YEAR_DIM, dtype=np.float32))
        assert feats.detected_years == ()

    def test_текст_без_года_включает_has_text_но_нули_one_hot(self) -> None:
        feats = build_ocr_features(["Какой-то текст без года"])

        assert feats.has_text == 1.0
        assert np.array_equal(feats.year_one_hot, np.zeros(OCR_YEAR_DIM, dtype=np.float32))
        assert feats.detected_years == ()

    def test_one_hot_в_правильной_позиции(self) -> None:
        feats = build_ocr_features(["Дата: 2023"])

        expected = np.zeros(OCR_YEAR_DIM, dtype=np.float32)
        expected[2023 - SUPPORTED_YEAR_RANGE.start] = 1.0
        assert np.array_equal(feats.year_one_hot, expected)
        assert feats.detected_years == (2023,)

    def test_несколько_годов_все_единицы_в_one_hot(self) -> None:
        feats = build_ocr_features(["2020 и 2024"])

        assert feats.year_one_hot[2020 - SUPPORTED_YEAR_RANGE.start] == 1.0
        assert feats.year_one_hot[2024 - SUPPORTED_YEAR_RANGE.start] == 1.0
        assert feats.year_one_hot.sum() == 2.0

    def test_to_vector_правильная_форма_и_тип(self) -> None:
        feats = build_ocr_features(["2024"])

        v = feats.to_vector()

        assert v.shape == (OCR_FEATURE_DIM,)
        assert v.dtype == np.float32
        assert v[-1] == 1.0  # has_text

    def test_только_пробелы_не_считаются_текстом(self) -> None:
        feats = build_ocr_features(["   ", ""])

        assert feats.has_text == 0.0


class TestOcrFeatureExtractor:
    def test_extractor_использует_результаты_easyocr_фильтрует_уверенность(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = OcrFeatureExtractor()

        class StubReader:
            def readtext(self, img: np.ndarray, **kw):
                return [
                    (None, "Привет 2023", 0.95),
                    (None, "low conf 2050", 0.1),  # ниже порога 0.3
                ]

        monkeypatch.setattr(extractor, "_ensure_reader", lambda: StubReader())

        res = extractor.extract(np.zeros((100, 100, 3), dtype=np.uint8))

        assert res.has_text == 1.0
        assert res.detected_years == (2023,)

    def test_extractor_уменьшает_большие_изображения(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = OcrFeatureExtractor(max_side=512)
        captured: dict[str, tuple[int, int, int]] = {}

        class StubReader:
            def readtext(self, img: np.ndarray, **kw) -> list:
                captured["shape"] = img.shape
                return []

        monkeypatch.setattr(extractor, "_ensure_reader", lambda: StubReader())

        big = np.zeros((2000, 1000, 3), dtype=np.uint8)
        extractor.extract(big)

        h, w, _ = captured["shape"]
        assert max(h, w) == 512

    def test_extractor_не_уменьшает_маленькие_изображения(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = OcrFeatureExtractor(max_side=1024)
        captured: dict[str, tuple[int, int, int]] = {}

        class StubReader:
            def readtext(self, img: np.ndarray, **kw) -> list:
                captured["shape"] = img.shape
                return []

        monkeypatch.setattr(extractor, "_ensure_reader", lambda: StubReader())

        small = np.zeros((400, 300, 3), dtype=np.uint8)
        extractor.extract(small)

        assert captured["shape"] == (400, 300, 3)

    def test_extractor_ленивая_инициализация_не_грузит_модель_на_создании(self) -> None:
        extractor = OcrFeatureExtractor()

        assert extractor._reader is None  # type: ignore[attr-defined]

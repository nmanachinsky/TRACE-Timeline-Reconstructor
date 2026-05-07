"""OCR-признаки для извлечения года из текста на фото (скриншоты, документы).

Идея: на скриншотах часто видна дата (заголовки чатов, дата-стампы); если EasyOCR
найдёт год вида 20YY в допустимом диапазоне 2005..2030, можно сразу резко поднять
вероятность соответствующего года в классификаторе. Для большинства "природных"
фото OCR не найдёт текста и вектор останется нулевым — это ожидаемое поведение.

Логика парсинга годов (`extract_years_from_texts`, `build_ocr_features`) — чистая
и тестируется без загрузки EasyOCR. Сам OCR-инференс инкапсулирован в
`OcrFeatureExtractor` с ленивой инициализацией.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

SUPPORTED_YEAR_RANGE: range = range(2005, 2031)  # 26 лет: 2005..2030 включительно
OCR_YEAR_DIM: int = len(SUPPORTED_YEAR_RANGE)
OCR_FEATURE_DIM: int = OCR_YEAR_DIM + 1  # +1 — флаг has_text

_YEAR_PATTERN = re.compile(r"\b(20[0-2]\d)\b")
_DEFAULT_LANGUAGES: tuple[str, ...] = ("ru", "en")
_DEFAULT_MAX_SIDE: int = 1024
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.3


@dataclass(frozen=True)
class OcrFeatures:
    """Признак OCR для одного фото."""

    year_one_hot: np.ndarray
    has_text: float
    detected_years: tuple[int, ...]

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.year_one_hot, [self.has_text]]).astype(np.float32)


def extract_years_from_texts(texts: Iterable[str]) -> tuple[int, ...]:
    """Уникальные годы из набора OCR-фраз; диапазон ограничен SUPPORTED_YEAR_RANGE.

    Why: ограничение по диапазону отсекает шум вроде телефонных номеров или
    почтовых индексов, которые иногда содержат подряд 4 цифры, начинающиеся на 20
    в обратной транскрипции (но они не попадают в 2005..2030).
    """
    seen: list[int] = []
    for text in texts:
        for match in _YEAR_PATTERN.finditer(text):
            year = int(match.group(1))
            if year in SUPPORTED_YEAR_RANGE and year not in seen:
                seen.append(year)
    return tuple(seen)


def build_ocr_features(texts: Sequence[str]) -> OcrFeatures:
    """Строит one-hot годов + флаг has_text на основе списка распознанных фраз."""
    years = extract_years_from_texts(texts)
    one_hot = np.zeros(OCR_YEAR_DIM, dtype=np.float32)
    for year in years:
        one_hot[year - SUPPORTED_YEAR_RANGE.start] = 1.0
    has_text = 1.0 if any(t.strip() for t in texts) else 0.0
    return OcrFeatures(
        year_one_hot=one_hot,
        has_text=has_text,
        detected_years=years,
    )


class OcrFeatureExtractor:
    """Обёртка над `easyocr.Reader` с ленивой инициализацией и ресайзом.

    Why: EasyOCR грузит ~500MB моделей при первом обращении, ресайз до 1024px
    ускоряет инференс на 4K-фотках в 4-10 раз без значимой потери точности на
    крупных датах/заголовках.
    """

    def __init__(
        self,
        languages: tuple[str, ...] = _DEFAULT_LANGUAGES,
        gpu: bool = False,
        max_side: int = _DEFAULT_MAX_SIDE,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._languages = languages
        self._gpu = gpu
        self._max_side = max_side
        self._confidence_threshold = confidence_threshold
        self._reader: object | None = None

    def _ensure_reader(self) -> object:
        if self._reader is not None:
            return self._reader
        import easyocr  # тяжёлый импорт — только при первом инференсе

        self._reader = easyocr.Reader(list(self._languages), gpu=self._gpu, verbose=False)
        return self._reader

    def extract(self, bgr_image: np.ndarray) -> OcrFeatures:
        reader = self._ensure_reader()
        resized = self._resize_if_needed(bgr_image)
        results = reader.readtext(resized, detail=1, paragraph=False)  # type: ignore[attr-defined]
        texts = [
            text
            for _, text, confidence in results
            if confidence >= self._confidence_threshold
        ]
        return build_ocr_features(texts)

    def _resize_if_needed(self, bgr_image: np.ndarray) -> np.ndarray:
        height, width = bgr_image.shape[:2]
        longest_side = max(height, width)
        if longest_side <= self._max_side:
            return bgr_image
        import cv2  # локальный импорт, чтобы тесты без cv2 не зависели

        scale = self._max_side / longest_side
        new_size = (int(round(width * scale)), int(round(height * scale)))
        return cv2.resize(bgr_image, new_size, interpolation=cv2.INTER_AREA)

"""Тесты face-эмбеддингов.

Чистая логика агрегации (без загрузки insightface) + проверка обёртки на стабах.
Реальные ONNX-модели грузятся только в интеграционном пайплайне.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.faces import (
    FACE_EMBEDDING_DIM,
    FACE_FEATURE_DIM,
    FaceFeatureExtractor,
    FaceFeatures,
    aggregate_face_embeddings,
)


class TestAggregateFaceEmbeddings:
    def test_пустой_список_даёт_нулевой_вектор_и_has_face_ноль(self) -> None:
        result = aggregate_face_embeddings([])

        assert result.face_count == 0
        assert result.has_face == 0.0
        assert result.embedding.shape == (FACE_EMBEDDING_DIM,)
        assert result.embedding.dtype == np.float32
        assert np.array_equal(result.embedding, np.zeros(FACE_EMBEDDING_DIM, dtype=np.float32))

    def test_среднее_двух_эмбеддингов(self) -> None:
        e1 = np.full(FACE_EMBEDDING_DIM, 0.2, dtype=np.float32)
        e2 = np.full(FACE_EMBEDDING_DIM, 0.8, dtype=np.float32)

        result = aggregate_face_embeddings([e1, e2])

        assert result.face_count == 2
        assert result.has_face == 1.0
        assert np.allclose(result.embedding, 0.5)

    def test_невалидная_размерность_бросает_ошибку(self) -> None:
        bad = np.ones(100, dtype=np.float32)

        with pytest.raises(ValueError):
            aggregate_face_embeddings([bad])

    def test_to_vector_конкатенирует_эмбеддинг_и_флаг(self) -> None:
        result = aggregate_face_embeddings([np.ones(FACE_EMBEDDING_DIM, dtype=np.float32)])

        vec = result.to_vector()

        assert vec.shape == (FACE_FEATURE_DIM,)
        assert vec.dtype == np.float32
        assert vec[-1] == 1.0
        assert np.allclose(vec[:-1], 1.0)

    def test_to_vector_для_пустых_лиц(self) -> None:
        result = aggregate_face_embeddings([])

        vec = result.to_vector()

        assert vec.shape == (FACE_FEATURE_DIM,)
        assert vec[-1] == 0.0
        assert np.array_equal(vec[:-1], np.zeros(FACE_EMBEDDING_DIM, dtype=np.float32))


class TestFaceFeaturesDataclass:
    def test_фрозен_не_позволяет_изменять_поля(self) -> None:
        result = aggregate_face_embeddings([])

        with pytest.raises(AttributeError):
            result.face_count = 99  # type: ignore[misc]


class TestFaceFeatureExtractor:
    def test_extractor_возвращает_no_face_для_пустого_результата(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extractor = FaceFeatureExtractor()

        class StubApp:
            def get(self, img: np.ndarray) -> list:
                return []

        monkeypatch.setattr(extractor, "_ensure_app", lambda: StubApp())

        result = extractor.extract(np.zeros((100, 100, 3), dtype=np.uint8))

        assert result.face_count == 0
        assert result.has_face == 0.0

    def test_extractor_использует_normed_embedding_из_insightface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = FaceFeatureExtractor()

        class FakeFace:
            normed_embedding = np.full(FACE_EMBEDDING_DIM, 0.1, dtype=np.float32)

        class StubApp:
            def get(self, img: np.ndarray) -> list:
                return [FakeFace(), FakeFace()]

        monkeypatch.setattr(extractor, "_ensure_app", lambda: StubApp())

        result = extractor.extract(np.zeros((100, 100, 3), dtype=np.uint8))

        assert result.face_count == 2
        assert result.has_face == 1.0
        assert np.allclose(result.embedding, 0.1)

    def test_extractor_ленивая_инициализация_не_грузит_модель_на_создании(self) -> None:
        extractor = FaceFeatureExtractor()

        # Просто факт того, что конструктор не упал и app не загружен
        assert extractor._app is None  # type: ignore[attr-defined]

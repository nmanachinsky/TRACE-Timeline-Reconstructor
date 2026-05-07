"""Smoke-тесты ResNet-feature extractor и кэша.

Реальный инференс ResNet-50 требует загрузки 100MB весов — отмечено как slow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.resnet import (
    ResNetEmbeddings,
    load_resnet_cache,
    save_resnet_cache,
)


class TestResNetCache:
    def test_round_trip_сохраняет_векторы_и_id(self, tmp_path: Path) -> None:
        embeddings = ResNetEmbeddings(
            ids=("a", "b", "c"),
            vectors=np.arange(6, dtype=np.float32).reshape(3, 2),
        )
        cache_path = tmp_path / "resnet.npz"

        save_resnet_cache(cache_path, embeddings)
        loaded = load_resnet_cache(cache_path)

        assert loaded.ids == embeddings.ids
        assert np.array_equal(loaded.vectors, embeddings.vectors)
        assert loaded.vectors.dtype == np.float32

    def test_to_dict_возвращает_id_to_vector(self) -> None:
        embeddings = ResNetEmbeddings(
            ids=("x", "y"),
            vectors=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

        as_dict = embeddings.to_dict()

        assert set(as_dict.keys()) == {"x", "y"}
        assert np.array_equal(as_dict["x"], np.array([1.0, 2.0], dtype=np.float32))


@pytest.mark.slow
def test_resnet_возвращает_векторы_правильной_формы(tmp_path: Path, make_jpeg) -> None:
    """Реальный прогон pretrained ResNet-50 на двух картинках."""
    from src.features.resnet import ResNetFeatureExtractor

    img_a = make_jpeg(tmp_path / "a.jpg", color=(50, 100, 150))
    img_b = make_jpeg(tmp_path / "b.jpg", color=(200, 50, 50))

    extractor = ResNetFeatureExtractor()
    vectors = extractor.encode_paths([img_a, img_b], batch_size=2)

    assert vectors.shape == (2, extractor.embedding_dim)
    assert vectors.dtype == np.float32
    # Разный контент → разные эмбеддинги
    assert not np.allclose(vectors[0], vectors[1])


@pytest.mark.slow
def test_resnet_пустой_список_возвращает_пустой_массив() -> None:
    from src.features.resnet import ResNetFeatureExtractor

    extractor = ResNetFeatureExtractor()
    vectors = extractor.encode_paths([])

    assert vectors.shape == (0, extractor.embedding_dim)

"""Тесты загрузки FeatureBundle: core + опциональные M2-признаки."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.common import FeatureBundle, load_feature_bundle


def _save_cache(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    np.savez_compressed(path, ids=np.asarray(ids), vectors=vectors.astype(np.float32))


def _build_core_caches(features_dir: Path, ids: list[str]) -> None:
    n = len(ids)
    _save_cache(features_dir / "resnet.npz", ids, np.ones((n, 8), dtype=np.float32))
    _save_cache(features_dir / "color.npz", ids, np.ones((n, 4), dtype=np.float32) * 2.0)
    _save_cache(features_dir / "lighting.npz", ids, np.ones((n, 2), dtype=np.float32) * 3.0)


class TestLoadFeatureBundleCore:
    def test_загружает_только_core_по_умолчанию(self, tmp_path: Path) -> None:
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        ids = ["a", "b", "c"]
        _build_core_caches(features_dir, ids)

        bundle = load_feature_bundle(features_dir)

        assert bundle.ids == tuple(sorted(ids))
        assert bundle.faces is None
        assert bundle.ocr is None
        assert bundle.has_m2_features is False
        assert bundle.full_matrix.shape == (3, 8 + 4 + 2)

    def test_бросает_ошибку_при_непересекающихся_id(self, tmp_path: Path) -> None:
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        _save_cache(features_dir / "resnet.npz", ["a"], np.ones((1, 8), dtype=np.float32))
        _save_cache(features_dir / "color.npz", ["b"], np.ones((1, 4), dtype=np.float32))
        _save_cache(features_dir / "lighting.npz", ["c"], np.ones((1, 2), dtype=np.float32))

        with pytest.raises(ValueError, match="не пересекаются"):
            load_feature_bundle(features_dir)


class TestLoadFeatureBundleFull:
    def test_загружает_faces_и_ocr_когда_попросили(self, tmp_path: Path) -> None:
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        ids = ["a", "b"]
        _build_core_caches(features_dir, ids)
        _save_cache(features_dir / "faces.npz", ids, np.ones((2, 513), dtype=np.float32) * 0.1)
        _save_cache(features_dir / "ocr.npz", ids, np.ones((2, 27), dtype=np.float32) * 0.2)

        bundle = load_feature_bundle(features_dir, include_faces=True, include_ocr=True)

        assert bundle.faces is not None and bundle.faces.shape == (2, 513)
        assert bundle.ocr is not None and bundle.ocr.shape == (2, 27)
        assert bundle.has_m2_features is True
        assert bundle.full_matrix.shape == (2, 8 + 4 + 2 + 513 + 27)

    def test_только_faces_без_ocr(self, tmp_path: Path) -> None:
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        ids = ["x", "y"]
        _build_core_caches(features_dir, ids)
        _save_cache(features_dir / "faces.npz", ids, np.ones((2, 513), dtype=np.float32))

        bundle = load_feature_bundle(features_dir, include_faces=True, include_ocr=False)

        assert bundle.faces is not None
        assert bundle.ocr is None
        assert bundle.full_matrix.shape == (2, 8 + 4 + 2 + 513)

    def test_пересечение_id_сужается_до_общих_по_всем_кэшам(self, tmp_path: Path) -> None:
        """Если faces.npz содержит подмножество id — bundle ограничен этим подмножеством."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        core_ids = ["a", "b", "c"]
        _build_core_caches(features_dir, core_ids)
        # Faces только для двух из трёх — bundle обрежется до этих двух
        _save_cache(features_dir / "faces.npz", ["a", "c"], np.ones((2, 513), dtype=np.float32))
        _save_cache(features_dir / "ocr.npz", ["a", "c"], np.ones((2, 27), dtype=np.float32))

        bundle = load_feature_bundle(features_dir, include_faces=True, include_ocr=True)

        assert bundle.ids == ("a", "c")
        assert bundle.full_matrix.shape[0] == 2


class TestFeatureBundleProperties:
    def test_resnet_dim_возвращает_второе_измерение_resnet(self) -> None:
        bundle = FeatureBundle(
            ids=("a",),
            resnet=np.zeros((1, 2048), dtype=np.float32),
            color=np.zeros((1, 4), dtype=np.float32),
            lighting=np.zeros((1, 5), dtype=np.float32),
        )

        assert bundle.resnet_dim == 2048

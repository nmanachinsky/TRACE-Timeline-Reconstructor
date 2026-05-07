"""Smoke-тесты ансамбля и cluster-consensus на синтетических признаках."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.classifier import (
    FeatureLayout,
    load_classifier,
    save_classifier,
    train_classifier,
)
from src.models.clustering import (
    ClusterAssignment,
    apply_cluster_consensus,
    cluster_resnet_embeddings,
)


def _make_synthetic_dataset(
    seed: int = 0, n_per_class: int = 40, n_classes: int = 3
) -> tuple[np.ndarray, list[str]]:
    """Признаки: 2048 ResNet (псевдо) + 5 цвет/свет, классы линейно разделимы."""
    rng = np.random.default_rng(seed)
    centers = np.eye(n_classes, 2048).astype(np.float32) * 5.0
    extras = rng.normal(size=(n_classes, 5)).astype(np.float32)

    rows = []
    labels: list[str] = []
    for ci in range(n_classes):
        cluster = centers[ci] + rng.normal(scale=0.5, size=(n_per_class, 2048)).astype(np.float32)
        extra = extras[ci] + rng.normal(scale=0.3, size=(n_per_class, 5)).astype(np.float32)
        rows.append(np.hstack([cluster, extra]))
        labels.extend([f"class_{ci}"] * n_per_class)
    return np.vstack(rows), labels


class TestTrainedClassifier:
    def test_должен_достичь_высокой_точности_на_разделимых_данных(self) -> None:
        x, y = _make_synthetic_dataset(seed=42)
        layout = FeatureLayout(resnet_start=0, resnet_end=2048)

        clf = train_classifier(x, y, layout)
        predictions = clf.predict_labels(x)

        accuracy = sum(p == t for p, t in zip(predictions, y)) / len(y)
        assert accuracy > 0.95

    def test_round_trip_сохранение_и_загрузка(self, tmp_path) -> None:
        x, y = _make_synthetic_dataset(seed=1, n_per_class=20, n_classes=2)
        layout = FeatureLayout(resnet_start=0, resnet_end=2048)
        clf = train_classifier(x, y, layout)

        path = tmp_path / "clf.joblib"
        save_classifier(path, clf)
        loaded = load_classifier(path)

        original_pred = clf.predict_labels(x)
        loaded_pred = loaded.predict_labels(x)
        assert original_pred == loaded_pred


class TestClustering:
    def test_кластеризация_находит_ожидаемое_число_групп(self) -> None:
        rng = np.random.default_rng(0)
        # Три плотных кластера в 16-мерном пространстве (PCA уменьшит до них)
        cluster_a = rng.normal(loc=[0] * 16, scale=0.2, size=(20, 16))
        cluster_b = rng.normal(loc=[5] * 16, scale=0.2, size=(20, 16))
        cluster_c = rng.normal(loc=[-5] * 16, scale=0.2, size=(20, 16))
        vectors = np.vstack([cluster_a, cluster_b, cluster_c]).astype(np.float32)
        ids = [f"id_{i}" for i in range(60)]

        assignment = cluster_resnet_embeddings(
            ids, vectors, min_cluster_size=4, pca_components=None
        )

        assert assignment.cluster_count >= 2

    def test_consensus_усиливает_train_метки_в_кластере(self) -> None:
        # 4 элемента в одном кластере: 3 train + 1 test
        ids = ("a_train", "b_train", "c_train", "d_test")
        labels = np.array([0, 0, 0, 0], dtype=np.int32)  # все в кластере 0
        assignment = ClusterAssignment(ids=ids, labels=labels)

        classes = np.array(["winter", "summer"])
        # Базовая модель уверена, что test → summer
        base_proba = np.array([[0.2, 0.8]], dtype=np.float32)
        train_labels = {"a_train": "winter", "b_train": "winter", "c_train": "winter"}

        smoothed = apply_cluster_consensus(
            assignment,
            test_ids=("d_test",),
            base_proba=base_proba,
            classes=classes,
            train_labels_by_id=train_labels,
            consensus_weight=0.7,
        )

        # После сглаживания должно сместиться в сторону winter (доминирующего в кластере)
        assert smoothed[0, 0] > base_proba[0, 0]
        assert smoothed[0, 1] < base_proba[0, 1]

    def test_consensus_не_трогает_шумовые_точки(self) -> None:
        ids = ("test_noise",)
        labels = np.array([-1], dtype=np.int32)
        assignment = ClusterAssignment(ids=ids, labels=labels)
        classes = np.array(["winter", "summer"])
        base_proba = np.array([[0.3, 0.7]], dtype=np.float32)

        smoothed = apply_cluster_consensus(
            assignment,
            test_ids=("test_noise",),
            base_proba=base_proba,
            classes=classes,
            train_labels_by_id={},
        )

        np.testing.assert_array_equal(smoothed, base_proba)

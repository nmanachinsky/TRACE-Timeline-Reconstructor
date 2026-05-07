"""HDBSCAN-кластеризация для согласования предсказаний внутри 'событий'.

Идея: фотографии одного события (одной серии съёмки) должны иметь близкие
ResNet-эмбеддинги. После кластеризации test-фото получают предсказание,
усиленное мнением соседей по кластеру (включая train-фото с известными метками).

PCA снижает размерность до CLUSTERING.pca_components, чтобы уменьшить шум и
ускорить HDBSCAN на 4486×2048-матрице.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import hdbscan
import numpy as np
from sklearn.decomposition import PCA

from src.config import CLUSTERING, SEED


NOISE_LABEL = -1


@dataclass(frozen=True)
class ClusterAssignment:
    ids: tuple[str, ...]
    labels: np.ndarray  # shape (N,), -1 = noise

    @property
    def cluster_count(self) -> int:
        valid = self.labels[self.labels >= 0]
        return int(np.unique(valid).size) if valid.size else 0


def cluster_resnet_embeddings(
    ids: Sequence[str],
    vectors: np.ndarray,
    min_cluster_size: int = CLUSTERING.min_cluster_size,
    pca_components: int | None = CLUSTERING.pca_components,
) -> ClusterAssignment:
    """HDBSCAN на эмбеддингах (опционально с предварительным PCA)."""
    reduced = _reduce(vectors, pca_components)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean"
    )
    labels = clusterer.fit_predict(reduced).astype(np.int32)
    return ClusterAssignment(ids=tuple(ids), labels=labels)


def apply_cluster_consensus(
    assignment: ClusterAssignment,
    test_ids: Sequence[str],
    base_proba: np.ndarray,
    classes: np.ndarray,
    train_labels_by_id: dict[str, str],
    consensus_weight: float = 0.5,
) -> np.ndarray:
    """Сглаживает предсказания: blend(base_proba, голоса train-соседей по кластеру).

    Для каждого test-фото берётся cluster, в который оно попало. Если cluster —
    шум (-1), вероятности не меняются. Иначе берётся нормированный голос train-меток
    внутри кластера и смешивается: `(1-w) * base + w * cluster_vote`.
    """
    label_to_index = {lbl: i for i, lbl in enumerate(classes)}
    cluster_votes = _build_cluster_votes(
        assignment, train_labels_by_id, label_to_index, n_classes=len(classes)
    )
    id_to_cluster = dict(zip(assignment.ids, assignment.labels))

    smoothed = base_proba.copy()
    for row, sid in enumerate(test_ids):
        cluster = id_to_cluster.get(sid)
        if cluster is None or cluster == NOISE_LABEL:
            continue
        votes = cluster_votes.get(int(cluster))
        if votes is None or votes.sum() == 0:
            continue
        normalized = votes / votes.sum()
        smoothed[row] = (1.0 - consensus_weight) * base_proba[row] + consensus_weight * normalized
    return smoothed


def _build_cluster_votes(
    assignment: ClusterAssignment,
    train_labels_by_id: dict[str, str],
    label_to_index: dict[str, int],
    n_classes: int,
) -> dict[int, np.ndarray]:
    cluster_votes: dict[int, np.ndarray] = {}
    for sid, cluster in zip(assignment.ids, assignment.labels):
        if cluster == NOISE_LABEL or sid not in train_labels_by_id:
            continue
        idx = label_to_index.get(train_labels_by_id[sid])
        if idx is None:
            continue
        votes = cluster_votes.setdefault(int(cluster), np.zeros(n_classes, dtype=np.float32))
        votes[idx] += 1.0
    return cluster_votes


def _reduce(vectors: np.ndarray, pca_components: int | None) -> np.ndarray:
    if pca_components is None or pca_components >= vectors.shape[1]:
        return vectors.astype(np.float32)
    n = min(pca_components, max(1, min(vectors.shape) - 1))
    pca = PCA(n_components=n, random_state=SEED)
    return pca.fit_transform(vectors).astype(np.float32)

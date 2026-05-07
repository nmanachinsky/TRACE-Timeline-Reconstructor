"""Soft-voting ансамбль kNN + RandomForest + LogisticRegression.

- kNN с cosine-метрикой работает на ResNet-эмбеддингах (семантические соседи).
- RandomForest — на полном векторе признаков, переносит дисбаланс через class_weight.
- LogisticRegression — даёт калиброванные вероятности для voting.

Финальное предсказание — взвешенное среднее предсказанных вероятностей.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CLASSIFIER, RESNET, SEED


@dataclass(frozen=True)
class FeatureLayout:
    """Описывает, какие колонки полного вектора соответствуют ResNet-эмбеддингу."""

    resnet_start: int
    resnet_end: int  # exclusive

    def slice_resnet(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.resnet_start : self.resnet_end]


@dataclass
class TrainedClassifier:
    knn: KNeighborsClassifier
    rf: RandomForestClassifier
    lr: LogisticRegression
    scaler: StandardScaler
    label_encoder: LabelEncoder
    layout: FeatureLayout
    voting_weights: tuple[float, float, float]

    def predict_proba(self, x_full: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(x_full)
        resnet_part = self.layout.slice_resnet(x_full)

        proba_knn = self.knn.predict_proba(resnet_part)
        proba_rf = self.rf.predict_proba(scaled)
        proba_lr = self.lr.predict_proba(scaled)

        proba = (
            self.voting_weights[0] * proba_knn
            + self.voting_weights[1] * proba_rf
            + self.voting_weights[2] * proba_lr
        )
        return proba

    def predict_labels(self, x_full: np.ndarray) -> list[str]:
        proba = self.predict_proba(x_full)
        indices = proba.argmax(axis=1)
        return list(self.label_encoder.inverse_transform(indices))

    @property
    def classes_(self) -> np.ndarray:
        return self.label_encoder.classes_


def train_classifier(
    x_full: np.ndarray,
    y_labels: list[str],
    layout: FeatureLayout,
    voting_weights: tuple[float, float, float] = CLASSIFIER.voting_weights,
) -> TrainedClassifier:
    """Обучает все три модели на одном train-наборе. Возвращает готовый ансамбль."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_full)
    resnet_part = layout.slice_resnet(x_full)

    encoder = LabelEncoder().fit(y_labels)
    y_encoded = encoder.transform(y_labels)

    knn = KNeighborsClassifier(
        n_neighbors=min(CLASSIFIER.knn_neighbors, len(y_encoded)),
        weights="distance",
        metric="cosine",
    ).fit(resnet_part, y_encoded)

    rf = RandomForestClassifier(
        n_estimators=CLASSIFIER.rf_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    ).fit(x_scaled, y_encoded)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=CLASSIFIER.lr_max_iter,
        C=CLASSIFIER.lr_c,
        random_state=SEED,
    ).fit(x_scaled, y_encoded)

    return TrainedClassifier(
        knn=knn,
        rf=rf,
        lr=lr,
        scaler=scaler,
        label_encoder=encoder,
        layout=layout,
        voting_weights=voting_weights,
    )


def save_classifier(path: Path, model: TrainedClassifier) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_classifier(path: Path) -> TrainedClassifier:
    return joblib.load(path)


def default_resnet_layout(full_dim: int, color_dim: int, lighting_dim: int) -> FeatureLayout:
    """Layout: ResNet идёт первым в полном векторе, дальше color и lighting."""
    return FeatureLayout(resnet_start=0, resnet_end=RESNET.embedding_dim)

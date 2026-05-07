"""Общие утилиты для train/predict/evaluate: загрузка кэшей, выравнивание признаков."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureBundle:
    """Загруженные кэши признаков, согласованные по stripped_id."""

    ids: tuple[str, ...]
    resnet: np.ndarray  # (N, resnet_dim)
    color: np.ndarray   # (N, color_dim)
    lighting: np.ndarray  # (N, lighting_dim)

    @property
    def full_matrix(self) -> np.ndarray:
        """ResNet → color → lighting в одном векторе. Layout стабилен."""
        return np.hstack([self.resnet, self.color, self.lighting]).astype(np.float32)

    @property
    def resnet_dim(self) -> int:
        return self.resnet.shape[1]


def load_feature_bundle(features_dir: Path) -> FeatureBundle:
    """Загружает resnet/color/lighting npz и приводит к общему порядку id."""
    resnet_data = _load_cache(features_dir / "resnet.npz")
    color_data = _load_cache(features_dir / "color.npz")
    lighting_data = _load_cache(features_dir / "lighting.npz")

    common_ids = sorted(
        set(resnet_data[0]) & set(color_data[0]) & set(lighting_data[0])
    )
    if not common_ids:
        raise ValueError("Кэши признаков не пересекаются по stripped_id")

    resnet_aligned = _align(resnet_data, common_ids)
    color_aligned = _align(color_data, common_ids)
    lighting_aligned = _align(lighting_data, common_ids)

    return FeatureBundle(
        ids=tuple(common_ids),
        resnet=resnet_aligned,
        color=color_aligned,
        lighting=lighting_aligned,
    )


def _load_cache(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    ids = [str(s) for s in data["ids"].tolist()]
    vectors = data["vectors"].astype(np.float32)
    return ids, vectors


def _align(data: tuple[list[str], np.ndarray], ordered_ids: list[str]) -> np.ndarray:
    ids, vectors = data
    index = {sid: i for i, sid in enumerate(ids)}
    rows = [vectors[index[sid]] for sid in ordered_ids]
    return np.vstack(rows).astype(np.float32)


def load_ground_truth(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(splits_dir: Path) -> tuple[list[str], list[str]]:
    train = json.loads((splits_dir / "train_ids.json").read_text(encoding="utf-8"))
    test = json.loads((splits_dir / "test_ids.json").read_text(encoding="utf-8"))
    return list(train), list(test)


def labels_for(ids: list[str], gt: dict[str, dict]) -> list[str]:
    return [gt[i]["class_label"] for i in ids]

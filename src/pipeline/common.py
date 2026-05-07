"""Общие утилиты для train/predict/evaluate: загрузка кэшей, выравнивание признаков.

`FeatureBundle` поддерживает два режима:
- core (M1): resnet + color + lighting — обязательные кэши
- full (M2): core + faces + ocr — дополнительно, опционально через `include_faces=True`

Layout полного вектора стабилен: [resnet | color | lighting | faces | ocr].
ResNet всегда первым — это инвариант для `FeatureLayout.slice_resnet`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureBundle:
    """Загруженные кэши признаков, согласованные по stripped_id.

    `faces` и `ocr` опциональны (None → не используются в `full_matrix`).
    """

    ids: tuple[str, ...]
    resnet: np.ndarray
    color: np.ndarray
    lighting: np.ndarray
    faces: np.ndarray | None = None
    ocr: np.ndarray | None = None

    @property
    def full_matrix(self) -> np.ndarray:
        """Конкатенация в стабильном порядке. M2-признаки добавляются в хвост."""
        parts: list[np.ndarray] = [self.resnet, self.color, self.lighting]
        if self.faces is not None:
            parts.append(self.faces)
        if self.ocr is not None:
            parts.append(self.ocr)
        return np.hstack(parts).astype(np.float32)

    @property
    def resnet_dim(self) -> int:
        return self.resnet.shape[1]

    @property
    def has_m2_features(self) -> bool:
        return self.faces is not None or self.ocr is not None


def load_feature_bundle(
    features_dir: Path,
    *,
    include_faces: bool = False,
    include_ocr: bool = False,
) -> FeatureBundle:
    """Загружает обязательные core-кэши и опционально M2-кэши.

    Все массивы выравниваются по пересечению id → так гарантируется, что строки в
    `full_matrix` и в `ids` соответствуют одному и тому же фото.
    """
    resnet_data = _load_cache(features_dir / "resnet.npz")
    color_data = _load_cache(features_dir / "color.npz")
    lighting_data = _load_cache(features_dir / "lighting.npz")

    id_sets: list[set[str]] = [set(resnet_data[0]), set(color_data[0]), set(lighting_data[0])]

    faces_data: tuple[list[str], np.ndarray] | None = None
    if include_faces:
        faces_data = _load_cache(features_dir / "faces.npz")
        id_sets.append(set(faces_data[0]))

    ocr_data: tuple[list[str], np.ndarray] | None = None
    if include_ocr:
        ocr_data = _load_cache(features_dir / "ocr.npz")
        id_sets.append(set(ocr_data[0]))

    common_ids = sorted(set.intersection(*id_sets))
    if not common_ids:
        raise ValueError("Кэши признаков не пересекаются по stripped_id")

    return FeatureBundle(
        ids=tuple(common_ids),
        resnet=_align(resnet_data, common_ids),
        color=_align(color_data, common_ids),
        lighting=_align(lighting_data, common_ids),
        faces=_align(faces_data, common_ids) if faces_data is not None else None,
        ocr=_align(ocr_data, common_ids) if ocr_data is not None else None,
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

"""ResNet-50 эмбеддинги через torchvision.

Используется pretrained ResNet-50 (ImageNet). Снимаем последний `fc`-слой и берём
выход GlobalAvgPool после layer4 → 2048-мерный вектор. Препроцессинг — стандартный
от torchvision.models.ResNet50_Weights.DEFAULT.

Кэширование: эмбеддинги сохраняются в .npz по stripped_id, чтобы пересборка
была дешёвой.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from src.config import RESNET


@dataclass(frozen=True)
class ResNetEmbeddings:
    ids: tuple[str, ...]
    vectors: np.ndarray  # shape (N, embedding_dim)

    def to_dict(self) -> dict[str, np.ndarray]:
        return {sid: self.vectors[i] for i, sid in enumerate(self.ids)}


class ResNetFeatureExtractor:
    """Тонкая обёртка над torchvision-ResNet, возвращает 2048-d вектор на изображение."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()
        backbone.eval()
        self._device = torch.device(device) if device else self._auto_device()
        self._model = backbone.to(self._device)
        self._preprocess = weights.transforms()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def embedding_dim(self) -> int:
        return RESNET.embedding_dim

    def encode_paths(
        self, image_paths: Sequence[Path], batch_size: int = RESNET.batch_size
    ) -> np.ndarray:
        """Возвращает (N, embedding_dim) тензор эмбеддингов для списка путей."""
        if not image_paths:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for batch in _batched(image_paths, batch_size):
            tensors = torch.stack([self._load_and_preprocess(p) for p in batch])
            tensors = tensors.to(self._device)
            with torch.inference_mode():
                features = self._model(tensors)
            chunks.append(features.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            return self._preprocess(rgb)

    @staticmethod
    def _auto_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_resnet_cache(path: Path, embeddings: ResNetEmbeddings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(embeddings.ids), vectors=embeddings.vectors)


def load_resnet_cache(path: Path) -> ResNetEmbeddings:
    data = np.load(path, allow_pickle=False)
    ids = tuple(str(s) for s in data["ids"].tolist())
    vectors = data["vectors"].astype(np.float32)
    return ResNetEmbeddings(ids=ids, vectors=vectors)


def _batched(seq: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

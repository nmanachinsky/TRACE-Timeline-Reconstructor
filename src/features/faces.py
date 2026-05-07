"""Face-эмбеддинги через insightface (RetinaFace детекция + ArcFace эмбеддинги).

Логика разделена надвое: чистая агрегация эмбеддингов (быстро тестируется без ONNX)
и тонкая обёртка `FaceFeatureExtractor` над `insightface.app.FaceAnalysis` с
ленивой инициализацией модели.

Выходной вектор имеет размерность `FACE_FEATURE_DIM` = эмбеддинг (512) + флаг
`has_face` (1.0 / 0.0). Когда лиц на фото не обнаружено — нулевой эмбеддинг и
`has_face=0`, чтобы классификатор мог отдельно обработать "пейзажный" случай.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

FACE_EMBEDDING_DIM: int = 512
FACE_FEATURE_DIM: int = FACE_EMBEDDING_DIM + 1  # +1 — флаг has_face

_DEFAULT_MODEL_NAME = "buffalo_s"  # лёгкая модель: RetinaFace-Mb + ArcFace MBF
_DEFAULT_DET_SIZE: tuple[int, int] = (640, 640)


@dataclass(frozen=True)
class FaceFeatures:
    """Агрегированный по фото face-признак."""

    embedding: np.ndarray
    face_count: int
    has_face: float

    def to_vector(self) -> np.ndarray:
        """Полный вектор для конкатенации с другими признаками."""
        return np.concatenate([self.embedding, [self.has_face]]).astype(np.float32)


def aggregate_face_embeddings(face_embeddings: Iterable[np.ndarray]) -> FaceFeatures:
    """Среднее по эмбеддингам лиц на одном фото; пустое — нулевой вектор + has_face=0.

    Why: insightface возвращает по эмбеддингу на каждое детектированное лицо, а
    нашему классификатору нужен один фиксированный вектор на изображение.
    """
    embs = list(face_embeddings)
    if not embs:
        return FaceFeatures(
            embedding=np.zeros(FACE_EMBEDDING_DIM, dtype=np.float32),
            face_count=0,
            has_face=0.0,
        )

    stacked = np.vstack(embs).astype(np.float32)
    if stacked.shape[1] != FACE_EMBEDDING_DIM:
        raise ValueError(
            f"Ожидался эмбеддинг размерности {FACE_EMBEDDING_DIM}, получено {stacked.shape[1]}"
        )

    return FaceFeatures(
        embedding=stacked.mean(axis=0).astype(np.float32),
        face_count=len(embs),
        has_face=1.0,
    )


class FaceFeatureExtractor:
    """Тонкая обёртка над `insightface.app.FaceAnalysis` с ленивой инициализацией.

    Загрузка ONNX-моделей дорогая (~2-5 сек), поэтому модель грузится только при
    первом вызове `extract`. Это позволяет дешёво создавать экземпляр в тестах и
    мокать `_ensure_app` без затрат на сеть/диск.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        det_size: tuple[int, int] = _DEFAULT_DET_SIZE,
    ) -> None:
        self._model_name = model_name
        self._providers = providers
        self._det_size = det_size
        self._app: object | None = None

    def _ensure_app(self) -> object:
        if self._app is not None:
            return self._app
        from insightface.app import FaceAnalysis  # локальный импорт — тяжёлый ONNX-стек

        app = FaceAnalysis(name=self._model_name, providers=list(self._providers))
        ctx_id = 0 if "CUDAExecutionProvider" in self._providers else -1
        app.prepare(ctx_id=ctx_id, det_size=self._det_size)
        self._app = app
        return app

    def extract(self, bgr_image: np.ndarray) -> FaceFeatures:
        """Детектирует лица и возвращает агрегированный признак."""
        app = self._ensure_app()
        faces = app.get(bgr_image)  # type: ignore[attr-defined]
        embeddings = [
            face.normed_embedding
            for face in faces
            if hasattr(face, "normed_embedding")
        ]
        return aggregate_face_embeddings(embeddings)

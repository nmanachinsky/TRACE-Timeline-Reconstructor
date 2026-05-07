"""Извлечение признаков для произвольного списка изображений.

В отличие от `src.features.extractor` (CLI-оркестратор поверх ground_truth.json),
эта функция принимает on-the-fly список `(sample_id, path)` — что нужно для
Wizard-режима, где reference и target разные папки и общего GT нет.

Кэширование: артефакты сохраняются в `features_dir/{resnet,color,lighting,faces,ocr}.npz`.
При повторном вызове на той же папке кэши переиспользуются (быстро).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import RESNET
from src.features.color import build_color_feature_vector
from src.features.lighting import build_lighting_feature_vector
from src.features.resnet import (
    ResNetEmbeddings,
    ResNetFeatureExtractor,
    save_resnet_cache,
)
from src.pipeline.common import FeatureBundle, load_feature_bundle

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class FeatureItem:
    """Элемент входа: пара (id, путь к изображению)."""

    sample_id: str
    path: Path


def extract_features_for_items(
    items: list[FeatureItem],
    features_dir: Path,
    *,
    use_m2: bool = False,
    batch_size: int = RESNET.batch_size,
    gpu: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> FeatureBundle:
    """Извлекает признаки для списка элементов и возвращает FeatureBundle.

    Если кэш уже существует и совпадает по id — этап пропускается. Это позволяет
    в Wizard-режиме мгновенно перейти к следующим шагам после первого прогона.
    """
    if not items:
        raise ValueError("Список items пуст — нечего извлекать")

    features_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = [it.sample_id for it in items]

    _ensure_resnet_cache(items, features_dir / "resnet.npz", batch_size, expected_ids, progress_cb)
    _report(progress_cb, 0.45, "ResNet готов")

    _ensure_color_lighting_cache(
        items,
        color_path=features_dir / "color.npz",
        lighting_path=features_dir / "lighting.npz",
        expected_ids=expected_ids,
        progress_cb=progress_cb,
    )
    _report(progress_cb, 0.7, "Цвет и освещение готовы")

    if use_m2:
        _ensure_faces_cache(items, features_dir / "faces.npz", expected_ids, gpu, progress_cb)
        _report(progress_cb, 0.85, "Лица готовы")
        _ensure_ocr_cache(items, features_dir / "ocr.npz", expected_ids, gpu, progress_cb)
        _report(progress_cb, 0.95, "OCR готов")

    bundle = load_feature_bundle(features_dir, include_faces=use_m2, include_ocr=use_m2)
    _report(progress_cb, 1.0, "Признаки загружены")
    return bundle


# --- ResNet -------------------------------------------------------------------


def _ensure_resnet_cache(
    items: list[FeatureItem],
    cache_path: Path,
    batch_size: int,
    expected_ids: list[str],
    progress_cb: ProgressCallback | None,
) -> None:
    if _cache_matches(cache_path, expected_ids):
        return
    _report(progress_cb, 0.05, "Загрузка ResNet-50")
    extractor = ResNetFeatureExtractor()
    paths = [it.path for it in items]
    ids = [it.sample_id for it in items]
    _report(progress_cb, 0.15, f"Эмбеддинги ResNet ({len(paths)} фото)")
    vectors = extractor.encode_paths(paths, batch_size=batch_size)
    save_resnet_cache(cache_path, ResNetEmbeddings(ids=tuple(ids), vectors=vectors))


# --- Color + lighting ---------------------------------------------------------


def _ensure_color_lighting_cache(
    items: list[FeatureItem],
    *,
    color_path: Path,
    lighting_path: Path,
    expected_ids: list[str],
    progress_cb: ProgressCallback | None,
) -> None:
    if _cache_matches(color_path, expected_ids) and _cache_matches(lighting_path, expected_ids):
        return
    ids: list[str] = []
    color_vectors: list[np.ndarray] = []
    lighting_vectors: list[np.ndarray] = []

    total = len(items)
    progress_step = max(1, total // 20)
    for i, item in enumerate(items):
        bgr = _read_bgr(item.path)
        if bgr is None:
            continue
        ids.append(item.sample_id)
        color_vectors.append(build_color_feature_vector(bgr))
        lighting_vectors.append(build_lighting_feature_vector(bgr))
        if (i + 1) % progress_step == 0:
            _report(progress_cb, 0.5 + 0.2 * ((i + 1) / total), f"Цвет/свет {i + 1}/{total}")

    if not ids:
        raise RuntimeError("Не удалось извлечь color/lighting ни для одного файла")
    _save_npz(color_path, ids, np.vstack(color_vectors))
    _save_npz(lighting_path, ids, np.vstack(lighting_vectors))


# --- Faces (M2) ---------------------------------------------------------------


def _ensure_faces_cache(
    items: list[FeatureItem],
    cache_path: Path,
    expected_ids: list[str],
    gpu: bool,
    progress_cb: ProgressCallback | None,
) -> None:
    if _cache_matches(cache_path, expected_ids):
        return
    from src.features.faces import FaceFeatureExtractor

    providers = (
        ("CUDAExecutionProvider", "CPUExecutionProvider")
        if gpu
        else ("CPUExecutionProvider",)
    )
    extractor = FaceFeatureExtractor(providers=providers)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    total = len(items)
    progress_step = max(1, total // 20)
    for i, item in enumerate(items):
        bgr = _read_bgr(item.path)
        if bgr is None:
            continue
        feats = extractor.extract(bgr)
        ids.append(item.sample_id)
        vectors.append(feats.to_vector())
        if (i + 1) % progress_step == 0:
            _report(progress_cb, 0.75 + 0.1 * ((i + 1) / total), f"Лица {i + 1}/{total}")

    _save_npz(cache_path, ids, np.vstack(vectors))


# --- OCR (M2) -----------------------------------------------------------------


def _ensure_ocr_cache(
    items: list[FeatureItem],
    cache_path: Path,
    expected_ids: list[str],
    gpu: bool,
    progress_cb: ProgressCallback | None,
) -> None:
    if _cache_matches(cache_path, expected_ids):
        return
    from src.features.ocr import OcrFeatureExtractor

    extractor = OcrFeatureExtractor(gpu=gpu)
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    total = len(items)
    progress_step = max(1, total // 20)
    for i, item in enumerate(items):
        bgr = _read_bgr(item.path)
        if bgr is None:
            continue
        feats = extractor.extract(bgr)
        ids.append(item.sample_id)
        vectors.append(feats.to_vector())
        if (i + 1) % progress_step == 0:
            _report(progress_cb, 0.88 + 0.07 * ((i + 1) / total), f"OCR {i + 1}/{total}")

    _save_npz(cache_path, ids, np.vstack(vectors))


# --- IO helpers ---------------------------------------------------------------


def _read_bgr(path: Path) -> np.ndarray | None:
    """Чтение через np.fromfile + cv2.imdecode для unicode-путей (Windows)."""
    import cv2

    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _cache_matches(cache_path: Path, expected_ids: list[str]) -> bool:
    """True, если в .npz уже есть строго те же id (порядок не важен)."""
    if not cache_path.exists():
        return False
    try:
        data = np.load(cache_path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    cached = {str(s) for s in data["ids"].tolist()}
    return cached == set(expected_ids)


def _save_npz(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(ids), vectors=vectors.astype(np.float32))


def _report(cb: ProgressCallback | None, fraction: float, message: str) -> None:
    if cb is None:
        return
    cb(max(0.0, min(1.0, fraction)), message)

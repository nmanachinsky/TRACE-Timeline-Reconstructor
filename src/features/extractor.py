"""CLI-оркестратор извлечения признаков.

Запуск: `uv run python -m src.features.extractor --stage=core`.

Стадии:
- core: ResNet-50 эмбеддинги, цвет, свет.
- full: core + (M2: faces, OCR — реализуется в фазе M2).

Артефакты сохраняются в data/features/{resnet,color,lighting}.npz, чтобы train
и predict могли быстро их подгружать без повторного инференса.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.config import FEATURES_DIR, GROUND_TRUTH_PATH, RESNET
from src.features.color import build_color_feature_vector
from src.features.lighting import build_lighting_feature_vector
from src.features.resnet import (
    ResNetEmbeddings,
    ResNetFeatureExtractor,
    save_resnet_cache,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()

    gt = _load_ground_truth(Path(args.ground_truth))
    items = sorted(gt.items())
    print(f"[features] {len(items)} изображений из ground_truth.json")

    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("core", "full"):
        _run_resnet(items, features_dir / "resnet.npz", args.batch_size)
        _run_color_and_lighting(
            items, features_dir / "color.npz", features_dir / "lighting.npz"
        )

    if args.stage == "full":
        print("[features] M2 признаки (faces, OCR) ещё не реализованы")

    print(f"[features] готово за {time.time() - started:.1f}s")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Извлечение визуальных признаков")
    parser.add_argument("--stage", choices=("core", "full"), default="core")
    parser.add_argument("--ground-truth", default=str(GROUND_TRUTH_PATH))
    parser.add_argument("--features-dir", default=str(FEATURES_DIR))
    parser.add_argument("--batch-size", type=int, default=RESNET.batch_size)
    return parser.parse_args(argv)


def _load_ground_truth(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_resnet(
    items: list[tuple[str, dict]], cache_path: Path, batch_size: int
) -> None:
    print(f"[features] ResNet-50: {cache_path}")
    extractor = ResNetFeatureExtractor()
    print(f"[features] device={extractor.device}")
    paths = [Path(meta["stripped_path"]) for _, meta in items]
    ids = [sid for sid, _ in items]

    started = time.time()
    vectors = extractor.encode_paths(paths, batch_size=batch_size)
    elapsed = time.time() - started
    print(f"[features] ResNet inference: {elapsed:.1f}s ({len(paths) / max(elapsed, 1e-6):.1f} img/s)")

    save_resnet_cache(cache_path, ResNetEmbeddings(ids=tuple(ids), vectors=vectors))


def _run_color_and_lighting(
    items: list[tuple[str, dict]], color_path: Path, lighting_path: Path
) -> None:
    """Читает каждый файл один раз и считает оба признака на одном BGR-массиве."""
    import cv2  # локальный импорт, чтобы извлечение ResNet не платило за загрузку cv2

    print(f"[features] color+lighting (single read): {color_path}, {lighting_path}")
    ids: list[str] = []
    color_vectors: list[np.ndarray] = []
    lighting_vectors: list[np.ndarray] = []

    progress_step = max(1, len(items) // 20)
    started = time.time()
    for i, (sid, meta) in enumerate(items):
        path = meta["stripped_path"]
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            print(f"[features] WARN: пустой файл {path}, пропускаю")
            continue
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[features] WARN: не декодируется {path}, пропускаю")
            continue
        ids.append(sid)
        color_vectors.append(build_color_feature_vector(bgr))
        lighting_vectors.append(build_lighting_feature_vector(bgr))
        if (i + 1) % progress_step == 0:
            elapsed = time.time() - started
            rate = (i + 1) / max(elapsed, 1e-6)
            print(f"[features] color+lighting: {i + 1}/{len(items)} ({rate:.0f} img/s)")

    _save_npz(color_path, ids, np.vstack(color_vectors))
    _save_npz(lighting_path, ids, np.vstack(lighting_vectors))


def _save_npz(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(ids), vectors=vectors.astype(np.float32))


if __name__ == "__main__":
    sys.exit(main())

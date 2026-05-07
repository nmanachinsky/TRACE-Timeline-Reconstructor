"""CLI-оркестратор извлечения признаков.

Запуск: `uv run python -m src.features.extractor --stage=core|full`.

Стадии:
- core: ResNet-50 эмбеддинги, цвет, свет.
- full: core + faces (insightface) + OCR (EasyOCR).

Артефакты сохраняются в data/features/{resnet,color,lighting,faces,ocr}.npz,
чтобы train и predict могли быстро их подгружать без повторного инференса.
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

# Faces / OCR импортируются лениво только в --stage=full, чтобы core-прогон не
# платил за тяжёлый ONNX-стек и easyocr.


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()

    gt = _load_ground_truth(Path(args.ground_truth))
    items = sorted(gt.items())
    print(f"[features] {len(items)} изображений из ground_truth.json")

    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("core", "full"):
        if not args.skip_resnet:
            _run_resnet(items, features_dir / "resnet.npz", args.batch_size)
        if not args.skip_color_lighting:
            _run_color_and_lighting(
                items, features_dir / "color.npz", features_dir / "lighting.npz"
            )

    if args.stage == "full":
        if not args.skip_faces:
            _run_faces(items, features_dir / "faces.npz", gpu=args.gpu)
        if not args.skip_ocr:
            _run_ocr(items, features_dir / "ocr.npz", gpu=args.gpu)

    print(f"[features] готово за {time.time() - started:.1f}s")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Извлечение визуальных признаков")
    parser.add_argument("--stage", choices=("core", "full"), default="core")
    parser.add_argument("--ground-truth", default=str(GROUND_TRUTH_PATH))
    parser.add_argument("--features-dir", default=str(FEATURES_DIR))
    parser.add_argument("--batch-size", type=int, default=RESNET.batch_size)
    parser.add_argument("--gpu", action="store_true", help="ONNXRuntime GPU для faces (если установлен onnxruntime-gpu)")
    parser.add_argument("--skip-resnet", action="store_true", help="Пропустить ResNet (если есть кэш)")
    parser.add_argument("--skip-color-lighting", action="store_true", help="Пропустить color+lighting (если есть кэш)")
    parser.add_argument("--skip-faces", action="store_true", help="Пропустить faces в --stage=full")
    parser.add_argument("--skip-ocr", action="store_true", help="Пропустить OCR в --stage=full")
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
    print(f"[features] color+lighting (single read): {color_path}, {lighting_path}")
    ids: list[str] = []
    color_vectors: list[np.ndarray] = []
    lighting_vectors: list[np.ndarray] = []

    progress_step = max(1, len(items) // 20)
    started = time.time()
    for i, (sid, meta) in enumerate(items):
        bgr = _read_bgr(meta["stripped_path"])
        if bgr is None:
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


def _run_faces(items: list[tuple[str, dict]], cache_path: Path, gpu: bool) -> None:
    """Извлечение face-эмбеддингов через insightface (RetinaFace + ArcFace)."""
    import cv2

    from src.features.faces import FaceFeatureExtractor

    providers = (
        ("CUDAExecutionProvider", "CPUExecutionProvider") if gpu else ("CPUExecutionProvider",)
    )
    print(f"[features] faces (insightface): {cache_path}, providers={providers}")
    extractor = FaceFeatureExtractor(providers=providers)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    progress_step = max(1, len(items) // 20)
    started = time.time()
    face_count_total = 0

    for i, (sid, meta) in enumerate(items):
        path = meta["stripped_path"]
        bgr = _read_bgr(path)
        if bgr is None:
            continue
        feats = extractor.extract(bgr)
        face_count_total += feats.face_count
        ids.append(sid)
        vectors.append(feats.to_vector())
        if (i + 1) % progress_step == 0:
            elapsed = time.time() - started
            rate = (i + 1) / max(elapsed, 1e-6)
            print(
                f"[features] faces: {i + 1}/{len(items)} ({rate:.1f} img/s), "
                f"всего лиц: {face_count_total}"
            )

    _save_npz(cache_path, ids, np.vstack(vectors))
    print(f"[features] faces готово: всего лиц найдено {face_count_total}")


def _run_ocr(items: list[tuple[str, dict]], cache_path: Path, gpu: bool) -> None:
    """OCR через EasyOCR с регулярным извлечением годов."""
    from src.features.ocr import OcrFeatureExtractor

    print(f"[features] OCR (EasyOCR): {cache_path}, gpu={gpu}")
    extractor = OcrFeatureExtractor(gpu=gpu)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    progress_step = max(1, len(items) // 20)
    started = time.time()
    has_text_count = 0
    has_year_count = 0

    for i, (sid, meta) in enumerate(items):
        path = meta["stripped_path"]
        bgr = _read_bgr(path)
        if bgr is None:
            continue
        feats = extractor.extract(bgr)
        if feats.has_text > 0:
            has_text_count += 1
        if feats.detected_years:
            has_year_count += 1
        ids.append(sid)
        vectors.append(feats.to_vector())
        if (i + 1) % progress_step == 0:
            elapsed = time.time() - started
            rate = (i + 1) / max(elapsed, 1e-6)
            print(
                f"[features] OCR: {i + 1}/{len(items)} ({rate:.2f} img/s), "
                f"текст={has_text_count}, год={has_year_count}"
            )

    _save_npz(cache_path, ids, np.vstack(vectors))
    print(
        f"[features] OCR готово: фото с текстом {has_text_count}, "
        f"с распознанным годом {has_year_count}"
    )


def _read_bgr(path: str) -> np.ndarray | None:
    """Чтение изображения с поддержкой не-ASCII путей (Windows + кириллица)."""
    import cv2

    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        print(f"[features] WARN: пустой файл {path}, пропускаю")
        return None
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[features] WARN: не декодируется {path}, пропускаю")
        return None
    return bgr


def _save_npz(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(ids), vectors=vectors.astype(np.float32))


if __name__ == "__main__":
    sys.exit(main())

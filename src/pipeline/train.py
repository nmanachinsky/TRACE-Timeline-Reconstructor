"""CLI обучения: загружает кэши признаков, тренирует ансамбль, сохраняет модель."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import FEATURES_DIR, GROUND_TRUTH_PATH, MODELS_DIR, SPLITS_DIR
from src.models.classifier import (
    FeatureLayout,
    save_classifier,
    train_classifier,
)
from src.pipeline.common import (
    labels_for,
    load_feature_bundle,
    load_ground_truth,
    load_split,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()

    bundle = load_feature_bundle(Path(args.features_dir))
    gt = load_ground_truth(Path(args.ground_truth))
    train_ids, _ = load_split(Path(args.splits_dir))

    print(f"[train] всего фото в кэшах: {len(bundle.ids)}")
    print(f"[train] train ids: {len(train_ids)}")

    train_set = set(train_ids)
    train_indices = [i for i, sid in enumerate(bundle.ids) if sid in train_set]
    if not train_indices:
        print("[train] ОШИБКА: нет пересечения train_ids с кэшем признаков")
        return 1

    x_full = bundle.full_matrix[train_indices]
    train_id_order = [bundle.ids[i] for i in train_indices]
    y_labels = labels_for(train_id_order, gt)
    print(f"[train] X.shape={x_full.shape}, классов={len(set(y_labels))}")

    layout = FeatureLayout(resnet_start=0, resnet_end=bundle.resnet_dim)
    classifier = train_classifier(x_full, y_labels, layout)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_classifier(out_dir / "classifier.joblib", classifier)
    print(f"[train] сохранено в {out_dir / 'classifier.joblib'}")
    print(f"[train] готово за {time.time() - started:.1f}s")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение ансамбля")
    parser.add_argument("--features-dir", default=str(FEATURES_DIR))
    parser.add_argument("--ground-truth", default=str(GROUND_TRUTH_PATH))
    parser.add_argument("--splits-dir", default=str(SPLITS_DIR))
    parser.add_argument("--features", choices=("core", "full"), default="core")
    parser.add_argument("--out", default=str(MODELS_DIR / "m1"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

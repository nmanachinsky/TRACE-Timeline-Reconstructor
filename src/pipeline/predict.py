"""CLI предсказания: загружает модель, гонит test, применяет cluster-consensus."""

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
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    DATA_DIR,
    FEATURES_DIR,
    GROUND_TRUTH_PATH,
    MODELS_DIR,
    SPLITS_DIR,
)
from src.models.classifier import load_classifier
from src.models.clustering import (
    apply_cluster_consensus,
    cluster_resnet_embeddings,
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
    train_ids, test_ids = load_split(Path(args.splits_dir))

    classifier = load_classifier(Path(args.models) / "classifier.joblib")
    print(f"[predict] классов в модели: {len(classifier.classes_)}")

    test_set = set(test_ids)
    test_indices = [i for i, sid in enumerate(bundle.ids) if sid in test_set]
    test_id_order = [bundle.ids[i] for i in test_indices]
    print(f"[predict] test фото: {len(test_id_order)}")

    full_matrix = bundle.full_matrix
    test_x = full_matrix[test_indices]
    base_proba = classifier.predict_proba(test_x)
    print("[predict] базовые предсказания готовы")

    if args.consensus:
        assignment = cluster_resnet_embeddings(bundle.ids, bundle.resnet)
        print(f"[predict] кластеров найдено: {assignment.cluster_count}")
        train_labels_by_id = dict(zip(train_ids, labels_for(train_ids, gt)))
        smoothed = apply_cluster_consensus(
            assignment,
            test_ids=test_id_order,
            base_proba=base_proba,
            classes=classifier.classes_,
            train_labels_by_id=train_labels_by_id,
            consensus_weight=args.consensus_weight,
        )
        final_proba = smoothed
    else:
        final_proba = base_proba

    predicted_indices = final_proba.argmax(axis=1)
    predicted_labels = list(classifier.classes_[predicted_indices])

    payload = _build_payload(test_id_order, gt, predicted_labels, final_proba, classifier.classes_)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[predict] сохранено в {out_path}")
    print(f"[predict] готово за {time.time() - started:.1f}s")
    return 0


def _build_payload(
    test_ids: list[str],
    gt: dict[str, dict],
    predicted_labels: list[str],
    proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, dict]:
    payload: dict[str, dict] = {}
    for i, sid in enumerate(test_ids):
        true_meta = gt[sid]
        topk_idx = np.argsort(-proba[i])[:3]
        payload[sid] = {
            "predicted_label": predicted_labels[i],
            "predicted_year": int(predicted_labels[i].split("-")[0]),
            "predicted_season": predicted_labels[i].split("-")[1],
            "true_label": true_meta["class_label"],
            "true_year": true_meta["year"],
            "true_season": true_meta["season"],
            "true_timestamp": true_meta["timestamp"],
            "source": true_meta["source"],
            "stripped_path": true_meta["stripped_path"],
            "top3": [
                {"label": str(classes[idx]), "proba": float(proba[i, idx])}
                for idx in topk_idx
            ],
        }
    return payload


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Предсказание + cluster consensus")
    parser.add_argument("--features-dir", default=str(FEATURES_DIR))
    parser.add_argument("--ground-truth", default=str(GROUND_TRUTH_PATH))
    parser.add_argument("--splits-dir", default=str(SPLITS_DIR))
    parser.add_argument("--models", default=str(MODELS_DIR / "m1"))
    parser.add_argument("--out", default=str(DATA_DIR / "predictions_m1.json"))
    parser.add_argument("--consensus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--consensus-weight", type=float, default=0.4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

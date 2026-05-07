"""Интеграционный smoke-тест: train → predict → evaluate на синтетических признаках.

Не использует реальную ResNet-модель — записывает фиктивные .npz прямо.
Это страхует, что CLI-обёртки и common-утилиты работоспособны.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline import evaluate as evaluate_module
from src.pipeline import predict as predict_module
from src.pipeline import train as train_module


def _build_synthetic_features(
    tmp_path: Path,
    n_per_class: int = 25,
    n_classes: int = 3,
    *,
    include_m2: bool = False,
) -> tuple[Path, Path, Path]:
    """Кладёт synthetic resnet/color/lighting (+опционально faces/ocr) npz и ground_truth.json + splits."""
    rng = np.random.default_rng(0)
    features_dir = tmp_path / "features"
    splits_dir = tmp_path / "splits"
    gt_path = tmp_path / "ground_truth.json"
    features_dir.mkdir()
    splits_dir.mkdir()

    seasons = ["winter", "spring", "summer"]
    years = [2022, 2023, 2024]

    ids: list[str] = []
    resnet_rows = []
    color_rows = []
    lighting_rows = []
    gt: dict[str, dict] = {}

    for ci in range(n_classes):
        center_resnet = rng.normal(loc=ci * 5.0, scale=0.4, size=(n_per_class, 2048)).astype(np.float32)
        center_color = rng.normal(loc=ci * 0.3, scale=0.05, size=(n_per_class, 514)).astype(np.float32)
        center_light = rng.normal(loc=ci * 0.1, scale=0.03, size=(n_per_class, 5)).astype(np.float32)
        for k in range(n_per_class):
            sid = f"id_{ci}_{k:03d}"
            ids.append(sid)
            resnet_rows.append(center_resnet[k])
            color_rows.append(center_color[k])
            lighting_rows.append(center_light[k])
            gt[sid] = {
                "stripped_path": f"/dummy/{sid}.jpg",
                "timestamp": f"{years[ci]}-{(ci + 1) * 2:02d}-15T12:00:00",
                "year": years[ci],
                "season": seasons[ci],
                "class_label": f"{years[ci]}-{seasons[ci]}",
                "source": "json_sidecar",
                "originals": [],
            }

    np.savez_compressed(features_dir / "resnet.npz", ids=np.asarray(ids), vectors=np.vstack(resnet_rows))
    np.savez_compressed(features_dir / "color.npz", ids=np.asarray(ids), vectors=np.vstack(color_rows))
    np.savez_compressed(features_dir / "lighting.npz", ids=np.asarray(ids), vectors=np.vstack(lighting_rows))

    if include_m2:
        # Синтетические M2-кэши: faces (513) и ocr (27). Class-specific сдвиги — чтобы
        # модель могла научиться на этих признаках при разделимых классах.
        faces_rows: list[np.ndarray] = []
        ocr_rows: list[np.ndarray] = []
        for ci in range(n_classes):
            face_center = rng.normal(loc=ci * 0.4, scale=0.05, size=(n_per_class, 513)).astype(np.float32)
            ocr_center = rng.normal(loc=ci * 0.2, scale=0.03, size=(n_per_class, 27)).astype(np.float32)
            for k in range(n_per_class):
                faces_rows.append(face_center[k])
                ocr_rows.append(ocr_center[k])
        np.savez_compressed(features_dir / "faces.npz", ids=np.asarray(ids), vectors=np.vstack(faces_rows))
        np.savez_compressed(features_dir / "ocr.npz", ids=np.asarray(ids), vectors=np.vstack(ocr_rows))

    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")

    train_ids = [sid for sid in ids if int(sid.split("_")[2]) < 18]
    test_ids = [sid for sid in ids if int(sid.split("_")[2]) >= 18]
    (splits_dir / "train_ids.json").write_text(json.dumps(train_ids), encoding="utf-8")
    (splits_dir / "test_ids.json").write_text(json.dumps(test_ids), encoding="utf-8")

    return features_dir, splits_dir, gt_path


@pytest.mark.integration
def test_full_pipeline_chain(tmp_path: Path) -> None:
    features_dir, splits_dir, gt_path = _build_synthetic_features(tmp_path)
    models_dir = tmp_path / "models"
    pred_path = tmp_path / "predictions.json"
    metrics_path = tmp_path / "metrics.json"

    train_exit = train_module.main([
        "--features-dir", str(features_dir),
        "--ground-truth", str(gt_path),
        "--splits-dir", str(splits_dir),
        "--out", str(models_dir),
    ])
    assert train_exit == 0
    assert (models_dir / "classifier.joblib").exists()

    predict_exit = predict_module.main([
        "--features-dir", str(features_dir),
        "--ground-truth", str(gt_path),
        "--splits-dir", str(splits_dir),
        "--models", str(models_dir),
        "--out", str(pred_path),
        "--no-consensus",
    ])
    assert predict_exit == 0
    assert pred_path.exists()
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    assert len(payload) > 0

    eval_exit = evaluate_module.main([
        "--predictions", str(pred_path),
        "--out", str(metrics_path),
    ])
    assert eval_exit == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "accuracy_year_season" in metrics
    # На разделимых данных accuracy должен быть высоким
    assert metrics["accuracy_year_season"] > 0.8


@pytest.mark.integration
def test_full_pipeline_chain_with_m2_features(tmp_path: Path) -> None:
    """train --features=full → predict --features=full → evaluate с faces+ocr кэшами."""
    features_dir, splits_dir, gt_path = _build_synthetic_features(tmp_path, include_m2=True)
    models_dir = tmp_path / "models"
    pred_path = tmp_path / "predictions.json"
    metrics_path = tmp_path / "metrics.json"

    train_exit = train_module.main([
        "--features-dir", str(features_dir),
        "--ground-truth", str(gt_path),
        "--splits-dir", str(splits_dir),
        "--features", "full",
        "--out", str(models_dir),
    ])
    assert train_exit == 0
    assert (models_dir / "classifier.joblib").exists()

    predict_exit = predict_module.main([
        "--features-dir", str(features_dir),
        "--ground-truth", str(gt_path),
        "--splits-dir", str(splits_dir),
        "--models", str(models_dir),
        "--out", str(pred_path),
        "--features", "full",
        "--no-consensus",
    ])
    assert predict_exit == 0
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    assert len(payload) > 0

    eval_exit = evaluate_module.main([
        "--predictions", str(pred_path),
        "--out", str(metrics_path),
    ])
    assert eval_exit == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    # Те же ожидания, что в core: разделимые данные дают высокую точность
    assert metrics["accuracy_year_season"] > 0.8

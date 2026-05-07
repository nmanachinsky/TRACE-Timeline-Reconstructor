"""Высокоуровневый API для Wizard-режима Streamlit-приложения.

Принимает на вход две директории:
- `reference_dir` — фото с сохранёнными EXIF/JSON-сайдкарами (обучающая выборка),
- `target_dir`    — фото без метаданных (выборка для классификации).

Готовит artefacts в `work_dir`, обучает локальный классификатор «под пользователя»
и сортирует target-файлы. Все долгие операции принимают callback вида
`progress_cb(fraction: float, message: str)` — для прогресс-бара UI.

Архитектурный принцип: TRACE никогда не скачивает обученные веса. Профиль
формируется заново на каждой reference-папке и валидируется на её 30%-выборке.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import (
    CLUSTERING,
    MODELS_DIR,
    RESNET,
    SPLIT,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.models.classifier import (
    FeatureLayout,
    TrainedClassifier,
    save_classifier,
    train_classifier,
)
from src.models.clustering import (
    apply_cluster_consensus,
    cluster_resnet_embeddings,
)
from src.pipeline.common import FeatureBundle, load_feature_bundle
from src.pipeline.feature_pipeline import (
    FeatureItem,
    extract_features_for_items,
)
from src.prepare.ground_truth import extract_ground_truth
from src.prepare.split import LabeledSample, train_test_split_stratified

ProgressCallback = Callable[[float, str], None]

REFERENCE_PREFIX = "ref_"
TARGET_PREFIX = "tgt_"
DEFAULT_WORK_ROOT: Path = MODELS_DIR / "wizard"
MIN_REFERENCE_FILES_FOR_TRAINING = 8
MIN_VALIDATION_FILES = 2

_SEASON_CENTER_MONTH = {"winter": 1, "spring": 4, "summer": 7, "autumn": 10}


# --- Errors -------------------------------------------------------------------


class WizardError(RuntimeError):
    """Ошибка пайплайна Wizard, понятная пользователю — пробрасывается в UI."""


# --- Step 1: Analyse paths ----------------------------------------------------


@dataclass(frozen=True)
class ReferenceFile:
    """Один эталонный файл с извлечённой меткой времени."""

    path: Path
    sample_id: str
    timestamp: datetime
    year: int
    season: str
    class_label: str
    source: str  # "json_sidecar" | "filename" | "folder"


@dataclass(frozen=True)
class ReferenceAnalysis:
    """Результат сканирования reference_dir — что нашли и какие классы покроем."""

    directory: Path
    total_image_files: int
    files: tuple[ReferenceFile, ...]
    skipped_paths: tuple[Path, ...]

    @property
    def files_with_metadata(self) -> int:
        return len(self.files)

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(sorted({f.year for f in self.files}))

    @property
    def seasons(self) -> tuple[str, ...]:
        return tuple(sorted({f.season for f in self.files}))

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(Counter(f.class_label for f in self.files))


@dataclass(frozen=True)
class TargetAnalysis:
    """Сканирование target_dir — путь и количество поддерживаемых файлов."""

    directory: Path
    files: tuple[Path, ...]

    @property
    def total_image_files(self) -> int:
        return len(self.files)


def analyze_reference(reference_dir: Path) -> ReferenceAnalysis:
    """Сканирует reference_dir, извлекает GT для каждого фото.

    Возвращает агрегированную статистику без побочных эффектов на диске.
    """
    if not reference_dir.exists() or not reference_dir.is_dir():
        raise WizardError(f"Reference Dir не найдена или не является папкой: {reference_dir}")

    image_paths = _collect_image_paths(reference_dir)
    files: list[ReferenceFile] = []
    skipped: list[Path] = []
    for path in image_paths:
        record = extract_ground_truth(path)
        if record is None:
            skipped.append(path)
            continue
        files.append(
            ReferenceFile(
                path=path,
                sample_id=_path_sample_id(path, REFERENCE_PREFIX),
                timestamp=record.timestamp,
                year=record.year,
                season=record.season,
                class_label=record.class_label,
                source=record.source.value,
            )
        )

    return ReferenceAnalysis(
        directory=reference_dir,
        total_image_files=len(image_paths),
        files=tuple(files),
        skipped_paths=tuple(skipped),
    )


def analyze_target(target_dir: Path) -> TargetAnalysis:
    """Сканирует target_dir на поддерживаемые изображения. GT не извлекается."""
    if not target_dir.exists() or not target_dir.is_dir():
        raise WizardError(f"Target Dir не найдена или не является папкой: {target_dir}")

    image_paths = _collect_image_paths(target_dir)
    return TargetAnalysis(directory=target_dir, files=tuple(image_paths))


# --- Step 2: Train profile ----------------------------------------------------


@dataclass(frozen=True)
class WizardConfig:
    """Параметры пайплайна. Для безопасной передачи через session_state."""

    reference_dir: Path
    target_dir: Path
    output_dir: Path
    use_m2: bool = False
    test_size: float = SPLIT.test_size
    consensus_weight: float = 0.4
    use_consensus: bool = True
    cluster_min_size: int = CLUSTERING.min_cluster_size
    cluster_pca_components: int = CLUSTERING.pca_components
    work_root: Path = DEFAULT_WORK_ROOT
    batch_size: int = RESNET.batch_size
    use_gpu: bool = False

    @property
    def session_dir(self) -> Path:
        return self.work_root / _path_hash(self.reference_dir)

    @property
    def reference_features_dir(self) -> Path:
        return self.session_dir / "reference_features"

    @property
    def classifier_path(self) -> Path:
        return self.session_dir / "classifier.joblib"

    @property
    def target_features_dir(self) -> Path:
        return self.session_dir / "target_features" / _path_hash(self.target_dir)


@dataclass(frozen=True)
class ValidationMetrics:
    """Метрики обученной модели на 30%-валидации эталонной выборки."""

    n_train: int
    n_val: int
    accuracy_year: float
    accuracy_year_season: float
    macro_f1_year_season: float
    mae_months: float
    confusion_matrix: np.ndarray
    confusion_labels: tuple[str, ...]
    classification_report: dict


@dataclass(frozen=True)
class TrainingResult:
    """Артефакты обучения: модель, эмбеддинги reference, метрики на валидации."""

    classifier: TrainedClassifier
    classifier_path: Path
    feature_bundle: FeatureBundle
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    train_labels_by_id: dict[str, str]
    val_metrics: ValidationMetrics


def train_reference_profile(
    config: WizardConfig,
    reference: ReferenceAnalysis,
    progress_cb: ProgressCallback | None = None,
) -> TrainingResult:
    """Обучает классификатор на reference-выборке, валидирует на её 30%.

    Извлечённые признаки кэшируются в `config.reference_features_dir` для
    ускорения повторных запусков на той же папке.
    """
    if reference.files_with_metadata < MIN_REFERENCE_FILES_FOR_TRAINING:
        raise WizardError(
            f"В Reference Dir слишком мало фото с метаданными "
            f"({reference.files_with_metadata}). Нужно минимум "
            f"{MIN_REFERENCE_FILES_FOR_TRAINING} для обучения профиля."
        )

    config.session_dir.mkdir(parents=True, exist_ok=True)

    items = [FeatureItem(sample_id=f.sample_id, path=f.path) for f in reference.files]
    bundle = extract_features_for_items(
        items,
        features_dir=config.reference_features_dir,
        use_m2=config.use_m2,
        batch_size=config.batch_size,
        gpu=config.use_gpu,
        progress_cb=_scaled(progress_cb, 0.0, 0.55, "Эталонные признаки"),
    )

    samples = [
        LabeledSample(stripped_id=f.sample_id, year=f.year, season=f.season)
        for f in reference.files
        if f.sample_id in bundle.ids
    ]
    split = train_test_split_stratified(samples, test_size=config.test_size)
    if len(split.train_ids) == 0:
        raise WizardError("После split нет ни одной обучающей строки — слишком мало данных")

    gt_map: dict[str, ReferenceFile] = {f.sample_id: f for f in reference.files}
    _report_progress(progress_cb, 0.65, "Обучение ансамбля (kNN + RF + LR)")
    classifier, train_ids, val_ids = _fit_classifier(bundle, split, gt_map)

    save_classifier(config.classifier_path, classifier)

    val_metrics = _evaluate_on_validation(
        classifier=classifier,
        bundle=bundle,
        val_ids=val_ids,
        train_count=len(train_ids),
        gt_map=gt_map,
    )
    _report_progress(progress_cb, 1.0, "Профиль обучен")

    train_labels_by_id = {sid: gt_map[sid].class_label for sid in train_ids}
    return TrainingResult(
        classifier=classifier,
        classifier_path=config.classifier_path,
        feature_bundle=bundle,
        train_ids=tuple(train_ids),
        val_ids=tuple(val_ids),
        train_labels_by_id=train_labels_by_id,
        val_metrics=val_metrics,
    )


def _fit_classifier(
    bundle: FeatureBundle,
    split,
    gt_map: dict[str, ReferenceFile],
) -> tuple[TrainedClassifier, list[str], list[str]]:
    train_set = set(split.train_ids)
    val_set = set(split.test_ids)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for i, sid in enumerate(bundle.ids):
        if sid in train_set:
            train_indices.append(i)
        elif sid in val_set:
            val_indices.append(i)

    if not train_indices:
        raise WizardError("Train ids не пересекаются с кэшем признаков")

    full_matrix = bundle.full_matrix
    train_x = full_matrix[train_indices]
    train_id_order = [bundle.ids[i] for i in train_indices]
    train_labels = [gt_map[sid].class_label for sid in train_id_order]

    layout = FeatureLayout(resnet_start=0, resnet_end=bundle.resnet_dim)
    classifier = train_classifier(train_x, train_labels, layout)
    val_id_order = [bundle.ids[i] for i in val_indices]
    return classifier, train_id_order, val_id_order


def _evaluate_on_validation(
    *,
    classifier: TrainedClassifier,
    bundle: FeatureBundle,
    val_ids: list[str],
    train_count: int,
    gt_map: dict[str, ReferenceFile],
) -> ValidationMetrics:
    if len(val_ids) < MIN_VALIDATION_FILES:
        return _empty_validation_metrics(n_train=train_count, n_val=len(val_ids))

    full_matrix = bundle.full_matrix
    val_indices = [i for i, sid in enumerate(bundle.ids) if sid in set(val_ids)]
    val_x = full_matrix[val_indices]
    val_id_order = [bundle.ids[i] for i in val_indices]
    val_true_labels = [gt_map[sid].class_label for sid in val_id_order]
    val_proba = classifier.predict_proba(val_x)
    val_predicted_labels = list(classifier.classes_[val_proba.argmax(axis=1)])

    classes = sorted(set(val_true_labels) | set(val_predicted_labels))
    cm = confusion_matrix(val_true_labels, val_predicted_labels, labels=classes)

    true_years = [int(lbl.split("-")[0]) for lbl in val_true_labels]
    pred_years = [int(lbl.split("-")[0]) for lbl in val_predicted_labels]
    mae = _mae_in_months(val_id_order, val_predicted_labels, gt_map)

    return ValidationMetrics(
        n_train=train_count,
        n_val=len(val_id_order),
        accuracy_year=float(accuracy_score(true_years, pred_years)),
        accuracy_year_season=float(accuracy_score(val_true_labels, val_predicted_labels)),
        macro_f1_year_season=float(
            f1_score(val_true_labels, val_predicted_labels, average="macro", zero_division=0)
        ),
        mae_months=mae,
        confusion_matrix=cm,
        confusion_labels=tuple(classes),
        classification_report=classification_report(
            val_true_labels, val_predicted_labels, zero_division=0, output_dict=True
        ),
    )


def _empty_validation_metrics(*, n_train: int, n_val: int) -> ValidationMetrics:
    return ValidationMetrics(
        n_train=n_train,
        n_val=n_val,
        accuracy_year=0.0,
        accuracy_year_season=0.0,
        macro_f1_year_season=0.0,
        mae_months=0.0,
        confusion_matrix=np.zeros((0, 0), dtype=int),
        confusion_labels=(),
        classification_report={},
    )


def _mae_in_months(
    val_id_order: list[str],
    predicted_labels: list[str],
    gt_map: dict[str, ReferenceFile],
) -> float:
    deltas: list[float] = []
    for sid, predicted in zip(val_id_order, predicted_labels):
        true_ts = gt_map[sid].timestamp
        try:
            pred_year_str, pred_season = predicted.split("-", 1)
            pred_year = int(pred_year_str)
        except ValueError:
            continue
        pred_dt = datetime(pred_year, _SEASON_CENTER_MONTH.get(pred_season, 6), 15)
        deltas.append(abs((pred_dt - true_ts).days) / 30.4375)
    return float(np.mean(deltas)) if deltas else 0.0


# --- Step 3: Inference --------------------------------------------------------


@dataclass(frozen=True)
class TargetPrediction:
    """Предсказание для одного target-файла."""

    path: Path
    sample_id: str
    label: str
    year: int
    season: str
    confidence: float
    top3: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class InferenceResult:
    predictions: tuple[TargetPrediction, ...]
    cluster_count: int
    consensus_applied: bool


def infer_target(
    config: WizardConfig,
    training: TrainingResult,
    target: TargetAnalysis,
    progress_cb: ProgressCallback | None = None,
) -> InferenceResult:
    """Извлекает признаки для target-файлов и применяет обученный классификатор."""
    if target.total_image_files == 0:
        raise WizardError("В Target Dir не найдено поддерживаемых изображений")

    items = [
        FeatureItem(sample_id=_path_sample_id(p, TARGET_PREFIX), path=p)
        for p in target.files
    ]
    target_bundle = extract_features_for_items(
        items,
        features_dir=config.target_features_dir,
        use_m2=config.use_m2,
        batch_size=config.batch_size,
        gpu=config.use_gpu,
        progress_cb=_scaled(progress_cb, 0.0, 0.7, "Признаки целевых фото"),
    )

    _check_dim_match(training.feature_bundle, target_bundle)

    _report_progress(progress_cb, 0.75, "Применяю классификатор")
    target_x = target_bundle.full_matrix
    base_proba = training.classifier.predict_proba(target_x)

    consensus_applied = False
    cluster_count = 0
    final_proba = base_proba
    if config.use_consensus:
        _report_progress(progress_cb, 0.85, "Сглаживание по кластерам (HDBSCAN)")
        try:
            final_proba, cluster_count = _apply_consensus(
                config=config,
                training=training,
                target_bundle=target_bundle,
                base_proba=base_proba,
            )
            consensus_applied = True
        except ValueError:
            consensus_applied = False

    classes = training.classifier.classes_
    predicted_indices = final_proba.argmax(axis=1)
    predictions: list[TargetPrediction] = []
    for i, sid in enumerate(target_bundle.ids):
        sorted_indices = np.argsort(-final_proba[i])[:3]
        top3 = tuple(
            (str(classes[j]), float(final_proba[i, j])) for j in sorted_indices
        )
        label = str(classes[predicted_indices[i]])
        year, season = _split_label(label)
        predictions.append(
            TargetPrediction(
                path=_lookup_path_by_id(items, sid),
                sample_id=sid,
                label=label,
                year=year,
                season=season,
                confidence=float(final_proba[i, predicted_indices[i]]),
                top3=top3,
            )
        )

    _report_progress(progress_cb, 1.0, "Предсказания готовы")
    return InferenceResult(
        predictions=tuple(predictions),
        cluster_count=cluster_count,
        consensus_applied=consensus_applied,
    )


def _check_dim_match(reference_bundle: FeatureBundle, target_bundle: FeatureBundle) -> None:
    ref_dim = reference_bundle.full_matrix.shape[1]
    tgt_dim = target_bundle.full_matrix.shape[1]
    if ref_dim != tgt_dim:
        raise WizardError(
            f"Размерность признаков target ({tgt_dim}) не совпадает с reference ({ref_dim}). "
            "Скорее всего использованы разные значения 'M2' (faces+OCR)."
        )


def _apply_consensus(
    *,
    config: WizardConfig,
    training: TrainingResult,
    target_bundle: FeatureBundle,
    base_proba: np.ndarray,
) -> tuple[np.ndarray, int]:
    combined_ids = list(training.feature_bundle.ids) + list(target_bundle.ids)
    combined_resnet = np.vstack([training.feature_bundle.resnet, target_bundle.resnet])
    assignment = cluster_resnet_embeddings(
        combined_ids,
        combined_resnet,
        min_cluster_size=config.cluster_min_size,
        pca_components=config.cluster_pca_components,
    )
    smoothed = apply_cluster_consensus(
        assignment,
        test_ids=list(target_bundle.ids),
        base_proba=base_proba,
        classes=training.classifier.classes_,
        train_labels_by_id=training.train_labels_by_id,
        consensus_weight=config.consensus_weight,
    )
    return smoothed, assignment.cluster_count


# --- Step 4: Apply ------------------------------------------------------------


@dataclass(frozen=True)
class ApplyReport:
    """Отчёт о раскладке файлов: успехи, пропуски, ошибки."""

    moved: int
    skipped: int
    errors: int
    output_dir: Path
    error_log: tuple[str, ...]


def apply_predictions(
    predictions: list[TargetPrediction] | tuple[TargetPrediction, ...],
    output_dir: Path,
    *,
    operation: str = "copy",
    confidence_threshold: float = 0.5,
) -> ApplyReport:
    """Раскладывает целевые файлы по подпапкам `output_dir/YYYY-Season/`.

    Файлы с уверенностью ниже порога пропускаются. Существующие копии того же
    размера не перезаписываются (идемпотентно).
    """
    if operation not in {"copy", "move"}:
        raise WizardError(f"Неподдерживаемая операция: {operation!r}")

    output_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0
    errors = 0
    error_log: list[str] = []
    for pred in predictions:
        if pred.confidence < confidence_threshold:
            skipped += 1
            continue
        if not pred.path.exists():
            errors += 1
            error_log.append(f"{pred.path.name}: исходный файл не найден")
            continue

        bucket = output_dir / pred.label
        bucket.mkdir(parents=True, exist_ok=True)
        destination = bucket / pred.path.name
        if destination.exists() and destination.stat().st_size == pred.path.stat().st_size:
            skipped += 1
            continue

        try:
            if operation == "copy":
                shutil.copy2(pred.path, destination)
            else:
                shutil.move(str(pred.path), str(destination))
            moved += 1
        except OSError as exc:
            errors += 1
            error_log.append(f"{pred.path.name}: {exc}")

    return ApplyReport(
        moved=moved,
        skipped=skipped,
        errors=errors,
        output_dir=output_dir,
        error_log=tuple(error_log),
    )


# --- Helpers ------------------------------------------------------------------


def reload_feature_bundle(features_dir: Path, *, use_m2: bool) -> FeatureBundle:
    """Тонкая обёртка — в session_state мы храним только пути и подгружаем заново."""
    return load_feature_bundle(features_dir, include_faces=use_m2, include_ocr=use_m2)


def _collect_image_paths(folder: Path) -> list[Path]:
    """Рекурсивный обход папки. Сортируется по строковому представлению пути."""
    paths: list[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: str(p).lower())


def _path_hash(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _path_sample_id(path: Path, prefix: str) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def _split_label(label: str) -> tuple[int, str]:
    parts = label.split("-", 1)
    if len(parts) != 2:
        return 0, ""
    try:
        return int(parts[0]), parts[1]
    except ValueError:
        return 0, ""


def _lookup_path_by_id(items: list[FeatureItem], sample_id: str) -> Path:
    for it in items:
        if it.sample_id == sample_id:
            return it.path
    raise KeyError(sample_id)


def _report_progress(cb: ProgressCallback | None, fraction: float, message: str) -> None:
    if cb is None:
        return
    cb(max(0.0, min(1.0, fraction)), message)


def _scaled(
    cb: ProgressCallback | None, start: float, end: float, prefix: str
) -> ProgressCallback | None:
    """Возвращает callback, отображающий [0,1] подзадачи в [start,end] общего бара."""
    if cb is None:
        return None
    span = end - start

    def scaled_cb(fraction: float, message: str) -> None:
        cb(start + span * max(0.0, min(1.0, fraction)), f"{prefix}: {message}")

    return scaled_cb


def write_predictions_dump(predictions: tuple[TargetPrediction, ...], path: Path) -> None:
    """Сериализует предсказания в JSON. Полезно для воспроизводимости/отладки."""
    payload = [
        {
            "path": str(p.path),
            "sample_id": p.sample_id,
            "label": p.label,
            "year": p.year,
            "season": p.season,
            "confidence": p.confidence,
            "top3": [{"label": lbl, "proba": prob} for lbl, prob in p.top3],
        }
        for p in predictions
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

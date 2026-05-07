"""Глобальная конфигурация проекта: пути, гиперпараметры, seed.

Все остальные модули должны брать константы отсюда — никаких магических чисел и
относительных путей в логике.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
ORIGINALS_DIR: Path = DATA_DIR / "originals"
STRIPPED_DIR: Path = DATA_DIR / "stripped"
FEATURES_DIR: Path = DATA_DIR / "features"
SPLITS_DIR: Path = DATA_DIR / "splits"
GROUND_TRUTH_PATH: Path = DATA_DIR / "ground_truth.json"
GT_CONFLICTS_PATH: Path = PROJECT_ROOT / "reports" / "gt_conflicts.json"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

SEED: int = 42

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".heic", ".png")
EXCLUDED_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".avi")

JSON_SIDECAR_SUFFIXES: tuple[str, ...] = (
    ".supplemental-metadata.json",
    ".supplemental-metada.json",
    ".supplemental-met.json",
    ".supplemental-me.json",
    ".supplemental-m.json",
)


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.30
    random_state: int = SEED
    min_class_size: int = 2


@dataclass(frozen=True)
class ResNetConfig:
    batch_size: int = 16
    image_size: int = 224
    embedding_dim: int = 2048


@dataclass(frozen=True)
class ColorConfig:
    hsv_bins: tuple[int, int, int] = (8, 8, 8)
    snow_value_threshold: float = 0.85
    snow_saturation_threshold: float = 0.15
    snow_pixel_ratio: float = 0.25
    foliage_hue_min_deg: float = 60.0
    foliage_hue_max_deg: float = 150.0
    foliage_saturation_threshold: float = 0.30


@dataclass(frozen=True)
class ClassifierConfig:
    knn_neighbors: int = 7
    rf_estimators: int = 300
    lr_max_iter: int = 2000
    lr_c: float = 1.0
    voting_weights: tuple[float, float, float] = (0.3, 0.5, 0.2)


@dataclass(frozen=True)
class ClusteringConfig:
    min_cluster_size: int = 4
    pca_components: int = 128


SPLIT = SplitConfig()
RESNET = ResNetConfig()
COLOR = ColorConfig()
CLASSIFIER = ClassifierConfig()
CLUSTERING = ClusteringConfig()


SEASON_BY_MONTH: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def season_for_month(month: int) -> str:
    """Сезон по номеру месяца. Зима = декабрь+январь+февраль и т.д."""
    if month not in SEASON_BY_MONTH:
        raise ValueError(f"Месяц должен быть в диапазоне 1-12, получено: {month}")
    return SEASON_BY_MONTH[month]

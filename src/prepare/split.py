"""Стратифицированный train/test split по (год, сезон).

Редкие классы (<min_count экземпляров) объединяются в (year, 'any'), иначе
StratifiedShuffleSplit падает на one-shot классах. После split выкидывать
эти "any"-фото из (year, season) метрики, но они полезны для year-метрики.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sklearn.model_selection import StratifiedShuffleSplit

from src.config import SPLIT


@dataclass(frozen=True)
class LabeledSample:
    stripped_id: str
    year: int
    season: str


@dataclass(frozen=True)
class SplitResult:
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def stratify_label(samples: list[LabeledSample], min_count: int = SPLIT.min_class_size) -> list[str]:
    """Возвращает строки-метки 'YYYY-season' с объединением редких в 'YYYY-any'."""
    counts = Counter((s.year, s.season) for s in samples)
    return [
        f"{s.year}-{s.season}" if counts[(s.year, s.season)] >= min_count else f"{s.year}-any"
        for s in samples
    ]


def train_test_split_stratified(
    samples: list[LabeledSample],
    test_size: float = SPLIT.test_size,
    random_state: int = SPLIT.random_state,
    min_count: int = SPLIT.min_class_size,
) -> SplitResult:
    """Стратифицированный split с устойчивостью к редким классам."""
    if not samples:
        return SplitResult(train_ids=(), test_ids=())

    labels = stratify_label(samples, min_count=min_count)
    label_counts = Counter(labels)
    rare_labels = {lbl for lbl, c in label_counts.items() if c < 2}

    rare_indices = [i for i, lbl in enumerate(labels) if lbl in rare_labels]
    keepable_indices = [i for i, lbl in enumerate(labels) if lbl not in rare_labels]

    if not keepable_indices:
        return SplitResult(
            train_ids=tuple(samples[i].stripped_id for i in range(len(samples))),
            test_ids=(),
        )

    keepable_labels = [labels[i] for i in keepable_indices]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    keepable_train_pos, keepable_test_pos = next(
        splitter.split(keepable_indices, keepable_labels)
    )

    train_set = {keepable_indices[p] for p in keepable_train_pos}
    test_set = {keepable_indices[p] for p in keepable_test_pos}
    # Все одиночки уходят в train (нельзя оценивать класс с 1 фото)
    train_set.update(rare_indices)

    train_ids = tuple(
        samples[i].stripped_id for i in sorted(train_set)
    )
    test_ids = tuple(
        samples[i].stripped_id for i in sorted(test_set)
    )
    return SplitResult(train_ids=train_ids, test_ids=test_ids)

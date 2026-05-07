"""Тесты стратифицированного train/test split по (год, сезон).

Требования:
- Стратификация сохраняет распределение классов (доли отличаются < 5% по абсолюту).
- Редкие классы (<2 фото) объединяются в (year, 'any'), чтобы StratifiedShuffleSplit
  не упал на one-shot классах.
- Воспроизводимость: одинаковый seed → одинаковый split.
- ID не пересекаются между train и test.
"""

from __future__ import annotations

from collections import Counter

from src.prepare.split import LabeledSample, stratify_label, train_test_split_stratified


def _samples_from(spec: list[tuple[str, int, str, int]]) -> list[LabeledSample]:
    """Утилита: список (id_prefix, count, season, year) → LabeledSample[].

    ID уникальны через глобальный счётчик — иначе разные группы коллизятся.
    """
    out: list[LabeledSample] = []
    counter = 0
    for prefix, count, season, year in spec:
        for _ in range(count):
            out.append(
                LabeledSample(stripped_id=f"{prefix}_{counter:04d}", year=year, season=season)
            )
            counter += 1
    return out


class TestStratifyLabel:
    def test_объединяет_классы_с_одним_фото_в_год_any(self) -> None:
        samples = [
            LabeledSample("a", 2019, "winter"),
            LabeledSample("b", 2019, "winter"),
            LabeledSample("c", 2019, "summer"),  # одиночка
            LabeledSample("d", 2020, "summer"),  # одиночка
        ]
        labels = stratify_label(samples, min_count=2)

        # Одиночные (2019,summer) и (2020,summer) должны стать year-any
        assert labels[0] == "2019-winter"
        assert labels[2] == "2019-any"
        assert labels[3] == "2020-any"

    def test_не_трогает_классы_с_достаточным_размером(self) -> None:
        samples = [
            LabeledSample("a", 2022, "winter"),
            LabeledSample("b", 2022, "winter"),
            LabeledSample("c", 2022, "winter"),
        ]
        labels = stratify_label(samples, min_count=2)
        assert all(label == "2022-winter" for label in labels)


class TestTrainTestSplit:
    def test_должен_давать_воспроизводимый_split(self) -> None:
        samples = _samples_from([
            ("p", 50, "winter", 2022),
            ("p", 50, "summer", 2022),
            ("p", 30, "spring", 2021),
        ])

        split_a = train_test_split_stratified(samples, test_size=0.30, random_state=42)
        split_b = train_test_split_stratified(samples, test_size=0.30, random_state=42)

        assert split_a.train_ids == split_b.train_ids
        assert split_a.test_ids == split_b.test_ids

    def test_train_и_test_не_пересекаются(self) -> None:
        samples = _samples_from([
            ("p", 100, "winter", 2022),
            ("p", 100, "summer", 2022),
            ("p", 50, "spring", 2021),
        ])

        split = train_test_split_stratified(samples, test_size=0.30, random_state=42)

        assert set(split.train_ids).isdisjoint(set(split.test_ids))
        assert len(split.train_ids) + len(split.test_ids) == len(samples)

    def test_test_size_близок_к_заданному(self) -> None:
        samples = _samples_from([
            ("p", 100, "winter", 2022),
            ("p", 100, "summer", 2022),
            ("p", 100, "spring", 2021),
        ])
        split = train_test_split_stratified(samples, test_size=0.30, random_state=42)
        ratio = len(split.test_ids) / len(samples)

        assert 0.27 <= ratio <= 0.33

    def test_сохраняет_распределение_классов(self) -> None:
        # 200 winter+2022, 100 summer+2022, 60 spring+2021, 30 autumn+2020
        samples = _samples_from([
            ("p", 200, "winter", 2022),
            ("p", 100, "summer", 2022),
            ("p", 60, "spring", 2021),
            ("p", 30, "autumn", 2020),
        ])
        split = train_test_split_stratified(samples, test_size=0.30, random_state=42)

        full_dist = Counter(f"{s.year}-{s.season}" for s in samples)
        train_lookup = {s.stripped_id: f"{s.year}-{s.season}" for s in samples}
        train_dist = Counter(train_lookup[i] for i in split.train_ids)
        test_dist = Counter(train_lookup[i] for i in split.test_ids)

        for cls, total in full_dist.items():
            full_share = total / sum(full_dist.values())
            train_share = train_dist.get(cls, 0) / max(sum(train_dist.values()), 1)
            test_share = test_dist.get(cls, 0) / max(sum(test_dist.values()), 1)
            assert abs(full_share - train_share) < 0.05, f"Train drift {cls}: {train_share} vs {full_share}"
            assert abs(full_share - test_share) < 0.05, f"Test drift {cls}: {test_share} vs {full_share}"

    def test_должен_обработать_классы_с_одним_элементом(self) -> None:
        # (2019, summer) — единственный экземпляр; не должно падать
        samples = _samples_from([
            ("p", 50, "winter", 2022),
            ("p", 50, "summer", 2022),
            ("p", 1, "summer", 2019),
        ])

        split = train_test_split_stratified(samples, test_size=0.30, random_state=42)

        assert len(split.train_ids) + len(split.test_ids) == 101

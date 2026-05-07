"""Тесты извлечения ground truth даты для фотографии.

Источники с приоритетом:
1) JSON-сайдкар Google Takeout (поле photoTakenTime.timestamp).
2) Дата в имени файла (несколько паттернов).
3) Год по имени родительской папки (день=15, месяц=6 — середина года).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.prepare.ground_truth import (
    GroundTruthRecord,
    GroundTruthSource,
    extract_ground_truth,
    parse_filename_date,
    parse_folder_year,
)


class TestParseFilenameDate:
    def test_должен_распарсить_паттерн_IMG_YYYYMMDD_HHMMSS(self) -> None:
        result = parse_filename_date("IMG_20190412_142843.jpg")
        assert result == datetime(2019, 4, 12, 14, 28, 43)

    def test_должен_распарсить_паттерн_IMG_без_подчёркивания(self) -> None:
        result = parse_filename_date("IMG20240128182236.heic")
        assert result == datetime(2024, 1, 28, 18, 22, 36)

    def test_должен_распарсить_паттерн_IMG_только_дата(self) -> None:
        result = parse_filename_date("IMG_20210715.jpg")
        assert result is not None
        assert result.year == 2021
        assert result.month == 7
        assert result.day == 15

    def test_должен_распарсить_unix_timestamp_миллисекунды(self) -> None:
        # 1635621754424 ms = 2021-10-30 18:42:34 UTC
        result = parse_filename_date("1635621754424.jpg")
        assert result is not None
        assert result.year == 2021
        assert result.month == 10

    def test_должен_распарсить_unix_timestamp_секунды(self) -> None:
        # 1706455356 = 2024-01-28 15:22:36 UTC
        result = parse_filename_date("1706455356.jpg")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1

    def test_должен_вернуть_None_для_случайного_хеша(self) -> None:
        assert parse_filename_date("e8_rl7McYfY.jpg") is None
        assert parse_filename_date("--RXkFwGhz4.jpg") is None

    def test_должен_проигнорировать_невалидную_дату(self) -> None:
        # 99 января не существует
        assert parse_filename_date("IMG_20210199_120000.jpg") is None

    def test_должен_отвергнуть_год_вне_диапазона(self) -> None:
        # 1999 — слишком рано
        assert parse_filename_date("IMG_19990412_142843.jpg") is None
        # 2099 — слишком далеко в будущее
        assert parse_filename_date("IMG_20990412_142843.jpg") is None

    def test_должен_распарсить_android_screenshot_короткий_формат(self) -> None:
        result = parse_filename_date("Screenshot_2020-06-09-17-19-27-34.jpg")
        assert result == datetime(2020, 6, 9, 17, 19, 27)

    def test_должен_распарсить_android_screenshot_с_приложением(self) -> None:
        result = parse_filename_date(
            "Screenshot_2019-06-16-10-05-34-414_com.vkontakt.png"
        )
        assert result == datetime(2019, 6, 16, 10, 5, 34)

    def test_должен_распарсить_дату_в_сером_формате_YYYY_MM_DD(self) -> None:
        # Telegram-style: 2022-04-15_19-30-22.jpg
        result = parse_filename_date("photo_2022-04-15_19-30-22.jpg")
        assert result == datetime(2022, 4, 15, 19, 30, 22)


class TestParseFolderYear:
    def test_должен_распарсить_русскую_папку_фото_год(self) -> None:
        assert parse_folder_year("Фото 2022 г") == 2022

    def test_должен_распарсить_папку_только_с_годом(self) -> None:
        assert parse_folder_year("2021") == 2021

    def test_должен_вернуть_None_для_папки_без_года(self) -> None:
        assert parse_folder_year("originals") is None
        assert parse_folder_year("random") is None


class TestExtractGroundTruth:
    def test_должен_использовать_JSON_сайдкар_приоритетно(
        self, tmp_path: Path, make_jpeg, make_json_sidecar
    ) -> None:
        folder = tmp_path / "Фото 2022 г"
        image = make_jpeg(folder / "IMG_20210101_000000.jpg")
        # JSON говорит 2022 год, имя файла — 2021 год → должен победить JSON
        make_json_sidecar(image, photo_taken_unix=1656600000)  # 2022-06-30

        record = extract_ground_truth(image)

        assert record is not None
        assert record.source == GroundTruthSource.JSON_SIDECAR
        assert record.timestamp.year == 2022

    def test_должен_использовать_имя_файла_если_JSON_отсутствует(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        folder = tmp_path / "Фото 2021 г"
        image = make_jpeg(folder / "IMG_20210715_120000.jpg")

        record = extract_ground_truth(image)

        assert record is not None
        assert record.source == GroundTruthSource.FILENAME
        assert record.timestamp == datetime(2021, 7, 15, 12, 0, 0)

    def test_должен_упасть_на_папку_если_другие_источники_пусты(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        folder = tmp_path / "Фото 2020 г"
        image = make_jpeg(folder / "random_hash_no_date.jpg")

        record = extract_ground_truth(image)

        assert record is not None
        assert record.source == GroundTruthSource.FOLDER
        assert record.timestamp.year == 2020

    def test_должен_вернуть_None_если_все_источники_пусты(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        folder = tmp_path / "no_year_folder"
        image = make_jpeg(folder / "random.jpg")

        record = extract_ground_truth(image)

        assert record is None

    def test_должен_зафиксировать_конфликт_JSON_и_имени_файла(
        self, tmp_path: Path, make_jpeg, make_json_sidecar
    ) -> None:
        folder = tmp_path / "Фото 2022 г"
        image = make_jpeg(folder / "IMG_20210715_120000.jpg")
        make_json_sidecar(image, photo_taken_unix=1656600000)  # 2022-06-30

        record = extract_ground_truth(image)

        assert record is not None
        # Конфликт: JSON говорит июнь 2022, имя — июль 2021 → разница > 1 месяц
        assert record.has_conflict is True
        assert record.conflicting_filename_timestamp is not None
        assert record.conflicting_filename_timestamp.year == 2021

    def test_не_должен_фиксировать_конфликт_при_малой_разнице(
        self, tmp_path: Path, make_jpeg, make_json_sidecar
    ) -> None:
        folder = tmp_path / "Фото 2022 г"
        # Имя файла — 30 июня 12:00:00
        image = make_jpeg(folder / "IMG_20220630_120000.jpg")
        # JSON — 30 июня 14:00:00 UTC того же дня
        make_json_sidecar(image, photo_taken_unix=1656597600)

        record = extract_ground_truth(image)

        assert record is not None
        assert record.has_conflict is False

    def test_должен_находить_JSON_с_усечённым_суффиксом(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        folder = tmp_path / "Фото 2024 г"
        image = make_jpeg(folder / "IMG20240128182236.heic")
        # Реальный артефакт Google Takeout — суффикс может быть обрезан
        truncated_sidecar = folder / f"{image.name}.supplemental-met.json"
        truncated_sidecar.write_text(
            '{"photoTakenTime": {"timestamp": "1706455356"}}', encoding="utf-8"
        )

        record = extract_ground_truth(image)

        assert record is not None
        assert record.source == GroundTruthSource.JSON_SIDECAR
        assert record.timestamp.year == 2024
        assert record.timestamp.month == 1


class TestGroundTruthRecord:
    def test_year_возвращает_год_из_timestamp(self) -> None:
        record = GroundTruthRecord(
            timestamp=datetime(2023, 5, 15, 10, 0, 0),
            source=GroundTruthSource.JSON_SIDECAR,
        )
        assert record.year == 2023

    def test_season_возвращает_сезон_из_месяца(self) -> None:
        winter_record = GroundTruthRecord(
            timestamp=datetime(2023, 1, 15), source=GroundTruthSource.FILENAME
        )
        summer_record = GroundTruthRecord(
            timestamp=datetime(2023, 7, 15), source=GroundTruthSource.FILENAME
        )

        assert winter_record.season == "winter"
        assert summer_record.season == "summer"

    def test_class_label_возвращает_строку_год_сезон(self) -> None:
        record = GroundTruthRecord(
            timestamp=datetime(2023, 4, 15), source=GroundTruthSource.FILENAME
        )
        assert record.class_label == "2023-spring"


@pytest.mark.parametrize(
    ("month", "expected_season"),
    [
        (1, "winter"), (2, "winter"), (12, "winter"),
        (3, "spring"), (4, "spring"), (5, "spring"),
        (6, "summer"), (7, "summer"), (8, "summer"),
        (9, "autumn"), (10, "autumn"), (11, "autumn"),
    ],
)
def test_season_for_month_корректно_распределяет_сезоны(month: int, expected_season: str) -> None:
    from src.config import season_for_month

    assert season_for_month(month) == expected_season

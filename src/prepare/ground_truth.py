"""Извлечение ground truth даты для фотографии из трёх источников.

Приоритет:
1. JSON-сайдкар Google Takeout (`<image>.supplemental-metadata.json` и обрезанные варианты).
2. Дата в имени файла: IMG_YYYYMMDD_HHMMSS, IMGYYYYMMDDHHMMSS, IMG_YYYYMMDD, Unix-timestamp.
3. Год в имени родительской папки (день=15, месяц=6 — середина года).

Конфликты между JSON и именем файла (разница > MAX_CONFLICT_DAYS) фиксируются в записи.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.config import JSON_SIDECAR_SUFFIXES, season_for_month


class GroundTruthSource(str, Enum):
    JSON_SIDECAR = "json_sidecar"
    FILENAME = "filename"
    FOLDER = "folder"


MIN_VALID_YEAR = 2005
MAX_VALID_YEAR = 2030
FOLDER_FALLBACK_DAY = 15
FOLDER_FALLBACK_MONTH = 6
MAX_CONFLICT_DAYS = 30

UNIX_SECONDS_MIN = 1_100_000_000   # 2004-11-12
UNIX_SECONDS_MAX = 1_900_000_000   # 2030-03-23
UNIX_MILLIS_MIN = UNIX_SECONDS_MIN * 1000
UNIX_MILLIS_MAX = UNIX_SECONDS_MAX * 1000

_FILENAME_DATETIME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"IMG[_-](\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})"),
    re.compile(r"IMG(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})"),
    # Android Screenshot_YYYY-MM-DD-HH-MM-SS[-XX...]
    re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})"),
    # Telegram/общий формат: YYYY-MM-DD_HH-MM-SS
    re.compile(r"(\d{4})-(\d{2})-(\d{2})[_ ](\d{2})-(\d{2})-(\d{2})"),
)
_FILENAME_DATE_ONLY_PATTERN = re.compile(r"IMG[_-](\d{4})(\d{2})(\d{2})(?:[_.-]|$)")
_UNIX_TIMESTAMP_PATTERN = re.compile(r"^(\d{10,13})(?:[_.-]|\.\w+$|$)")
_FOLDER_YEAR_PATTERN = re.compile(r"(?:^|\D)(20\d{2})(?:\D|$)")


@dataclass(frozen=True)
class GroundTruthRecord:
    timestamp: datetime
    source: GroundTruthSource
    has_conflict: bool = False
    conflicting_filename_timestamp: datetime | None = None

    @property
    def year(self) -> int:
        return self.timestamp.year

    @property
    def season(self) -> str:
        return season_for_month(self.timestamp.month)

    @property
    def class_label(self) -> str:
        return f"{self.year}-{self.season}"


def parse_filename_date(filename: str) -> datetime | None:
    """Извлекает дату из имени файла. None — если дата не распознана или невалидна."""
    name = Path(filename).name

    for pattern in _FILENAME_DATETIME_PATTERNS:
        match = pattern.search(name)
        if match:
            return _safe_make_datetime(*(int(g) for g in match.groups()))

    date_only = _FILENAME_DATE_ONLY_PATTERN.search(name)
    if date_only:
        year, month, day = (int(g) for g in date_only.groups())
        return _safe_make_datetime(year, month, day, 12, 0, 0)

    unix_match = _UNIX_TIMESTAMP_PATTERN.match(name)
    if unix_match:
        return _parse_unix_timestamp(unix_match.group(1))

    return None


def parse_folder_year(folder_name: str) -> int | None:
    """Извлекает год из имени папки (например, 'Фото 2022 г' → 2022)."""
    match = _FOLDER_YEAR_PATTERN.search(folder_name)
    if not match:
        return None
    year = int(match.group(1))
    if not _is_year_valid(year):
        return None
    return year


def extract_ground_truth(image_path: Path) -> GroundTruthRecord | None:
    """Возвращает GroundTruthRecord или None, если ни один источник не сработал."""
    json_timestamp = _read_json_sidecar_timestamp(image_path)
    filename_timestamp = parse_filename_date(image_path.name)

    if json_timestamp is not None:
        return _make_record_with_conflict_detection(json_timestamp, filename_timestamp)

    if filename_timestamp is not None:
        return GroundTruthRecord(
            timestamp=filename_timestamp,
            source=GroundTruthSource.FILENAME,
        )

    folder_year = parse_folder_year(image_path.parent.name)
    if folder_year is not None:
        fallback_dt = datetime(folder_year, FOLDER_FALLBACK_MONTH, FOLDER_FALLBACK_DAY)
        return GroundTruthRecord(timestamp=fallback_dt, source=GroundTruthSource.FOLDER)

    return None


def _read_json_sidecar_timestamp(image_path: Path) -> datetime | None:
    for suffix in JSON_SIDECAR_SUFFIXES:
        candidate = image_path.with_name(image_path.name + suffix)
        if candidate.exists():
            return _parse_sidecar_file(candidate)
    return None


def _parse_sidecar_file(path: Path) -> datetime | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    timestamp_raw = payload.get("photoTakenTime", {}).get("timestamp")
    if not timestamp_raw:
        return None
    try:
        return datetime.utcfromtimestamp(int(timestamp_raw))
    except (ValueError, OSError):
        return None


def _make_record_with_conflict_detection(
    json_timestamp: datetime,
    filename_timestamp: datetime | None,
) -> GroundTruthRecord:
    if filename_timestamp is None:
        return GroundTruthRecord(
            timestamp=json_timestamp,
            source=GroundTruthSource.JSON_SIDECAR,
        )

    delta = abs((json_timestamp - filename_timestamp).days)
    has_conflict = delta > MAX_CONFLICT_DAYS
    return GroundTruthRecord(
        timestamp=json_timestamp,
        source=GroundTruthSource.JSON_SIDECAR,
        has_conflict=has_conflict,
        conflicting_filename_timestamp=filename_timestamp if has_conflict else None,
    )


def _safe_make_datetime(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime | None:
    if not _is_year_valid(year):
        return None
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def _parse_unix_timestamp(raw: str) -> datetime | None:
    value = int(raw)
    if UNIX_MILLIS_MIN <= value <= UNIX_MILLIS_MAX:
        seconds = value / 1000.0
    elif UNIX_SECONDS_MIN <= value <= UNIX_SECONDS_MAX:
        seconds = float(value)
    else:
        return None
    try:
        return datetime.utcfromtimestamp(seconds)
    except (ValueError, OSError):
        return None


def _is_year_valid(year: int) -> bool:
    return MIN_VALID_YEAR <= year <= MAX_VALID_YEAR

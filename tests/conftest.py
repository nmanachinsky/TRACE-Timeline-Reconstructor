"""Общие pytest-фикстуры для всех тестов."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Callable, Iterator

import pytest
from PIL import Image


@pytest.fixture
def tmp_dataset_root(tmp_path: Path) -> Iterator[Path]:
    """Временный каркас датасета: data/Фото 2022 г/ и т.п. с пустыми файлами."""
    root = tmp_path / "data"
    root.mkdir()
    yield root


@pytest.fixture
def make_json_sidecar() -> Callable[..., Path]:
    """Фабрика JSON-сайдкаров Google Takeout формата."""

    def _make(image_path: Path, photo_taken_unix: int, formatted: str = "") -> Path:
        sidecar = image_path.parent / f"{image_path.name}.supplemental-metadata.json"
        sidecar.write_text(
            json.dumps(
                {
                    "title": image_path.name,
                    "photoTakenTime": {
                        "timestamp": str(photo_taken_unix),
                        "formatted": formatted,
                    },
                }
            ),
            encoding="utf-8",
        )
        return sidecar

    return _make


@pytest.fixture
def make_jpeg() -> Callable[..., Path]:
    """Фабрика небольших JPEG-файлов с настраиваемым цветом и опциональным EXIF."""

    def _make(
        path: Path,
        size: tuple[int, int] = (32, 32),
        color: tuple[int, int, int] = (200, 200, 200),
        exif_datetime: str | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", size, color=color)
        if exif_datetime:
            exif = img.getexif()
            exif[0x9003] = exif_datetime  # DateTimeOriginal
            buf = io.BytesIO()
            img.save(buf, format="JPEG", exif=exif.tobytes())
            path.write_bytes(buf.getvalue())
        else:
            img.save(path, format="JPEG")
        return path

    return _make


@pytest.fixture
def make_png() -> Callable[..., Path]:
    """Фабрика PNG-файлов."""

    def _make(path: Path, size: tuple[int, int] = (32, 32), color: tuple[int, int, int] = (200, 200, 200)) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color=color).save(path, format="PNG")
        return path

    return _make

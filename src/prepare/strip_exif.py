"""Стриппинг EXIF-метаданных и анонимное переименование.

Фотографии копируются в каталог stripped/ с именем `photo_<sha1>.<ext>`. Это
устраняет два канала утечки даты: EXIF DateTimeOriginal и паттерны даты в имени.

HEIC исходники перекодируются в JPEG (для всего остального пайплайна это удобнее
и снимает зависимость от HEIF-энкодера на чтение).
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

from src.config import SUPPORTED_IMAGE_EXTENSIONS

register_heif_opener()

STRIPPED_ID_LENGTH = 16
STRIPPED_PREFIX = "photo_"
JPEG_QUALITY = 92

_OUTPUT_SUFFIX_BY_INPUT: dict[str, str] = {
    ".jpg": ".jpg",
    ".jpeg": ".jpg",
    ".heic": ".jpg",
    ".png": ".png",
}


@dataclass(frozen=True)
class StrippedFile:
    original_path: Path
    stripped_path: Path
    stripped_id: str


def compute_stripped_id(image_path: Path) -> str:
    """Возвращает первые 16 hex-символов sha1 от байт файла."""
    digest = hashlib.sha1(image_path.read_bytes()).hexdigest()
    return digest[:STRIPPED_ID_LENGTH]


def strip_image(image_path: Path, output_dir: Path) -> StrippedFile:
    """Создаёт копию без EXIF в output_dir под анонимным именем."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stripped_id = compute_stripped_id(image_path)
    output_suffix = _OUTPUT_SUFFIX_BY_INPUT[image_path.suffix.lower()]
    stripped_path = output_dir / f"{STRIPPED_PREFIX}{stripped_id}{output_suffix}"

    if not stripped_path.exists():
        _save_without_metadata(image_path, stripped_path)

    return StrippedFile(
        original_path=image_path,
        stripped_path=stripped_path,
        stripped_id=stripped_id,
    )


def strip_directory(source_dir: Path, output_dir: Path) -> list[StrippedFile]:
    """Стриппит все поддерживаемые изображения из source_dir рекурсивно."""
    results: list[StrippedFile] = []
    for path in _iter_supported_images(source_dir):
        results.append(strip_image(path, output_dir))
    return results


def _iter_supported_images(source_dir: Path):
    seen_lower: set[str] = set()
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        key = str(path).lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        yield path


def _save_without_metadata(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        img.load()
        target_mode = "RGB" if dst.suffix == ".jpg" else _safe_mode(img.mode)
        clean = Image.new(target_mode, img.size)
        clean.paste(img.convert(target_mode))
        save_kwargs: dict[str, object] = {}
        if dst.suffix == ".jpg":
            save_format = "JPEG"
            save_kwargs["quality"] = JPEG_QUALITY
            save_kwargs["exif"] = b""
        else:
            save_format = "PNG"
        buffer = io.BytesIO()
        clean.save(buffer, format=save_format, **save_kwargs)
        dst.write_bytes(buffer.getvalue())


def _safe_mode(mode: str) -> str:
    return mode if mode in ("RGB", "RGBA", "L") else "RGB"

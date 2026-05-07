"""Тесты стриппинга EXIF и анонимного переименования.

После стриппинга:
- В файле не должно остаться EXIF-метаданных.
- Имя файла не должно содержать дату (защита от утечки в имени).
- sha1-id стабилен и одинаков для одинакового содержимого.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PIL import Image

from src.prepare.strip_exif import (
    StrippedFile,
    compute_stripped_id,
    strip_directory,
    strip_image,
)


_DATE_LIKE_PATTERN = re.compile(r"\d{4}[_-]?\d{2}[_-]?\d{2}|\d{10,13}")


class TestComputeStrippedId:
    def test_id_стабилен_для_одного_содержимого(self, tmp_path: Path, make_jpeg) -> None:
        a = make_jpeg(tmp_path / "IMG_20210101_120000.jpg", color=(100, 100, 100))
        b = make_jpeg(tmp_path / "IMG_20220202_120000.jpg", color=(100, 100, 100))

        id_a = compute_stripped_id(a)
        id_b = compute_stripped_id(b)

        assert id_a == id_b
        assert len(id_a) == 16

    def test_id_отличается_для_разного_содержимого(self, tmp_path: Path, make_jpeg) -> None:
        a = make_jpeg(tmp_path / "a.jpg", color=(10, 20, 30))
        b = make_jpeg(tmp_path / "b.jpg", color=(200, 100, 50))

        assert compute_stripped_id(a) != compute_stripped_id(b)


class TestStripImage:
    def test_должен_удалить_exif_из_jpeg(self, tmp_path: Path, make_jpeg) -> None:
        src = make_jpeg(
            tmp_path / "src.jpg",
            exif_datetime="2021:07:15 12:34:56",
        )
        dst_dir = tmp_path / "stripped"

        result = strip_image(src, dst_dir)

        assert result.stripped_path.exists()
        assert result.stripped_path != src
        with Image.open(result.stripped_path) as out:
            exif = out.getexif()
            assert exif.get(0x9003) is None  # DateTimeOriginal
            assert len(dict(exif)) == 0 or all(v in (None, b"") for v in exif.values())

    def test_должен_дать_имя_без_даты(self, tmp_path: Path, make_jpeg) -> None:
        src = make_jpeg(tmp_path / "IMG_20210715_123456.jpg")
        dst_dir = tmp_path / "stripped"

        result = strip_image(src, dst_dir)

        new_name = result.stripped_path.name
        assert new_name.startswith("photo_")
        assert _DATE_LIKE_PATTERN.search(new_name) is None

    def test_должен_сохранить_расширение(self, tmp_path: Path, make_jpeg, make_png) -> None:
        jpg = make_jpeg(tmp_path / "a.jpg")
        png = make_png(tmp_path / "b.png")

        jpg_result = strip_image(jpg, tmp_path / "out")
        png_result = strip_image(png, tmp_path / "out")

        assert jpg_result.stripped_path.suffix == ".jpg"
        assert png_result.stripped_path.suffix == ".png"

    def test_должен_вернуть_метаданные_StrippedFile(self, tmp_path: Path, make_jpeg) -> None:
        src = make_jpeg(tmp_path / "IMG_20210715_120000.jpg", color=(50, 60, 70))
        dst_dir = tmp_path / "stripped"

        result = strip_image(src, dst_dir)

        assert isinstance(result, StrippedFile)
        assert result.original_path == src
        assert result.stripped_id == compute_stripped_id(src)
        assert result.stripped_path.parent == dst_dir

    def test_должен_быть_идемпотентен(self, tmp_path: Path, make_jpeg) -> None:
        src = make_jpeg(tmp_path / "a.jpg", color=(123, 45, 67))
        dst_dir = tmp_path / "stripped"

        first = strip_image(src, dst_dir)
        second = strip_image(src, dst_dir)

        assert first.stripped_path == second.stripped_path
        assert first.stripped_id == second.stripped_id


class TestStripDirectory:
    def test_должен_обработать_все_изображения(
        self, tmp_path: Path, make_jpeg, make_png
    ) -> None:
        src_dir = tmp_path / "Фото 2022 г"
        make_jpeg(src_dir / "IMG_20220701_120000.jpg")
        make_jpeg(src_dir / "IMG_20220702_120000.jpg")
        make_png(src_dir / "IMG_20220703_120000.png")
        # Файл, который не должен попасть в результат:
        (src_dir / "metadata.json").write_text("{}", encoding="utf-8")
        dst_dir = tmp_path / "stripped"

        results = strip_directory(src_dir, dst_dir)

        assert len(results) == 3
        for r in results:
            assert r.stripped_path.exists()
            assert "photo_" in r.stripped_path.name

    def test_должен_дедуплицировать_одинаковое_содержимое(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        src_dir = tmp_path / "Фото 2022 г"
        make_jpeg(src_dir / "a.jpg", color=(50, 50, 50))
        make_jpeg(src_dir / "b.jpg", color=(50, 50, 50))  # тот же контент
        dst_dir = tmp_path / "stripped"

        results = strip_directory(src_dir, dst_dir)

        # Оба файла указывают на один stripped_id (контент идентичен)
        assert len({r.stripped_id for r in results}) == 1
        # Но оба исходника зарегистрированы
        assert len(results) == 2

    def test_должен_пропустить_неподдерживаемые_расширения(
        self, tmp_path: Path, make_jpeg
    ) -> None:
        src_dir = tmp_path / "Фото 2022 г"
        make_jpeg(src_dir / "a.jpg")
        (src_dir / "video.mp4").write_bytes(b"fake mp4")
        (src_dir / "doc.pdf").write_bytes(b"fake pdf")
        dst_dir = tmp_path / "stripped"

        results = strip_directory(src_dir, dst_dir)

        assert len(results) == 1
        assert results[0].original_path.name == "a.jpg"


@pytest.mark.parametrize(
    "filename",
    [
        "IMG_20210715_123456.jpg",
        "Screenshot_2020-06-09-17-19-27-34.jpg",
        "1635621754424.jpg",
        "IMG20240128182236.heic",
    ],
)
def test_новое_имя_никогда_не_содержит_дату(
    tmp_path: Path, make_jpeg, filename: str
) -> None:
    src = make_jpeg(tmp_path / filename)
    result = strip_image(src, tmp_path / "stripped")

    assert _DATE_LIKE_PATTERN.search(result.stripped_path.name) is None

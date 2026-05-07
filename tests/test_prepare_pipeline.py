"""Интеграционный тест: prepare/pipeline.py на маленьком синтетическом датасете."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.prepare.pipeline import main


@pytest.mark.integration
def test_должен_выполнить_полный_prepare_на_синтетическом_датасете(
    tmp_path: Path, make_jpeg, make_json_sidecar
) -> None:
    data_dir = tmp_path / "data"
    folder_2022 = data_dir / "Фото 2022 г"
    folder_2023 = data_dir / "Фото 2023 г"

    # 2022: 10 фото — 5 winter (jan), 5 summer (jun)
    for i in range(5):
        img = make_jpeg(folder_2022 / f"IMG_2022010{i+1}_120000.jpg", color=(10 * i, 20 * i, 30 * i))
        make_json_sidecar(img, photo_taken_unix=1641038400 + i * 86400)
    for i in range(5):
        img = make_jpeg(folder_2022 / f"IMG_2022060{i+1}_120000.jpg", color=(50 + i, 60 + i, 70 + i))
        make_json_sidecar(img, photo_taken_unix=1654084800 + i * 86400)

    # 2023: 10 фото — 5 spring (apr), 5 autumn (oct)
    for i in range(5):
        img = make_jpeg(folder_2023 / f"IMG_2023040{i+1}_120000.jpg", color=(80 + i, 90 + i, 100 + i))
        make_json_sidecar(img, photo_taken_unix=1680350400 + i * 86400)
    for i in range(5):
        img = make_jpeg(folder_2023 / f"IMG_2023100{i+1}_120000.jpg", color=(110 + i, 120 + i, 130 + i))
        make_json_sidecar(img, photo_taken_unix=1696118400 + i * 86400)

    stripped_dir = tmp_path / "stripped"
    splits_dir = tmp_path / "splits"
    gt_path = tmp_path / "ground_truth.json"
    conflicts_path = tmp_path / "gt_conflicts.json"

    exit_code = main([
        "--data-dir", str(data_dir),
        "--dst-stripped", str(stripped_dir),
        "--dst-originals", str(tmp_path / "originals"),
        "--gt-path", str(gt_path),
        "--splits-dir", str(splits_dir),
        "--conflicts-path", str(conflicts_path),
    ])

    assert exit_code == 0
    # Все 20 stripped файлов созданы
    stripped_files = list(stripped_dir.iterdir())
    assert len(stripped_files) == 20
    assert all(f.name.startswith("photo_") for f in stripped_files)

    # ground_truth.json содержит 20 записей
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    assert len(gt) == 20
    sample_entry = next(iter(gt.values()))
    assert {"timestamp", "year", "season", "class_label", "source"}.issubset(sample_entry)

    # splits валидны и не пересекаются
    train_ids = json.loads((splits_dir / "train_ids.json").read_text(encoding="utf-8"))
    test_ids = json.loads((splits_dir / "test_ids.json").read_text(encoding="utf-8"))
    assert set(train_ids).isdisjoint(set(test_ids))
    assert len(train_ids) + len(test_ids) == 20

    # Конфликтов быть не должно (JSON и filename совпадают)
    conflicts = json.loads(conflicts_path.read_text(encoding="utf-8"))
    assert conflicts["count"] == 0


@pytest.mark.integration
def test_должен_логировать_конфликт_при_расхождении_источников(
    tmp_path: Path, make_jpeg, make_json_sidecar
) -> None:
    data_dir = tmp_path / "data"
    folder = data_dir / "Фото 2022 г"
    img = make_jpeg(folder / "IMG_20210715_120000.jpg")  # имя — 2021
    make_json_sidecar(img, photo_taken_unix=1656600000)  # JSON — 2022

    # Ещё одно валидное фото, чтобы split не упал
    img2 = make_jpeg(folder / "IMG_20220815_120000.jpg")
    make_json_sidecar(img2, photo_taken_unix=1660564800)  # 2022-08-15

    conflicts_path = tmp_path / "gt_conflicts.json"
    main([
        "--data-dir", str(data_dir),
        "--dst-stripped", str(tmp_path / "stripped"),
        "--dst-originals", str(tmp_path / "originals"),
        "--gt-path", str(tmp_path / "ground_truth.json"),
        "--splits-dir", str(tmp_path / "splits"),
        "--conflicts-path", str(conflicts_path),
    ])

    payload = json.loads(conflicts_path.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert "IMG_20210715" in payload["items"][0]["image"]

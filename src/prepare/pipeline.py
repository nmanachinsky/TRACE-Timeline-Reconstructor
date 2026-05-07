"""CLI-оркестратор подготовки данных.

Запуск: `uv run python -m src.prepare.pipeline`.

Шаги:
1. Сканирует data/source/Фото YYYY г/, извлекает ground truth для каждого фото.
2. Стриппит EXIF, складывает копии без метаданных в data/stripped/.
3. Создаёт data/originals/ как зеркало оригиналов (для сравнения и UI).
4. Сохраняет data/ground_truth.json — маппинг stripped_id → дата + метаданные источника.
5. Делает стратифицированный train/test split по (год, сезон), сохраняет в data/splits/.
6. Логирует расхождения между источниками в reports/gt_conflicts.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from src.config import (
    GROUND_TRUTH_PATH,
    GT_CONFLICTS_PATH,
    ORIGINALS_DIR,
    SOURCE_DIR,
    SPLITS_DIR,
    STRIPPED_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.prepare.ground_truth import GroundTruthRecord, GroundTruthSource, extract_ground_truth
from src.prepare.split import LabeledSample, train_test_split_stratified
from src.prepare.strip_exif import StrippedFile, strip_image


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_dir = Path(args.source_dir)

    started = time.time()
    print(f"[prepare] source_dir={source_dir}")

    image_paths = list(_iter_source_images(source_dir))
    print(f"[prepare] найдено изображений: {len(image_paths)}")

    records, conflicts, no_gt = _build_ground_truth(image_paths)
    print(
        f"[prepare] ground_truth: ok={len(records)}, "
        f"конфликтов JSON-vs-filename={len(conflicts)}, без GT={len(no_gt)}"
    )

    stripped_files = _strip_all(records, args.dst_stripped)
    if args.copy_originals:
        _mirror_originals(records, args.dst_originals)

    samples = _to_labeled_samples(stripped_files, records)
    split = train_test_split_stratified(samples)
    print(f"[prepare] split: train={len(split.train_ids)}, test={len(split.test_ids)}")

    _write_ground_truth_json(records, stripped_files, args.gt_path)
    _write_split_json(split, args.splits_dir)
    _write_conflicts(conflicts, args.conflicts_path)

    duration = time.time() - started
    print(f"[prepare] готово за {duration:.1f}s")
    print(f"[prepare] источники GT: {Counter(r.source.value for r in records.values())}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подготовка датасета для TRACE")
    parser.add_argument("--source-dir", default=str(SOURCE_DIR),
                        help="Каталог с папками 'Фото YYYY г' (default: data/source)")
    parser.add_argument("--dst-stripped", default=str(STRIPPED_DIR))
    parser.add_argument("--dst-originals", default=str(ORIGINALS_DIR))
    parser.add_argument("--gt-path", default=str(GROUND_TRUTH_PATH))
    parser.add_argument("--splits-dir", default=str(SPLITS_DIR))
    parser.add_argument("--conflicts-path", default=str(GT_CONFLICTS_PATH))
    parser.add_argument("--copy-originals", action="store_true",
                        help="Зеркалить оригиналы в data/originals/ (расход места)")
    return parser.parse_args(argv)


def _iter_source_images(source_dir: Path):
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Каталог исходников не найден: {source_dir}. "
            "Создайте data/source/ и положите туда папки 'Фото YYYY г'."
        )
    for folder in sorted(source_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("Фото"):
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                yield path


def _build_ground_truth(
    image_paths: list[Path],
) -> tuple[dict[Path, GroundTruthRecord], list[dict[str, str]], list[Path]]:
    records: dict[Path, GroundTruthRecord] = {}
    conflicts: list[dict[str, str]] = []
    no_gt: list[Path] = []
    for path in image_paths:
        record = extract_ground_truth(path)
        if record is None:
            no_gt.append(path)
            continue
        records[path] = record
        if record.has_conflict and record.conflicting_filename_timestamp:
            conflicts.append({
                "image": str(path),
                "json_timestamp": record.timestamp.isoformat(),
                "filename_timestamp": record.conflicting_filename_timestamp.isoformat(),
            })
    return records, conflicts, no_gt


def _strip_all(
    records: dict[Path, GroundTruthRecord], output_dir: str
) -> dict[Path, StrippedFile]:
    out_path = Path(output_dir)
    result: dict[Path, StrippedFile] = {}
    for source_path in records:
        result[source_path] = strip_image(source_path, out_path)
    return result


def _mirror_originals(
    records: dict[Path, GroundTruthRecord], originals_dir: str
) -> None:
    out_path = Path(originals_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for source_path in records:
        rel_target = out_path / source_path.parent.name / source_path.name
        rel_target.parent.mkdir(parents=True, exist_ok=True)
        if not rel_target.exists():
            shutil.copy2(source_path, rel_target)


def _to_labeled_samples(
    stripped: dict[Path, StrippedFile], records: dict[Path, GroundTruthRecord]
) -> list[LabeledSample]:
    seen: set[str] = set()
    out: list[LabeledSample] = []
    for source_path, sf in stripped.items():
        if sf.stripped_id in seen:
            continue
        seen.add(sf.stripped_id)
        gt = records[source_path]
        out.append(LabeledSample(stripped_id=sf.stripped_id, year=gt.year, season=gt.season))
    return out


def _write_ground_truth_json(
    records: dict[Path, GroundTruthRecord],
    stripped: dict[Path, StrippedFile],
    out_path_str: str,
) -> None:
    payload: dict[str, dict[str, object]] = {}
    for source_path, sf in stripped.items():
        gt = records[source_path]
        bucket = payload.setdefault(sf.stripped_id, {
            "stripped_path": str(sf.stripped_path),
            "timestamp": gt.timestamp.isoformat(),
            "year": gt.year,
            "season": gt.season,
            "class_label": gt.class_label,
            "source": gt.source.value,
            "originals": [],
        })
        bucket.setdefault("originals", []).append(str(source_path))
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_split_json(split, splits_dir_str: str) -> None:
    splits_dir = Path(splits_dir_str)
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train_ids.json").write_text(
        json.dumps(list(split.train_ids), indent=2), encoding="utf-8"
    )
    (splits_dir / "test_ids.json").write_text(
        json.dumps(list(split.test_ids), indent=2), encoding="utf-8"
    )


def _write_conflicts(conflicts: list[dict[str, str]], path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"count": len(conflicts), "items": conflicts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Заглушки для совместимости с интерфейсом dataclass-сериализации
_ = asdict, GroundTruthSource


if __name__ == "__main__":
    sys.exit(main())

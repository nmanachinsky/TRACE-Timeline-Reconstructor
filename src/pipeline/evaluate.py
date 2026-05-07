"""CLI оценки: считает accuracy, macro-F1, MAE, генерирует confusion matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import DATA_DIR, REPORTS_DIR, season_for_month

SEASON_INDEX = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))

    if not payload:
        print("[evaluate] пустые предсказания — нечего оценивать")
        return 1

    rows = list(payload.values())
    metrics = _compute_metrics(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    _save_confusion_matrix(rows, out_path.parent / f"{out_path.stem}_confusion.png")

    _print_summary(metrics)
    return 0


def _compute_metrics(rows: list[dict]) -> dict:
    y_true_full = [r["true_label"] for r in rows]
    y_pred_full = [r["predicted_label"] for r in rows]
    y_true_year = [r["true_year"] for r in rows]
    y_pred_year = [r["predicted_year"] for r in rows]

    # MAE в месяцах: оцениваем разницу между предсказанной центральной датой и реальной
    mae_months = _mae_in_months(rows)

    return {
        "n_samples": len(rows),
        "n_classes": len(set(y_true_full)),
        "accuracy_year": float(accuracy_score(y_true_year, y_pred_year)),
        "accuracy_year_season": float(accuracy_score(y_true_full, y_pred_full)),
        "macro_f1_year_season": float(
            f1_score(y_true_full, y_pred_full, average="macro", zero_division=0)
        ),
        "weighted_f1_year_season": float(
            f1_score(y_true_full, y_pred_full, average="weighted", zero_division=0)
        ),
        "mae_months": mae_months,
        "classification_report": classification_report(
            y_true_full, y_pred_full, zero_division=0, output_dict=True
        ),
    }


def _mae_in_months(rows: list[dict]) -> float:
    """Среднее абсолютное отклонение между предсказанной центральной датой и реальной."""
    deltas: list[float] = []
    for row in rows:
        true_dt = datetime.fromisoformat(row["true_timestamp"])
        pred_dt = _season_center(row["predicted_year"], row["predicted_season"])
        deltas.append(abs((pred_dt - true_dt).days) / 30.4375)
    return float(np.mean(deltas))


_SEASON_CENTER_MONTH = {"winter": 1, "spring": 4, "summer": 7, "autumn": 10}


def _season_center(year: int, season: str) -> datetime:
    month = _SEASON_CENTER_MONTH.get(season, 6)
    return datetime(year, month, 15)


def _save_confusion_matrix(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted({r["true_label"] for r in rows} | {r["predicted_label"] for r in rows})
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["predicted_label"] for r in rows]
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.4), max(6, len(classes) * 0.4)))
    cax = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title("Confusion matrix (year-season)")
    fig.colorbar(cax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _print_summary(metrics: dict) -> None:
    print(f"[evaluate] N = {metrics['n_samples']}, классов = {metrics['n_classes']}")
    print(f"[evaluate] accuracy@year = {metrics['accuracy_year']:.3f}")
    print(f"[evaluate] accuracy@(year,season) = {metrics['accuracy_year_season']:.3f}")
    print(f"[evaluate] macro-F1@(year,season) = {metrics['macro_f1_year_season']:.3f}")
    print(f"[evaluate] weighted-F1@(year,season) = {metrics['weighted_f1_year_season']:.3f}")
    print(f"[evaluate] MAE = {metrics['mae_months']:.2f} мес.")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка предсказаний")
    parser.add_argument("--predictions", default=str(DATA_DIR / "predictions_m1.json"))
    parser.add_argument("--out", default=str(REPORTS_DIR / "metrics_m1.json"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())


# season_for_month имеется в config, но не используется здесь. Пусть остаётся импортируемым.
_ = season_for_month, SEASON_INDEX

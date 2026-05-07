"""Streamlit-демо для TRACE: визуализация таймлайна и метрик.

Запуск: `uv run streamlit run src/app/streamlit_app.py`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import DATA_DIR, REPORTS_DIR

PREDICTIONS_FILE = DATA_DIR / "predictions_m1.json"
METRICS_FILE = REPORTS_DIR / "metrics_m1.json"

SEASON_RU = {"winter": "зима", "spring": "весна", "summer": "лето", "autumn": "осень"}
SEASON_ORDER = {"winter": 1, "spring": 2, "summer": 3, "autumn": 4}


def main() -> None:
    st.set_page_config(page_title="TRACE Timeline", layout="wide")
    st.title("TRACE — Восстановленный таймлайн")
    st.caption("Курсовой проект: восстановление (год, сезон) фото без EXIF")

    if not PREDICTIONS_FILE.exists():
        st.error(f"Не найден файл предсказаний: {PREDICTIONS_FILE}")
        st.info("Запустите: `make prepare && make features-core && make train-m1 && make predict-m1`")
        return

    predictions = json.loads(PREDICTIONS_FILE.read_text(encoding="utf-8"))
    metrics = (
        json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        if METRICS_FILE.exists()
        else None
    )

    _render_metrics(metrics)
    df = _build_dataframe(predictions)
    tabs = st.tabs(["Таймлайн", "Per-class отчёт", "Confusion matrix", "Браузер фото"])

    with tabs[0]:
        _render_timeline(df)
    with tabs[1]:
        _render_per_class(df)
    with tabs[2]:
        _render_confusion(df)
    with tabs[3]:
        _render_browser(predictions)


def _render_metrics(metrics: dict | None) -> None:
    if metrics is None:
        st.warning("Файл метрик не найден — запустите `make eval-m1`")
        return
    cols = st.columns(4)
    cols[0].metric("accuracy@year", f"{metrics['accuracy_year']:.1%}")
    cols[1].metric("accuracy@(year,season)", f"{metrics['accuracy_year_season']:.1%}")
    cols[2].metric("macro-F1", f"{metrics['macro_f1_year_season']:.3f}")
    cols[3].metric("MAE, мес.", f"{metrics['mae_months']:.2f}")


def _build_dataframe(predictions: dict[str, dict]) -> pd.DataFrame:
    rows = [
        {
            "id": sid,
            "true_year": meta["true_year"],
            "true_season": meta["true_season"],
            "predicted_year": meta["predicted_year"],
            "predicted_season": meta["predicted_season"],
            "true_label": meta["true_label"],
            "predicted_label": meta["predicted_label"],
            "correct": meta["true_label"] == meta["predicted_label"],
            "year_correct": meta["true_year"] == meta["predicted_year"],
            "stripped_path": meta["stripped_path"],
        }
        for sid, meta in predictions.items()
    ]
    return pd.DataFrame(rows)


def _render_timeline(df: pd.DataFrame) -> None:
    st.subheader("Распределение предсказаний по (год, сезон)")
    counts = (
        df.groupby(["predicted_year", "predicted_season"])
        .size()
        .reset_index(name="count")
    )
    counts["season_order"] = counts["predicted_season"].map(SEASON_ORDER)
    counts = counts.sort_values(["predicted_year", "season_order"])
    counts["bucket"] = counts["predicted_year"].astype(str) + "-" + counts["predicted_season"]
    st.bar_chart(counts.set_index("bucket")["count"])

    st.subheader("Точность по годам")
    year_acc = df.groupby("true_year").apply(
        lambda g: pd.Series({"accuracy": g["correct"].mean(), "n": len(g)})
    )
    st.dataframe(year_acc.style.format({"accuracy": "{:.1%}", "n": "{:.0f}"}))


def _render_per_class(df: pd.DataFrame) -> None:
    st.subheader("Per-class точность по (год, сезон)")
    per_class = df.groupby("true_label").apply(
        lambda g: pd.Series({"n": len(g), "accuracy": g["correct"].mean()})
    )
    st.dataframe(per_class.style.format({"accuracy": "{:.1%}", "n": "{:.0f}"}))


def _render_confusion(df: pd.DataFrame) -> None:
    st.subheader("Confusion matrix")
    image_path = REPORTS_DIR / "metrics_m1_confusion.png"
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.info("Запустите `make eval-m1` чтобы сгенерировать матрицу")


def _render_browser(predictions: dict[str, dict]) -> None:
    st.subheader("Браузер тестовых фото")
    options = sorted(predictions.keys())
    selected = st.selectbox("Выберите фото", options)
    meta = predictions[selected]
    cols = st.columns([1, 1])
    with cols[0]:
        path = Path(meta["stripped_path"])
        if path.exists():
            with Image.open(path) as img:
                st.image(img, caption=path.name, use_container_width=True)
        else:
            st.warning(f"Файл не найден: {path}")
    with cols[1]:
        st.markdown(f"**Предсказание**: {meta['predicted_year']}, {SEASON_RU[meta['predicted_season']]}")
        st.markdown(
            f"**Истина**: {meta['true_year']}, {SEASON_RU[meta['true_season']]} — {meta['true_timestamp']}"
        )
        st.markdown(f"**Источник GT**: {meta['source']}")
        st.markdown("**Top-3 классов**:")
        for entry in meta["top3"]:
            st.markdown(f"- {entry['label']} — {entry['proba']:.1%}")


if __name__ == "__main__":
    main()

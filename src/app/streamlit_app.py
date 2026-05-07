"""Streamlit-демо для TRACE: визуализация предсказаний и метрик.

Запуск: `uv run streamlit run src/app/streamlit_app.py`.

Поддерживает переключение между моделями M1 и M2, сравнение метрик, интерактивные
графики на Plotly, браузер тестовых фото с разбором M2-признаков (faces, OCR).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.config import DATA_DIR, FEATURES_DIR, REPORTS_DIR

SEASONS_EN = ("winter", "spring", "summer", "autumn")
SEASON_RU = {"winter": "Зима", "spring": "Весна", "summer": "Лето", "autumn": "Осень"}
SEASON_ORDER = {s: i for i, s in enumerate(SEASONS_EN)}
SEASON_EMOJI = {"winter": "❄️", "spring": "🌱", "summer": "☀️", "autumn": "🍂"}

# Палитра — мягкие оттенки сезонов
SEASON_COLORS = {
    "winter": "#7AB8E0",
    "spring": "#9FCB7E",
    "summer": "#F2C94C",
    "autumn": "#E89B5A",
}

OCR_YEAR_OFFSET = 2005  # совпадает с SUPPORTED_YEAR_RANGE.start в src/features/ocr.py


@dataclass(frozen=True)
class ModelView:
    """Описывает один вариант модели (M1/M2/consensus) — пути к артефактам."""

    label: str
    predictions_path: Path
    metrics_path: Path
    confusion_image: Path


MODEL_VIEWS: tuple[ModelView, ...] = (
    ModelView(
        "M2 (full: ResNet + цвет + свет + faces + OCR)",
        DATA_DIR / "predictions_m2.json",
        REPORTS_DIR / "metrics_m2.json",
        REPORTS_DIR / "metrics_m2_confusion.png",
    ),
    ModelView(
        "M1 (core: ResNet + цвет + свет)",
        DATA_DIR / "predictions_m1.json",
        REPORTS_DIR / "metrics_m1.json",
        REPORTS_DIR / "metrics_m1_confusion.png",
    ),
    ModelView(
        "M2 + cluster consensus",
        DATA_DIR / "predictions_m2_consensus.json",
        REPORTS_DIR / "metrics_m2_consensus.json",
        REPORTS_DIR / "metrics_m2_consensus_confusion.png",
    ),
)


def main() -> None:
    st.set_page_config(
        page_title="TRACE Timeline",
        page_icon="🕰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    st.title("🕰️ TRACE — Timeline Reconstructor")
    st.caption(
        "Восстановление хронологии (год + сезон) фото без EXIF "
        "на основе классических CV/ML инструментов."
    )

    chosen_view = _sidebar_model_picker()
    if not chosen_view.predictions_path.exists():
        _missing_view_warning(chosen_view)
        return

    predictions = _load_json(chosen_view.predictions_path)
    metrics = _load_json(chosen_view.metrics_path) if chosen_view.metrics_path.exists() else None
    df = _build_dataframe(predictions)
    df = _apply_sidebar_filters(df)

    _render_kpi_block(metrics, chosen_view, df)

    tabs = st.tabs(
        ["📊 Обзор", "📅 Таймлайн", "🎯 Матрица ошибок", "🔍 Браузер фото", "⚠️ Где модель ошибается"]
    )
    with tabs[0]:
        _render_overview(df, metrics)
    with tabs[1]:
        _render_timeline(df)
    with tabs[2]:
        _render_confusion(df, chosen_view)
    with tabs[3]:
        _render_browser(df, predictions)
    with tabs[4]:
        _render_errors(df)


def _inject_styles() -> None:
    """Лёгкая шлифовка визуального стиля поверх дефолта Streamlit."""
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
            div[data-testid="stMetricValue"] { font-size: 1.6rem; }
            div[data-testid="stMetricDelta"] { font-size: 0.9rem; }
            section[data-testid="stSidebar"] { background-color: #fafbfc; }
            .stTabs [data-baseweb="tab-list"] button { padding: 0.5rem 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_model_picker() -> ModelView:
    st.sidebar.header("⚙️ Модель и фильтры")
    available = [v for v in MODEL_VIEWS if v.predictions_path.exists()]
    if not available:
        st.sidebar.error("Не найдено ни одного файла предсказаний")
        return MODEL_VIEWS[0]

    chosen_label = st.sidebar.radio(
        "Какой вариант модели смотрим",
        options=[v.label for v in available],
        index=0,
        key="model_view",
    )
    return next(v for v in available if v.label == chosen_label)


def _missing_view_warning(view: ModelView) -> None:
    st.error(f"Не найден файл предсказаний: `{view.predictions_path}`")
    st.info(
        "Запустите пайплайн:\n```\nmake prepare\nmake features-core\n"
        "make train-m1\nmake predict-m1\nmake eval-m1\n```\n"
        "Для M2 — также `make features-full && make train-m2 && make predict-m2 && make eval-m2`."
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_dataframe(predictions: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for sid, meta in predictions.items():
        rows.append(
            {
                "id": sid,
                "true_year": meta["true_year"],
                "true_season": meta["true_season"],
                "true_label": meta["true_label"],
                "predicted_year": meta["predicted_year"],
                "predicted_season": meta["predicted_season"],
                "predicted_label": meta["predicted_label"],
                "correct": meta["true_label"] == meta["predicted_label"],
                "year_correct": meta["true_year"] == meta["predicted_year"],
                "year_diff": meta["predicted_year"] - meta["true_year"],
                "stripped_path": meta["stripped_path"],
                "source": meta.get("source", "n/a"),
                "true_timestamp": meta.get("true_timestamp", ""),
            }
        )
    return pd.DataFrame(rows)


def _apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("---")
    years = sorted(df["true_year"].unique())
    selected_years = st.sidebar.multiselect("Годы (true)", options=years, default=years)
    selected_seasons = st.sidebar.multiselect(
        "Сезоны (true)",
        options=list(SEASONS_EN),
        default=list(SEASONS_EN),
        format_func=lambda s: f"{SEASON_EMOJI[s]} {SEASON_RU[s]}",
    )
    only_errors = st.sidebar.checkbox("Только ошибочные", value=False)

    filtered = df[df["true_year"].isin(selected_years) & df["true_season"].isin(selected_seasons)]
    if only_errors:
        filtered = filtered[~filtered["correct"]]
    st.sidebar.caption(f"Фотографий в выборке: **{len(filtered)}** из {len(df)}")
    return filtered


def _render_kpi_block(
    metrics: dict | None, view: ModelView, df: pd.DataFrame
) -> None:
    st.subheader(f"📌 {view.label}")
    if metrics is None:
        st.warning("Файл метрик не найден — запустите соответствующий `make eval-*`")
        return

    baseline = _load_baseline_metrics(view)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "accuracy@year",
        f"{metrics['accuracy_year']:.1%}",
        delta=_delta(metrics, baseline, "accuracy_year", as_pp=True),
    )
    col2.metric(
        "accuracy@(год,сезон)",
        f"{metrics['accuracy_year_season']:.1%}",
        delta=_delta(metrics, baseline, "accuracy_year_season", as_pp=True),
    )
    col3.metric(
        "macro-F1",
        f"{metrics['macro_f1_year_season']:.3f}",
        delta=_delta(metrics, baseline, "macro_f1_year_season"),
    )
    col4.metric(
        "MAE, мес",
        f"{metrics['mae_months']:.2f}",
        delta=_delta(metrics, baseline, "mae_months", invert_color=True),
    )
    col5.metric("Тестовых фото в выборке", f"{len(df):,}".replace(",", " "))


def _load_baseline_metrics(view: ModelView) -> dict | None:
    """Если смотрим M2 — сравниваем с M1; иначе baseline отсутствует."""
    if "M2" not in view.label or "M1" in view.label:
        return None
    m1_path = REPORTS_DIR / "metrics_m1.json"
    return _load_json(m1_path) if m1_path.exists() else None


def _delta(metrics: dict, baseline: dict | None, key: str, *, as_pp: bool = False, invert_color: bool = False) -> str | None:
    if baseline is None:
        return None
    diff = metrics[key] - baseline[key]
    formatted = f"{diff * 100:+.1f} пп" if as_pp else f"{diff:+.3f}"
    if invert_color:
        # Для MAE: меньше — лучше. Streamlit покажет красным/зелёным в зависимости от знака,
        # поэтому переворачиваем знак, добавив "vs M1" в текст.
        formatted = f"{diff:+.2f} мес vs M1"
    else:
        formatted += " vs M1"
    return formatted


def _render_overview(df: pd.DataFrame, metrics: dict | None) -> None:
    st.subheader("Точность по истинным годам")
    year_acc = (
        df.groupby("true_year")
        .agg(accuracy=("correct", "mean"), year_acc=("year_correct", "mean"), n=("correct", "size"))
        .reset_index()
    )
    fig = px.bar(
        year_acc,
        x="true_year",
        y=["year_acc", "accuracy"],
        barmode="group",
        labels={"value": "Точность", "true_year": "Истинный год", "variable": "метрика"},
        color_discrete_map={"year_acc": "#5B9BD5", "accuracy": "#70AD47"},
    )
    fig.update_layout(height=380, legend_title_text="")
    fig.for_each_trace(
        lambda t: t.update(name="accuracy@year" if t.name == "year_acc" else "accuracy@(год,сезон)")
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Синий — попадание в год, зелёный — точное попадание (год + сезон).")

    if metrics is not None and "classification_report" in metrics:
        _render_top_classes(metrics)


def _render_top_classes(metrics: dict) -> None:
    report = metrics["classification_report"]
    rows = []
    for cls, payload in report.items():
        if not isinstance(payload, dict) or cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append(
            {"класс": cls, "n": int(payload["support"]), "F1": payload["f1-score"]}
        )
    df_cls = pd.DataFrame(rows).sort_values("F1", ascending=False)
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("##### 🥇 Топ-5 лучших классов (по F1)")
        st.dataframe(
            df_cls.head(5).style.format({"F1": "{:.3f}", "n": "{:.0f}"}),
            hide_index=True,
            width="stretch",
        )
    with col_right:
        st.markdown("##### 🪨 Топ-5 худших классов (n ≥ 5)")
        worst = df_cls[df_cls["n"] >= 5].sort_values("F1").head(5)
        st.dataframe(
            worst.style.format({"F1": "{:.3f}", "n": "{:.0f}"}),
            hide_index=True,
            width="stretch",
        )


def _render_timeline(df: pd.DataFrame) -> None:
    st.subheader("Распределение предсказаний по (год, сезон)")
    counts = (
        df.groupby(["predicted_year", "predicted_season"])
        .size()
        .reset_index(name="count")
    )
    counts["season_order"] = counts["predicted_season"].map(SEASON_ORDER)
    counts = counts.sort_values(["predicted_year", "season_order"])
    counts["bucket"] = (
        counts["predicted_year"].astype(str)
        + " "
        + counts["predicted_season"].map(SEASON_EMOJI)
    )

    fig = px.bar(
        counts,
        x="bucket",
        y="count",
        color="predicted_season",
        color_discrete_map=SEASON_COLORS,
        labels={"bucket": "(год, сезон)", "count": "Фото в test"},
    )
    fig.update_layout(height=420, showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Истина vs предсказание")
    pivot = (
        df.groupby(["true_year", "predicted_year"])
        .size()
        .reset_index(name="count")
    )
    fig2 = px.density_heatmap(
        pivot,
        x="true_year",
        y="predicted_year",
        z="count",
        color_continuous_scale="Blues",
        labels={"true_year": "Истинный год", "predicted_year": "Предсказанный год", "count": "Фото"},
    )
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, width="stretch")


def _render_confusion(df: pd.DataFrame, view: ModelView) -> None:
    st.subheader("Интерактивная матрица ошибок (год × сезон)")
    labels = sorted(set(df["true_label"].astype(str)) | set(df["predicted_label"].astype(str)))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for _, row in df.iterrows():
        matrix[label_to_idx[row["true_label"]], label_to_idx[row["predicted_label"]]] += 1

    # Нормализация по строкам — относительная вероятность
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)

    fig = go.Figure(
        data=go.Heatmap(
            z=norm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="истина: %{y}<br>предсказание: %{x}<br>"
            "доля: %{z:.1%}<br>фото: %{text}<extra></extra>",
            colorbar={"title": "доля<br>в строке"},
        )
    )
    fig.update_layout(
        height=560,
        xaxis_title="Предсказание",
        yaxis_title="Истина",
        xaxis={"side": "bottom"},
    )
    st.plotly_chart(fig, width="stretch")

    if view.confusion_image.exists():
        with st.expander("Посмотреть статичный confusion matrix (sklearn)"):
            st.image(str(view.confusion_image), width="stretch")


def _render_browser(df: pd.DataFrame, predictions: dict[str, dict]) -> None:
    st.subheader("Браузер тестовых фотографий")
    if df.empty:
        st.info("В текущей выборке нет фото — снимите фильтры в сайдбаре.")
        return

    options = df["id"].tolist()
    selected = st.selectbox(
        "Выберите фото",
        options=options,
        format_func=lambda sid: f"{sid[:12]}…  → {predictions[sid]['predicted_label']} (истина {predictions[sid]['true_label']})",
    )
    meta = predictions[selected]
    cols = st.columns([1, 1])
    with cols[0]:
        _render_photo(meta["stripped_path"])
    with cols[1]:
        _render_photo_meta(meta, selected)


def _render_photo(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        st.warning(f"Файл не найден: `{path}`")
        return
    try:
        with Image.open(path) as img:
            st.image(img, caption=path.name, width="stretch")
    except (OSError, ValueError) as exc:
        st.error(f"Не удалось открыть изображение: {exc}")


def _render_photo_meta(meta: dict, sid: str) -> None:
    correct = meta["true_label"] == meta["predicted_label"]
    badge = "✅ верно" if correct else "❌ ошибка"
    st.markdown(f"### {badge}")

    pred_season_ru = SEASON_RU.get(meta["predicted_season"], meta["predicted_season"])
    true_season_ru = SEASON_RU.get(meta["true_season"], meta["true_season"])
    pred_emoji = SEASON_EMOJI.get(meta["predicted_season"], "")
    true_emoji = SEASON_EMOJI.get(meta["true_season"], "")
    st.markdown(f"**Предсказание:** {meta['predicted_year']}, {pred_emoji} {pred_season_ru}")
    st.markdown(
        f"**Истина:** {meta['true_year']}, {true_emoji} {true_season_ru} "
        f"(`{meta.get('true_timestamp', 'n/a')}`)"
    )
    st.caption(f"Источник GT: `{meta.get('source', 'n/a')}`")

    st.markdown("**Top-3 классов от ансамбля:**")
    for entry in meta.get("top3", []):
        st.markdown(f"- `{entry['label']}` — {entry['proba']:.1%}")

    m2_meta = _load_m2_meta_for(sid)
    if m2_meta is not None:
        st.markdown("---")
        st.markdown("**🔬 M2-признаки этого фото:**")
        face_status = "обнаружены" if m2_meta["has_face"] else "не найдены"
        st.markdown(f"- Лица: **{face_status}**")
        if m2_meta["ocr_years"]:
            years_txt = ", ".join(str(y) for y in m2_meta["ocr_years"])
            st.markdown(f"- OCR извлёк годы: **{years_txt}**")
        elif m2_meta["has_text"]:
            st.markdown("- OCR нашёл текст, но без распознанного года")
        else:
            st.markdown("- OCR: текста не найдено")


@st.cache_data(show_spinner=False)
def _load_faces_cache() -> dict[str, np.ndarray] | None:
    path = FEATURES_DIR / "faces.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {str(k): v for k, v in zip(data["ids"].tolist(), data["vectors"])}


@st.cache_data(show_spinner=False)
def _load_ocr_cache() -> dict[str, np.ndarray] | None:
    path = FEATURES_DIR / "ocr.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {str(k): v for k, v in zip(data["ids"].tolist(), data["vectors"])}


def _load_m2_meta_for(sid: str) -> dict | None:
    faces = _load_faces_cache()
    ocr = _load_ocr_cache()
    if faces is None and ocr is None:
        return None

    has_face = bool(faces[sid][-1] > 0.5) if faces is not None and sid in faces else False
    if ocr is not None and sid in ocr:
        ocr_vec = ocr[sid]
        has_text = bool(ocr_vec[-1] > 0.5)
        year_one_hot = ocr_vec[:-1]
        ocr_years = [
            OCR_YEAR_OFFSET + i for i, value in enumerate(year_one_hot) if value > 0.5
        ]
    else:
        has_text = False
        ocr_years = []

    return {"has_face": has_face, "has_text": has_text, "ocr_years": ocr_years}


def _render_errors(df: pd.DataFrame) -> None:
    st.subheader("Самые частые ошибочные пары (истина → предсказание)")
    errors = df[~df["correct"]]
    if errors.empty:
        st.success("В текущей выборке нет ошибок 🎉")
        return

    pair_counts = Counter(
        zip(errors["true_label"], errors["predicted_label"])
    )
    pairs_df = pd.DataFrame(
        [
            {"истина": true, "предсказание": pred, "n": count}
            for (true, pred), count in pair_counts.most_common(20)
        ]
    )
    st.dataframe(pairs_df, hide_index=True, width="stretch")

    st.subheader("Распределение ошибок по разнице лет")
    diff_counts = (
        errors["year_diff"]
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"index": "Δ лет", "year_diff": "n"})
    )
    diff_counts.columns = ["Δ лет", "n"]
    fig = px.bar(diff_counts, x="Δ лет", y="n", color_discrete_sequence=["#E89B5A"])
    fig.update_layout(height=320, xaxis_tickmode="linear")
    st.plotly_chart(fig, width="stretch")
    st.caption("Δ = predicted_year − true_year. 0 на этой диаграмме нет — это были корректные предсказания.")


if __name__ == "__main__":
    main()

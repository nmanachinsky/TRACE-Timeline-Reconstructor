"""Компоненты шагов Wizard-интерфейса TRACE.

Каждая функция `render_step_*` отвечает за один шаг мастера и работает с
`st.session_state` как single source of truth. Состояние прогрессивно
накапливается: шаг N разблокирован, только если шаг N-1 успешно завершён.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.file_picker import is_picker_available, pick_directory
from src.app.theme import (
    PRIMARY,
    SEASON_COLORS,
    SEASON_EMOJI,
    SEASON_RU,
    apply_plotly_layout,
    badge,
)
from src.pipeline.wizard import (
    InferenceResult,
    ReferenceAnalysis,
    TargetAnalysis,
    TrainingResult,
    WizardConfig,
    WizardError,
    analyze_reference,
    analyze_target,
    apply_predictions,
    infer_target,
    train_reference_profile,
    write_predictions_dump,
)

STATE_KEY_REFERENCE_PATH = "wizard_reference_path"
STATE_KEY_TARGET_PATH = "wizard_target_path"
STATE_KEY_OUTPUT_PATH = "wizard_output_path"
STATE_KEY_REFERENCE_ANALYSIS = "wizard_reference_analysis"
STATE_KEY_TARGET_ANALYSIS = "wizard_target_analysis"
STATE_KEY_TRAINING = "wizard_training_result"
STATE_KEY_INFERENCE = "wizard_inference_result"


# --- Step 1: Data input -------------------------------------------------------


def render_step_data_input() -> None:
    st.markdown("### Шаг 1. Выбор данных")
    st.caption(
        "Укажите две папки: эталонную с фото, у которых сохранены EXIF/JSON-метаданные, "
        "и целевую — с фото без метаданных, которые нужно отсортировать."
    )

    col_ref, col_target = st.columns(2)
    with col_ref:
        _render_path_input(
            label="📂 Reference Dir (эталонные фото с EXIF/JSON)",
            state_key=STATE_KEY_REFERENCE_PATH,
            picker_title="Выберите Reference Dir",
            placeholder=r"например, D:\Photos\Archive_2018",
            help_text="Здесь должны быть фото, у которых есть JSON-сайдкары Google Takeout "
            "или дата в имени файла — приложение использует их как обучающую выборку.",
        )
    with col_target:
        _render_path_input(
            label="📁 Target Dir (фото без метаданных)",
            state_key=STATE_KEY_TARGET_PATH,
            picker_title="Выберите Target Dir",
            placeholder=r"например, D:\Photos\Unsorted",
            help_text="Папка с фотографиями, у которых утрачены EXIF и осмысленные имена. "
            "Эти фото будут получать предсказание (год, сезон).",
        )

    if st.button("🔍 Проанализировать файлы", type="primary"):
        _run_analysis()

    _render_analysis_summary()


def _run_analysis() -> None:
    reference_path = _resolved_path(st.session_state.get(STATE_KEY_REFERENCE_PATH))
    target_path = _resolved_path(st.session_state.get(STATE_KEY_TARGET_PATH))
    if reference_path is None or target_path is None:
        st.error("Заполните обе папки — Reference и Target")
        return

    try:
        with st.spinner("Сканирую Reference Dir и извлекаю метаданные…"):
            reference = analyze_reference(reference_path)
        with st.spinner("Сканирую Target Dir…"):
            target = analyze_target(target_path)
    except WizardError as exc:
        st.error(str(exc))
        return

    st.session_state[STATE_KEY_REFERENCE_ANALYSIS] = reference
    st.session_state[STATE_KEY_TARGET_ANALYSIS] = target
    st.session_state.pop(STATE_KEY_TRAINING, None)
    st.session_state.pop(STATE_KEY_INFERENCE, None)


def _render_analysis_summary() -> None:
    reference: ReferenceAnalysis | None = st.session_state.get(STATE_KEY_REFERENCE_ANALYSIS)
    target: TargetAnalysis | None = st.session_state.get(STATE_KEY_TARGET_ANALYSIS)
    if reference is None or target is None:
        st.info("Нажмите «Проанализировать файлы», чтобы увидеть сводку.")
        return

    st.markdown("#### Сводка по эталонной папке")
    cols = st.columns(4)
    cols[0].metric("Всего изображений", _humanize(reference.total_image_files))
    cols[1].metric("С метаданными", _humanize(reference.files_with_metadata))
    cols[2].metric("Лет покрыто", len(reference.years))
    cols[3].metric("Классов (год-сезон)", len(reference.class_counts))

    if not reference.files:
        st.warning(
            "В Reference Dir не найдено фото с метаданными. "
            "Положите туда фото с EXIF/JSON-сайдкарами или с датой в имени."
        )
        return

    years_str = ", ".join(str(y) for y in reference.years)
    seasons_str = ", ".join(
        f"{SEASON_EMOJI.get(s, '')} {SEASON_RU.get(s, s)}" for s in reference.seasons
    )
    st.markdown(f"**Годы в эталоне:** {years_str}")
    st.markdown(f"**Сезоны в эталоне:** {seasons_str}")

    sources = Counter(f.source for f in reference.files)
    sources_str = ", ".join(f"{name}: {count}" for name, count in sources.most_common())
    st.caption(f"Источники меток: {sources_str}")

    with st.expander("Распределение по классам"):
        rows = [
            {"класс": cls, "n": cnt} for cls, cnt in sorted(reference.class_counts.items())
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if reference.skipped_paths:
        with st.expander(f"Файлы без распознанной даты ({len(reference.skipped_paths)})"):
            st.caption(
                "Дата не извлекается ни из EXIF/JSON-сайдкара, ни из имени файла, "
                "ни из имени родительской папки. Эти фото не используются для обучения."
            )
            for p in reference.skipped_paths[:200]:
                st.code(str(p))

    st.markdown("#### Сводка по целевой папке")
    cols2 = st.columns(2)
    cols2[0].metric("Всего изображений в Target", _humanize(target.total_image_files))
    cols2[1].metric(
        "Reference Dir и Target Dir совпадают?",
        "да" if reference.directory == target.directory else "нет",
    )


# --- Step 2: Train ------------------------------------------------------------


def render_step_train(config: WizardConfig) -> None:
    st.markdown("### Шаг 2. Обучение визуального профиля")
    st.caption(
        "Извлекаем признаки из эталонной папки, делим её на train/validation, "
        "обучаем ансамбль и оцениваем качество. Никакие веса не скачиваются — "
        "профиль формируется заново под вашу подборку."
    )

    reference: ReferenceAnalysis | None = st.session_state.get(STATE_KEY_REFERENCE_ANALYSIS)
    if reference is None:
        st.info("Сначала пройдите шаг 1.")
        return
    if reference.files_with_metadata == 0:
        st.error("Нет фото с метаданными в Reference Dir — обучаться не на чем.")
        return

    if st.button("🧠 Извлечь признаки и Обучить", type="primary"):
        _run_training(config, reference)

    _render_training_summary()


def _run_training(config: WizardConfig, reference: ReferenceAnalysis) -> None:
    progress_bar = st.progress(0.0, text="Подготовка…")

    def progress_cb(fraction: float, message: str) -> None:
        progress_bar.progress(fraction, text=message)

    try:
        result = train_reference_profile(config, reference, progress_cb=progress_cb)
    except WizardError as exc:
        progress_bar.empty()
        st.error(str(exc))
        return
    finally:
        progress_bar.empty()

    st.session_state[STATE_KEY_TRAINING] = result
    st.session_state.pop(STATE_KEY_INFERENCE, None)
    st.success("Профиль обучен")


def _render_training_summary() -> None:
    training: TrainingResult | None = st.session_state.get(STATE_KEY_TRAINING)
    if training is None:
        st.info("Нажмите «Извлечь признаки и Обучить», чтобы получить метрики.")
        return

    metrics = training.val_metrics
    st.markdown("#### Метрики на валидации (30% эталонной выборки)")
    cols = st.columns(5)
    cols[0].metric("Train-фото", _humanize(metrics.n_train))
    cols[1].metric("Val-фото", _humanize(metrics.n_val))
    cols[2].metric("Точность по году", f"{metrics.accuracy_year:.1%}" if metrics.n_val else "—")
    cols[3].metric(
        "Точность (год+сезон)",
        f"{metrics.accuracy_year_season:.1%}" if metrics.n_val else "—",
    )
    cols[4].metric("MAE, мес", f"{metrics.mae_months:.2f}" if metrics.n_val else "—")

    if metrics.n_val < 2 or metrics.confusion_matrix.size == 0:
        st.warning(
            "Валидационная выборка пустая или слишком мала — метрики не считаются. "
            "Это нормально для маленьких эталонов; продолжайте к Шагу 3."
        )
        return

    _render_confusion(metrics.confusion_matrix, list(metrics.confusion_labels))
    _render_per_class_table(metrics.classification_report)


def _render_confusion(matrix: np.ndarray, labels: list[str]) -> None:
    if matrix.size == 0:
        return
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
    fig.update_layout(xaxis_title="Предсказание", yaxis_title="Истина")
    apply_plotly_layout(fig, height=480)
    st.plotly_chart(fig, width="stretch")


def _render_per_class_table(report: dict) -> None:
    if not report:
        return
    rows = []
    for cls, payload in report.items():
        if not isinstance(payload, dict):
            continue
        if cls in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "класс": cls,
                "n": int(payload.get("support", 0)),
                "precision": payload.get("precision", 0.0),
                "recall": payload.get("recall", 0.0),
                "f1": payload.get("f1-score", 0.0),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    with st.expander("Метрики по классам"):
        st.dataframe(
            df.style.format(
                {"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}", "n": "{:.0f}"}
            ),
            hide_index=True,
            width="stretch",
        )


# --- Step 3: Inference --------------------------------------------------------


def render_step_inference(config: WizardConfig) -> None:
    st.markdown("### Шаг 3. Восстановление хронологии")
    st.caption(
        "Извлекаем признаки из Target Dir, прогоняем через обученный классификатор "
        "и (опционально) сглаживаем предсказания через HDBSCAN-кластеры."
    )

    training: TrainingResult | None = st.session_state.get(STATE_KEY_TRAINING)
    target: TargetAnalysis | None = st.session_state.get(STATE_KEY_TARGET_ANALYSIS)
    if training is None or target is None:
        st.info("Сначала пройдите шаги 1 и 2.")
        return
    if target.total_image_files == 0:
        st.error("В Target Dir нет поддерживаемых изображений.")
        return

    if st.button("🕰️ Восстановить хронологию", type="primary"):
        _run_inference(config, training, target)

    _render_inference_summary()


def _run_inference(
    config: WizardConfig, training: TrainingResult, target: TargetAnalysis
) -> None:
    progress_bar = st.progress(0.0, text="Подготовка…")

    def progress_cb(fraction: float, message: str) -> None:
        progress_bar.progress(fraction, text=message)

    try:
        result = infer_target(config, training, target, progress_cb=progress_cb)
    except WizardError as exc:
        progress_bar.empty()
        st.error(str(exc))
        return
    finally:
        progress_bar.empty()

    st.session_state[STATE_KEY_INFERENCE] = result
    st.success(f"Готово. Обработано фото: {len(result.predictions)}")


def _render_inference_summary() -> None:
    result: InferenceResult | None = st.session_state.get(STATE_KEY_INFERENCE)
    if result is None:
        st.info("Нажмите «Восстановить хронологию», чтобы увидеть результат.")
        return

    df = _predictions_to_dataframe(result)
    cols = st.columns(4)
    cols[0].metric("Всего фото", _humanize(len(df)))
    cols[1].metric("Средняя уверенность", f"{df['confidence'].mean():.1%}")
    cols[2].metric(
        "Высокая уверенность (≥80%)",
        _humanize(int((df["confidence"] >= 0.8).sum())),
    )
    cols[3].metric(
        "Кластеров (HDBSCAN)" if result.consensus_applied else "Consensus",
        _humanize(result.cluster_count) if result.consensus_applied else "выкл",
    )

    tabs = st.tabs(["📋 Таблица", "📊 Распределение по годам"])
    with tabs[0]:
        _render_predictions_table(df)
    with tabs[1]:
        _render_year_histogram(df)


def _render_predictions_table(df: pd.DataFrame) -> None:
    view = df.assign(
        сезон=df["season"].map(lambda s: f"{SEASON_EMOJI.get(s, '')} {SEASON_RU.get(s, s)}"),
        confidence_pct=df["confidence"].map(lambda v: f"{v:.1%}"),
    )[["filename", "year", "сезон", "confidence_pct", "top3_str", "path"]].rename(
        columns={
            "filename": "Файл",
            "year": "Год",
            "confidence_pct": "Уверенность",
            "top3_str": "Top-3",
            "path": "Путь",
        }
    )
    st.dataframe(view, hide_index=True, width="stretch")
    csv = view.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "💾 Скачать CSV", data=csv, file_name="trace_predictions.csv", mime="text/csv"
    )


def _render_year_histogram(df: pd.DataFrame) -> None:
    counts = (
        df.groupby(["year", "season"]).size().reset_index(name="count")
    )
    if counts.empty:
        st.info("Нет данных для построения гистограммы.")
        return
    counts["bucket"] = counts["year"].astype(str) + " " + counts["season"].map(SEASON_EMOJI)
    counts = counts.sort_values(["year", "season"])
    fig = px.bar(
        counts,
        x="bucket",
        y="count",
        color="season",
        color_discrete_map=SEASON_COLORS,
        labels={"bucket": "(год, сезон)", "count": "Фото", "season": "Сезон"},
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    apply_plotly_layout(fig, height=420)
    st.plotly_chart(fig, width="stretch")


# --- Step 4: Apply ------------------------------------------------------------


def render_step_apply(config: WizardConfig) -> None:
    st.markdown("### Шаг 4. Раскладка по папкам")
    st.caption(
        "Копируем или переносим target-фото в `<Output Dir>/YYYY-Season/`. "
        "Файлы ниже выбранного порога уверенности пропускаются."
    )

    result: InferenceResult | None = st.session_state.get(STATE_KEY_INFERENCE)
    if result is None:
        st.info("Сначала пройдите шаг 3.")
        return

    _render_path_input(
        label="📁 Output Dir (куда раскладывать)",
        state_key=STATE_KEY_OUTPUT_PATH,
        picker_title="Выберите Output Dir",
        placeholder=r"например, D:\Photos\Sorted",
        help_text="Папка будет создана при необходимости. Внутри появятся подпапки YYYY-season.",
    )

    confidence_threshold = st.slider(
        "Минимальный порог уверенности",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Фото с уверенностью ниже порога не трогаются.",
    )
    operation = st.radio(
        "Операция",
        options=["copy", "move"],
        horizontal=True,
        format_func=lambda v: "Копировать (безопасно)" if v == "copy" else "Переместить",
    )

    eligible_count = sum(1 for p in result.predictions if p.confidence >= confidence_threshold)
    st.caption(
        f"Будет обработано **{eligible_count}** из {len(result.predictions)} фото "
        f"при текущем пороге."
    )

    output_path = _resolved_path(st.session_state.get(STATE_KEY_OUTPUT_PATH))
    if output_path is None:
        st.warning("Укажите Output Dir.")
        return

    if st.button("📦 Разложить по папкам", type="primary"):
        _run_apply(result, output_path, operation, confidence_threshold)


def _run_apply(
    result: InferenceResult,
    output_path: Path,
    operation: str,
    threshold: float,
) -> None:
    try:
        with st.spinner(f"{'Копирую' if operation == 'copy' else 'Переношу'} файлы…"):
            report = apply_predictions(
                result.predictions,
                output_path,
                operation=operation,
                confidence_threshold=threshold,
            )
            write_predictions_dump(result.predictions, output_path / "_trace_predictions.json")
    except WizardError as exc:
        st.error(str(exc))
        return

    st.success(
        f"Готово: обработано **{report.moved}**, пропущено **{report.skipped}**, "
        f"ошибок **{report.errors}**."
    )
    if report.errors:
        with st.expander(f"Лог ошибок ({report.errors})"):
            for line in report.error_log:
                st.code(line, language="bash")


# --- Shared helpers -----------------------------------------------------------


def _render_path_input(
    *,
    label: str,
    state_key: str,
    picker_title: str,
    placeholder: str,
    help_text: str,
) -> None:
    """Поле ввода пути + кнопка «Обзор» с системным диалогом выбора папки."""
    col_input, col_button = st.columns([6, 1])
    with col_input:
        st.text_input(
            label,
            key=state_key,
            placeholder=placeholder,
            help=help_text,
            label_visibility="visible",
        )
    with col_button:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        picker_disabled = not is_picker_available()
        if st.button(
            "📂 Обзор…",
            key=f"{state_key}_picker",
            disabled=picker_disabled,
            help=(
                "Открыть системный проводник"
                if not picker_disabled
                else "Tk недоступен — введите путь вручную"
            ),
        ):
            chosen = pick_directory(picker_title)
            if chosen is not None:
                st.session_state[state_key] = str(chosen)
                st.rerun()


def _resolved_path(value: str | None) -> Path | None:
    if not value:
        return None
    expanded = Path(value).expanduser()
    return expanded if expanded.exists() else None


def _humanize(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _predictions_to_dataframe(result: InferenceResult) -> pd.DataFrame:
    rows = []
    for p in result.predictions:
        d = asdict(p)
        d["path"] = str(p.path)
        d["filename"] = p.path.name
        d["top3_str"] = ", ".join(f"{lbl} {prob:.0%}" for lbl, prob in p.top3) if p.top3 else ""
        rows.append(d)
    return pd.DataFrame(rows)


# --- Wizard hero --------------------------------------------------------------


def render_wizard_status_banner(active_step: int) -> None:
    """Лента из 4 чипов-статусов сверху над контентом."""
    titles = (
        "1. Данные",
        "2. Профиль",
        "3. Хронология",
        "4. Раскладка",
    )
    chips: list[str] = []
    for idx, title in enumerate(titles, start=1):
        kind = "good" if idx < active_step else ("warn" if idx == active_step else "")
        if kind:
            chips.append(badge(title, kind))
        else:
            chips.append(
                f"<span class='trace-badge' style='background:#E2E8F0;color:{PRIMARY};'>"
                f"{title}</span>"
            )
    st.markdown(
        " &nbsp; ".join(chips),
        unsafe_allow_html=True,
    )

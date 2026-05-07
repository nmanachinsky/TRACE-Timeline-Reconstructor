"""Точка входа Streamlit-приложения TRACE — Personal Archive Restorer.

Единый Wizard из четырёх шагов:
1. Выбор Reference Dir и Target Dir.
2. Извлечение признаков из Reference Dir и обучение визуального профиля
   с валидацией на 30%-выборке.
3. Inference на Target Dir с опциональным cluster-consensus.
4. Раскладка целевых файлов по подпапкам YYYY-Season.

В сайдбаре — глобальные настройки модели (M1/M2, кластеризация). Никаких
скачиваний весов: профиль формируется заново под каждую reference-папку.

Запуск: `uv run streamlit run src/app/streamlit_app.py` или `make app`.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.app.theme import hero, inject_styles
from src.app.wizard_steps import (
    STATE_KEY_INFERENCE,
    STATE_KEY_REFERENCE_ANALYSIS,
    STATE_KEY_REFERENCE_PATH,
    STATE_KEY_TARGET_PATH,
    STATE_KEY_TRAINING,
    render_step_apply,
    render_step_data_input,
    render_step_inference,
    render_step_train,
    render_wizard_status_banner,
)
from src.config import CLUSTERING, RESNET, SPLIT
from src.pipeline.wizard import DEFAULT_WORK_ROOT, WizardConfig

SIDEBAR_KEY_USE_M2 = "wizard_use_m2"
SIDEBAR_KEY_USE_CONSENSUS = "wizard_use_consensus"
SIDEBAR_KEY_CONSENSUS_WEIGHT = "wizard_consensus_weight"
SIDEBAR_KEY_CLUSTER_MIN_SIZE = "wizard_cluster_min_size"
SIDEBAR_KEY_PCA_COMPONENTS = "wizard_pca_components"
SIDEBAR_KEY_TEST_SIZE = "wizard_test_size"
SIDEBAR_KEY_BATCH_SIZE = "wizard_batch_size"
SIDEBAR_KEY_USE_GPU = "wizard_use_gpu"


def main() -> None:
    st.set_page_config(
        page_title="TRACE — Personal Archive Restorer",
        page_icon="🕰️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "TRACE — Personal Archive Restorer. Локальное восстановление "
                "хронологии личных фотоархивов."
            )
        },
    )
    inject_styles()
    _render_hero()
    _render_sidebar()

    config = _build_config_from_session()
    active_step = _detect_active_step()
    render_wizard_status_banner(active_step)

    st.markdown("---")
    render_step_data_input()

    st.markdown("---")
    render_step_train(config)

    st.markdown("---")
    render_step_inference(config)

    st.markdown("---")
    render_step_apply(config)


def _render_hero() -> None:
    hero(
        "🕰️ TRACE — Personal Archive Restorer",
        "Локальное восстановление хронологии личных фотоархивов. "
        "TRACE учится «на лету» на части архива с EXIF/JSON-метаданными "
        "(одежда, интерьеры, питомцы, особенности камеры) и применяет полученный "
        "профиль для сортировки фотографий, потерявших метаданные.",
    )


def _render_sidebar() -> None:
    st.sidebar.subheader("⚙️ Параметры модели")

    st.sidebar.toggle(
        "M2: добавить лица + OCR (медленнее)",
        key=SIDEBAR_KEY_USE_M2,
        value=False,
        help=(
            "M1 — ResNet + цвет + освещение (быстро). "
            "M2 — те же признаки + face-эмбеддинги insightface + OCR EasyOCR. "
            "M2 точнее, но требует optional-deps `m2` (uv sync --extra m2)."
        ),
    )

    st.sidebar.toggle(
        "GPU для тяжёлых моделей (если есть CUDA)",
        key=SIDEBAR_KEY_USE_GPU,
        value=False,
    )
    st.sidebar.slider(
        "Batch size (ResNet inference)",
        min_value=4,
        max_value=64,
        value=RESNET.batch_size,
        step=4,
        key=SIDEBAR_KEY_BATCH_SIZE,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Train / Validation split")
    st.sidebar.slider(
        "Доля валидации (от Reference Dir)",
        min_value=0.10,
        max_value=0.50,
        value=SPLIT.test_size,
        step=0.05,
        key=SIDEBAR_KEY_TEST_SIZE,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔗 Cluster Consensus (HDBSCAN)")
    st.sidebar.toggle(
        "Включить consensus",
        key=SIDEBAR_KEY_USE_CONSENSUS,
        value=True,
        help=(
            "Сглаживает предсказания: фото внутри одного визуального кластера "
            "получают «голос» соседей с известными метками."
        ),
    )
    st.sidebar.slider(
        "Сила consensus",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        key=SIDEBAR_KEY_CONSENSUS_WEIGHT,
    )
    st.sidebar.slider(
        "Минимальный размер кластера",
        min_value=2,
        max_value=20,
        value=CLUSTERING.min_cluster_size,
        step=1,
        key=SIDEBAR_KEY_CLUSTER_MIN_SIZE,
    )
    st.sidebar.slider(
        "PCA компонент для HDBSCAN",
        min_value=16,
        max_value=512,
        value=CLUSTERING.pca_components,
        step=16,
        key=SIDEBAR_KEY_PCA_COMPONENTS,
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Сбросить состояние мастера"):
        _reset_session_state()
        st.rerun()


def _build_config_from_session() -> WizardConfig:
    reference = _resolved_path(st.session_state.get(STATE_KEY_REFERENCE_PATH))
    target = _resolved_path(st.session_state.get(STATE_KEY_TARGET_PATH))
    output = _resolved_path(st.session_state.get("wizard_output_path"))

    return WizardConfig(
        reference_dir=reference or Path("."),
        target_dir=target or Path("."),
        output_dir=output or Path("."),
        use_m2=bool(st.session_state.get(SIDEBAR_KEY_USE_M2, False)),
        use_gpu=bool(st.session_state.get(SIDEBAR_KEY_USE_GPU, False)),
        batch_size=int(st.session_state.get(SIDEBAR_KEY_BATCH_SIZE, RESNET.batch_size)),
        test_size=float(st.session_state.get(SIDEBAR_KEY_TEST_SIZE, SPLIT.test_size)),
        use_consensus=bool(st.session_state.get(SIDEBAR_KEY_USE_CONSENSUS, True)),
        consensus_weight=float(st.session_state.get(SIDEBAR_KEY_CONSENSUS_WEIGHT, 0.4)),
        cluster_min_size=int(
            st.session_state.get(SIDEBAR_KEY_CLUSTER_MIN_SIZE, CLUSTERING.min_cluster_size)
        ),
        cluster_pca_components=int(
            st.session_state.get(SIDEBAR_KEY_PCA_COMPONENTS, CLUSTERING.pca_components)
        ),
        work_root=DEFAULT_WORK_ROOT,
    )


def _resolved_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser()


def _detect_active_step() -> int:
    if STATE_KEY_INFERENCE in st.session_state:
        return 4
    if STATE_KEY_TRAINING in st.session_state:
        return 3
    if STATE_KEY_REFERENCE_ANALYSIS in st.session_state:
        return 2
    return 1


def _reset_session_state() -> None:
    keys = [
        STATE_KEY_REFERENCE_PATH,
        STATE_KEY_TARGET_PATH,
        STATE_KEY_REFERENCE_ANALYSIS,
        "wizard_target_analysis",
        STATE_KEY_TRAINING,
        STATE_KEY_INFERENCE,
        "wizard_output_path",
    ]
    for key in keys:
        st.session_state.pop(key, None)


if __name__ == "__main__":
    main()

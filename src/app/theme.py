"""Единая тема Streamlit-приложения: палитра, CSS-инъекции, Plotly-template.

Принципы:
- Высокий контраст: основной текст почти-чёрный (#0F172A) на светло-сером фоне.
- Сезонные цвета зафиксированы в SEASON_COLORS — используются и в графиках, и в бейджах.
- Plotly use plotly_white шаблона, явно прописываем цвет осей и подписей,
  чтобы не сливались с фоном.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Палитра ----------------------------------------------------------------------

PRIMARY = "#2563EB"          # blue-600 — primary accent
ACCENT_GOOD = "#059669"      # emerald-600 — успех / положительная дельта
ACCENT_BAD = "#DC2626"       # red-600 — ошибка / отрицательная дельта
ACCENT_WARN = "#D97706"      # amber-600 — предупреждение

TEXT_PRIMARY = "#0F172A"     # slate-900
TEXT_SECONDARY = "#475569"   # slate-600 — для caption, не сливается со светлым
TEXT_MUTED = "#64748B"       # slate-500

SURFACE_BG = "#F8FAFC"       # slate-50 — основной фон
SURFACE_CARD = "#FFFFFF"     # белые карточки
SURFACE_BORDER = "#E2E8F0"   # slate-200 — границы

SEASON_COLORS: dict[str, str] = {
    "winter": "#3B82F6",   # blue-500
    "spring": "#10B981",   # emerald-500
    "summer": "#F59E0B",   # amber-500
    "autumn": "#EA580C",   # orange-600
}

SEASON_RU: dict[str, str] = {
    "winter": "Зима",
    "spring": "Весна",
    "summer": "Лето",
    "autumn": "Осень",
}

SEASON_EMOJI: dict[str, str] = {
    "winter": "❄️",
    "spring": "🌱",
    "summer": "☀️",
    "autumn": "🍂",
}


# CSS-инъекции -----------------------------------------------------------------

_CUSTOM_CSS = f"""
<style>
    .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 600;
    }}

    /* st.caption — стандарт слишком бледный, делаем slate-600 */
    [data-testid="stCaptionContainer"], .stCaption {{
        color: {TEXT_SECONDARY} !important;
    }}

    /* Метрики — крупнее значение, сабтекст контрастнее */
    div[data-testid="stMetricValue"] {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_SECONDARY};
        font-weight: 500;
    }}
    div[data-testid="stMetricDelta"] {{
        font-size: 0.85rem;
        font-weight: 500;
    }}

    /* Карточки-метрики — добавляем рамку/подложку для разделения */
    div[data-testid="stMetric"] {{
        background-color: {SURFACE_CARD};
        border: 1px solid {SURFACE_BORDER};
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }}

    /* Сайдбар */
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE_CARD};
        border-right: 1px solid {SURFACE_BORDER};
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {TEXT_PRIMARY} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: {TEXT_MUTED} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.25rem;
        border-bottom: 1px solid {SURFACE_BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.6rem 1.1rem;
        font-weight: 500;
        color: {TEXT_SECONDARY};
    }}
    .stTabs [aria-selected="true"] {{
        color: {PRIMARY} !important;
    }}

    /* Кнопки */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 500;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {PRIMARY};
        border-color: {PRIMARY};
    }}

    /* DataFrame — лёгкая рамка, чёрный текст */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {SURFACE_BORDER};
        border-radius: 8px;
    }}

    /* Code-блоки и алёрты — text-color на тёмный, фоны не трогаем */
    code, pre {{
        color: {TEXT_PRIMARY};
    }}

    /* Trace-бейджи */
    .trace-badge {{
        display: inline-block;
        padding: 0.18rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        line-height: 1.4;
    }}
    .trace-badge-good {{ background: #D1FAE5; color: {ACCENT_GOOD}; }}
    .trace-badge-bad  {{ background: #FEE2E2; color: {ACCENT_BAD}; }}
    .trace-badge-warn {{ background: #FEF3C7; color: {ACCENT_WARN}; }}

    /* Hero-блок: серая подложка вокруг шапки */
    .trace-hero {{
        background-color: {SURFACE_CARD};
        border: 1px solid {SURFACE_BORDER};
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }}
    .trace-hero h2 {{ margin: 0 0 0.4rem 0; }}
    .trace-hero p {{ color: {TEXT_SECONDARY}; margin: 0; }}
</style>
"""


def inject_styles() -> None:
    """Вставляет общий CSS — вызывать один раз в main."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    """Заголовок раздела в карточке-обложке."""
    st.markdown(
        f"""
        <div class="trace-hero">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "good") -> str:
    """HTML-бейдж 'good'/'bad'/'warn' для рендера через st.markdown(unsafe_allow_html=True)."""
    klass = {"good": "trace-badge-good", "bad": "trace-badge-bad", "warn": "trace-badge-warn"}.get(
        kind, "trace-badge-good"
    )
    return f'<span class="trace-badge {klass}">{text}</span>'


# Plotly -----------------------------------------------------------------------


def apply_plotly_layout(fig: go.Figure, *, height: int = 380) -> go.Figure:
    """Единая стилизация для всех Plotly-графиков: контрастные оси, белый фон."""
    fig.update_layout(
        height=height,
        plot_bgcolor=SURFACE_CARD,
        paper_bgcolor=SURFACE_CARD,
        font={"color": TEXT_PRIMARY, "family": "sans-serif", "size": 13},
        margin={"l": 50, "r": 30, "t": 40, "b": 50},
        legend={"bgcolor": SURFACE_CARD, "bordercolor": SURFACE_BORDER, "borderwidth": 1},
    )
    fig.update_xaxes(gridcolor=SURFACE_BORDER, linecolor=SURFACE_BORDER, tickcolor=SURFACE_BORDER)
    fig.update_yaxes(gridcolor=SURFACE_BORDER, linecolor=SURFACE_BORDER, tickcolor=SURFACE_BORDER)
    return fig

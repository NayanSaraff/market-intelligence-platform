"""Responsive Streamlit helpers for charts, metrics, tables, and layout rows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from textwrap import dedent
from typing import Any
import html

import pandas as pd
import streamlit as st

from .table_config import render_fixed_width_html_table, render_fixed_width_table


MOBILE_BREAKPOINT = 768
TABLET_BREAKPOINT = 1200


def _user_agent() -> str:
    try:
        return str(st.context.headers.get("User-Agent", "") or "")
    except Exception:
        return ""


def get_viewport_width(viewport_width: int | None = None) -> int:
    """Return a best-effort viewport width using session overrides or user-agent hints."""
    if viewport_width is not None:
        return int(viewport_width)

    override = st.session_state.get("responsive_viewport_width")
    if isinstance(override, (int, float)) and override > 0:
        return int(override)

    ua = _user_agent().lower()
    if any(token in ua for token in ("iphone", "android", "mobile", "ipod")):
        return 390
    if any(token in ua for token in ("ipad", "tablet", "kindle")):
        return 1024
    return 1440


def is_mobile(viewport_width: int | None = None) -> bool:
    return get_viewport_width(viewport_width) < MOBILE_BREAKPOINT


def is_tablet(viewport_width: int | None = None) -> bool:
    width = get_viewport_width(viewport_width)
    return MOBILE_BREAKPOINT <= width < TABLET_BREAKPOINT


def is_desktop(viewport_width: int | None = None) -> bool:
    return get_viewport_width(viewport_width) >= TABLET_BREAKPOINT


def get_column_count(
    viewport_width: int | None = None,
    *,
    desktop: int = 4,
    tablet: int = 2,
    mobile: int = 1,
) -> int:
    width = get_viewport_width(viewport_width)
    if width < MOBILE_BREAKPOINT:
        return mobile
    if width < TABLET_BREAKPOINT:
        return tablet
    return desktop


def render_responsive_metrics(
    metrics: Sequence[dict[str, Any]],
    *,
    viewport_width: int | None = None,
    grid_min_width: int = 180,
) -> None:
    """Render KPI cards in an auto-wrapping responsive grid."""
    if not metrics:
        return

    cards = []
    for metric in metrics:
        label = html.escape(str(metric.get("label", "")))
        value = html.escape(str(metric.get("value", "")))
        accent = html.escape(str(metric.get("accent", "#00d4ff")))
        subtitle = html.escape(str(metric.get("subtitle", "")))
        tone = "pos" if metric.get("is_positive") is True else "neg" if metric.get("is_positive") is False else ""
        subtitle_html = f'<div class="responsive-metric-subtitle">{subtitle}</div>' if subtitle else ""
        cards.append(
            f'<div class="responsive-metric-card" style="border-top-color:{accent};">'
            f'<div class="responsive-metric-label">{label}</div>'
            f'<div class="responsive-metric-value {tone}" style="color:{accent};">{value}</div>'
            f'{subtitle_html}'
            f'</div>'
        )

    st.markdown(
        dedent(
            f'''
            <div class="responsive-metric-grid" style="grid-template-columns: repeat(auto-fit, minmax({grid_min_width}px, 1fr));">
                {''.join(cards)}
            </div>
            '''
        ),
        unsafe_allow_html=True,
    )


def render_responsive_table(
    df: pd.DataFrame,
    *,
    height: int = 300,
    hide_index: bool = True,
    editable: bool = False,
    use_container_width: bool = True,
) -> None:
    render_fixed_width_table(
        df,
        height=height,
        hide_index=hide_index,
        editable=editable,
        use_container_width=use_container_width,
    )


def render_responsive_html_table(df: pd.DataFrame, *, title: str | None = None, max_rows: int | None = None) -> None:
    html_table = render_fixed_width_html_table(df, title=title, max_rows=max_rows)
    if html_table:
        st.markdown(html_table, unsafe_allow_html=True)


def _apply_chart_responsiveness(fig: Any, viewport_width: int | None = None, height: int | None = None) -> int:
    width = get_viewport_width(viewport_width)
    mobile = width < MOBILE_BREAKPOINT
    tablet = MOBILE_BREAKPOINT <= width < TABLET_BREAKPOINT

    resolved_height = height or (320 if mobile else 380 if tablet else 460)
    title_size = 16 if mobile else 18 if tablet else 22
    axis_size = 10 if mobile else 11 if tablet else 12
    legend_size = 9 if mobile else 10 if tablet else 11
    margin = dict(l=12, r=12, t=42 if mobile else 48, b=20 if mobile else 24)

    if hasattr(fig, "update_layout"):
        fig.update_layout(
            height=resolved_height,
            margin=margin,
            title=dict(font=dict(size=title_size)),
            legend=dict(font=dict(size=legend_size), orientation="h" if mobile else "h"),
            font=dict(size=11 if mobile else 12),
        )
        if hasattr(fig.layout, "xaxis"):
            fig.update_xaxes(tickfont=dict(size=axis_size), automargin=True)
        if hasattr(fig.layout, "yaxis"):
            fig.update_yaxes(tickfont=dict(size=axis_size), automargin=True)

    return resolved_height


def render_responsive_chart(fig: Any, *, viewport_width: int | None = None, height: int | None = None, key: str | None = None) -> None:
    resolved_height = _apply_chart_responsiveness(fig, viewport_width=viewport_width, height=height)
    st.plotly_chart(fig, use_container_width=True, height=resolved_height, key=key)


def render_responsive_panels(
    renderers: Sequence[Callable[[], None]],
    *,
    viewport_width: int | None = None,
    desktop: int = 3,
    tablet: int = 2,
    mobile: int = 1,
    gap: str = "small",
) -> None:
    """Render a row of panels that collapses to stacked sections on smaller devices."""
    if not renderers:
        return

    count = get_column_count(viewport_width, desktop=desktop, tablet=tablet, mobile=mobile)
    if count <= 1:
        for render in renderers:
            render()
        return

    cols = st.columns(min(count, len(renderers)), gap=gap)
    for idx, render in enumerate(renderers):
        with cols[idx % len(cols)]:
            render()


def inject_responsive_css() -> None:
    st.markdown(
        dedent(
            """
<style>
:root {
    --responsive-title-desktop: 32px;
    --responsive-title-tablet: 24px;
    --responsive-title-mobile: 20px;
    --responsive-section-desktop: 24px;
    --responsive-section-tablet: 20px;
    --responsive-section-mobile: 18px;
    --responsive-body-desktop: 16px;
    --responsive-body-mobile: 14px;
    --responsive-table-desktop: 14px;
    --responsive-table-mobile: 12px;
}

.responsive-metric-grid {
    display: grid;
    gap: 12px;
    width: 100%;
    margin: 8px 0 12px;
}

.responsive-metric-card {
    background: linear-gradient(180deg, rgba(20, 26, 34, 0.98), rgba(16, 21, 28, 0.98));
    border: 1px solid #1f2a38;
    border-top: 2px solid #00d4ff;
    border-radius: 8px;
    padding: 12px 14px;
    min-height: 84px;
    overflow: hidden;
}

.responsive-metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}

.responsive-metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1rem, 2.5vw, 1.55rem);
    font-weight: 700;
    line-height: 1.05;
    word-break: break-word;
}

.responsive-metric-value.pos { color: #00ff88; }
.responsive-metric-value.neg { color: #ff3366; }

.responsive-metric-subtitle {
    margin-top: 6px;
    color: #64748b;
    font-size: 11px;
}

.responsive-html-table-title {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e6edf5;
    margin: 8px 0 10px;
    font-size: 11px;
}

.responsive-html-table-shell {
    overflow-x: auto;
    width: 100%;
    border: 1px solid #1f2a38;
    border-radius: 8px;
    background: rgba(16, 21, 28, 0.92);
}

.responsive-html-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

.responsive-html-table th,
.responsive-html-table td {
    border-bottom: 1px solid rgba(31, 42, 56, 0.9);
    padding: 10px 12px;
    overflow-wrap: anywhere;
    word-break: break-word;
    vertical-align: top;
}

.responsive-html-table th {
    position: sticky;
    top: 0;
    background: #141a22;
    z-index: 1;
    color: #00d4ff;
    text-align: left;
}

@media (max-width: 1199px) {
    .responsive-metric-card { min-height: 76px; }
    .responsive-html-table { font-size: 12px; }
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
        padding-top: 0.45rem !important;
        padding-bottom: 0.8rem !important;
    }

    h1, .main-title, [data-testid="stMarkdownContainer"] h1 {
        font-size: var(--responsive-title-mobile) !important;
        line-height: 1.15 !important;
    }

    h2, [data-testid="stMarkdownContainer"] h2 {
        font-size: var(--responsive-section-mobile) !important;
    }

    p, li, div, span {
        font-size: var(--responsive-body-mobile);
    }

    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.6rem !important;
    }

    [data-testid="column"] {
        min-width: 100% !important;
        flex-basis: 100% !important;
    }

    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 320px !important;
    }

    .stButton>button,
    .stDownloadButton>button {
        width: 100% !important;
    }

    .responsive-metric-grid {
        gap: 10px;
    }

    .responsive-metric-card {
        padding: 10px 12px;
        min-height: 72px;
    }

    .responsive-metric-value {
        font-size: clamp(0.95rem, 5vw, 1.2rem);
    }

    .responsive-html-table {
        font-size: var(--responsive-table-mobile);
    }
}

@media (max-width: 480px) {
    .ticker-tape {
        display: none !important;
    }

    .terminal-shell-inner {
        grid-template-columns: 1fr !important;
        gap: 8px !important;
    }

    .terminal-right {
        justify-content: flex-start !important;
    }

    .panel-header {
        padding: 8px 10px;
        font-size: 10px;
    }

    .responsive-metric-card {
        min-height: 68px;
    }
}
</style>
            """
        ),
        unsafe_allow_html=True,
    )
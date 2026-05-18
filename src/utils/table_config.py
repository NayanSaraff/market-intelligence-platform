"""Shared table-width policy for Streamlit dataframes, editors, and HTML tables."""

from __future__ import annotations

import html
from numbers import Number
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st


COMMON_COLUMN_WIDTHS: dict[str, int] = {
    "stock": 120,
    "ticker": 120,
    "symbol": 120,
    "company": 180,
    "actual": 110,
    "predicted": 110,
    "sector": 140,
    "industry": 150,
    "model": 130,
    "persistence": 110,
    "date": 110,
    "day": 110,
    "rmse": 100,
    "mae": 100,
    "mape": 100,
    "mse": 100,
    "r2": 90,
    "r²": 90,
    "weight": 100,
    "weight (%)": 100,
    "weight_%": 100,
    "return": 100,
    "return (%)": 100,
    "forecast_5d_ret_%": 110,
    "garch_vol_%": 110,
    "current_annvol_pct": 120,
    "var_95_pct": 110,
    "var_95_%": 110,
    "forecast_decision": 120,
    "ensemble": 110,
    "allocation": 140,
    "allocation (₹)": 140,
    "amount_inr": 140,
    "entry_price": 120,
    "shares_to_buy": 120,
    "portfolio value": 150,
    "reason": 300,
    "comments": 300,
    "comment": 280,
    "signal": 120,
    "action": 120,
    "status": 120,
    "price": 110,
    "close": 110,
    "open": 110,
    "high": 110,
    "low": 110,
    "volume": 120,
    "volatility": 120,
    "risk": 120,
    "forecast": 120,
}

DEFAULT_WIDTHS = {
    "identifier": 120,
    "numeric": 100,
    "currency": 140,
    "percentage": 100,
    "date": 110,
    "text": 240,
    "long_text": 320,
}


@dataclass(frozen=True)
class TableColumnMeta:
    name: str
    width: int
    kind: str


def _normalize_column_name(column_name: str) -> str:
    return column_name.strip().lower()


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.empty:
        return False
    sample = series.dropna().head(3)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.67


def infer_column_width(column_name: str, series: pd.Series | None = None) -> TableColumnMeta:
    normalized_name = _normalize_column_name(column_name)

    for key, width in COMMON_COLUMN_WIDTHS.items():
        if key in normalized_name:
            if any(token in key for token in ("reason", "comments", "comment")):
                kind = "long_text"
            elif any(token in key for token in ("date", "day")):
                kind = "date"
            elif any(token in key for token in ("allocation", "portfolio value")):
                kind = "currency"
            elif any(token in key for token in ("weight", "return", "%")):
                kind = "percentage"
            elif any(token in key for token in ("rmse", "mape", "mae", "mse", "r2")):
                kind = "numeric"
            elif any(token in key for token in ("stock", "ticker", "symbol", "model", "sector", "signal", "action", "status")):
                kind = "identifier"
            else:
                kind = "numeric"
            return TableColumnMeta(name=column_name, width=width, kind=kind)

    if series is not None and _is_datetime_like(series):
        return TableColumnMeta(name=column_name, width=DEFAULT_WIDTHS["date"], kind="date")

    if series is not None and pd.api.types.is_numeric_dtype(series):
        width = DEFAULT_WIDTHS["currency"] if any(token in normalized_name for token in ("allocation", "value", "price")) else DEFAULT_WIDTHS["numeric"]
        if any(token in normalized_name for token in ("weight", "return", "yield", "margin", "rate", "%")):
            width = DEFAULT_WIDTHS["percentage"]
        return TableColumnMeta(name=column_name, width=width, kind="numeric")

    sample = series.dropna().astype(str).head(20) if series is not None else pd.Series(dtype=str)
    max_content = int(sample.map(len).max()) if not sample.empty else 0
    header_len = len(column_name)
    inferred = max(header_len * 10, max_content * 8)

    if inferred >= 28:
        return TableColumnMeta(name=column_name, width=max(DEFAULT_WIDTHS["long_text"], min(inferred, 360)), kind="long_text")

    if inferred >= 18:
        return TableColumnMeta(name=column_name, width=max(DEFAULT_WIDTHS["text"], min(inferred + 80, 300)), kind="text")

    return TableColumnMeta(name=column_name, width=max(DEFAULT_WIDTHS["identifier"], min(inferred + 60, 160)), kind="identifier")


def build_fixed_column_config(df: pd.DataFrame) -> dict[str, Any]:
    column_config: dict[str, Any] = {}

    for column in df.columns:
        meta = infer_column_width(str(column), df[column])
        normalized_name = _normalize_column_name(str(column))

        if meta.kind == "date":
            column_config[column] = st.column_config.DateColumn(
                str(column),
                width=meta.width,
                alignment="center",
            )
        elif meta.kind == "percentage" or any(token in normalized_name for token in ("weight", "return", "yield", "rate", "%")):
            column_config[column] = st.column_config.NumberColumn(
                str(column),
                width=meta.width,
                format="%.2f",
                alignment="right",
            )
        elif meta.kind == "currency" or any(token in normalized_name for token in ("allocation", "portfolio value", "value", "price")):
            column_config[column] = st.column_config.NumberColumn(
                str(column),
                width=meta.width,
                format="₹%.2f",
                alignment="right",
            )
        elif meta.kind == "numeric":
            column_config[column] = st.column_config.NumberColumn(
                str(column),
                width=meta.width,
                alignment="right",
            )
        else:
            max_chars = 60 if meta.kind == "long_text" else None
            column_config[column] = st.column_config.TextColumn(
                str(column),
                width=meta.width,
                alignment="left",
                max_chars=max_chars,
            )

    return column_config


def prepare_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a DataFrame for Streamlit rendering without mutating the input."""
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[column]):
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce")
    return display_df


def _format_html_value(value: Any, column_name: str) -> str:
    if pd.isna(value):
        return ""

    normalized = _normalize_column_name(column_name)
    if "date" in normalized or "day" in normalized:
        try:
            return pd.to_datetime(value, errors="coerce").strftime("%Y-%m-%d")
        except Exception:
            return html.escape(str(value))

    if any(token in normalized for token in ("weight", "return", "yield", "rate", "%", "vol", "var")):
        try:
            return f"{float(value):.2f}%"
        except Exception:
            return html.escape(str(value))

    if any(token in normalized for token in ("allocation", "portfolio value", "amount", "value", "price", "entry")):
        try:
            return f"₹{float(value):,.2f}"
        except Exception:
            return html.escape(str(value))

    if isinstance(value, Number):
        try:
            return f"{float(value):.2f}"
        except Exception:
            return html.escape(str(value))

    return html.escape(str(value))


def render_fixed_width_table(
    df: pd.DataFrame,
    *,
    height: int = 300,
    hide_index: bool = True,
    editable: bool = False,
    use_container_width: bool = True,
) -> None:
    """Render a table with fixed column widths and stable layout."""
    if df is None or df.empty:
        return

    display_df = prepare_display_frame(df)
    column_config = build_fixed_column_config(display_df)

    if editable:
        st.data_editor(
            display_df,
            width="stretch" if use_container_width else None,
            height=height,
            hide_index=hide_index,
            column_config=column_config,
            disabled=False,
        )
    else:
        st.dataframe(
            display_df,
            width="stretch" if use_container_width else None,
            height=height,
            hide_index=hide_index,
            column_config=column_config,
        )


def render_fixed_width_html_table(df: pd.DataFrame, *, title: str | None = None, max_rows: int | None = None) -> str:
    """Return a fixed-layout HTML table string that uses the same width policy."""
    if df is None or df.empty:
        return ""

    display_df = df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)

    rows = []
    for _, row in display_df.iterrows():
        cells = []
        for column in display_df.columns:
            meta = infer_column_width(str(column), display_df[column])
            value = _format_html_value(row[column], str(column))
            align = "right" if meta.kind in {"numeric", "currency", "percentage"} else ("center" if meta.kind == "date" else "left")
            cells.append(
                f'<td style="width:{meta.width}px;text-align:{align};">{value}</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    header_cells = []
    for column in display_df.columns:
        meta = infer_column_width(str(column), display_df[column])
        header_cells.append(f'<th style="width:{meta.width}px;">{html.escape(str(column))}</th>')

    title_html = f'<div class="responsive-html-table-title">{html.escape(title)}</div>' if title else ""
    return (
        f'{title_html}'
        '<div class="responsive-html-table-shell">'
        '<table class="responsive-html-table">'
        '<thead><tr>' + "".join(header_cells) + '</tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody>'
        '</table></div>'
    )
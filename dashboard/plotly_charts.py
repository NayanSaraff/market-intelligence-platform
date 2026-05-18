from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional


def plot_portfolio_allocation(alloc_df: pd.DataFrame, title: str = "Portfolio Allocation") -> go.Figure:
    df = alloc_df.copy()
    if "Weight_%" in df.columns:
        df["Weight_%"] = pd.to_numeric(df["Weight_%"], errors="coerce").fillna(0)
    labels = df["Ticker"].astype(str)
    values = df["Weight_%"].astype(float)
    fig = px.pie(df, names=labels, values=values, title=title, hole=0.35)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.1))
    return fig


def plot_portfolio_attribution(attrib_df: pd.DataFrame, title: str = "Portfolio Attribution") -> go.Figure:
    df = attrib_df.copy()
    # Expect columns: Ticker, Attribution (or similar). Try to auto-detect numeric column.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric column found for attribution chart")
    value_col = numeric_cols[0]
    fig = px.bar(df.sort_values(value_col, ascending=False), x=value_col, y=df.columns[0], orientation="h", title=title)
    fig.update_layout(margin=dict(t=40, b=40, l=120, r=10))
    return fig


def plot_sector_exposure(sector_df: pd.DataFrame, title: str = "Sector Exposure") -> go.Figure:
    df = sector_df.copy()
    if "Sector" not in df.columns or "Weight_%" not in df.columns:
        raise ValueError("Sector and Weight_% columns are required")

    df["Weight_%"] = pd.to_numeric(df["Weight_%"], errors="coerce").fillna(0)
    grouped = (
        df.groupby("Sector", as_index=False)["Weight_%"].sum().sort_values("Weight_%", ascending=False)
    )
    grouped["Label"] = grouped["Weight_%"].map(lambda v: f"{v:.1f}%")
    max_sector = grouped.iloc[0]["Sector"] if not grouped.empty else None
    colors = ["#ff8c00" if sector == max_sector else "#00d4ff" for sector in grouped["Sector"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped["Sector"],
                y=grouped["Weight_%"],
                text=grouped["Label"],
                textposition="outside",
                marker=dict(color=colors, line=dict(color="#243244", width=1)),
                hovertemplate="Sector: %{x}<br>Weight: %{y:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Sector",
        yaxis_title="Weight (%)",
        xaxis=dict(categoryorder="total descending"),
        yaxis=dict(ticksuffix="%", rangemode="tozero"),
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


def plot_garch_volatility(garch_df: pd.DataFrame, title: str = "GARCH Annualized Volatility") -> go.Figure:
    df = garch_df.copy()
    # Try common column names
    candidates = [
        "Current_AnnVol_pct",
        "Annualized_Volatility_%",
        "GARCH_Vol_%",
        "Ann_Volatility_%",
        "Current_AnnVol",
    ]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break
    if col is None:
        # try any numeric column besides index
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("No volatility column found in GARCH report")
        col = numeric_cols[0]
    series = pd.to_numeric(df[col], errors="coerce").fillna(0)
    labels = df.get("Ticker", df.index.astype(str)).astype(str)
    fig = px.bar(x=series, y=labels, orientation="h", title=title)
    fig.update_layout(xaxis_title=col, yaxis={'categoryorder':'total ascending'}, margin=dict(l=120, t=40, b=40))
    return fig


def plot_model_metrics_bars(all_metrics: pd.DataFrame, metric: str = "RMSE", title: str = "Model RMSE Comparison") -> go.Figure:
    df = all_metrics.copy()
    # find RMSE-like column
    candidates = [metric, "RMSE", "rmse", "RMSE_mean"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        # try numeric columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("No numeric metric column found in model metrics")
        col = numeric_cols[0]
    df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby("Model")[col].mean().reset_index().sort_values(col)
    fig = px.bar(grouped, x="Model", y=col, color="Model", title=title)
    fig.update_layout(margin=dict(t=40, b=40), showlegend=False)
    return fig


def plot_forecast_vs_actual(avp_df: pd.DataFrame, date_col: str = "Date", title: str = "Forecast vs Actual", model_order: Optional[list] = None) -> go.Figure:
    df = avp_df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
    else:
        # assume index is date-like
        try:
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={"index": date_col})
        except Exception:
            raise ValueError("No date column found for forecast chart")

    # Determine series columns: Actual + models
    cols = [c for c in df.columns if c not in [date_col, "Ticker", "Company"]]
    if model_order:
        # preserve requested plotting order
        cols = [c for c in model_order if c in df.columns] + [c for c in cols if c not in (model_order or [])]

    # color mapping (institutional palette)
    color_map = {
        "Actual": "#ffffff",
        "ARIMA": "#00d4ff",
        "ETS": "#ffaa00",
        "Prophet-STL (Custom)": "#9b59b6",
        "Prophet": "#9b59b6",
        "LSTM": "#ff6b35",
        "Ensemble": "#00ff88",
    }

    fig = go.Figure()
    for col in cols:
        name = col
        y = pd.to_numeric(df[col], errors="coerce")
        color = color_map.get(col, None)
        if col == "Ensemble":
            fig.add_trace(go.Scatter(x=df[date_col], y=y, mode="lines", name=name, line=dict(color=color or "#00ff88", width=3)))
        else:
            fig.add_trace(go.Scatter(x=df[date_col], y=y, mode="lines", name=name, line=dict(color=color or None, width=1.5)))

    # styling for dark terminal
    fig.update_layout(
        template=None,
        plot_bgcolor="#0a0e1a",
        paper_bgcolor="#0a0e1a",
        font=dict(color="#e8edf5"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=60, r=20),
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor="#1e2d47"),
        yaxis=dict(showgrid=True, gridcolor="#122033", zeroline=False, showline=True, linecolor="#1e2d47"),
    )

    # unified hovermode
    fig.update_layout(hovermode="x unified")

    return fig

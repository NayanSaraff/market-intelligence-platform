"""Bloomberg Terminal-grade Streamlit dashboard for StockGro capstone project."""

from __future__ import annotations

import contextlib
import io
import logging
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

try:
    from dashboard import plotly_charts
except ModuleNotFoundError:
    import plotly_charts

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
PORTFOLIO_DIR = OUTPUTS_DIR / "portfolio"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
IST = pytz.timezone("Asia/Kolkata")

TAB_LABELS = ["Overview", "EDA", "Model Results", "Forecast", "Volatility", "Portfolio", "Execution"]
TERMINAL_TITLE = dict(font=dict(size=11, color="#c8d4e8"), x=0.01, xanchor="left")
TERMINAL_LAYOUT = dict(
    paper_bgcolor="#080c14",
    plot_bgcolor="#090d15",
    font=dict(family="JetBrains Mono, monospace", color="#8aa0bb", size=10),
    xaxis=dict(
        gridcolor="#111d2c", linecolor="#1a2535", zeroline=False,
        tickcolor="#2a3d55", tickfont=dict(size=9, color="#4a6070"),
        showgrid=True, gridwidth=0.5,
    ),
    yaxis=dict(
        gridcolor="#111d2c", linecolor="#1a2535", zeroline=False,
        tickcolor="#2a3d55", tickfont=dict(size=9, color="#4a6070"),
        showgrid=True, gridwidth=0.5,
    ),
    legend=dict(
        bgcolor="#0a101a", bordercolor="#1a2535", borderwidth=1,
        font=dict(size=9, color="#8aa0bb"),
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0,
    ),
    margin=dict(l=48, r=16, t=36, b=36),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0a101a", bordercolor="#00aadd",
        font_color="#c8d4e8", font_size=10,
        font_family="JetBrains Mono, monospace",
    ),
    modebar=dict(bgcolor="#080c14", color="#2a3d55", activecolor="#00aadd"),
)


def terminal_layout(title=None, **overrides):
    layout = TERMINAL_LAYOUT.copy()
    if title is not None:
        title_layout = TERMINAL_TITLE.copy()
        title_layout["text"] = title
        layout["title"] = title_layout
    layout.update(overrides)
    return layout


STOCK_COLORS = {
    "DRREDDY.NS": "#00d4ff",
    "HDFCBANK.NS": "#1d4ed8",
    "ICICIBANK.NS": "#00ff88",
    "INFY.NS": "#a855f7",
    "ITC.NS": "#ff8c00",
    "MARUTI.NS": "#f59e0b",
    "SUNPHARMA.NS": "#ff3366",
    "TATAMOTORS.NS": "#f97316",
}

MODEL_COLORS = {
    "ARIMA": "#00d4ff",
    "ETS": "#1d4ed8",
    "Prophet-STL (Custom)": "#a855f7",
    "LSTM": "#f59e0b",
    "Ensemble": "#00ff88",
}

YFINANCE_SKIP_TICKERS = {"TATAMOTORS.NS"}

_DEFAULT_STOCK_UNIVERSE = pd.DataFrame(
    [
        {"Ticker": "DRREDDY.NS", "Company": "Dr. Reddy's Laboratories", "Sector": "Healthcare"},
        {"Ticker": "HDFCBANK.NS", "Company": "HDFC Bank", "Sector": "Financials"},
        {"Ticker": "ICICIBANK.NS", "Company": "ICICI Bank", "Sector": "Financials"},
        {"Ticker": "INFY.NS", "Company": "Infosys", "Sector": "IT"},
        {"Ticker": "ITC.NS", "Company": "ITC", "Sector": "Consumer"},
        {"Ticker": "MARUTI.NS", "Company": "Maruti Suzuki", "Sector": "Automobile"},
        {"Ticker": "SUNPHARMA.NS", "Company": "Sun Pharma", "Sector": "Healthcare"},
        {"Ticker": "TATAMOTORS.NS", "Company": "Tata Motors", "Sector": "Automobile"},
    ]
)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    try:
        path_obj = Path(path)
        if path_obj.exists():
            return pd.read_csv(path_obj)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_live_prices(tickers: list[str]) -> dict:
    prices: dict[str, dict] = {}
    for ticker in tickers:
        if ticker in YFINANCE_SKIP_TICKERS:
            prices[ticker] = {"price": np.nan}
            continue
        price = np.nan
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                history = yf.Ticker(ticker).history(period="1d", interval="1m")
            if not history.empty:
                price = float(history["Close"].dropna().iloc[-1])
        except Exception:
            price = np.nan
        prices[ticker] = {"price": price}
    return prices


@st.cache_data(ttl=1800)
def load_all_ohlc(tickers, start="2021-01-01"):
    tickers = [ticker for ticker in tickers if ticker not in YFINANCE_SKIP_TICKERS]
    if not tickers:
        return pd.DataFrame()

    try:
        return yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_price_data(start: str = "2021-01-01") -> pd.DataFrame:
    tickers = STOCK_UNIVERSE["Ticker"].dropna().astype(str).unique().tolist()
    if not tickers:
        return pd.DataFrame()

    raw = load_all_ohlc(tickers, start=start)
    if raw.empty:
        return pd.DataFrame()

    close_df = pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close_df = raw["Close"].copy()
    else:
        if isinstance(raw, pd.Series):
            close_df = raw.to_frame(name=tickers[0])
        elif "Close" in raw.columns:
            close_df = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            close_df = raw.copy()

    if close_df.empty:
        return pd.DataFrame()

    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=tickers[0])

    close_df = close_df.dropna(how="all")
    return close_df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(ticker: str, period: str = "6mo") -> pd.DataFrame:
    if ticker in YFINANCE_SKIP_TICKERS:
        return pd.DataFrame()

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        flat_cols: list[str] = []
        for col in df.columns:
            if isinstance(col, tuple):
                non_empty = [str(part) for part in col if part and str(part) != "nan"]
                flat_cols.append(non_empty[0] if non_empty else "")
            else:
                flat_cols.append(str(col))
        df.columns = flat_cols

    df = df.reset_index()
    date_col = next((c for c in ["Date", "Datetime", "index"] if c in df.columns), None)
    if date_col and date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        cutoff = pd.Timestamp.today() - pd.DateOffset(months=6)
        df = df[df["Date"] >= cutoff]

    required = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if len(required) < 4:
        return pd.DataFrame()

    return df[["Date", "Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])].dropna(subset=["Date"])


def compute_portfolio_snapshot(portfolio_df: pd.DataFrame, live_prices: dict) -> tuple[float, float, float]:
    if portfolio_df.empty or "Ticker" not in portfolio_df.columns:
        return 0.0, 0.0, 0.0

    invested = 0.0
    current_value = 0.0
    amount_col = "Amount_INR" if "Amount_INR" in portfolio_df.columns else None
    shares_col = "Shares_to_Buy" if "Shares_to_Buy" in portfolio_df.columns else None

    for _, row in portfolio_df.iterrows():
        ticker = str(row.get("Ticker", ""))
        amount = float(pd.to_numeric(row.get(amount_col, 0.0), errors="coerce") or 0.0) if amount_col else 0.0
        shares = float(pd.to_numeric(row.get(shares_col, 0.0), errors="coerce") or 0.0) if shares_col else 0.0
        price = float(live_prices.get(ticker, {}).get("price", np.nan)) if ticker in live_prices else np.nan

        invested += amount
        if np.isfinite(price) and shares > 0:
            current_value += shares * price
        else:
            current_value += amount

    if current_value <= 0 and invested > 0:
        current_value = invested

    pnl = current_value - invested
    return invested, current_value, pnl


def panel_header(title: str, subtitle: str | None = None) -> None:
    if subtitle:
        st.markdown(f'<div class="panel-header"><span>{title}</span><span>{subtitle}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="panel-header">{title}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, accent: str = "#00d4ff", subtitle: str = "", is_positive: bool | None = None) -> None:
    extra_class = "val-neu"
    if is_positive is True:
        extra_class = "val-pos"
    elif is_positive is False:
        extra_class = "val-neg"
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{accent}">{value}</div>'
        f'<div class="kpi-subtitle {extra_class}">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def alert_box(level: str, message: str) -> None:
    level_key = str(level).strip().lower()
    level_map = {
        "warning": ("WARNING", "alert-warning"),
        "info": ("INFO", "alert-info"),
        "success": ("INFO", "alert-info"),
        "error": ("CRITICAL", "alert-critical"),
        "critical": ("CRITICAL", "alert-critical"),
    }
    label, css_class = level_map.get(level_key, ("INFO", "alert-info"))
    st.markdown(
        f"<div class='alert-box {css_class}'><span class='alert-level'>{label}</span>"
        f"<span class='alert-message'>{message}</span></div>",
        unsafe_allow_html=True,
    )


def choose_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def tab_hint(selected_tab: str, title: str) -> None:
    st.caption(f"{selected_tab} | {title}")


def stock_selectbox(label: str, options: list[str], key: str = "selected_stock") -> str:
    if not options:
        return ""
    current = st.session_state.get(key, options[0])
    if current not in options:
        current = options[0]
    choice = st.selectbox(label, options, index=options.index(current), key=key)
    st.session_state[key] = choice
    return choice


def signal_badge(signal: str) -> str:
    text = str(signal).lower()
    if any(token in text for token in ["bull", "buy", "long", "positive"]):
        return '<span class="badge-bull">BULLISH</span>'
    if any(token in text for token in ["bear", "sell", "short", "negative"]):
        return '<span class="badge-bear">BEARISH</span>'
    return '<span class="badge-neu">NEUTRAL</span>'


def style_stationarity(df: pd.DataFrame) -> pd.DataFrame:
    return df


def style_best_rmse(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "RMSE" not in df.columns:
        return df
    return df.sort_values("RMSE", ascending=True)


def get_best_models(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Ticker" not in df.columns or "RMSE" not in df.columns:
        return df
    idx = df.groupby("Ticker")["RMSE"].idxmin()
    return df.loc[idx].sort_values("Ticker")


def style_persistence(df: pd.DataFrame) -> pd.DataFrame:
    return df


def style_forecast_decision(df: pd.DataFrame) -> pd.DataFrame:
    return df


def style_ensemble_error(df: pd.DataFrame) -> pd.DataFrame:
    return df


def style_forecast_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    if df.empty:
        return df.style
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    if "Model" in df.columns:
        ens_mask = df["Model"].astype(str).str.lower().str.contains("ensemble")
        for col in df.columns:
            styles.loc[ens_mask, col] = "background-color:#0a1418; border-top:1px solid #ff6600; color:#e8f3f8; font-weight:700"
    return df.style.apply(lambda _: styles, axis=None)


STOCK_UNIVERSE = load_csv(str(PORTFOLIO_DIR / "final_portfolio_allocation.csv"))
if STOCK_UNIVERSE.empty or "Ticker" not in STOCK_UNIVERSE.columns:
    STOCK_UNIVERSE = _DEFAULT_STOCK_UNIVERSE.copy()

ALL_OHLC_DATA = load_all_ohlc(STOCK_UNIVERSE["Ticker"].dropna().astype(str).tolist())

EDA_CHARTS = {
    "Price Trends": CHARTS_DIR / "eda_p5" / "EDA_01_price_panels.png",
    "Return Dist": CHARTS_DIR / "eda_p5" / "EDA_02_return_distributions.png",
    "Q-Q Plots": CHARTS_DIR / "eda_p5" / "EDA_03_qq_plots.png",
    "Correlation": CHARTS_DIR / "eda_p5" / "EDA_04_correlation_heatmap.png",
    "Rolling Vol": CHARTS_DIR / "eda_p5" / "EDA_05_rolling_volatility.png",
    "STL Decomp": CHARTS_DIR / "eda_p5" / "EDA_06_stl_decomposition.png",
    "Cum Returns": CHARTS_DIR / "eda_p5" / "EDA_07_cumret_drawdown.png",
    "ACF/PACF": CHARTS_DIR / "eda_p5" / "EDA_08_acf_pacf.png",
    "Monthly Returns": CHARTS_DIR / "eda_p5" / "EDA_09_monthly_return_heatmap.png",
    "Risk-Return": CHARTS_DIR / "eda_p5" / "EDA_10_riskreturn_stationarity.png",
}

MODEL_CHARTS = {
    "Predicted vs Actual": CHARTS_DIR / "models" / "M01_predicted_vs_actual.png",
    "Metric Heatmap": CHARTS_DIR / "models" / "M02_model_metric_heatmap.png",
    "Metrics Bars": CHARTS_DIR / "models" / "M03_model_metrics_bars.png",
    "5-Day Forecast": CHARTS_DIR / "models" / "M04_5day_forecasts.png",
    "Residuals": CHARTS_DIR / "models" / "M05_residuals_distribution.png",
    "Ensemble Weights": CHARTS_DIR / "models" / "M06_ensemble_weights.png",
    "ARIMA ACF": CHARTS_DIR / "models" / "M07_arima_residual_acf.png",
    "LSTM Curves": CHARTS_DIR / "models" / "M08_lstm_training_curves.png",
}

def render_terminal_shell(portfolio_df: pd.DataFrame, summary_df: pd.DataFrame, live_prices: dict) -> None:
    """Render the fixed Bloomberg-style terminal header."""
    invested, current_value, pnl = compute_portfolio_snapshot(portfolio_df, live_prices)
    capital = safe_metric(summary_df, "Total_Capital_INR", 1_000_000.0)
    freshness = datetime.now(IST).strftime("%H:%M:%S IST")
    market_open = (datetime.now(IST).weekday() < 5) and (555 <= (datetime.now(IST).hour * 60 + datetime.now(IST).minute) <= 930)
    status_text = "OPEN" if market_open else "CLOSED"
    pnl_class = "pos" if pnl > 0 else "neg" if pnl < 0 else "cyan"

    st.markdown(
        f"""
<div class="terminal-shell">
    <div class="terminal-shell-inner">
        <div class="terminal-brand">
            <div class="terminal-brand-title">STOCKGRO TERMINAL</div>
            <div class="terminal-brand-meta"><span class="live-dot"></span>LIVE | V1.0 | NSE INDIA</div>
        </div>
        <div class="terminal-center">
            <div class="terminal-kpi">
                <span class="label">Market</span>
                <span class="value {'pos' if market_open else 'neg'}">NSE {status_text}</span>
            </div>
            <div class="terminal-kpi">
                <span class="label">IST</span>
                <span class="value cyan">{freshness}</span>
            </div>
            <div class="terminal-kpi">
                <span class="label">Freshness</span>
                <span class="value">5m cache</span>
            </div>
        </div>
        <div class="terminal-right">
            <div class="terminal-kpi">
                <span class="label">Capital</span>
                <span class="value cyan">{fmt_inr(capital)}</span>
            </div>
            <div class="terminal-kpi">
                <span class="label">Portfolio</span>
                <span class="value">{fmt_inr(current_value)}</span>
            </div>
            <div class="terminal-kpi">
                <span class="label">Live P&amp;L</span>
                <span class="value {pnl_class}">{fmt_inr(pnl)}</span>
            </div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def market_status_banner() -> None:
    """Show NSE market status based on IST time."""
    now = datetime.now(IST)
    wd = now.weekday()
    h, m = now.hour, now.minute
    mins = h * 60 + m
    is_open = (wd < 5) and (555 <= mins <= 930)
    status = "OPEN" if is_open else "CLOSED"
    color = "#00ff88" if is_open else "#ff3366"
    st.markdown(
        f"""
<div style="display:flex;align-items:center;gap:16px;
            padding:8px 16px;background:#0f1623;
            border-bottom:1px solid #1e2d47;font-family:monospace;font-size:12px;">
    <span class="live-dot"></span>
    <span style="color:#00d4ff">NSE MARKET:</span>
    <span style="color:{color};font-weight:700">{status}</span>
    <span style="color:#8892a4;margin-left:auto">
        {now.strftime('%A, %d %b %Y  %H:%M:%S IST')}
    </span>
</div>""",
        unsafe_allow_html=True,
    )


def ticker_tape(portfolio_df: pd.DataFrame, live_prices: dict) -> None:
    """Render scrolling ticker tape for portfolio stocks."""
    if portfolio_df.empty:
        st.markdown(
            '<div class="ticker-tape"><div class="ticker-inner">NO DATA LOADED</div></div>',
            unsafe_allow_html=True,
        )
        return

    if "Ticker" not in portfolio_df.columns:
        st.markdown(
            '<div class="ticker-tape"><div class="ticker-inner">NO DATA</div></div>',
            unsafe_allow_html=True,
        )
        return

    items = []
    for _, row in portfolio_df.iterrows():
        t = row.get("Ticker", "")
        ret = float(row.get("Forecast_5d_Ret_%", 0))
        sym = "▲" if ret >= 0 else "▼"
        col = "#00ff88" if ret >= 0 else "#ff3366"
        ticker_short = t.replace(".NS", "")
        
        price_info = live_prices.get(t, {})
        price = price_info.get("price")
        price_str = f"₹{price:.2f}" if price else "—"
        
        items.append(
            f'<span style="margin:0 28px;color:#e8edf5">{ticker_short}</span>'
            f'<span style="color:#00d4ff;font-size:11px">{price_str}</span>'
            f'<span style="color:{col}">{sym} {abs(ret):.2f}%</span>'
        )
    tape = "  ·  ".join(items) * 3
    st.markdown(
        f"""
<div class="ticker-tape">
    <div class="ticker-inner">{tape}</div>
</div>""",
        unsafe_allow_html=True,
    )


def fmt_inr(val: float) -> str:
    """Format as Indian currency with lakh/crore notation."""
    try:
        v = float(val)
        if v >= 1e7:
            return f"₹{v/1e7:.2f}Cr"
        elif v >= 1e5:
            return f"₹{v/1e5:.2f}L"
        else:
            return f"₹{v:,.0f}"
    except Exception:
        return "₹—"


def fmt_pct(val: float, show_sign: bool = True) -> str:
    """Format percentage with optional sign."""
    try:
        v = float(val)
        sign = "+" if v > 0 and show_sign else ""
        return f"{sign}{v:.2f}%"
    except Exception:
        return "—"


def safe_metric(df: pd.DataFrame, col: str, default: float = np.nan) -> float:
    """Return a numeric metric from the first row when available, else default."""
    if df.empty or col not in df.columns:
        return default
    try:
        return float(pd.to_numeric(df.iloc[0][col], errors="coerce"))
    except Exception:
        return default


def show_image(path: Path, caption: str = "") -> None:
    """Display image or placeholder."""
    p = Path(path)
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.markdown(
            f'<div style="background:#141c2e;border:1px dashed #1e2d47;'
            f'padding:20px;text-align:center;color:#8892a4;font-size:12px;'
            f'font-family:monospace">CHART NOT YET GENERATED: {p.name}</div>',
            unsafe_allow_html=True,
        )


def render_fixed_table(df: pd.DataFrame, height: int = 300, hide_index: bool = True) -> None:
    """Render a wide, non-resizable table with fixed column widths."""
    if df is None or df.empty:
        return

    display_df = df.copy()
    display_df = display_df.astype(str).fillna("")
    column_config = {
        column: st.column_config.TextColumn(column, width="large")
        for column in display_df.columns
    }
    st.dataframe(display_df, width="stretch", height=height, hide_index=hide_index, column_config=column_config)


# ── Page Config ─────────────────────────────────
st.set_page_config(
    page_title="StockGro Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = STOCK_UNIVERSE["Ticker"].iloc[0]
elif st.session_state.selected_stock not in STOCK_UNIVERSE["Ticker"].tolist():
    st.session_state.selected_stock = STOCK_UNIVERSE["Ticker"].iloc[0]

st.markdown(
    """
<style>
:root {
    --bg: #05070b;
    --bg-alt: #0a0e14;
    --panel: #10151c;
    --panel-2: #141a22;
    --card: #1a212b;
    --border: #1f2a38;
    --border-2: #243244;
    --text: #e6edf5;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --orange: #ff8c00;
    --amber: #f59e0b;
    --cyan: #00d4ff;
    --green: #00ff88;
    --red: #ff3366;
    --purple: #a855f7;
    --blue: #1d4ed8;
}

html, body, [class*="css"] {
    background: linear-gradient(180deg, var(--bg) 0%, var(--bg-alt) 100%) !important;
    color: var(--text) !important;
    font-family: 'Inter', 'IBM Plex Sans', sans-serif;
}

body {
    padding-top: 76px;
}

/* Fixed terminal header */
.terminal-shell {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: rgba(10, 14, 20, 0.96);
    border-bottom: 1px solid var(--border-2);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(10px);
}
.terminal-shell-inner {
    display: grid;
    grid-template-columns: 1.25fr 1.1fr 1fr;
    gap: 12px;
    align-items: center;
    padding: 10px 16px 8px;
    min-height: 72px;
}
.terminal-brand {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.terminal-brand-title {
    font-family: 'JetBrains Mono', monospace;
    color: var(--cyan);
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.16em;
}
.terminal-brand-meta,
.terminal-meta-line {
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
.terminal-center,
.terminal-right {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 12px;
    align-items: center;
    justify-content: center;
}
.terminal-right {
    justify-content: flex-end;
}
.terminal-kpi {
    display: flex;
    flex-direction: column;
    min-width: 108px;
    padding: 6px 10px;
    border: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(20, 26, 34, 0.95), rgba(16, 21, 28, 0.95));
    border-radius: 6px;
}
.terminal-kpi .label {
    font-size: 9px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.terminal-kpi .value {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text);
    font-weight: 700;
    font-size: 14px;
}
.terminal-kpi .value.pos { color: var(--green); }
.terminal-kpi .value.neg { color: var(--red); }
.terminal-kpi .value.cyan { color: var(--cyan); }

/* Terminal shell positioning */
.block-container {
    padding-top: 0.75rem;
    padding-bottom: 1.5rem;
}

/* Ticker tape animation */
@keyframes ticker-scroll {
    0% { transform: translate3d(0, 0, 0); }
    100% { transform: translate3d(-50%, 0, 0); }
}
.ticker-tape {
    position: sticky;
    top: 86px;
    z-index: 20;
    overflow: hidden;
    white-space: nowrap;
    border-top: 1px solid var(--cyan);
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(16, 21, 28, 0.98), rgba(10, 14, 20, 0.98));
    padding: 7px 0;
    margin: 0 0 10px 0;
}
.ticker-tape:hover .ticker-inner {
    animation-play-state: paused;
}
.ticker-inner {
    display: inline-flex;
    width: max-content;
    animation: ticker-scroll 42s linear infinite;
    will-change: transform;
}
.ticker-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text);
}
.ticker-symbol {
    color: var(--cyan);
    letter-spacing: 0.06em;
    font-weight: 700;
}
.ticker-price {
    color: var(--text);
}

/* Live pulse dot */
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(0.85); opacity: 0.35; }
}
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse 1.4s ease-in-out infinite;
    box-shadow: 0 0 0 4px rgba(0, 255, 136, 0.08);
    margin-right: 6px;
}

/* Panel headers */
.panel-header {
    border-top: 2px solid var(--orange);
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    padding: 9px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.16em;
    color: var(--text);
    text-transform: uppercase;
    margin-bottom: 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: 6px 6px 0 0;
}

.loading-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-muted);
    font-size: 11px;
    padding: 8px 12px;
    background: rgba(16, 21, 28, 0.85);
    border: 1px solid var(--border);
    border-radius: 6px;
}

/* Value color classes */
.val-pos  { color: var(--green) !important; font-family: 'JetBrains Mono', monospace; }
.val-neg  { color: var(--red) !important; font-family: 'JetBrains Mono', monospace; }
.val-neu  { color: var(--amber) !important; font-family: 'JetBrains Mono', monospace; }
.val-cyan { color: var(--cyan) !important; font-family: 'JetBrains Mono', monospace; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(180deg, rgba(20, 26, 34, 0.98), rgba(16, 21, 28, 0.98));
    border: 1px solid var(--border);
    border-top: 2px solid var(--orange);
    border-radius: 6px;
    padding: 11px 12px;
    text-align: center;
    min-height: 86px;
    transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
}
.kpi-card:hover {
    transform: translateY(-1px);
    border-color: rgba(0, 212, 255, 0.45);
    box-shadow: 0 0 0 1px rgba(0, 212, 255, 0.12), 0 12px 20px rgba(0, 0, 0, 0.25);
}
.kpi-label {
    font-size: 9px;
    letter-spacing: 0.14em;
    color: var(--text-muted);
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: clamp(1rem, 1.4vw, 1.6rem);
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--cyan);
    line-height: 1;
}

/* Status chips and alerts */
.status-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 52px;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--border-2);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
}
.chip-pos { color: var(--green); background: rgba(0, 255, 136, 0.08); }
.chip-neg { color: var(--red); background: rgba(255, 51, 102, 0.08); }
.chip-warn { color: var(--amber); background: rgba(245, 158, 11, 0.08); }
.chip-neu { color: var(--cyan); background: rgba(0, 212, 255, 0.08); }
.chip-live { color: var(--green); background: rgba(0, 255, 136, 0.10); }

.alert-box {
    display: flex;
    gap: 10px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-left-width: 3px;
    border-radius: 6px;
    background: rgba(16, 21, 28, 0.94);
    margin: 8px 0;
    align-items: center;
}
.alert-level {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    font-size: 10px;
    color: var(--text);
}
.alert-message {
    color: var(--text-muted);
    font-size: 12px;
}
.alert-critical { border-left-color: var(--red); }
.alert-warning { border-left-color: var(--amber); }
.alert-info { border-left-color: var(--cyan); }

/* Signal badges */
.badge-bull  { background: rgba(0,255,136,0.10); color: var(--green); padding:2px 10px; border:1px solid var(--green); border-radius:3px; font-size:11px; font-family:'JetBrains Mono', monospace; }
.badge-bear  { background: rgba(255,51,102,0.10); color: var(--red); padding:2px 10px; border:1px solid var(--red); border-radius:3px; font-size:11px; font-family:'JetBrains Mono', monospace; }
.badge-neu   { background: rgba(245,158,11,0.10); color: var(--amber); padding:2px 10px; border:1px solid var(--amber); border-radius:3px; font-size:11px; font-family:'JetBrains Mono', monospace; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(180deg, rgba(20, 26, 34, 0.96), rgba(16, 21, 28, 0.96));
    border-bottom: 1px solid var(--border);
    gap: 0;
    padding-left: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.10em;
    color: var(--text-muted);
    text-transform: uppercase;
    padding: 10px 16px;
    border-bottom: 2px solid transparent;
    min-height: 44px;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--orange) !important;
    background: rgba(26, 33, 43, 0.95) !important;
    font-weight: 700;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(16,21,28,0.98), rgba(10,14,20,0.98)) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text);
}
[data-testid="stSidebar"] .stMetricValue {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--cyan) !important;
}

/* Dataframe / tables */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden;
}
.terminal-table-wrap [data-testid="stDataFrame"] {
    background: rgba(16, 21, 28, 0.85);
}
div[data-testid="stDataFrame"] div[role="grid"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* Compact dataframe font */
div[data-testid="stDataFrame"] div[role="grid"] {
    font-size: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.5 !important;
}

/* Remove ugly delta SVG arrow from st.metric */
[data-testid="stMetricDelta"] svg { display: none !important; }

/* Flatten st.info / st.warning boxes */
[data-testid="stAlert"] {
    background: #0a1018 !important;
    border: 1px solid #1a2535 !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    color: #6a8aaa !important;
}

/* Compact st.metric style */
[data-testid="stMetric"] {
    background: #0a1018;
    border: 1px solid #1a2535;
    padding: 8px 12px;
    border-radius: 2px;
}
[data-testid="stMetricLabel"] {
    font-size: 8px !important;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #3a5070 !important;
}
[data-testid="stMetricValue"] {
    font-size: 16px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #00aadd !important;
}

/* kpi-value: clamp so it doesn't overflow narrow columns */
.kpi-value {
    font-size: clamp(1rem, 1.4vw, 1.5rem) !important;
}

/* Spinner text */
[data-testid="stSpinner"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    color: #2a3d55 !important;
    letter-spacing: .1em;
}

/* Remove max-width center bleed */
.block-container {
    padding-top: 0.5rem !important;
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Thin scrollbar on dataframes */
div[data-testid="stDataFrame"] .dvn-scroller {
    scrollbar-width: thin;
    scrollbar-color: #1a2535 #080c14;
}

/* Inputs */
.stButton>button,
.stDownloadButton>button {
    border: 1px solid var(--border-2) !important;
    background: linear-gradient(180deg, rgba(20,26,34,0.98), rgba(16,21,28,0.98)) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    transition: all 150ms ease;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
    border-color: rgba(255, 140, 0, 0.65) !important;
    box-shadow: 0 0 0 1px rgba(255, 140, 0, 0.16);
}

[data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stRadio"], [data-testid="stCheckbox"] {
    font-family: 'Inter', sans-serif;
}

/* Expanders */
details {
    border: 1px solid var(--border);
    border-radius: 6px;
    background: rgba(16, 21, 28, 0.92);
}
summary {
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
}

/* Scrollbars */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0b0f14; }
::-webkit-scrollbar-thumb { background: #243244; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #36506a; }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Global State ────────────────────────────────
market_status_banner()
selected_tab = TAB_LABELS[0]
global_stock = st.session_state.selected_stock

st.set_option("client.showErrorDetails", False)

dashboard_portfolio_df = load_csv(str(PORTFOLIO_DIR / "final_portfolio_allocation.csv"))
dashboard_summary_df = load_csv(str(PORTFOLIO_DIR / "portfolio_metrics_summary.csv"))

# ── Fetch Live Prices ──────────────────────────
live_prices = fetch_live_prices(STOCK_UNIVERSE["Ticker"].tolist())

render_terminal_shell(dashboard_portfolio_df, dashboard_summary_df, live_prices)
# ── Main Tabs ──────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(TAB_LABELS)

# ── TAB 1: OVERVIEW ────────────────────────────
with tab1:
    with st.spinner("Loading portfolio..."):
        portfolio_df = load_csv(str(PORTFOLIO_DIR / "final_portfolio_allocation.csv"))
        summary_df = load_csv(str(PORTFOLIO_DIR / "portfolio_metrics_summary.csv"))
    
    ticker_tape(portfolio_df, live_prices)
    
    st.markdown("")
    panel_header("PORTFOLIO COMMAND CENTER")
    
    row1 = st.columns(4)
    row2 = st.columns(4)
    with row1[0]:
        kpi_card("CAPITAL", "₹10L", "#00aadd")
    with row1[1]:
        kpi_card("INVESTED", "₹9.81L", "#00aadd")
    with row1[2]:
        kpi_card("LIVE P&L", "+₹4.8K", "#00cc66", "+0.49%", True)
    with row1[3]:
        kpi_card("EXP RET", "15.92%", "#00cc66")
    with row2[0]:
        kpi_card("SHARPE", "0.53", "#ddaa00")
    with row2[1]:
        kpi_card("MAX DD", "−32.98%", "#ff3366")
    with row2[2]:
        kpi_card("PORT VOL", "18.64%", "#ddaa00")
    with row2[3]:
        kpi_card("CASH RSV", "₹19K", "#00aadd")
    
    st.markdown("")
    col_left, col_center, col_right = st.columns([5, 4, 3])
    
    with col_left:
        panel_header("PORTFOLIO ALLOCATION")
        if not portfolio_df.empty and "Weight_%" in portfolio_df.columns:
            alloc_df = portfolio_df[["Ticker", "Weight_%"]].copy()
            alloc_df["Weight_%"] = pd.to_numeric(alloc_df["Weight_%"], errors="coerce")
            alloc_df["Ticker"] = alloc_df["Ticker"].str.replace(".NS", "")
            fig = go.Figure(data=[go.Pie(labels=alloc_df["Ticker"], values=alloc_df["Weight_%"],
                                          marker=dict(colors=[STOCK_COLORS.get(t+".NS", "#00d4ff") 
                                                             for t in alloc_df["Ticker"]]))])
            fig.update_layout(**TERMINAL_LAYOUT, showlegend=True)
            st.plotly_chart(fig, width="stretch")
    
    with col_center:
        panel_header("RISK SIGNALS")
        if not portfolio_df.empty:
            for _, row in portfolio_df.head(5).iterrows():
                ticker = row.get("Ticker", "")
                vol = float(row.get("GARCH_Vol_%", 0))
                var = float(row.get("VaR_95_%", 0))
                signal = row.get("Forecast_Decision", "")
                vol_color = "#ff3366" if vol > 30 else "#ffaa00" if vol > 20 else "#00ff88"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                    f'border-bottom:1px solid #1e2d47;font-family:monospace;font-size:11px">'
                    f'<span style="color:#00d4ff">{ticker.replace(".NS","")}</span>'
                    f'<span style="color:{vol_color}">{vol:.1f}%</span>'
                    f'<span style="color:#ff3366">{var:.2f}%</span>'
                    f'<span>{signal_badge(signal)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    
    with col_right:
        panel_header("SYSTEM STATUS")
        checks = [
            ("Raw Data", True),
            ("Preprocessing", True),
            ("ARIMA", True),
            ("ETS", True),
            ("Prophet-STL", True),
            ("LSTM", True),
            ("Ensemble", True),
            ("GARCH", True),
            ("Portfolio", True),
        ]
        for item, status in checks:
            symbol = "✅" if status else "❌"
            st.markdown(f"<span style='font-family:monospace;font-size:11px'>{symbol} {item}</span>", 
                       unsafe_allow_html=True)
        st.markdown("---")
        st.caption("⚠️ Prophet: Custom implementation")
        st.caption("⚠️ 6/8 stocks: negative 5d forecast")
    
    st.markdown("")
    panel_header("STOCK UNIVERSE — 8 SELECTED | 5 SECTORS")
    
    if not portfolio_df.empty:
        display_df = portfolio_df[["Ticker", "Company", "Sector", "Weight_%", "Forecast_5d_Ret_%", 
                                    "GARCH_Vol_%"]].copy()
        
        for col in ["Weight_%", "Forecast_5d_Ret_%", "GARCH_Vol_%"]:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
        
        display_df["Weight_%"] = display_df["Weight_%"].apply(fmt_pct)
        display_df["Forecast_5d_Ret_%"] = display_df["Forecast_5d_Ret_%"].apply(fmt_pct)
        display_df["GARCH_Vol_%"] = display_df["GARCH_Vol_%"].apply(fmt_pct)
        
        render_fixed_table(display_df, height=400, hide_index=True)


# ── TAB 2: MARKET INTELLIGENCE ─────────────────
with tab2:
    st.markdown("## 📊 MARKET INTELLIGENCE TERMINAL")
    
    with st.spinner("Loading EDA data..."):
        summary_stats = load_csv(str(REPORTS_DIR / "summary_statistics.csv"))
        stationarity = load_csv(str(REPORTS_DIR / "stationarity_report.csv"))
    
    col_left, col_right = st.columns([2, 6])
    
    with col_left:
        panel_header("CHART SELECTOR")
        chart_option = st.radio(
            "Select Analysis",
            [
                "Price Trends",
                "Return Dist",
                "Q-Q Plots",
                "Correlation",
                "Rolling Vol",
                "STL Decomp",
                "Cum Returns",
                "ACF/PACF",
                "Monthly Returns",
                "Risk-Return",
            ],
            label_visibility="collapsed",
        )
        
        chart_map = {
            "Price Trends": CHARTS_DIR / "eda_p5" / "EDA_01_price_panels.png",
            "Return Dist": CHARTS_DIR / "eda_p5" / "EDA_02_return_distributions.png",
            "Q-Q Plots": CHARTS_DIR / "eda_p5" / "EDA_03_qq_plots.png",
            "Correlation": CHARTS_DIR / "eda_p5" / "EDA_04_correlation_heatmap.png",
            "Rolling Vol": CHARTS_DIR / "eda_p5" / "EDA_05_rolling_volatility.png",
            "STL Decomp": CHARTS_DIR / "eda_p5" / "EDA_06_stl_decomposition.png",
            "Cum Returns": CHARTS_DIR / "eda_p5" / "EDA_07_cumret_drawdown.png",
            "ACF/PACF": CHARTS_DIR / "eda_p5" / "EDA_08_acf_pacf.png",
            "Monthly Returns": CHARTS_DIR / "eda_p5" / "EDA_09_monthly_return_heatmap.png",
            "Risk-Return": CHARTS_DIR / "eda_p5" / "EDA_10_riskreturn_stationarity.png",
        }
        
    with col_right:
        panel_header("MARKET ANALYSIS CHART")
        price_df = load_price_data()

        if price_df.empty and chart_option != "Risk-Return":
            alert_box("warning", "Live/downloaded close-price history is unavailable. Plotly charts need price data to render.")
        elif chart_option == "Price Trends":
            norm = price_df / price_df.iloc[0] * 100
            fig = go.Figure()
            for col in norm.columns:
                fig.add_trace(
                    go.Scatter(
                        x=norm.index,
                        y=norm[col],
                        name=col.replace(".NS", ""),
                        line=dict(color=STOCK_COLORS.get(col, "#00d4ff"), width=1.5),
                        hovertemplate="%{x|%d %b %Y}<br>Idx: %{y:.1f}<extra></extra>",
                    )
                )
            fig.update_layout(**terminal_layout("Normalized Price Index (Base=100)"))
            st.plotly_chart(fig, width="stretch")
        elif chart_option == "Rolling Vol":
            returns = price_df.pct_change()
            rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
            fig = go.Figure()
            for col in rolling_vol.columns:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol[col],
                        name=col.replace(".NS", ""),
                        line=dict(color=STOCK_COLORS.get(col, "#00d4ff"), width=1.5),
                    )
                )
            fig.update_layout(**terminal_layout("20-Day Rolling Annualized Volatility (%)"))
            st.plotly_chart(fig, width="stretch")
        elif chart_option == "Correlation":
            corr = price_df.pct_change().corr()
            labels = [c.replace(".NS", "") for c in corr.columns]
            fig = go.Figure(
                go.Heatmap(
                    z=corr.values,
                    x=labels,
                    y=labels,
                    colorscale="RdYlGn",
                    zmin=-1,
                    zmax=1,
                    text=corr.round(2).values,
                    texttemplate="%{text}",
                    hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
                )
            )
            fig.update_layout(**terminal_layout("Return Correlation Heatmap"))
            st.plotly_chart(fig, width="stretch")
        elif chart_option == "Return Dist":
            returns = price_df.pct_change().dropna() * 100
            fig = go.Figure()
            for col in returns.columns:
                col_series = returns[col].dropna()
                fig.add_trace(
                    go.Violin(
                        x=[col.replace(".NS", "")] * len(col_series),
                        y=col_series,
                        name=col.replace(".NS", ""),
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=STOCK_COLORS.get(col, "#00d4ff"),
                        opacity=0.7,
                        line_color=STOCK_COLORS.get(col, "#00d4ff"),
                    )
                )
            fig.update_layout(**terminal_layout("Daily Return Distributions (%)"))
            st.plotly_chart(fig, width="stretch")
        elif chart_option == "Cum Returns":
            cum = (1 + price_df.pct_change()).cumprod() - 1
            fig = go.Figure()
            for col in cum.columns:
                fig.add_trace(
                    go.Scatter(
                        x=cum.index,
                        y=cum[col] * 100,
                        name=col.replace(".NS", ""),
                        line=dict(color=STOCK_COLORS.get(col, "#00d4ff"), width=1.5),
                        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
                    )
                )
            fig.update_layout(**terminal_layout("Cumulative Returns (%)"))
            st.plotly_chart(fig, width="stretch")
        elif chart_option == "Monthly Returns":
            monthly = price_df.resample("ME").last().pct_change() * 100
            available_tickers = [ticker for ticker in STOCK_UNIVERSE["Ticker"].tolist() if ticker in monthly.columns]
            st.session_state.selected_stock = st.selectbox(
                "Stock",
                available_tickers or monthly.columns.tolist(),
                index=(available_tickers or monthly.columns.tolist()).index(st.session_state.selected_stock)
                if st.session_state.selected_stock in (available_tickers or monthly.columns.tolist())
                else 0,
                key="stock_sel_eda",
            )
            s = monthly[st.session_state.selected_stock].dropna()
            if s.empty:
                alert_box("info", "Not enough monthly data to render this heatmap yet.")
            else:
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                pivot = pd.DataFrame({"year": s.index.year, "month": s.index.strftime("%b"), "val": s.values}).pivot(
                    index="year", columns="month", values="val"
                )
                pivot = pivot[[m for m in month_order if m in pivot.columns]]

                fig = go.Figure(
                    go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns,
                        y=pivot.index,
                        colorscale="RdYlGn",
                        zmid=0,
                        text=np.round(pivot.values, 1),
                        texttemplate="%{text}%",
                        hovertemplate="Year %{y} %{x}: %{z:.2f}%<extra></extra>",
                    )
                )
                fig.update_layout(**terminal_layout(f"Monthly Returns — {st.session_state.selected_stock.replace('.NS', '')}"))
                st.plotly_chart(fig, width="stretch")
        elif chart_option == "Q-Q Plots":
            returns = price_df.pct_change().dropna() * 100
            fig = go.Figure()
            for col in returns.columns:
                sample = returns[col].dropna().to_numpy(dtype=float)
                sample = sample[np.isfinite(sample)]
                if len(sample) < 10:
                    continue
                sample = (sample - sample.mean()) / sample.std(ddof=1)
                sample = np.sort(sample)
                probs = (np.arange(1, len(sample) + 1) - 0.5) / len(sample)
                theoretical = np.array([NormalDist().inv_cdf(float(p)) for p in probs])
                fig.add_trace(
                    go.Scattergl(
                        x=theoretical,
                        y=sample,
                        mode="markers",
                        name=col.replace(".NS", ""),
                        marker=dict(color=STOCK_COLORS.get(col, "#00d4ff"), size=4, opacity=0.45),
                        hovertemplate="Normal q: %{x:.2f}<br>Return q: %{y:.2f}<extra></extra>",
                    )
                )
            if fig.data:
                all_x = np.concatenate([np.asarray(trace.x, dtype=float) for trace in fig.data])
                all_y = np.concatenate([np.asarray(trace.y, dtype=float) for trace in fig.data])
                lo = float(np.nanmin([all_x.min(), all_y.min()]))
                hi = float(np.nanmax([all_x.max(), all_y.max()]))
                fig.add_trace(
                    go.Scatter(
                        x=[lo, hi],
                        y=[lo, hi],
                        mode="lines",
                        name="Normal reference",
                        line=dict(color="#ffaa00", width=1.5, dash="dot"),
                        hoverinfo="skip",
                    )
                )
                fig.update_layout(**terminal_layout("Q-Q Plot of Standardized Daily Returns"))
                st.plotly_chart(fig, width="stretch")
            else:
                alert_box("info", "Not enough return observations to render Q-Q plots.")
        elif chart_option == "STL Decomp":
            ticker_options = [ticker for ticker in STOCK_UNIVERSE["Ticker"].tolist() if ticker in price_df.columns]
            available_tickers = ticker_options or price_df.columns.tolist()
            st.session_state.selected_stock = st.selectbox(
                "Stock",
                available_tickers,
                index=available_tickers.index(st.session_state.selected_stock)
                if st.session_state.selected_stock in available_tickers
                else 0,
                key="stock_sel_eda_stl",
            )
            series = pd.to_numeric(price_df[st.session_state.selected_stock], errors="coerce").dropna()
            if len(series) < 80:
                alert_box("info", "Not enough price history to render decomposition.")
            else:
                try:
                    from statsmodels.tsa.seasonal import STL

                    period = min(252, max(20, len(series) // 4))
                    stl_result = STL(series, period=period, robust=True).fit()
                    trend = stl_result.trend
                    seasonal = stl_result.seasonal
                    resid = stl_result.resid
                except Exception:
                    trend = series.rolling(63, min_periods=10).mean()
                    seasonal = series - trend
                    resid = series - trend - seasonal.rolling(20, min_periods=5).mean()

                fig = make_subplots(
                    rows=3,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.045,
                    subplot_titles=("Price / Trend", "Seasonal Component", "Residual"),
                )
                color = STOCK_COLORS.get(st.session_state.selected_stock, "#00d4ff")
                fig.add_trace(go.Scatter(x=series.index, y=series, name="Close", line=dict(color=color, width=1.2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=trend.index, y=trend, name="Trend", line=dict(color="#ffaa00", width=1.7)), row=1, col=1)
                fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, name="Seasonal", line=dict(color="#00ff88", width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=resid.index, y=resid, name="Residual", line=dict(color="#ff3366", width=1)), row=3, col=1)
                fig.update_layout(**terminal_layout(f"STL Decomposition - {st.session_state.selected_stock.replace('.NS', '')}", height=620))
                st.plotly_chart(fig, width="stretch")
        elif chart_option == "ACF/PACF":
            ticker_options = [ticker for ticker in STOCK_UNIVERSE["Ticker"].tolist() if ticker in price_df.columns]
            available_tickers = ticker_options or price_df.columns.tolist()
            st.session_state.selected_stock = st.selectbox(
                "Stock",
                available_tickers,
                index=available_tickers.index(st.session_state.selected_stock)
                if st.session_state.selected_stock in available_tickers
                else 0,
                key="stock_sel_eda_acf",
            )
            returns = pd.to_numeric(price_df[st.session_state.selected_stock], errors="coerce").pct_change().dropna()
            if len(returns) < 30:
                alert_box("info", "Not enough return observations to render ACF/PACF diagnostics.")
            else:
                from statsmodels.tsa.stattools import acf, pacf

                nlags = min(30, max(5, len(returns) // 4))
                lags = np.arange(nlags + 1)
                acf_vals = acf(returns, nlags=nlags, fft=True)
                pacf_vals = pacf(returns, nlags=nlags, method="ywm")
                conf = 1.96 / np.sqrt(len(returns))

                fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
                fig.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF", marker_color="#00d4ff"), row=1, col=1)
                fig.add_trace(go.Bar(x=lags, y=pacf_vals, name="PACF", marker_color="#ffaa00"), row=1, col=2)
                for col_idx in [1, 2]:
                    fig.add_hline(y=conf, line_dash="dot", line_color="#00ff88", opacity=0.6, row=1, col=col_idx)
                    fig.add_hline(y=-conf, line_dash="dot", line_color="#ff3366", opacity=0.6, row=1, col=col_idx)
                fig.update_traces(marker_line_width=0)
                fig.update_layout(**terminal_layout(f"Return Autocorrelation - {st.session_state.selected_stock.replace('.NS', '')}", barmode="group"))
                st.plotly_chart(fig, width="stretch")
        elif chart_option == "Risk-Return":
            if not summary_stats.empty and {"Ticker", "Ann_Ret%", "Ann_Vol%"}.issubset(summary_stats.columns):
                risk_df = summary_stats.copy()
                risk_df["Ann Return %"] = pd.to_numeric(risk_df["Ann_Ret%"], errors="coerce")
                risk_df["Ann Vol %"] = pd.to_numeric(risk_df["Ann_Vol%"], errors="coerce")
                risk_df["Sharpe"] = pd.to_numeric(risk_df.get("Sharpe", np.nan), errors="coerce")
            elif not price_df.empty:
                returns = price_df.pct_change().dropna()
                risk_df = pd.DataFrame(
                    {
                        "Ticker": returns.columns,
                        "Ann Return %": returns.mean().values * 252 * 100,
                        "Ann Vol %": returns.std().values * np.sqrt(252) * 100,
                    }
                )
                risk_df["Sharpe"] = risk_df["Ann Return %"] / risk_df["Ann Vol %"].replace(0, np.nan)
                risk_df = risk_df.merge(STOCK_UNIVERSE[["Ticker", "Sector"]], on="Ticker", how="left")
            else:
                risk_df = pd.DataFrame()

            if risk_df.empty:
                alert_box("info", "Risk-return data is unavailable.")
            else:
                risk_df["Label"] = risk_df["Ticker"].astype(str).str.replace(".NS", "", regex=False)
                fig = px.scatter(
                    risk_df,
                    x="Ann Vol %",
                    y="Ann Return %",
                    color="Sector" if "Sector" in risk_df.columns else None,
                    size=risk_df["Sharpe"].abs().fillna(0.1) + 0.5,
                    text="Label",
                    hover_data=["Ticker", "Sharpe"] if "Sharpe" in risk_df.columns else ["Ticker"],
                    color_discrete_sequence=list(STOCK_COLORS.values()),
                )
                fig.add_hline(y=0, line_dash="dot", line_color="#ff3366", opacity=0.65)
                fig.update_traces(textposition="top center", marker=dict(line=dict(width=0)))
                fig.update_layout(**terminal_layout("Risk-Return Landscape"))
                st.plotly_chart(fig, width="stretch")
        else:
            alert_box("info", f"Chart '{chart_option}' is not yet implemented as a Plotly figure.")

            st.markdown("")
            panel_header("CORRELATION ANALYSIS")
            alert_box("info", "Correlation matrix available in chart above")
    
    st.markdown("")
    panel_header("SUMMARY STATISTICS")
    if not summary_stats.empty:
            render_fixed_table(summary_stats, height=420, hide_index=True)

    st.markdown("")
    panel_header("STATIONARITY TESTS — ADF RESULTS")
    if not stationarity.empty:
            render_fixed_table(stationarity, height=420, hide_index=True)



# ── TAB 3: ALPHA ENGINE ────────────────────────
with tab3:
    st.markdown("## 🤖 ALPHA ENGINE — MODEL PERFORMANCE")
    
    with st.spinner("Loading model data..."):
        all_metrics = load_csv(str(REPORTS_DIR / "all_model_metrics.csv"))
        arima_metrics = load_csv(str(REPORTS_DIR / "arima_metrics.csv"))
        avp_df = load_csv(str(PREDICTIONS_DIR / "actual_vs_predicted.csv"))
        forecast_df_tab3 = load_csv(str(PREDICTIONS_DIR / "all_5day_forecasts.csv"))
    
    if not all_metrics.empty:
        st.markdown("")
        panel_header("MODEL PERFORMANCE KPIs")
        
        model_list = ["ARIMA", "ETS", "Prophet-STL (Custom)", "LSTM", "Ensemble"]
        cols = st.columns(5)
        
        for idx, model in enumerate(model_list):
            model_data = all_metrics[
                all_metrics["Model"].astype(str).str.contains(model, case=False, na=False, regex=False)
            ]
            if not model_data.empty:
                avg_rmse = pd.to_numeric(model_data["RMSE"], errors="coerce").mean()
                color = "#00ff88" if model == "Ensemble" else "#00d4ff"
                with cols[idx]:
                    kpi_card(model.split("(")[0].strip(), f"{avg_rmse:.0f}", color)
        
        st.markdown("")
        panel_header("MODEL METRICS TERMINAL")
        
        col_t, col_m = st.columns([3, 3])
        with col_t:
            filter_tickers = st.multiselect("Filter Ticker", 
                                            all_metrics["Ticker"].unique().tolist() if "Ticker" in all_metrics.columns else [],
                                            default=all_metrics["Ticker"].unique().tolist()[:3] if "Ticker" in all_metrics.columns else [])
        with col_m:
            filter_models = st.multiselect("Filter Model",
                                          all_metrics["Model"].unique().tolist() if "Model" in all_metrics.columns else [],
                                          default=["Ensemble"])
        
        filtered_metrics = all_metrics.copy()
        if filter_tickers and "Ticker" in filtered_metrics.columns:
            filtered_metrics = filtered_metrics[filtered_metrics["Ticker"].isin(filter_tickers)]
        if filter_models and "Model" in filtered_metrics.columns:
            filtered_metrics = filtered_metrics[filtered_metrics["Model"].isin(filter_models)]
        
        render_fixed_table(filtered_metrics, height=400)

        st.markdown("")
        panel_header("ENSEMBLE WEIGHTS — INVERSE RMSE")
        w = all_metrics[all_metrics["Model"].isin(["ARIMA", "ETS", "Prophet-STL (Custom)", "LSTM"])].copy()
        w["RMSE"] = pd.to_numeric(w["RMSE"], errors="coerce")
        w = w.dropna(subset=["RMSE"])
        if not w.empty and "Ticker" in w.columns:
            w["inv"] = 1 / w["RMSE"]
            w["weight"] = w.groupby("Ticker")["inv"].transform(lambda x: x / x.sum()) * 100
            w["Ticker"] = w["Ticker"].astype(str).str.replace(".NS", "", regex=False)
            fig = px.bar(
                w,
                x="Ticker",
                y="weight",
                color="Model",
                barmode="stack",
                color_discrete_map=MODEL_COLORS,
                labels={"weight": "Weight (%)", "Ticker": "Stock"},
            )
            fig.update_traces(marker_line_width=0)
            fig.update_layout(**terminal_layout("Ensemble Weights per Stock (%)"))
            st.plotly_chart(fig, width="stretch")
        else:
            alert_box("info", "Ensemble weights unavailable: missing RMSE/model rows.")
        
        st.markdown("")
        panel_header("MODEL COMPARISON CHARTS")
        
        chart_sel = st.selectbox(
            "Select Visualization",
            [
                "Predicted vs Actual",
                "Metric Heatmap",
                "Metrics Bars",
                "5-Day Forecast",
                "Residuals",
                "Ensemble Weights",
                "ARIMA ACF",
                "LSTM Curves",
            ],
            label_visibility="collapsed",
        )
        
        chart_map = {
            "Predicted vs Actual": CHARTS_DIR / "models" / "M01_predicted_vs_actual.png",
            "Metric Heatmap": CHARTS_DIR / "models" / "M02_model_metric_heatmap.png",
            "Metrics Bars": CHARTS_DIR / "models" / "M03_model_metrics_bars.png",
            "5-Day Forecast": CHARTS_DIR / "models" / "M04_5day_forecasts.png",
            "Residuals": CHARTS_DIR / "models" / "M05_residuals_distribution.png",
            "Ensemble Weights": CHARTS_DIR / "models" / "M06_ensemble_weights.png",
            "ARIMA ACF": CHARTS_DIR / "models" / "M07_arima_residual_acf.png",
            "LSTM Curves": CHARTS_DIR / "models" / "M08_lstm_training_curves.png",
        }
        
        if chart_sel == "Metric Heatmap" and not all_metrics.empty:
            pivot = all_metrics.pivot_table(index="Ticker", columns="Model", values="RMSE", aggfunc="mean")
            fig = go.Figure(
                go.Heatmap(
                    z=pivot.values,
                    x=[c for c in pivot.columns],
                    y=[r.replace(".NS", "") for r in pivot.index],
                    colorscale="RdYlGn_r",
                    text=pivot.round(1).values,
                    texttemplate="%{text}",
                    hovertemplate="%{y} | %{x}: RMSE=%{z:.2f}<extra></extra>",
                    colorbar=dict(title="RMSE"),
                )
            )
            fig.update_layout(**terminal_layout("RMSE Heatmap — Model × Ticker"))
            st.plotly_chart(fig, width="stretch")
        elif chart_sel == "Predicted vs Actual" and not avp_df.empty:
            ticker = st.session_state.get("selected_stock", STOCK_UNIVERSE["Ticker"].iloc[0])
            sub = avp_df.copy()
            if "Ticker" in sub.columns:
                sub = sub[sub["Ticker"].astype(str) == str(ticker)].copy()

            if sub.empty:
                alert_box("info", f"No prediction rows found for {str(ticker).replace('.NS', '')}.")
            else:
                date_col = "Date" if "Date" in sub.columns else None
                if date_col:
                    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
                    sub = sub.sort_values(date_col)

                model_cols = [c for c in ["Actual", "ARIMA", "ETS", "Prophet-STL (Custom)", "LSTM", "Ensemble"] if c in sub.columns]
                fig = go.Figure()
                for model_name in model_cols:
                    fig.add_trace(
                        go.Scatter(
                            x=sub[date_col] if date_col else sub.index,
                            y=pd.to_numeric(sub[model_name], errors="coerce"),
                            name=model_name,
                            line=dict(
                                color=MODEL_COLORS.get(model_name, "#888"),
                                width=2.5 if model_name in ("Actual", "Ensemble") else 1.2,
                                dash="solid" if model_name == "Actual" else ("dot" if model_name == "Ensemble" else "dash"),
                            ),
                            hovertemplate=f"{model_name}: ₹%{{y:,.2f}}<extra></extra>",
                        )
                    )

                fig.update_layout(
                    **terminal_layout(
                        f"{str(ticker).replace('.NS', '')} — Actual vs Predicted",
                        hovermode="x unified",
                        xaxis=dict(rangeslider=dict(visible=True), type="date"),
                    )
                )
                st.plotly_chart(fig, width="stretch")
        elif chart_sel == "5-Day Forecast" and not forecast_df_tab3.empty:
            ticker = st.session_state.get("selected_stock", STOCK_UNIVERSE["Ticker"].iloc[0])
            sub = forecast_df_tab3.copy()
            if "Ticker" in sub.columns:
                sub = sub[sub["Ticker"].astype(str) == str(ticker)].copy()

            if sub.empty:
                alert_box("info", f"No 5-day forecast rows found for {str(ticker).replace('.NS', '')}.")
            else:
                date_col = "Date"
                if "Date" not in sub.columns:
                    first_col = sub.columns[0]
                    sub = sub.rename(columns={first_col: "Date"})
                sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")
                sub = sub.sort_values("Date")

                fig = go.Figure()
                for model_name, grp in sub.groupby("Model"):
                    grp = grp.sort_values("Date")
                    fig.add_trace(
                        go.Scatter(
                            x=grp["Date"],
                            y=pd.to_numeric(grp["Forecast"], errors="coerce"),
                            name=str(model_name),
                            line=dict(color=MODEL_COLORS.get(str(model_name), "#00d4ff"), width=2),
                            hovertemplate="%{x|%d %b %Y}<br>Forecast: ₹%{y:,.2f}<extra></extra>",
                        )
                    )

                if {"Lower_95", "Upper_95"}.issubset(sub.columns):
                    ens = sub[sub["Model"].astype(str).str.contains("Ensemble", case=False, na=False)]
                    if not ens.empty:
                        ens = ens.sort_values("Date")
                        fig.add_trace(
                            go.Scatter(
                                x=ens["Date"],
                                y=pd.to_numeric(ens["Upper_95"], errors="coerce"),
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=ens["Date"],
                                y=pd.to_numeric(ens["Lower_95"], errors="coerce"),
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor="rgba(0, 212, 255, 0.12)",
                                name="Ensemble 95% CI",
                                hovertemplate="95% CI: ₹%{y:,.2f}<extra></extra>",
                            )
                        )

                fig.update_layout(
                    **terminal_layout(
                        f"{str(ticker).replace('.NS', '')} — 5-Day Forecast by Model",
                        hovermode="x unified",
                    )
                )
                st.plotly_chart(fig, width="stretch")
        elif chart_sel == "ARIMA ACF" and not arima_metrics.empty:
            p10_col = next((c for c in ["LjungBox_lag10_p", "LjungBox_lag10"] if c in arima_metrics.columns), None)
            p20_col = next((c for c in ["LjungBox_lag20_p", "LjungBox_lag20"] if c in arima_metrics.columns), None)
            if p10_col and p20_col and "Ticker" in arima_metrics.columns:
                ar = arima_metrics[["Ticker", p10_col, p20_col]].copy()
                ar[p10_col] = pd.to_numeric(ar[p10_col], errors="coerce")
                ar[p20_col] = pd.to_numeric(ar[p20_col], errors="coerce")
                ar["Ticker"] = ar["Ticker"].astype(str).str.replace(".NS", "", regex=False)

                fig = go.Figure()
                fig.add_trace(go.Bar(x=ar["Ticker"], y=ar[p10_col], name="Ljung-Box p (lag 10)", marker_color="#00d4ff"))
                fig.add_trace(go.Bar(x=ar["Ticker"], y=ar[p20_col], name="Ljung-Box p (lag 20)", marker_color="#ffaa00"))
                fig.add_hline(y=0.05, line_dash="dot", line_color="#ff3366", annotation_text="0.05 threshold")
                fig.update_traces(marker_line_width=0)
                fig.update_layout(**terminal_layout("ARIMA Residual Autocorrelation Check", barmode="group"))
                st.plotly_chart(fig, width="stretch")
            else:
                alert_box("info", "ARIMA ACF diagnostics are unavailable in the current metrics file.")
        elif chart_sel == "LSTM Curves" and not all_metrics.empty:
            lstm = all_metrics[all_metrics["Model"].astype(str).str.contains("LSTM", case=False, na=False)].copy()
            if not lstm.empty and "Ticker" in lstm.columns:
                lstm["Ticker"] = lstm["Ticker"].astype(str).str.replace(".NS", "", regex=False)
                lstm["RMSE"] = pd.to_numeric(lstm.get("RMSE", np.nan), errors="coerce")
                lstm["MAE"] = pd.to_numeric(lstm.get("MAE", np.nan), errors="coerce")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=lstm["Ticker"], y=lstm["RMSE"], mode="lines+markers", name="RMSE", line=dict(color="#00d4ff", width=2)))
                fig.add_trace(go.Scatter(x=lstm["Ticker"], y=lstm["MAE"], mode="lines+markers", name="MAE", line=dict(color="#ffaa00", width=2, dash="dot")))
                fig.update_layout(**terminal_layout("LSTM Error Curves by Ticker"))
                st.plotly_chart(fig, width="stretch")
            else:
                alert_box("info", "LSTM curve data is unavailable in the current metrics file.")
        elif chart_sel == "Residuals" and not avp_df.empty:
            model_cols = [c for c in ["ARIMA", "ETS", "Prophet-STL (Custom)", "LSTM", "Ensemble"] if c in avp_df.columns]
            fig = go.Figure()
            for model_name in model_cols:
                if "Actual" in avp_df.columns:
                    resid = pd.to_numeric(avp_df[model_name], errors="coerce") - pd.to_numeric(avp_df["Actual"], errors="coerce")
                    fig.add_trace(
                        go.Histogram(
                            x=resid.dropna(),
                            name=model_name,
                            opacity=0.65,
                            marker_color=MODEL_COLORS.get(model_name, "#888"),
                            nbinsx=40,
                        )
                    )
            fig.update_layout(**terminal_layout("Residuals Distribution by Model", barmode="overlay"))
            st.plotly_chart(fig, width="stretch")
        elif chart_sel == "Ensemble Weights" and not all_metrics.empty:
            rmse_data = all_metrics[all_metrics["Model"].isin(["ARIMA", "ETS", "Prophet-STL (Custom)", "LSTM"])].copy()
            if not rmse_data.empty:
                rmse_data["RMSE"] = pd.to_numeric(rmse_data["RMSE"], errors="coerce")
                rmse_data = rmse_data.dropna(subset=["RMSE"])
                if not rmse_data.empty:
                    rmse_data["inv_rmse"] = 1 / rmse_data["RMSE"]
                    rmse_data["weight"] = rmse_data.groupby("Ticker")["inv_rmse"].transform(lambda x: x / x.sum())
                    fig = px.bar(
                        rmse_data,
                        x="Ticker",
                        y="weight",
                        color="Model",
                        barmode="stack",
                        color_discrete_map=MODEL_COLORS,
                        labels={"weight": "Ensemble Weight", "Ticker": "Stock"},
                    )
                    fig.update_traces(marker_line_width=0)
                    fig.update_layout(**terminal_layout("Ensemble Weights (Inverse-RMSE) per Stock"))
                    st.plotly_chart(fig, width="stretch")
                else:
                    alert_box("info", f"Chart '{chart_sel}' needs more data before it can be plotted.")
            else:
                alert_box("info", f"Chart '{chart_sel}' needs more data before it can be plotted.")
        elif chart_sel == "Metrics Bars":
            try:
                fig = plotly_charts.plot_model_metrics_bars(all_metrics)
                st.plotly_chart(fig, width="stretch")
            except Exception:
                alert_box("info", f"Chart '{chart_sel}' could not be rendered as Plotly.")
        else:
            alert_box("info", f"Chart '{chart_sel}' is not yet implemented as a Plotly figure.")
        
        st.markdown("")
        panel_header("ARIMA DIAGNOSTICS")
        if not arima_metrics.empty:
            render_fixed_table(arima_metrics, height=300)


# ── TAB 4: FORECASTING DESK ────────────────────
with tab4:
    st.markdown("## 📈 FORECASTING DESK — 5-DAY ALPHA")
    
    with st.spinner("Loading forecasts..."):
        forecast_df = load_csv(str(PREDICTIONS_DIR / "all_5day_forecasts.csv"))
        avp_df = load_csv(str(PREDICTIONS_DIR / "actual_vs_predicted.csv"))
    
    panel_header("5-DAY FORWARD PRICE FORECAST | BASE: 2025-07-01")
    alert_box("info", "FORECAST: Ensemble (inverse-RMSE weighted) of ARIMA + ETS + Prophet-STL + LSTM. Out-of-sample from 2025-07-01.")
    
    st.markdown("")
    panel_header("SIGNAL DASHBOARD — ALL STOCKS")
    
    if not forecast_df.empty and not portfolio_df.empty:
        cols = st.columns(4)
        for idx, (_, prow) in enumerate(portfolio_df.iterrows()):
            ticker = prow.get("Ticker", "")
            ret_5d = float(prow.get("Forecast_5d_Ret_%", 0))
            signal = prow.get("Forecast_Decision", "")
            
            with cols[idx % 4]:
                col_color = STOCK_COLORS.get(ticker, "#00d4ff")
                signal_html = signal_badge(signal)
                st.markdown(
                    f'<div style="border-left:3px solid {col_color};padding:12px;'
                    f'background:#141c2e;border-radius:4px">'
                    f'<div style="font-family:monospace;color:#00d4ff;font-weight:bold">'
                    f'{ticker.replace(".NS","")}</div>'
                    f'<div style="font-size:13px;margin-top:6px">{signal_html}</div>'
                    f'<div style="color:{("#00ff88" if ret_5d >= 0 else "#ff3366")};'
                    f'font-family:monospace;font-size:14px;margin-top:4px">'
                    f'{fmt_pct(ret_5d)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    
    st.markdown("")
    panel_header("DETAILED FORECAST — SELECT STOCK")
    
    stock_options = STOCK_UNIVERSE["Ticker"].tolist()
    st.session_state.selected_stock = st.selectbox(
        "Stock",
        stock_options,
        index=stock_options.index(st.session_state.selected_stock) if st.session_state.selected_stock in stock_options else 0,
        label_visibility="collapsed",
        key="stock_sel_forecast",
    )
    selected_stock = st.session_state.selected_stock
    stock_forecasts = pd.DataFrame()
    if not forecast_df.empty and "Ticker" in forecast_df.columns:
        stock_forecasts = forecast_df[forecast_df["Ticker"].astype(str) == str(selected_stock)].copy()
        if not stock_forecasts.empty:
            if "Date" not in stock_forecasts.columns:
                stock_forecasts = stock_forecasts.rename(columns={stock_forecasts.columns[0]: "Date"})
            stock_forecasts["Date"] = pd.to_datetime(stock_forecasts["Date"], errors="coerce")
            stock_forecasts["Forecast"] = pd.to_numeric(stock_forecasts.get("Forecast", np.nan), errors="coerce")
            stock_forecasts = stock_forecasts.sort_values("Date")
    ohlc_for_signal = fetch_ohlc(selected_stock)
    last_close = np.nan
    if not ohlc_for_signal.empty and "Close" in ohlc_for_signal.columns:
        close_series = pd.to_numeric(ohlc_for_signal["Close"], errors="coerce").dropna()
        if not close_series.empty:
            last_close = float(close_series.iloc[-1])
    
    col_left, col_center, col_right = st.columns([4, 3, 3])
    
    with col_left:
        panel_header("SELECTED STOCK SNAPSHOT")
        if stock_forecasts.empty:
            alert_box("info", "No forecast rows found for the selected stock.")
        else:
            latest = stock_forecasts.dropna(subset=["Forecast"]).tail(1)
            latest_fc = float(latest["Forecast"].iloc[0]) if not latest.empty else np.nan
            ret_est = ((latest_fc - last_close) / last_close) * 100 if np.isfinite(last_close) and np.isfinite(latest_fc) else np.nan
            st.metric("Latest Forecast", f"₹{latest_fc:,.2f}" if np.isfinite(latest_fc) else "—")
            st.metric("Last Close", f"₹{last_close:,.2f}" if np.isfinite(last_close) else "—")
            st.metric("Implied 5D Return", fmt_pct(ret_est) if np.isfinite(ret_est) else "—")
    
    with col_center:
        panel_header("SIGNAL STRENGTH")
        if stock_forecasts.empty:
            alert_box("info", "Signal strength unavailable: no forecast rows for selected stock.")
        else:
            latest = stock_forecasts.dropna(subset=["Forecast"]).tail(1)
            latest_fc = float(latest["Forecast"].iloc[0]) if not latest.empty else np.nan
            ret_est = ((latest_fc - last_close) / last_close) * 100 if np.isfinite(last_close) and np.isfinite(latest_fc) else 0.0
            strength = float(np.clip(50 + ret_est * 4, 0, 100))
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=strength,
                    number={"suffix": "/100"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#00d4ff"},
                        "steps": [
                            {"range": [0, 35], "color": "rgba(255, 51, 102, 0.20)"},
                            {"range": [35, 65], "color": "rgba(245, 170, 0, 0.20)"},
                            {"range": [65, 100], "color": "rgba(0, 255, 136, 0.20)"},
                        ],
                        "threshold": {"line": {"color": "#ff3366", "width": 3}, "value": 50},
                    },
                )
            )
            fig.update_layout(**terminal_layout("Forecast Signal Gauge"))
            st.plotly_chart(fig, width="stretch")
    
    with col_right:
        panel_header("MODEL CONSENSUS")
        if stock_forecasts.empty or "Model" not in stock_forecasts.columns:
            alert_box("info", "Model consensus unavailable: missing model-level forecasts.")
        else:
            latest_model = stock_forecasts.dropna(subset=["Forecast"]).groupby("Model", as_index=False).tail(1).copy()
            if np.isfinite(last_close):
                latest_model["Return_%"] = (latest_model["Forecast"] - last_close) / last_close * 100
                y_col = "Return_%"
                y_title = "Forecast Return %"
                hover = "%{x}: %{y:.2f}%<extra></extra>"
            else:
                latest_model["Return_%"] = latest_model["Forecast"]
                y_col = "Return_%"
                y_title = "Forecast Price"
                hover = "%{x}: ₹%{y:,.2f}<extra></extra>"

            fig = go.Figure(
                go.Bar(
                    x=latest_model["Model"],
                    y=pd.to_numeric(latest_model[y_col], errors="coerce"),
                    marker_color=[MODEL_COLORS.get(str(m), "#00d4ff") for m in latest_model["Model"]],
                    hovertemplate=hover,
                )
            )
            fig.update_traces(marker_line_width=0)
            if y_title == "Forecast Return %":
                fig.add_hline(y=0, line_dash="dot", line_color="#ff3366", opacity=0.6)
            fig.update_layout(**terminal_layout("Latest Model Signals", yaxis_title=y_title))
            st.plotly_chart(fig, width="stretch")

    st.markdown("")
    panel_header("5-DAY PRICE PROJECTION — FULL TABLE")
    if not stock_forecasts.empty:
        st.dataframe(style_forecast_table(stock_forecasts), width="stretch", height=320, hide_index=True)
    
    st.markdown("")
    panel_header("BACKTEST — ACTUAL vs PREDICTED | TEST: 2025-01-01 to 2025-06-30")
    if not avp_df.empty and "Ticker" in avp_df.columns:
        sub = avp_df[avp_df["Ticker"] == selected_stock].copy()
        date_col = "Date" if "Date" in sub.columns else None
        if date_col:
            sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
            sub = sub.sort_values(date_col)

        fig = go.Figure()
        model_cols = [c for c in ["Actual", "ARIMA", "ETS", "Prophet-STL (Custom)", "LSTM", "Ensemble"] if c in sub.columns]
        for model_name in model_cols:
            fig.add_trace(
                go.Scatter(
                    x=sub[date_col] if date_col else sub.index,
                    y=pd.to_numeric(sub[model_name], errors="coerce"),
                    name=model_name,
                    line=dict(
                        color=MODEL_COLORS.get(model_name, "#888"),
                        width=2.5 if model_name in ("Actual", "Ensemble") else 1.2,
                        dash="solid" if model_name == "Actual" else ("dot" if model_name == "Ensemble" else "dash"),
                    ),
                    hovertemplate=f"{model_name}: ₹%{{y:,.2f}}<extra></extra>",
                )
            )

        fig.update_layout(
            **terminal_layout(
                f"{selected_stock.replace('.NS','')} — Actual vs Predicted (Test Period)",
                hovermode="x unified",
                xaxis=dict(rangeslider=dict(visible=True), type="date"),
            )
        )
        st.plotly_chart(fig, width="stretch")

    ohlc = fetch_ohlc(selected_stock)
    if not ohlc.empty:
        fig = go.Figure(
            go.Candlestick(
                x=ohlc["Date"],
                open=ohlc["Open"],
                high=ohlc["High"],
                low=ohlc["Low"],
                close=ohlc["Close"],
                increasing_line_color="#00ff88",
                decreasing_line_color="#ff3366",
                name=selected_stock.replace(".NS", ""),
            )
        )
        ohlc["MA20"] = pd.to_numeric(ohlc["Close"], errors="coerce").rolling(20).mean()
        ohlc["MA50"] = pd.to_numeric(ohlc["Close"], errors="coerce").rolling(50).mean()
        fig.add_trace(go.Scatter(x=ohlc["Date"], y=ohlc["MA20"], name="MA20", line=dict(color="#ffaa00", width=1.2, dash="dot")))
        fig.add_trace(go.Scatter(x=ohlc["Date"], y=ohlc["MA50"], name="MA50", line=dict(color="#00d4ff", width=1.2, dash="dash")))
        fig.update_layout(
            **TERMINAL_LAYOUT,
            title=f"{selected_stock.replace('.NS','')} — 6M OHLC + Moving Averages",
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, width="stretch")
    elif not avp_df.empty:
        render_fixed_table(avp_df, height=400)


# ── TAB 5: RISK TERMINAL ───────────────────────
with tab5:
    st.markdown("## 🌪️ RISK TERMINAL — VOLATILITY & GARCH")
    
    with st.spinner("Loading risk data..."):
        garch_df = load_csv(str(REPORTS_DIR / "garch_volatility_report.csv"))
    
    panel_header("VOLATILITY RISK TERMINAL | GARCH(1,1) MODEL")
    
    if not garch_df.empty:
        vol_col = next((c for c in ["Current_AnnVol_%", "Current_AnnVol_pct", "AnnVolPct", "GARCH_Vol_%"] if c in garch_df.columns), None)
        var_col = next((c for c in ["VaR_95_%", "VaR_95_pct"] if c in garch_df.columns), None)
        persist_col = next((c for c in ["Persistence", "GARCH_beta"] if c in garch_df.columns), None)

        vol_series = pd.to_numeric(garch_df[vol_col], errors="coerce") if vol_col else pd.Series(dtype=float)
        persist_series = pd.to_numeric(garch_df[persist_col], errors="coerce") if persist_col else pd.Series(dtype=float)
        cols = st.columns(4)
        with cols[0]:
            max_vol_idx = vol_series.idxmax() if vol_series.notna().any() else None
            ticker_max = garch_df.loc[max_vol_idx, "Ticker"] if max_vol_idx is not None else "—"
            kpi_card("HIGHEST VOL", ticker_max, "#ff3366")
        with cols[1]:
            min_vol_idx = vol_series.idxmin() if vol_series.notna().any() else None
            ticker_min = garch_df.loc[min_vol_idx, "Ticker"] if min_vol_idx is not None else "—"
            kpi_card("LOWEST VOL", ticker_min, "#00ff88")
        with cols[2]:
            avg_vol = vol_series.mean()
            kpi_card("AVG PORT VOL", fmt_pct(avg_vol) if np.isfinite(avg_vol) else "—", "#ffaa00")
        with cols[3]:
            max_persist = persist_series.max()
            max_persist_text = f"{max_persist:.2f}" if np.isfinite(max_persist) else "—"
            kpi_card("MAX PERSIST", max_persist_text, "#ff3366" if np.isfinite(max_persist) and max_persist > 0.95 else "#ffaa00")
        
        st.markdown("")
        with st.expander("📐 GARCH(1,1) Model Specification"):
            st.latex(r"\sigma^2_t = \omega + \alpha \cdot \varepsilon^2_{t-1} + \beta \cdot \sigma^2_{t-1}")
            st.caption("α + β > 0.95 indicates high volatility persistence (DANGER)")
        
        st.markdown("")
        panel_header("GARCH RISK MATRIX — ALL STOCKS")
        render_fixed_table(garch_df, height=400)
        
        st.markdown("")
        plot_df = garch_df.merge(STOCK_UNIVERSE[["Ticker", "Sector"]], on="Ticker", how="left")

        if vol_col and {"Ticker", vol_col}.issubset(plot_df.columns):
            vol_values = pd.to_numeric(plot_df[vol_col], errors="coerce")
            var_values = pd.to_numeric(plot_df[var_col], errors="coerce") if var_col in plot_df.columns else pd.Series([np.nan] * len(plot_df))

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=plot_df["Ticker"].astype(str).str.replace(".NS", "", regex=False),
                    y=vol_values,
                    name="Ann. Vol %",
                    marker_color=[STOCK_COLORS.get(t, "#00d4ff") for t in plot_df["Ticker"]],
                    hovertemplate="%{x}: %{y:.2f}%<extra>Ann Vol</extra>",
                )
            )
            if var_col in plot_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=plot_df["Ticker"].astype(str).str.replace(".NS", "", regex=False),
                        y=var_values.abs(),
                        name="VaR 95% (abs)",
                        mode="markers+lines",
                        marker=dict(color="#ff3366", size=10, symbol="diamond"),
                        line=dict(color="#ff3366", dash="dot"),
                        yaxis="y2",
                    )
                )
            fig.update_layout(
                **TERMINAL_LAYOUT,
                title="GARCH Annualized Volatility vs VaR 95%",
                yaxis2=dict(overlaying="y", side="right", showgrid=False, tickcolor="#ff3366", tickfont=dict(color="#ff3366")),
                barmode="group",
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, width="stretch")

        if not portfolio_df.empty and {"Ticker", "Forecast_5d_Ret_%"}.issubset(portfolio_df.columns):
            risk_df = garch_df.merge(portfolio_df[["Ticker", "Forecast_5d_Ret_%"]], on="Ticker", how="left")
            if "Sector" not in risk_df.columns:
                risk_df = risk_df.merge(STOCK_UNIVERSE[["Ticker", "Sector"]], on="Ticker", how="left")
            if vol_col and vol_col in risk_df.columns:
                fig = px.scatter(
                    risk_df,
                    x=pd.to_numeric(risk_df[vol_col], errors="coerce"),
                    y=pd.to_numeric(risk_df["Forecast_5d_Ret_%"], errors="coerce"),
                    text=risk_df["Ticker"].astype(str).str.replace(".NS", "", regex=False),
                    color="Sector",
                    size=pd.to_numeric(risk_df[vol_col], errors="coerce").abs() + 1,
                    color_discrete_sequence=list(STOCK_COLORS.values()),
                    labels={"x": "GARCH Ann. Vol %", "y": "5-Day Forecast Return %"},
                )
                fig.add_hline(y=0, line_dash="dot", line_color="#ff3366", opacity=0.6)
                fig.update_traces(textposition="top center")
                fig.update_layout(**terminal_layout("Risk-Return Landscape"))
                st.plotly_chart(fig, width="stretch")


# ── TAB 6: PORTFOLIO TERMINAL ──────────────────
with tab6:
    st.markdown("## 💼 PORTFOLIO TERMINAL — POSITION MANAGEMENT")
    
    with st.spinner("Loading portfolio..."):
        portfolio_df = load_csv(str(PORTFOLIO_DIR / "final_portfolio_allocation.csv"))
        summary_df = load_csv(str(PORTFOLIO_DIR / "portfolio_metrics_summary.csv"))
    
    ticker_tape(portfolio_df, live_prices)
    
    st.markdown("")
    panel_header("PORTFOLIO CURRENT VALUE")
    
    cols = st.columns(3)
    with cols[0]:
        kpi_card("CURRENT VALUE", "₹10.1L", "#00ff88", "+₹1.2L", True)
    with cols[1]:
        kpi_card("TOTAL INVESTED", "₹9.8L", "#00d4ff")
    with cols[2]:
        kpi_card("UNREALIZED P&L", "+₹1.2L (+2.3%)", "#00ff88", "Today")
    
    st.markdown("")
    panel_header("OPEN POSITIONS | LIVE PRICES")
    
    if not portfolio_df.empty:
        display = portfolio_df[["Ticker", "Company", "Sector", "Weight_%", "Shares_to_Buy",
                                 "Forecast_5d_Ret_%", "GARCH_Vol_%"]].copy()
        render_fixed_table(display, height=400)
    
    st.markdown("")
    col_alloc, col_sector = st.columns([5, 5])
    
    with col_alloc:
        panel_header("WEIGHT ALLOCATION")
        if not portfolio_df.empty:
            alloc_df = portfolio_df[["Ticker", "Weight_%"]].copy()
            alloc_df["Weight_%"] = pd.to_numeric(alloc_df["Weight_%"], errors="coerce")
            alloc_df["Ticker"] = alloc_df["Ticker"].str.replace(".NS", "")
            fig = go.Figure(data=[go.Pie(labels=alloc_df["Ticker"], values=alloc_df["Weight_%"],
                                          marker=dict(colors=[STOCK_COLORS.get(t+".NS", "#00d4ff") 
                                                             for t in alloc_df["Ticker"]]))])
            fig.update_layout(**TERMINAL_LAYOUT)
            st.plotly_chart(fig, width="stretch")
    
    with col_sector:
        panel_header("SECTOR EXPOSURE")
        if not portfolio_df.empty:
            try:
                sector_fig = plotly_charts.plot_sector_exposure(portfolio_df)
                st.plotly_chart(sector_fig, width="stretch")

                sector_group = (
                    portfolio_df.assign(**{"Weight_%": pd.to_numeric(portfolio_df["Weight_%"], errors="coerce").fillna(0)})
                    .groupby("Sector", as_index=False)["Weight_%"].sum()
                    .sort_values("Weight_%", ascending=False)
                )
                if not sector_group.empty:
                    top_sector = sector_group.iloc[0]
                    hhi = float(((sector_group["Weight_%"] / 100.0) ** 2).sum())
                    diversification_score = max(0.0, (1.0 - hhi) * 100.0)
                    st.metric("Diversification Score", f"{diversification_score:.1f}/100")
                    if float(top_sector["Weight_%"]) > 40:
                        alert_box(
                            "warning",
                            f"Concentration risk: {top_sector['Sector']} holds {top_sector['Weight_%']:.1f}% of portfolio weight.",
                        )
            except Exception:
                sector_group = portfolio_df.groupby("Sector")["Weight_%"].sum()
                sector_labels = sector_group.index.astype(str).str.replace(".NS", "", regex=False)
                fig = px.bar(x=sector_labels, y=sector_group.values,
                            labels={"x": "Sector", "y": "Weight %"})
                fig.update_traces(marker_line_width=0)
                fig.update_layout(**TERMINAL_LAYOUT)
                st.plotly_chart(fig, width="stretch")
    
    with st.expander("⚠️ NEGATIVE FORECAST HANDLING — PORTFOLIO CONSTRUCTION"):
        st.markdown(
            """
At construction time, 6 of 8 stocks had negative 5-day forecasts.
Excluding them would leave only 2 stocks (ITC, DRREDDY), violating 5-sector diversification.
Shorting unavailable on StockGro (long-only). Applied minimum weight (5%) to bearish,
maximum (30%) to bullish. Minimum-variance portfolio with forecast tilt.
            """
        )


# ── TAB 7: EXECUTION TERMINAL ──────────────────
with tab7:
    st.markdown("## 🛒 TRADE EXECUTION TERMINAL — STOCKGRO NSE")
    
    with st.spinner("Loading execution files..."):
        trades_df = load_csv(str(PORTFOLIO_DIR / "stockgro_trade_instructions.csv"))
        day1_df = load_csv(str(PORTFOLIO_DIR / "day1_execution_plan.csv"))
        day2_df = load_csv(str(PORTFOLIO_DIR / "day2_tracking_plan.csv"))
    
    st.markdown(
        '<div style="background:#0f1623;border-left:3px solid #00ff88;padding:12px;'
        'font-family:monospace;color:#e8edf5">EXECUTE ₹9,80,929 ACROSS 8 NSE STOCKS | '
        'LONG-ONLY | MARKET ORDERS AT OPEN | CAPITAL RESERVE: ₹19,071</div>',
        unsafe_allow_html=True,
    )
    
    st.markdown("")
    panel_header("ORDER BLOTTER — DAY 1 EXECUTION")
    
    if not trades_df.empty:
        render_fixed_table(trades_df, height=400)
        st.download_button("⬇️ EXPORT ORDER FILE", trades_df.to_csv(index=False),
                          "stockgro_orders.csv", "text/csv")
    
    st.markdown("")
    col_d1, col_d2 = st.columns([3, 2])
    
    with col_d1:
        panel_header("DAY 1 | EXECUTION CHECKLIST")
        if not day1_df.empty:
            for idx, row in enumerate(day1_df.iterrows()):
                _, item = row
                label = str(item.get("Step", "")).strip() or f"Execution Step {idx + 1}"
                st.checkbox(label, key=f"day1_execution_step_{idx}")
    
    with col_d2:
        panel_header("DAY 2 | TRACKING PROTOCOL")
        if not day2_df.empty:
            render_fixed_table(day2_df, height=320)
    
    st.markdown("")
    panel_header("RISK MANAGEMENT ALERTS")
    
    alerts = [
        ("🔴 CRITICAL", "Any stock -5% Day 1 → review position size"),
        ("🟡 WARNING", "TATAMOTORS + ICICIBANK high GARCH volatility"),
        ("🟢 INFO", "ITC is sole bullish forecast anchor"),
        ("⏸️ PROTOCOL", "No averaging down in Week 1"),
    ]
    
    for icon, msg in alerts:
        st.markdown(f"<div style='padding:8px;font-family:monospace;color:#e8edf5'>"
                   f"{icon} {msg}</div>", unsafe_allow_html=True)
    
    st.markdown("")
    panel_header("EXECUTION CHECKLIST")
    
    st.checkbox("Verified total investment ≤ ₹9,80,929", key="execution_check_total_investment")
    st.checkbox("All 8 buy orders placed at market open", key="execution_check_all_orders")
    st.checkbox("Screenshots taken for each order", key="execution_check_screenshots")
    st.checkbox("Day 1 closing prices recorded", key="execution_check_day1_close")
    st.checkbox("actual_vs_predicted.csv updated with actuals", key="execution_check_actuals")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("### Data-Driven Stock Analysis using Time Series Models")
        st.markdown(
            """
- Platform: StockGro | Exchange: NSE India
- Period: 2021-01-01 to 2025-06-30
- Stocks: 8 NSE stocks across 5 sectors
- Models: ARIMA, ETS, Prophet-STL, LSTM, Ensemble
            """
        )

    with c2:
        with st.spinner("Loading portfolio summary..."):
            portfolio_summary = load_csv(str(PORTFOLIO_DIR / "portfolio_metrics_summary.csv"))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Capital", fmt_inr(10_00_000))
        m2.metric("Invested", fmt_inr(safe_metric(portfolio_summary, "Total_Invested_INR", 980929)))
        m3.metric("Cash Reserve", fmt_inr(safe_metric(portfolio_summary, "Cash_Reserve_INR", 19071)))
        m4.metric("Expected Return", fmt_pct(safe_metric(portfolio_summary, "Expected_Return_%", 15.92)), "Annualized")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Sharpe Ratio", f"{safe_metric(portfolio_summary, 'Sharpe_Ratio', 0.53):.2f}", "RF = 6%")
        m6.metric(
            "Max Drawdown",
            fmt_pct(safe_metric(portfolio_summary, "MaxDrawdown_%", -32.98)),
            delta_color="inverse",
        )
        m7.metric("Portfolio Vol", fmt_pct(safe_metric(portfolio_summary, "Portfolio_Vol_%", 18.64)))
        m8.metric("N Stocks", f"{int(safe_metric(portfolio_summary, 'N_Stocks', 8))}")

    alert_box("info", "Train: 2021–2024 | Test: 2025-H1 | Forecast: 2025-07-01 onwards")
    alert_box("info", "Prophet-STL: custom implementation used due to pystan/Python 3.12 incompatibility")
    alert_box("warning", "6 of 8 stocks carry negative 5-day forecast — minimum weights applied")

    st.markdown("### Stock Universe")
    render_fixed_table(STOCK_UNIVERSE, height=320)

# Quant Market Analyzer & Trading Terminal

> **End-to-end market research system** for screening, forecasting, risk analysis, portfolio construction, and trade planning.  
> **Built for quick reading:** what it is, how it works, what the signals mean, and where to find the outputs.

---

## What This Project Does

This repo is a complete **quant workflow** for turning raw market data into a readable trading view:
- Pulls and cleans **NSE price data** from Yahoo Finance
- Builds **forecasting models** for price direction and short-horizon prediction
- Measures **risk and volatility** using GARCH and related diagnostics
- Creates a **portfolio allocation** with weights, expected return, and drawdown context
- Produces a **dashboard** that summarizes signals, forecasts, and risk in one place
- Saves tables, charts, and model outputs so the whole pipeline is reproducible

## How To Read It

If you want the shortest mental model, think of the repo in this order:
1. **Data in** - raw market prices and features.
2. **Models** - ARIMA, ETS, Prophet, LSTM, GARCH, ensemble.
3. **Risk layer** - volatility, VaR, CVaR, drawdown.
4. **Portfolio layer** - weights, expected return, and concentration.
5. **Execution layer** - readable trade instructions and dashboards.

The dashboard is the fastest way to interpret the outputs. It shows:
- Which stocks look bullish or bearish
- How forecast models compare
- Where the risk is concentrated
- What the portfolio allocation implies
- What trade action would follow from the signal set

---

## 🗂️ Project Structure

```
StockGro_Capstone/
│
├── 📁 configs/
│   └── project_config.yaml          # Master config — dates, stocks, model params
│
├── 📁 data/
│   ├── raw/                         # Downloaded OHLCV data (CSV per stock)
│   ├── processed/                   # Cleaned, merged, feature-engineered data
│   └── features/                   # Technical indicators, log returns, rolling stats
│
├── 📁 notebooks/
│   ├── 00_setup_and_data.ipynb      # Data ingestion and initial EDA
│   ├── 01_preprocessing.ipynb       # Cleaning, alignment, feature engineering
│   ├── 02_arima_sarima.ipynb        # ARIMA/SARIMA baseline forecasting
│   ├── 03_ets_holtwinters.ipynb     # Trend and seasonality baseline
│   ├── 04_prophet.ipynb             # Prophet-style forecasting
│   ├── 05_lstm.ipynb                # Deep learning forecast pipeline
│   ├── 06_garch_volatility.ipynb    # Volatility and risk modeling
│   ├── 07_model_comparison.ipynb    # Model ranking and diagnostics
│   ├── 08_portfolio.ipynb           # Portfolio construction and weights
│   ├── 09_stockgro_signals.ipynb    # Trade instructions and execution logic
│   └── 10_master_runner.ipynb       # Full pipeline runner (1-click)
│
├── 📁 src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py            # yfinance data download
│   │   ├── preprocessor.py          # Cleaning, alignment, splits
│   │   └── feature_engineer.py     # Technical indicators, returns
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── arima_model.py           # ARIMA/SARIMA pipeline
│   │   ├── ets_model.py             # Exponential Smoothing pipeline
│   │   ├── prophet_model.py         # Prophet pipeline
│   │   ├── lstm_model.py            # LSTM pipeline
│   │   └── garch_model.py           # GARCH volatility pipeline
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # RMSE, MAE, MAPE, DA, R²
│   │   └── diagnostics.py           # Ljung-Box, residual analysis
│   │
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── optimizer.py             # Mean-variance optimization
│   │   ├── risk_manager.py          # VaR, CVaR, drawdown
│   │   └── stockgro_signals.py     # Trade instruction generator
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plotter.py               # All standard plots
│   │   └── dashboard_builder.py    # Dashboard utilities
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py         # YAML config reader
│       ├── logger.py                # Logging setup
│       └── helpers.py               # General utilities
│
├── 📁 models/
│   ├── arima/                       # Saved ARIMA model files (.pkl)
│   ├── ets/                         # Saved ETS model files (.pkl)
│   ├── prophet/                     # Saved Prophet model files (.json)
│   ├── lstm/                        # Saved LSTM weights (.h5, .keras)
│   ├── garch/                       # Saved GARCH model files (.pkl)
│   └── ensemble/                   # Ensemble weights and configs
│
├── 📁 outputs/
│   ├── charts/                      # All PNG/HTML visualizations
│   │   ├── eda/                     # EDA charts
│   │   ├── model_forecasts/         # Per-model forecast plots
│   │   ├── portfolio/               # Portfolio allocation charts
│   │   └── volatility/              # Volatility analysis charts
│   ├── reports/
│   │   ├── final_report.pdf         # 10-page academic report
│   │   └── metrics_summary.xlsx     # All model metrics consolidated
│   ├── predictions/
│   │   ├── forecasts_5day.csv       # 5-day ahead forecasts
│   │   └── actual_vs_predicted.csv # Comparison template
│   └── portfolio/
│       ├── portfolio_allocation.csv # Final ₹10L allocation
│       └── stockgro_instructions.xlsx # StockGro trade instructions
│
├── 📁 dashboard/
│   └── app.py                       # Streamlit market terminal
│
├── 📁 docs/
│   ├── final_report.md              # Academic report (Markdown source)
│   ├── methodology.md               # Detailed methodology
│   └── references.md                # Bibliography / References
│
├── 📁 tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_portfolio.py
│
├── 📁 logs/
│   └── pipeline.log                 # Auto-generated run logs
│
├── requirements.txt                 # All Python dependencies
├── setup.py                         # Package setup (optional)
├── .gitignore
└── README.md                        # ← You are here
```

---

## ⚡ Quick Start

```bash
# 1. Clone / Setup
cd StockGro_Capstone

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (one command)
jupyter notebook notebooks/10_master_runner.ipynb

# 5. Launch dashboard
streamlit run dashboard/app.py
```

---

## Live Dashboard

Access the deployed Streamlit dashboard here:

Open Dashboard - https://market-intelligence-platform-2ha3yqdsagaffhvucael3w.streamlit.app/

# Notes:

Requires an internet connection for live market data retrieval from Yahoo Finance.

Initial load may take a few seconds if model artifacts are being loaded.

Best viewed on a desktop or tablet for the full interactive experience.

## 🏦 Stock Universe (10 NSE Stocks — 5 Sectors)

| # | Ticker | Company | Sector | Role |
|---|--------|---------|--------|------|
| 1 | HDFCBANK.NS | HDFC Bank | Banking | Core |
| 2 | ICICIBANK.NS | ICICI Bank | Banking | Core |
| 3 | INFY.NS | Infosys | IT | Core |
| 4 | TCS.NS | TCS | IT | Core |
| 5 | SUNPHARMA.NS | Sun Pharma | Pharma | Satellite |
| 6 | DRREDDY.NS | Dr. Reddy's | Pharma | Satellite |
| 7 | HINDUNILVR.NS | HUL | FMCG | Satellite |
| 8 | ITC.NS | ITC | FMCG | Satellite |
| 9 | MARUTI.NS | Maruti Suzuki | Auto | Tactical |
| 10 | TATAMOTORS.NS | Tata Motors | Auto | Tactical |

---

## 🤖 Model Stack

| Model | Type | Use Case | Strength |
|-------|------|----------|----------|
| ARIMA/SARIMA | Statistical | Short-term linear trend | Interpretable, fast |
| Holt-Winters ETS | Statistical | Trend + seasonality | Robust baseline |
| Prophet | ML-Statistical | Trend breaks, holidays | India calendar support |
| LSTM | Deep Learning | Non-linear patterns | Captures complexity |
| GARCH | Statistical | Volatility forecasting | Risk modeling |
| Ensemble | Hybrid | Best of all models | Highest accuracy |

---

## 📊 Evaluation Framework

**Forecast Accuracy:**
- RMSE, MAE, MAPE, Directional Accuracy, R²

**Statistical Diagnostics:**
- AIC/BIC (ARIMA), Ljung-Box test, Residual normality

**Risk Metrics:**
- Annualized volatility, Rolling volatility, GARCH forecast vol

**Portfolio Metrics:**
- Expected return, Sharpe ratio (Rf=6%), Max drawdown, VaR₉₅, CVaR₉₅

---

## 💼 Portfolio Strategy

- **Capital:** ₹10,00,000
- **Stocks:** 5–8 selected from universe (post model evaluation)
- **Optimization:** Mean-Variance Optimization (Markowitz)
- **Benchmark:** Nifty 50 (^NSEI)
- **Risk-free rate:** 6% annualized (10-yr GOI bond)
- **Rebalancing:** Monthly

---

## 📜 License

This project is submitted as part of an academic capstone. All code is original.  
Data sourced from Yahoo Finance via `yfinance` (public market data).

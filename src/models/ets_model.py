"""
src/models/ets_model.py
══════════════════════════════════════════════════════════════════
Phase 6 — Holt-Winters Exponential Smoothing Module
StockGro Capstone Project

THEORY
------
Holt-Winters (Triple Exponential Smoothing) decomposes a series into:
  Level (α)  : current value with smoothing
  Trend (β)  : local slope with smoothing
  Season (γ) : periodic pattern (weekly for daily stock data)

Additive form  : Y_t = L_t + T_t + S_t + ε_t
Multiplicative : Y_t = L_t × T_t × S_t × ε_t

For stock prices (multiplicative trends, low seasonality) we test both
and select via AIC.
══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import warnings, logging
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ETS")

# ── Shared metrics ────────────────────────────────────────────
def _rmse(a, b): return float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))
def _mae(a, b):  return float(np.mean(np.abs(np.array(a)-np.array(b))))
def _mape(a, b):
    a, b = np.array(a), np.array(b)
    m = a != 0
    return float(np.mean(np.abs((a[m]-b[m])/a[m]))*100)
def _da(a, b):
    da = np.diff(np.array(a))>0; dp = np.diff(np.array(b))>0
    return float(np.mean(da==dp)*100)
def _r2(a, b):
    a = np.array(a)
    ss_r = np.sum((a-np.array(b))**2); ss_t = np.sum((a-a.mean())**2)
    return float(1 - ss_r/(ss_t+1e-10))


# ══════════════════════════════════════════════════════════════════════════════
class ETSForecaster:
    """
    Holt-Winters ETS forecaster.
    Automatically selects best trend/seasonal combination via AIC.

    Parameters
    ----------
    ticker  : str   NSE ticker symbol
    name    : str   Company display name
    season  : int   Seasonal period (5 = weekly trading cycle)
    """

    CONFIGS = [
        # (trend, seasonal, seasonal_periods)
        ('add',  None,  None,  'Holt-Additive'),
        ('add',  'add', 5,     'Holt-Winters-Add'),
        ('add',  'mul', 5,     'Holt-Winters-Mul'),
        ('mul',  'add', 5,     'Damped-Add'),
        ('mul',  'mul', 5,     'Damped-Mul'),
    ]

    def __init__(self, ticker: str, name: str, season: int = 5):
        self.ticker       = ticker
        self.name         = name
        self.season       = season
        self.best_config  = None
        self.best_label   = None
        self.model_fit    = None
        self.train_series = None
        self.test_series  = None
        self.predictions  = None
        self.forecast_5d  = None
        self.metrics      = {}
        self.aic          = None

    # ── Step 1: Select best configuration ────────────────────────────────────
    def fit_best(self, train_series: pd.Series) -> str:
        self.train_series = train_series.dropna()
        best_aic = np.inf
        best_fit = None
        best_lbl = None

        log.info(f"  [{self.ticker}] ETS model selection ...")

        for trend, seasonal, sp, label in self.CONFIGS:
            try:
                m = ExponentialSmoothing(
                    self.train_series,
                    trend           = trend,
                    seasonal        = seasonal,
                    seasonal_periods= sp,
                    initialization_method='estimated',
                ).fit(optimized=True, remove_bias=True)

                if m.aic < best_aic:
                    best_aic = m.aic
                    best_fit = m
                    best_lbl = label
                    self.best_config = (trend, seasonal, sp)

            except Exception as e:
                log.debug(f"    Config {label} failed: {e}")

        self.model_fit = best_fit
        self.best_label = best_lbl
        self.aic = best_aic
        log.info(f"  [{self.ticker}] Best ETS: {best_lbl}  AIC={best_aic:.2f}")
        return best_lbl

    # ── Step 2: Rolling 1-step-ahead predictions ──────────────────────────────
    def predict_test(self, test_series: pd.Series) -> pd.Series:
        self.test_series = test_series.dropna()
        trend, seasonal, sp = self.best_config
        history = list(self.train_series)
        preds   = []

        for true_val in self.test_series:
            try:
                m = ExponentialSmoothing(
                    history,
                    trend           = trend,
                    seasonal        = seasonal,
                    seasonal_periods= sp,
                    initialization_method='estimated',
                ).fit(optimized=True, remove_bias=True)
                p = float(m.forecast(1).iloc[0])
            except Exception:
                p = history[-1]   # fallback: carry last value
            preds.append(p)
            history.append(true_val)

        self.predictions = pd.Series(preds,
                                     index=self.test_series.index,
                                     name=f"ETS_{self.ticker}")
        log.info(f"  [{self.ticker}] ETS rolling prediction complete")
        return self.predictions

    # ── Step 3: 5-day forecast ────────────────────────────────────────────────
    def forecast_5_days(self) -> pd.DataFrame:
        full = pd.concat([self.train_series, self.test_series])
        trend, seasonal, sp = self.best_config
        m = ExponentialSmoothing(
            full, trend=trend, seasonal=seasonal,
            seasonal_periods=sp,
            initialization_method='estimated',
        ).fit(optimized=True, remove_bias=True)

        fc   = m.forecast(5)
        last = full.index[-1]
        future = pd.bdate_range(start=last, periods=6, freq='B')[1:]

        # ETS confidence intervals (approximate: ±1.96 × RMSE of training fit)
        resid_std = float(np.sqrt(np.mean(m.resid**2)))
        half_width = 1.96 * resid_std

        self.forecast_5d = pd.DataFrame({
            'Date'    : future,
            'Forecast': fc.values,
            'Lower_95': fc.values - half_width,
            'Upper_95': fc.values + half_width,
            'Model'   : f'ETS({self.best_label})',
            'Ticker'  : self.ticker,
        }).set_index('Date')

        log.info(f"  [{self.ticker}] ETS 5-day forecast: "
                 f"{self.forecast_5d['Forecast'].values.round(2)}")
        return self.forecast_5d

    # ── Step 4: Metrics ───────────────────────────────────────────────────────
    def compute_metrics(self) -> dict:
        y_true = self.test_series.values
        y_pred = self.predictions.values
        self.metrics = {
            'Ticker' : self.ticker,
            'Company': self.name,
            'Model'  : f'ETS({self.best_label})',
            'Config' : str(self.best_config),
            'AIC'    : round(self.aic, 2),
            'RMSE'   : round(_rmse(y_true, y_pred), 4),
            'MAE'    : round(_mae(y_true, y_pred), 4),
            'MAPE_%' : round(_mape(y_true, y_pred), 4),
            'DA_%'   : round(_da(y_true, y_pred), 2),
            'R2'     : round(_r2(y_true, y_pred), 4),
        }
        return self.metrics

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(self, train_series: pd.Series,
            test_series: pd.Series) -> dict:
        self.fit_best(train_series)
        self.predict_test(test_series)
        self.forecast_5_days()
        self.compute_metrics()
        return self.metrics

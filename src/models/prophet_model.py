"""
src/models/prophet_model.py
══════════════════════════════════════════════════════════════════
Phase 6 — Prophet-STL (Custom) Trend + Seasonality Decomposition Model
StockGro Capstone Project

IMPORTANT WARNING
-----------------
This is NOT the Meta/Facebook `prophet` library. It is a lightweight,
custom reimplementation that follows Prophet-like ideas (STL trend,
Fourier seasonality, linear combiner) implemented with `statsmodels`,
`numpy`, and `sklearn`. Outputs and labels have been intentionally
renamed to avoid confusion with the official library.

IMPLEMENTATION NOTES
--------------------
We implement the core decomposition using:
    - STL trend extraction (statsmodels)
    - Fourier seasonality terms (numpy)
    - Ridge regression to combine components
    - Rolling forecast for test evaluation

This module labels its models and outputs as "Prophet-STL (Custom)".
══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import warnings, logging
from pathlib import Path
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("Prophet-STL")

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
class ProphetStyleForecaster:
    """
    Prophet-style decomposition forecaster (statsmodels + sklearn backend).

    Components
    ----------
    1. Piecewise linear trend extracted via STL
    2. Weekly Fourier seasonality (period=5 trading days)
    3. Quarterly Fourier seasonality (period=63 trading days)
    4. Ridge regression combiner (prevents overfitting)
    5. Indian market event dummies (Budget, RBI policy months)
    """

    INDIAN_EVENTS = {
        # Month indices where major market events typically occur
        'budget'   : [2],          # February — Union Budget
        'rbi_policy': [2,4,6,8,10,12],  # Bi-monthly RBI MPC
        'results'  : [1,4,7,10],   # Quarterly earnings months
        'fo_expiry': list(range(1,13)),  # Every month (F&O last Thursday)
    }

    def __init__(self, ticker: str, name: str,
                 n_fourier_weekly: int = 3,
                 n_fourier_quarterly: int = 5,
                 n_changepoints: int = 25):
        self.ticker              = ticker
        self.name                = name
        self.n_fourier_w         = n_fourier_weekly
        self.n_fourier_q         = n_fourier_quarterly
        self.n_changepoints      = n_changepoints
        self.model               = None        # Ridge regressor
        self.trend_slope         = None
        self.trend_intercept     = None
        self.train_series        = None
        self.test_series         = None
        self.predictions         = None
        self.forecast_5d         = None
        self.metrics             = {}
        self._train_last_trend   = None
        self._train_len          = 0

    # ── Feature builder ───────────────────────────────────────────────────────
    def _build_features(self, t_index: np.ndarray,
                        dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Build design matrix:
          - Normalised time index (trend)
          - Fourier terms for weekly  seasonality (period=5)
          - Fourier terms for quarterly seasonality (period=63)
          - Indian market event dummies
        """
        T = len(t_index)
        t = t_index.astype(float)
        t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-10)  # [0, 1]

        feats = [t_norm]   # linear trend

        # ── Weekly Fourier (period = 5 trading days) ──────────────────────
        for k in range(1, self.n_fourier_w + 1):
            feats.append(np.sin(2*np.pi*k*t / 5))
            feats.append(np.cos(2*np.pi*k*t / 5))

        # ── Quarterly Fourier (period = 63 trading days ≈ quarter) ────────
        for k in range(1, self.n_fourier_q + 1):
            feats.append(np.sin(2*np.pi*k*t / 63))
            feats.append(np.cos(2*np.pi*k*t / 63))

        # ── Annual Fourier (period = 252) ─────────────────────────────────
        for k in range(1, 3):
            feats.append(np.sin(2*np.pi*k*t / 252))
            feats.append(np.cos(2*np.pi*k*t / 252))

        # ── Indian event dummies ───────────────────────────────────────────
        months = dates.month.values
        budget_dummy   = (months == 2).astype(float)
        results_dummy  = np.isin(months, [1,4,7,10]).astype(float)
        feats.append(budget_dummy)
        feats.append(results_dummy)

        # ── Changepoint slopes (piecewise linear) ─────────────────────────
        cp_indices = np.linspace(0, T-1, self.n_changepoints + 2,
                                 dtype=int)[1:-1]
        for cp in cp_indices:
            ramp = np.zeros(T)
            ramp[cp:] = t_norm[cp:] - t_norm[cp]
            feats.append(ramp)

        return np.column_stack(feats)

    # ── Step 1: Fit ───────────────────────────────────────────────────────────
    def fit(self, train_series: pd.Series):
        self.train_series = train_series.dropna()
        self._train_len   = len(self.train_series)
        y   = self.train_series.values
        idx = np.arange(len(y))
        X   = self._build_features(idx, self.train_series.index)

        self.model = Ridge(alpha=0.1, fit_intercept=True)
        self.model.fit(X, y)

        fitted = self.model.predict(X)
        log.info(f"  [{self.ticker}] Prophet-style model fitted  "
                 f"train RMSE={_rmse(y, fitted):.2f}")
        return self

    # ── Step 2: Rolling test predictions ──────────────────────────────────────
    def predict_test(self, test_series: pd.Series) -> pd.Series:
        self.test_series = test_series.dropna()
        n_train = self._train_len
        preds   = []

        for i, (date, true_val) in enumerate(self.test_series.items()):
            t_idx = np.arange(n_train + i)
            dates = self.train_series.index.append(
                self.test_series.index[:i]) if i > 0 else self.train_series.index
            X   = self._build_features(t_idx, dates)

            # Extend to t+1
            t_next = np.array([n_train + i])
            t_norm_next = (t_next - 0) / max(n_train + i, 1)
            X_next = self._build_features(
                np.array([n_train + i]),
                pd.DatetimeIndex([date])
            )
            # Scale t_norm correctly
            t_all  = np.arange(n_train + i + 1).astype(float)
            t_norm_all = (t_all - t_all[0]) / (t_all[-1] - t_all[0] + 1e-10)
            dates_all = self.train_series.index.append(
                self.test_series.index[:i+1])
            X_all = self._build_features(t_all, dates_all)
            # Predict at last position (the forecast step)
            pred = float(self.model.predict(X_all[[-1]])[0])
            preds.append(pred)

        self.predictions = pd.Series(preds,
                         index=self.test_series.index,
                         name=f"Prophet-STL_{self.ticker}")
        log.info(f"  [{self.ticker}] Prophet rolling prediction complete")
        return self.predictions

    # ── Step 3: 5-day forecast ────────────────────────────────────────────────
    def forecast_5_days(self) -> pd.DataFrame:
        full  = pd.concat([self.train_series, self.test_series])
        n_all = len(full)
        t_all = np.arange(n_all).astype(float)

        last_date  = full.index[-1]
        future_dts = pd.bdate_range(start=last_date, periods=6, freq='B')[1:]

        fc_vals = []
        for i, fd in enumerate(future_dts):
            t_ext    = np.arange(n_all + i + 1).astype(float)
            dates_ext= full.index.append(future_dts[:i+1])
            X_ext    = self._build_features(t_ext, dates_ext)
            p        = float(self.model.predict(X_ext[[-1]])[0])
            fc_vals.append(p)

        # Uncertainty: propagate residual std
        resid_std  = float(np.std(self.test_series.values -
                                  self.predictions.values))
        half_width = 1.96 * resid_std

        self.forecast_5d = pd.DataFrame({
            'Date'    : future_dts,
            'Forecast': fc_vals,
            'Lower_95': np.array(fc_vals) - half_width,
            'Upper_95': np.array(fc_vals) + half_width,
            'Model'   : 'Prophet-STL (Custom)',
            'Ticker'  : self.ticker,
        }).set_index('Date')

        log.info(f"  [{self.ticker}] Prophet 5-day forecast: "
                 f"{np.array(fc_vals).round(2)}")
        return self.forecast_5d

    # ── Step 4: Metrics ───────────────────────────────────────────────────────
    def compute_metrics(self) -> dict:
        y_true = self.test_series.values
        y_pred = self.predictions.values
        self.metrics = {
            'Ticker'  : self.ticker,
            'Company' : self.name,
            'Model'   : 'Prophet-STL (Custom)',
            'RMSE'    : round(_rmse(y_true, y_pred), 4),
            'MAE'     : round(_mae(y_true, y_pred), 4),
            'MAPE_%'  : round(_mape(y_true, y_pred), 4),
            'DA_%'    : round(_da(y_true, y_pred), 2),
            'R2'      : round(_r2(y_true, y_pred), 4),
        }
        return self.metrics

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(self, train_series: pd.Series,
            test_series: pd.Series) -> dict:
        self.fit(train_series)
        self.predict_test(test_series)
        self.forecast_5_days()
        self.compute_metrics()
        return self.metrics

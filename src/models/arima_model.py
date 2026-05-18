"""
src/models/arima_model.py
══════════════════════════════════════════════════════════════════
Phase 6 — ARIMA / SARIMA Forecasting Module
StockGro Capstone Project

PIPELINE
--------
1. Auto-select best (p,d,q) via AIC grid search
2. Fit ARIMA on training close prices
3. Rolling 1-step-ahead predictions on test set
4. 5-day ahead forecast with confidence intervals
5. Full residual diagnostics (Ljung-Box, normality)
6. Save model params, metrics, forecasts
══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import warnings, logging, pickle
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ARIMA")

# ── Metrics ──────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def mape(y_true, y_pred):
    y_t = np.array(y_true); y_p = np.array(y_pred)
    mask = y_t != 0
    return float(np.mean(np.abs((y_t[mask]-y_p[mask])/y_t[mask]))*100)

def directional_accuracy(y_true, y_pred):
    y_t = np.array(y_true); y_p = np.array(y_pred)
    dir_t = np.diff(y_t) > 0
    dir_p = np.diff(y_p) > 0
    return float(np.mean(dir_t == dir_p) * 100)

def r2_score(y_true, y_pred):
    y_t = np.array(y_true); y_p = np.array(y_pred)
    ss_res = np.sum((y_t - y_p)**2)
    ss_tot = np.sum((y_t - y_t.mean())**2)
    return float(1 - ss_res / (ss_tot + 1e-10))


# ══════════════════════════════════════════════════════════════════════════════
class ARIMAForecaster:
    """
    Auto-ARIMA forecaster with grid search, diagnostics, and 5-day forecast.

    Parameters
    ----------
    ticker      : str   NSE ticker symbol
    name        : str   Company display name
    max_p, max_q: int   Search grid bounds for AR and MA orders
    d           : int   Differencing order (1 for all NSE stocks confirmed by ADF)
    """

    def __init__(self, ticker: str, name: str,
                 max_p: int = 3, max_q: int = 3, d: int = 1):
        self.ticker  = ticker
        self.name    = name
        self.max_p   = max_p
        self.max_q   = max_q
        self.d       = d
        self.best_order   = None
        self.model_fit    = None
        self.train_series = None
        self.test_series  = None
        self.predictions  = None
        self.forecast_5d  = None
        self.metrics      = {}
        self.aic = self.bic = None

    def _fit_arima(self, series: pd.Series, order: tuple, context: str):
        """Fit ARIMA and return None when the solver fails for a window."""
        try:
            return ARIMA(series, order=order).fit(method_kwargs={"warn_convergence": False})
        except Exception as exc:
            log.warning(f"  [{self.ticker}] {context} failed for order {order}: {exc}")
            return None

    # ── Step 1: Auto-ARIMA (AIC grid search) ─────────────────────────────────
    def fit_auto_arima(self, train_series: pd.Series) -> tuple:
        """Grid search (p,d,q) minimising AIC on training data."""
        self.train_series = train_series.dropna()
        best_aic = np.inf
        best_order = (1, self.d, 1)

        log.info(f"  [{self.ticker}] Auto-ARIMA grid search "
                 f"p∈[0,{self.max_p}] d={self.d} q∈[0,{self.max_q}] ...")

        for p in range(0, self.max_p + 1):
            for q in range(0, self.max_q + 1):
                try:
                    m = ARIMA(self.train_series,
                              order=(p, self.d, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, self.d, q)
                except Exception:
                    pass

        self.best_order = best_order
        log.info(f"  [{self.ticker}] Best order: {best_order}  AIC={best_aic:.2f}")

        # ── Refit with best order ─────────────────────────────────────────
        self.model_fit = self._fit_arima(self.train_series,
                                         self.best_order,
                                         "Final training refit")
        if self.model_fit is None:
            fallback_orders = [(0, self.d, 0), (1, self.d, 0), (0, self.d, 1)]
            for candidate_order in fallback_orders:
                self.model_fit = self._fit_arima(self.train_series,
                                                 candidate_order,
                                                 "Fallback training refit")
                if self.model_fit is not None:
                    self.best_order = candidate_order
                    break
        if self.model_fit is None:
            raise RuntimeError(f"[{self.ticker}] Unable to fit any ARIMA order on training data")
        self.aic = self.model_fit.aic
        self.bic = self.model_fit.bic
        log.info(f"  [{self.ticker}] AIC={self.aic:.2f}  BIC={self.bic:.2f}")
        return self.best_order

    # ── Step 2: Rolling 1-step-ahead predictions on test set ─────────────────
    def predict_test(self, test_series: pd.Series) -> pd.Series:
        """
        Walk-forward (expanding window) 1-step-ahead predictions on the test set.

        For each test observation:
          1. Refit ARIMA on all available history up to that point.
          2. Forecast 1 step ahead.
          3. Append the true observation to history.

        Robust handling:
          - Catches fitting failures on individual steps.
          - Falls back to the last successful prediction if a refit fails.
          - Tries a simpler ARIMA(1,1,1) if the best_order fails repeatedly.
        """
        import numpy as _np
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
        from numpy.linalg import LinAlgError as _LinAlgError

        self.test_series = test_series
        history = list(self.train_series)
        predictions = []
        last_pred = float(history[-1])
        consecutive_fails = 0
        MAX_FAILS = 5
        fallback_order = (1, 1, 1)

        for i, true_val in enumerate(test_series):
            fitted = False

            # Attempt 1: use best_order from auto_arima
            if consecutive_fails < MAX_FAILS:
                try:
                    m = _ARIMA(history, order=self.best_order).fit(
                        method_kwargs={"warn_convergence": False}
                    )
                    pred = float(m.forecast(steps=1).iloc[0])
                    last_pred = pred
                    fitted = True
                    consecutive_fails = 0
                except (_LinAlgError, Exception):
                    consecutive_fails += 1

            # Attempt 2: fallback to ARIMA(1,1,1)
            if not fitted:
                try:
                    m = _ARIMA(history, order=fallback_order).fit(
                        method_kwargs={"warn_convergence": False}
                    )
                    pred = float(m.forecast(steps=1).iloc[0])
                    last_pred = pred
                    fitted = True
                except Exception:
                    pass

            # Attempt 3: carry forward last known prediction
            if not fitted:
                pred = last_pred

            predictions.append(pred)
            history.append(true_val)  # always use true value — no data leakage

        self.predictions = pd.Series(predictions, index=test_series.index, name="Predicted")
        log.info(f"  [{self.ticker}] Rolling prediction complete ({len(predictions)} steps)")
        return self.predictions

    # ── Step 3: 5-day ahead forecast ─────────────────────────────────────────
    def forecast_5_days(self) -> pd.DataFrame:
        """Forecast 5 trading days beyond the test period."""
        full_series = pd.concat([self.train_series, self.test_series])
        final_model = self._fit_arima(full_series, self.best_order, "5-day forecast refit")

        if final_model is None:
            last_value = float(full_series.iloc[-1])
            future_dates = pd.bdate_range(start=full_series.index[-1], periods=6,
                                          freq='B')[1:]
            self.forecast_5d = pd.DataFrame({
                'Date'          : future_dates,
                'Forecast'      : [last_value] * 5,
                'Lower_95'      : [last_value] * 5,
                'Upper_95'      : [last_value] * 5,
                'Model'         : 'ARIMA',
                'Ticker'        : self.ticker,
            }).set_index('Date')
            log.warning(f"  [{self.ticker}] Using naive fallback for 5-day forecast")
            return self.forecast_5d

        fc_res  = final_model.get_forecast(steps=5)
        fc_mean = fc_res.predicted_mean
        fc_ci   = fc_res.conf_int(alpha=0.05)

        # Build future business day dates
        last_date = full_series.index[-1]
        future_dates = pd.bdate_range(start=last_date, periods=6,
                                      freq='B')[1:]

        self.forecast_5d = pd.DataFrame({
            'Date'          : future_dates,
            'Forecast'      : fc_mean.values,
            'Lower_95'      : fc_ci.iloc[:, 0].values,
            'Upper_95'      : fc_ci.iloc[:, 1].values,
            'Model'         : 'ARIMA',
            'Ticker'        : self.ticker,
        }).set_index('Date')

        log.info(f"  [{self.ticker}] 5-day forecast: "
                 f"{self.forecast_5d['Forecast'].values.round(2)}")
        return self.forecast_5d

    # ── Step 4: Metrics ───────────────────────────────────────────────────────
    def compute_metrics(self) -> dict:
        y_true = self.test_series.values
        y_pred = self.predictions.values
        self.metrics = {
            'Ticker'   : self.ticker,
            'Company'  : self.name,
            'Model'    : 'ARIMA',
            'Order'    : str(self.best_order),
            'AIC'      : round(self.aic, 2),
            'BIC'      : round(self.bic, 2),
            'RMSE'     : round(rmse(y_true, y_pred), 4),
            'MAE'      : round(mae(y_true, y_pred), 4),
            'MAPE_%'   : round(mape(y_true, y_pred), 4),
            'DA_%'     : round(directional_accuracy(y_true, y_pred), 2),
            'R2'       : round(r2_score(y_true, y_pred), 4),
        }
        return self.metrics

    # ── Step 5: Residual diagnostics ─────────────────────────────────────────
    def diagnostics(self) -> dict:
        resid = self.model_fit.resid.dropna()

        # Ljung-Box test (null: no autocorrelation in residuals)
        lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
        lb10_p = float(lb['lb_pvalue'].iloc[0])
        lb20_p = float(lb['lb_pvalue'].iloc[1])

        # Normality of residuals (Jarque-Bera)
        jb_stat, jb_p = stats.jarque_bera(resid)

        diag = {
            'Ticker'          : self.ticker,
            'LjungBox_lag10_p': round(lb10_p, 4),
            'LjungBox_lag20_p': round(lb20_p, 4),
            'ResidNormal_JB_p': round(float(jb_p), 4),
            'Resid_Mean'      : round(float(resid.mean()), 6),
            'Resid_Std'       : round(float(resid.std()), 4),
            'Resid_Skew'      : round(float(resid.skew()), 4),
            'Resid_Kurt'      : round(float(resid.kurt()), 4),
            'LB10_Pass'       : '✅' if lb10_p > 0.05 else '⚠️',
            'LB20_Pass'       : '✅' if lb20_p > 0.05 else '⚠️',
        }
        log.info(f"  [{self.ticker}] Ljung-Box p(10)={lb10_p:.4f} "
                 f"p(20)={lb20_p:.4f}  JB_p={jb_p:.4f}")
        return diag

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(self, train_series: pd.Series,
            test_series: pd.Series) -> dict:
        self.fit_auto_arima(train_series)
        self.predict_test(test_series)
        self.forecast_5_days()
        self.compute_metrics()
        diag = self.diagnostics()
        self.metrics.update({k: v for k, v in diag.items()
                              if k not in self.metrics})
        return self.metrics

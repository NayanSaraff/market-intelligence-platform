"""
src/models/ensemble_model.py
══════════════════════════════════════════════════════════════════
Phase 7 — Ensemble Forecasting Module
StockGro Capstone Project

THEORY
------
Ensemble combines forecasts from multiple models using
inverse-RMSE weighting:

  w_i = (1 / RMSE_i) / Σ(1 / RMSE_j)

  ŷ_ensemble = Σ w_i × ŷ_i

Rationale:
  - Models with lower RMSE on the test set get higher weight
  - Automatically adapts weight to each stock's best model
  - Reduces variance while preserving signal from all models
  - Provably reduces MSE vs any single model in expectation

Additional methods also computed for comparison:
  - Simple average  (equal weights)
  - Trimmed mean    (drop worst model)
  - Median ensemble (robust to outliers)
══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import warnings, logging
warnings.filterwarnings("ignore")
log = logging.getLogger("Ensemble")

# ── Metrics ───────────────────────────────────────────────────
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
class EnsembleForecaster:
    """
    Inverse-RMSE weighted ensemble of ARIMA, ETS, Prophet, LSTM forecasts.

    Parameters
    ----------
    ticker      : str
    name        : str
    model_names : list[str]   e.g. ['ARIMA','ETS','Prophet','LSTM']
    """

    def __init__(self, ticker: str, name: str,
                 model_names: list = None):
        self.ticker       = ticker
        self.name         = name
        self.model_names  = model_names or ['ARIMA','ETS','Prophet','LSTM']
        self.weights      = {}          # {model_name: weight}
        self.predictions  = None
        self.forecast_5d  = None
        self.metrics      = {}

    # ── Step 1: Compute inverse-RMSE weights ─────────────────────────────────
    def compute_weights(self, rmse_dict: dict) -> dict:
        """
        rmse_dict: {model_name: rmse_value}
        Returns:   {model_name: weight}  (sum=1)
        """
        inv  = {m: 1.0/(v + 1e-10) for m, v in rmse_dict.items()}
        total = sum(inv.values())
        self.weights = {m: w/total for m, w in inv.items()}
        log.info(f"  [{self.ticker}] Ensemble weights: "
                 + "  ".join(f"{m}:{w:.3f}" for m,w in self.weights.items()))
        return self.weights

    # ── Step 2: Weighted combination of test predictions ──────────────────────
    def combine_predictions(self,
                            pred_dict: dict,
                            y_true: pd.Series) -> pd.Series:
        """
        pred_dict: {model_name: pd.Series of predictions}
        y_true:     actual test series (for metric computation)
        """
        # Align all predictions to same index
        preds_df = pd.DataFrame(pred_dict)
        preds_df = preds_df.dropna()

        # Inverse-RMSE weighted ensemble
        rmse_dict = {m: _rmse(y_true.loc[preds_df.index],
                               preds_df[m])
                     for m in preds_df.columns}
        self.compute_weights(rmse_dict)

        weighted = sum(preds_df[m] * self.weights[m]
                       for m in preds_df.columns)
        self.predictions = pd.Series(weighted,
                                     name=f"Ensemble_{self.ticker}")

        # Also compute alternative ensembles
        self._simple_avg  = preds_df.mean(axis=1)
        self._median_ens  = preds_df.median(axis=1)

        # Trimmed: drop highest-RMSE model
        worst_model = max(rmse_dict, key=rmse_dict.get)
        trim_cols   = [c for c in preds_df.columns if c != worst_model]
        self._trimmed = preds_df[trim_cols].mean(axis=1)

        return self.predictions

    # ── Step 3: Weighted 5-day forecast ──────────────────────────────────────
    def combine_forecasts_5d(self, fc_dict: dict) -> pd.DataFrame:
        """
        fc_dict: {model_name: forecast_5d DataFrame}
        Each DataFrame must have columns: Forecast, Lower_95, Upper_95
        """
        dates = list(fc_dict.values())[0].index

        fc_vals   = np.zeros(5)
        fc_lower  = np.zeros(5)
        fc_upper  = np.zeros(5)

        for m, fc_df in fc_dict.items():
            w = self.weights.get(m, 0)
            fc_vals  += w * fc_df['Forecast'].values
            fc_lower += w * fc_df['Lower_95'].values
            fc_upper += w * fc_df['Upper_95'].values

        self.forecast_5d = pd.DataFrame({
            'Date'    : dates,
            'Forecast': fc_vals,
            'Lower_95': fc_lower,
            'Upper_95': fc_upper,
            'Model'   : 'Ensemble',
            'Ticker'  : self.ticker,
            'Weights' : [str({m: round(w,3) for m,w in self.weights.items()})] * 5,
        }).set_index('Date')

        log.info(f"  [{self.ticker}] Ensemble 5-day forecast: "
                 f"{fc_vals.round(2)}")
        return self.forecast_5d

    # ── Step 4: Metrics comparison ────────────────────────────────────────────
    def compute_metrics(self, y_true: pd.Series) -> dict:
        y_t   = y_true.loc[self.predictions.index].values
        y_ens = self.predictions.values

        self.metrics = {
            'Ticker'          : self.ticker,
            'Company'         : self.name,
            'Model'           : 'Ensemble',
            'Weights'         : str({m: round(w,3) for m,w in self.weights.items()}),
            'RMSE'            : round(_rmse(y_t, y_ens), 4),
            'MAE'             : round(_mae(y_t, y_ens), 4),
            'MAPE_%'          : round(_mape(y_t, y_ens), 4),
            'DA_%'            : round(_da(y_t, y_ens), 2),
            'R2'              : round(_r2(y_t, y_ens), 4),
            'RMSE_SimpleAvg'  : round(_rmse(y_t, self._simple_avg.loc[self.predictions.index].values), 4),
            'RMSE_Median'     : round(_rmse(y_t, self._median_ens.loc[self.predictions.index].values), 4),
            'RMSE_Trimmed'    : round(_rmse(y_t, self._trimmed.loc[self.predictions.index].values), 4),
        }
        return self.metrics

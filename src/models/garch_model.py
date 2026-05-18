"""GARCH volatility forecaster.

Provides a lightweight wrapper around `arch` to produce volatility
forecasts for a price series. Fits on log-returns (train only) and
produces horizon volatility forecasts for use in downstream risk/portfolio
modules.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd

log = logging.getLogger("GARCH")


class GARCHForecaster:
    """Fit a simple GARCH(p,q) model to log-returns and forecast volatility.

    Methods:
    - `fit(series)`: fit GARCH on price series (index-preserving)
    - `forecast(horizon)`: produce horizon-step volatility forecast (array)
    - `predict_test(train, test)`: fit on `train` and forecast vol for `test` length
    """

    def __init__(self, p: int = 1, q: int = 1, vol: str = "Garch", dist: str = "normal"):
        self.p = int(p)
        self.q = int(q)
        self.vol = vol
        self.dist = dist
        self.model = None
        self.res = None

    def _prepare_returns(self, series: pd.Series) -> pd.Series:
        s = pd.Series(series).astype(float).dropna()
        # use log returns; multiply by 100 for percent-scale (arch convention)
        lr = np.log(s).diff().dropna() * 100.0
        return lr

    def fit(self, series: pd.Series) -> "GARCHForecaster":
        try:
            from arch import arch_model
        except Exception as e:
            log.error("arch package not available: %s", e)
            raise

        returns = self._prepare_returns(series)
        if len(returns) < max(10, self.p + self.q + 1):
            raise ValueError("Not enough data to fit GARCH model")

        self.model = arch_model(returns, p=self.p, q=self.q, vol=self.vol, dist=self.dist)
        # suppress convergence output
        self.res = self.model.fit(disp="off")
        return self

    def forecast(self, horizon: int = 5) -> np.ndarray:
        if self.res is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        fc = self.res.forecast(horizon=horizon, reindex=False)
        # `fc.variance` is a DataFrame; take last row and sqrt to get stddev
        var = fc.variance.iloc[-1].values
        vol = np.sqrt(var) / 100.0  # convert percent back to fractional stddev
        return vol

    def predict_test(self, train: pd.Series, test: pd.Series) -> pd.DataFrame:
        """Fit on `train` and produce per-index volatility forecast for `test` length.

        Returns a DataFrame with index=test.index and a single column `volatility`.
        """
        if test is None or len(test) == 0:
            return pd.DataFrame(columns=["volatility"])

        self.fit(train)
        try:
            vols = self.forecast(horizon=len(test))
        except Exception as e:
            log.warning("GARCH forecast failed: %s — falling back to constant vol", e)
            # fallback: use in-sample stddev of returns
            rets = self._prepare_returns(train)
            const_vol = rets.std() / 100.0
            vols = np.repeat(const_vol, len(test))

        return pd.DataFrame({"volatility": vols}, index=test.index)


__all__ = ["GARCHForecaster"]

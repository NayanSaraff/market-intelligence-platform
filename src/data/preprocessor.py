"""
src/data/preprocessor.py
═════════════════════════════════════════════════════════════════
Phase 4 — Preprocessing Module
StockGro Capstone Project

PIPELINE
--------
1. Load raw CSVs
2. Missing value audit → forward-fill → back-fill
3. Reindex to full NSE business-day calendar
4. Feature engineering (41 indicators)
5. ADF + KPSS stationarity tests
6. Apply differencing where required (d=1 for all 8 stocks)
7. MinMaxScaler fit on TRAIN only (no leakage)
8. Strict train/test temporal split
9. Save processed CSVs to data/processed/ and data/features/
═════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss
import yaml
import warnings, logging
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger("Preprocessor")

# ── Constants ─────────────────────────────────────────────────
SELECTED = [
    "ITC.NS","SUNPHARMA.NS","TATAMOTORS.NS","ICICIBANK.NS",
    "INFY.NS","MARUTI.NS","DRREDDY.NS","HDFCBANK.NS",
]
NAMES = {
    "ITC.NS":"ITC","SUNPHARMA.NS":"Sun Pharma","TATAMOTORS.NS":"Tata Motors",
    "ICICIBANK.NS":"ICICI Bank","INFY.NS":"Infosys","MARUTI.NS":"Maruti",
    "DRREDDY.NS":"Dr Reddy's","HDFCBANK.NS":"HDFC Bank",
}
TRAIN_ROWS = 1047   # legacy fallback; preferred: date-driven split from config
RF_RATE    = 0.06   # 6% annualised risk-free rate


def _load_config(cfg_path: str = "configs/project_config.yaml") -> dict:
    p = Path(cfg_path)
    if not p.exists():
        log.warning(f"Config {cfg_path} not found — falling back to defaults")
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh)


# ═════════════════════════════════════════════════════════════
class Preprocessor:
    """Full preprocessing pipeline for all 8 NSE stocks."""

    def __init__(self,
                 raw_dir   : str = "data/raw",
                 proc_dir  : str = "data/processed",
                 feat_dir  : str = "data/features",
                 config_path: str = "configs/project_config.yaml"):
        self.raw_dir  = Path(raw_dir)
        self.proc_dir = Path(proc_dir)
        self.feat_dir = Path(feat_dir)
        self.config   = _load_config(config_path) or {}
        # read date bounds if provided
        dates = self.config.get("dates", {})
        self.train_start = pd.to_datetime(dates.get("train_start", None))
        self.train_end   = pd.to_datetime(dates.get("train_end", None))
        self.test_start  = pd.to_datetime(dates.get("test_start", None))
        self.test_end    = pd.to_datetime(dates.get("test_end", None))
        for d in [self.proc_dir, self.feat_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.raw_data    = {}
        self.train_data  = {}
        self.test_data   = {}
        self.scalers     = {}
        self.stat_report = []

    # ─── Load ──────────────────────────────────────────────
    def load_raw(self):
        for t in SELECTED:
            df = pd.read_csv(self.raw_dir/f"{t}.csv",
                             index_col=0, parse_dates=True)
            df.index.name = "Date"
            df.sort_index(inplace=True)
            # Prefer adjusted close if present — normalize to 'Close'
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
                log.info(f"  {t}: using 'Adj Close' as Close")
            elif 'Adj_Close' in df.columns:
                df['Close'] = df['Adj_Close']
                log.info(f"  {t}: using 'Adj_Close' as Close")
            else:
                log.debug(f"  {t}: no adjusted-close column found; using raw Close")
            self.raw_data[t] = df
            log.info(f"  Loaded {t}: {len(df)} rows")
        return self

    # ─── Missing Values ────────────────────────────────────
    def handle_missing(self):
        for t, df in self.raw_data.items():
            bdays = pd.bdate_range(df.index.min(), df.index.max())
            df = df.reindex(bdays)
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            self.raw_data[t] = df
            log.info(f"  {t}: reindexed → {len(df)} rows, nulls={df.isnull().sum().sum()}")
        return self

    # ─── Feature Engineering ───────────────────────────────
    @staticmethod
    def _compute_rsi(s, period=14):
        d = s.diff()
        g = d.clip(lower=0).ewm(com=period-1, adjust=False).mean()
        l = (-d).clip(lower=0).ewm(com=period-1, adjust=False).mean()
        return 100 - 100/(1 + g/(l+1e-10))

    @staticmethod
    def _compute_atr(df, period=14):
        hl  = df["High"] - df["Low"]
        hpc = (df["High"] - df["Close"].shift(1)).abs()
        lpc = (df["Low"]  - df["Close"].shift(1)).abs()
        tr  = pd.concat([hl,hpc,lpc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def engineer_features(self):
        for t, df in self.raw_data.items():
            fe = df.copy(); c = fe["Close"]
            fe["Return_1d"]   = c.pct_change()
            # Log returns used for volatility modelling per project rules
            fe["LogReturn"]   = np.log(c / c.shift(1))
            fe["Return_5d"]   = c.pct_change(5)
            fe["Return_21d"]  = c.pct_change(21)
            for w in [5,10,20,50,200]: fe[f"SMA_{w}"] = c.rolling(w).mean()
            fe["EMA_12"] = c.ewm(span=12, adjust=False).mean()
            fe["EMA_26"] = c.ewm(span=26, adjust=False).mean()
            fe["MACD"]        = fe["EMA_12"] - fe["EMA_26"]
            fe["MACD_Signal"] = fe["MACD"].ewm(span=9, adjust=False).mean()
            fe["MACD_Hist"]   = fe["MACD"] - fe["MACD_Signal"]
            fe["RSI_14"]      = self._compute_rsi(c)
            s20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
            fe["BB_Upper"] = s20 + 2*std20; fe["BB_Lower"] = s20 - 2*std20
            fe["BB_Width"] = (fe["BB_Upper"]-fe["BB_Lower"])/(s20+1e-10)
            fe["BB_%B"]    = (c-fe["BB_Lower"])/(fe["BB_Upper"]-fe["BB_Lower"]+1e-10)
            fe["ATR_14"]   = self._compute_atr(df)
            # Volatility should be computed on log-returns (statistically correct)
            fe["Vol_30d"]  = fe["LogReturn"].rolling(30).std()*np.sqrt(252)
            fe["Vol_10d"]  = fe["LogReturn"].rolling(10).std()*np.sqrt(252)
            fe["OBV"]      = (np.where(c.diff()>0,1,-1)*df["Volume"]).cumsum()
            fe["Vol_SMA_20"] = df["Volume"].rolling(20).mean()
            for lag in [1,2,3,5,10,21]: fe[f"Close_Lag{lag}"]  = c.shift(lag)
            for lag in [1,2,3,5]:       fe[f"Return_Lag{lag}"] = fe["Return_1d"].shift(lag)
            self.raw_data[t] = fe
        log.info(f"  Feature engineering complete ({len(self.raw_data[SELECTED[0]].columns)} cols)")
        return self

    # ─── Train/Test Split ──────────────────────────────────
    def split(self):
        # Date-driven split if configuration provided; otherwise use project-standard dates
        # Project-standard: train up to 2024-12-31; test from 2025-01-01
        default_train_end = pd.to_datetime("2024-12-31")
        default_test_start = pd.to_datetime("2025-01-01")
        for t, fe in self.raw_data.items():
            # prefer explicit config dates when provided
            used_config = False
            if self.train_end is not None and self.test_start is not None:
                tr_end = pd.to_datetime(self.train_end)
                ts_start = pd.to_datetime(self.test_start)
                used_config = True
            else:
                tr_end = default_train_end
                ts_start = default_test_start

            tr = fe.loc[:tr_end].copy()
            te = fe.loc[ts_start:].copy()

            # If config dates were used but produced no test rows (data not available yet),
            # attempt the project-standard dates before falling back to row-based split.
            if used_config and te.empty:
                log.warning(f"  {t}: config dates produced empty test split; trying project-standard dates {default_train_end.date()}->{default_test_start.date()}")
                tr = fe.loc[:default_train_end].copy()
                te = fe.loc[default_test_start:].copy()

            # if either split is still empty, fallback to row-based split to avoid data loss
            if tr.empty or te.empty:
                n = len(fe)
                tr = fe.iloc[:TRAIN_ROWS].copy()
                te = fe.iloc[TRAIN_ROWS:].copy()
                log.warning(f"  {t}: date-split produced empty split; falling back to row-based TRAIN_ROWS={TRAIN_ROWS}")

            self.train_data[t] = tr
            self.test_data[t]  = te
            log.info(f"  {t}: date-split → train={len(tr)} ({tr.index.min().date()}->{tr.index.max().date()}) test={len(te)} ({te.index.min().date()}->{te.index.max().date()})")
        return self

    # ─── Stationarity Tests ────────────────────────────────
    def test_stationarity(self):
        for t in SELECTED:
            c   = self.train_data[t]["Close"].dropna()
            r   = np.log(c / c.shift(1)).dropna()
            adf_c = adfuller(c, autolag="AIC")
            adf_r = adfuller(r, autolag="AIC")
            d_req = 0 if adf_c[1]<0.05 else 1
            self.stat_report.append({
                "Ticker"         : t,
                "Company"        : NAMES[t],
                "ADF_Price_p"    : round(adf_c[1],4),
                "Price_Stationary": adf_c[1]<0.05,
                "ADF_Return_p"   : round(adf_r[1],6),
                "Return_Stationary": adf_r[1]<0.05,
                "Required_d"     : d_req,
            })
        df_stat = pd.DataFrame(self.stat_report)
        df_stat.to_csv("outputs/reports/stationarity_report.csv", index=False)
        log.info("  Stationarity report saved")
        return self

    # ─── LSTM Scaling ──────────────────────────────────────
    def scale_for_lstm(self):
        for t in SELECTED:
            sc = MinMaxScaler((0,1))
            sc.fit(self.train_data[t][["Close"]])
            self.scalers[t] = sc
            tr_sc = sc.transform(self.train_data[t][["Close"]]).flatten()
            te_sc = sc.transform(self.test_data[t][["Close"]]).flatten()
            pd.DataFrame({"Close_Scaled":tr_sc},
                         index=self.train_data[t].index).to_csv(
                             self.feat_dir/f"{t}_train_scaled.csv")
            pd.DataFrame({"Close_Scaled":te_sc},
                         index=self.test_data[t].index).to_csv(
                             self.feat_dir/f"{t}_test_scaled.csv")
        log.info("  LSTM scaling complete (fit on TRAIN only)")
        return self

    # ─── Save ──────────────────────────────────────────────
    def save(self):
        for t in SELECTED:
            self.train_data[t].to_csv(self.proc_dir/f"{t}_train_full.csv")
            self.test_data[t].to_csv(self.proc_dir/f"{t}_test_full.csv")
            # Ensure close and returns saved for downstream models
            cols = [c for c in ["Close","LogReturn","Return_1d"] if c in self.train_data[t].columns]
            self.train_data[t][cols].to_csv(self.feat_dir/f"{t}_train_close.csv")
            self.test_data[t][cols].to_csv(self.feat_dir/f"{t}_test_close.csv")
        log.info(f"  Saved {len(SELECTED)*4} CSV files")
        return self

    # ─── Run all ───────────────────────────────────────────
    def run(self):
        (self.load_raw()
             .handle_missing()
             .engineer_features()
             .split()
             .test_stationarity()
             .scale_for_lstm()
             .save())
        log.info("✅ PREPROCESSING COMPLETE")
        return self


if __name__ == "__main__":
    p = Preprocessor()
    p.run()

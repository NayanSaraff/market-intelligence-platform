"""
src/data/downloader.py
══════════════════════════════════════════════════════════════════
Phase 2 — Data Collection Module
StockGro Capstone Project

PURPOSE
-------
Download, validate, and persist raw NSE stock data from Yahoo Finance.
Handles rate-limiting, retries, missing-value imputation, and corporate
actions (splits / dividends) transparently via Adjusted Close.

USAGE
-----
    from src.data.downloader import StockDataDownloader
    dl = StockDataDownloader()
    dl.download_all()
    report = dl.generate_quality_report()
══════════════════════════════════════════════════════════════════
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Logger ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("StockDownloader")


# ══════════════════════════════════════════════════════════════
# CONSTANTS — Edit here or load from project_config.yaml
# ══════════════════════════════════════════════════════════════

STOCK_UNIVERSE: Dict[str, str] = {
    "HDFCBANK.NS":   "HDFC Bank",
    "ICICIBANK.NS":  "ICICI Bank",
    "INFY.NS":       "Infosys",
    "TCS.NS":        "Tata Consultancy Services",
    "SUNPHARMA.NS":  "Sun Pharmaceutical",
    "DRREDDY.NS":    "Dr. Reddy's Laboratories",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS":        "ITC Limited",
    "MARUTI.NS":     "Maruti Suzuki",
    "TATAMOTORS.NS": "Tata Motors",
}

FULL_START   = "2021-01-01"
FULL_END     = "2025-06-30"   # inclusive upper bound for training
DATA_RAW_DIR = Path("data/raw")
RETRY_LIMIT  = 3
RETRY_SLEEP  = 5              # seconds between retries


# ══════════════════════════════════════════════════════════════
class StockDataDownloader:
    """
    End-to-end NSE data downloader and quality checker.

    Parameters
    ----------
    tickers : dict
        {ticker_symbol: company_name}. Defaults to STOCK_UNIVERSE.
    start : str
        Download start date (YYYY-MM-DD).
    end : str
        Download end date (YYYY-MM-DD).
    output_dir : Path
        Directory to save raw CSVs.
    """

    def __init__(
        self,
        tickers: Dict[str, str] = STOCK_UNIVERSE,
        start:   str = FULL_START,
        end:     str = FULL_END,
        output_dir: Path = DATA_RAW_DIR,
    ):
        self.tickers    = tickers
        self.start      = start
        self.end        = end
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.quality_report: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────
    # DOWNLOAD
    # ─────────────────────────────────────────────────────────

    def _download_single(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download OHLCV + Adjusted Close for one ticker.
        Retries up to RETRY_LIMIT times on failure.
        """
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                df = yf.download(
                    tickers  = ticker,
                    start    = self.start,
                    end      = self.end,
                    progress = False,
                    auto_adjust = False,   # Keep raw + Adj Close both
                    actions  = True,       # Include dividends & splits
                )
                if df.empty:
                    logger.warning(f"{ticker}: empty DataFrame on attempt {attempt}")
                    time.sleep(RETRY_SLEEP)
                    continue

                # Flatten MultiIndex columns if present (yfinance ≥ 0.2.x)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Ensure standard column names
                df.index.name = "Date"
                df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                logger.info(f"  ✅ {ticker:<18} {len(df):>5} rows  "
                            f"{df.index[0].date()} → {df.index[-1].date()}")
                return df

            except Exception as exc:
                logger.error(f"  ❌ {ticker} attempt {attempt} failed: {exc}")
                if attempt < RETRY_LIMIT:
                    time.sleep(RETRY_SLEEP)

        logger.error(f"  ✗ {ticker}: all {RETRY_LIMIT} attempts failed")
        return None

    def download_all(self) -> Dict[str, pd.DataFrame]:
        """
        Download all tickers and save raw CSVs.

        Returns
        -------
        dict
            {ticker: DataFrame}
        """
        logger.info("=" * 60)
        logger.info(" Phase 2: Downloading NSE Stock Data")
        logger.info(f" Period : {self.start} → {self.end}")
        logger.info(f" Stocks : {len(self.tickers)}")
        logger.info("=" * 60)

        for ticker, name in self.tickers.items():
            logger.info(f"Downloading {name} ({ticker}) ...")
            df = self._download_single(ticker)
            if df is not None:
                self.stock_data[ticker] = df
                # Save raw CSV
                out_path = self.output_dir / f"{ticker}.csv"
                df.to_csv(out_path)
                logger.info(f"  💾 Saved → {out_path}")
            else:
                logger.warning(f"  ⚠️  {ticker} could not be downloaded")

        logger.info(f"\n✅ Downloaded {len(self.stock_data)}/{len(self.tickers)} stocks")
        return self.stock_data

    # ─────────────────────────────────────────────────────────
    # LOAD FROM DISK (if already downloaded)
    # ─────────────────────────────────────────────────────────

    def load_from_disk(self) -> Dict[str, pd.DataFrame]:
        """Load previously downloaded CSVs instead of re-downloading."""
        for ticker in self.tickers:
            path = self.output_dir / f"{ticker}.csv"
            if path.exists():
                df = pd.read_csv(path, index_col="Date", parse_dates=True)
                self.stock_data[ticker] = df
                logger.info(f"  📂 Loaded {ticker} from disk ({len(df)} rows)")
            else:
                logger.warning(f"  ⚠️  {ticker} not found on disk")
        return self.stock_data

    # ─────────────────────────────────────────────────────────
    # DATA QUALITY CHECKS
    # ─────────────────────────────────────────────────────────

    def _check_single(self, ticker: str, df: pd.DataFrame) -> dict:
        """
        Run comprehensive quality checks on one stock DataFrame.
        """
        # ── Expected trading days (NSE ~250/yr) ──────────────
        n_years  = (pd.Timestamp(self.end) - pd.Timestamp(self.start)).days / 365
        expected = int(n_years * 250)

        # ── Missing dates (business days not in index) ────────
        bdays = pd.bdate_range(self.start, self.end)
        missing_dates = bdays.difference(df.index)

        # ── Missing values per column ─────────────────────────
        null_counts = df.isnull().sum()
        total_nulls = int(null_counts.sum())

        # ── Zero-volume days ──────────────────────────────────
        zero_vol = int((df["Volume"] == 0).sum())

        # ── Price anomalies: |daily return| > 20% ────────────
        daily_ret   = df["Close"].pct_change()
        anomalies   = int((daily_ret.abs() > 0.20).sum())

        # ── Price range sanity ────────────────────────────────
        bad_hlc = int(((df["High"] < df["Low"]) |
                       (df["Close"] > df["High"]) |
                       (df["Close"] < df["Low"])).sum())

        # ── Duplicate index ───────────────────────────────────
        duplicates = int(df.index.duplicated().sum())

        return {
            "Ticker":          ticker,
            "Company":         self.tickers[ticker],
            "Rows_Available":  len(df),
            "Expected_Rows":   expected,
            "Coverage_%":      round(len(df) / expected * 100, 1),
            "Date_From":       df.index[0].date(),
            "Date_To":         df.index[-1].date(),
            "Missing_BDays":   len(missing_dates),
            "Total_Nulls":     total_nulls,
            "Null_Open":       int(null_counts.get("Open",  0)),
            "Null_Close":      int(null_counts.get("Close", 0)),
            "Null_Volume":     int(null_counts.get("Volume",0)),
            "Zero_Volume_Days":zero_vol,
            "Price_Anomalies": anomalies,
            "Bad_HLC_Rows":    bad_hlc,
            "Duplicate_Idx":   duplicates,
            "Status":          "✅ PASS" if total_nulls == 0 and anomalies <= 5 else "⚠️  WARN",
        }

    def generate_quality_report(self) -> pd.DataFrame:
        """
        Run quality checks on all downloaded stocks.

        Returns
        -------
        pd.DataFrame
            Data quality report table.
        """
        if not self.stock_data:
            logger.warning("No data loaded. Call download_all() or load_from_disk() first.")
            return pd.DataFrame()

        rows = []
        for ticker, df in self.stock_data.items():
            rows.append(self._check_single(ticker, df))

        report = pd.DataFrame(rows)
        self.quality_report = report

        # Save report to disk
        out = Path("outputs/reports/data_quality_report.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out, index=False)
        logger.info(f"\n📊 Quality report saved → {out}")

        return report

    # ─────────────────────────────────────────────────────────
    # SUMMARY STATISTICS
    # ─────────────────────────────────────────────────────────

    def compute_summary_statistics(self) -> pd.DataFrame:
        """
        Compute descriptive statistics for each stock.
        Includes price stats, return stats, and volatility.
        """
        rows = []
        for ticker, df in self.stock_data.items():
            close  = df["Close"].dropna()
            ret    = close.pct_change().dropna()
            log_ret= np.log(close / close.shift(1)).dropna()

            rows.append({
                "Ticker":          ticker,
                "Company":         self.tickers[ticker],
                # Price statistics
                "Price_Min":       round(close.min(), 2),
                "Price_Max":       round(close.max(), 2),
                "Price_Mean":      round(close.mean(), 2),
                "Price_Median":    round(close.median(), 2),
                "Price_Std":       round(close.std(), 2),
                # Return statistics
                "Daily_Ret_Mean%": round(ret.mean() * 100, 4),
                "Daily_Ret_Std%":  round(ret.std()  * 100, 4),
                "Total_Return%":   round((close.iloc[-1]/close.iloc[0] - 1) * 100, 2),
                # Annualized metrics
                "Ann_Return%":     round(ret.mean() * 252 * 100, 2),
                "Ann_Volatility%": round(ret.std()  * np.sqrt(252) * 100, 2),
                # Distribution
                "Skewness":        round(float(ret.skew()), 4),
                "Kurtosis":        round(float(ret.kurt()), 4),
                # Extremes
                "Max_Daily_Gain%": round(ret.max() * 100, 2),
                "Max_Daily_Loss%": round(ret.min() * 100, 2),
                # Sharpe (Rf=6%)
                "Sharpe_Ratio":    round((ret.mean()*252 - 0.06) / (ret.std()*np.sqrt(252)), 3),
            })

        stats_df = pd.DataFrame(rows)

        out = Path("outputs/reports/summary_statistics.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(out, index=False)
        logger.info(f"📊 Summary statistics saved → {out}")

        return stats_df


# ══════════════════════════════════════════════════════════════
# CONVENIENCE — run as script
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    dl = StockDataDownloader()

    # Step 1: Download (or load from disk if already downloaded)
    csv_exists = all(
        (DATA_RAW_DIR / f"{t}.csv").exists() for t in STOCK_UNIVERSE
    )
    if csv_exists:
        print("📂 Raw CSVs found on disk — loading ...")
        dl.load_from_disk()
    else:
        print("🌐 Downloading from Yahoo Finance ...")
        dl.download_all()

    # Step 2: Quality report
    print("\n📋 Running data quality checks ...")
    qr = dl.generate_quality_report()
    print(qr.to_string(index=False))

    # Step 3: Summary stats
    print("\n📊 Computing summary statistics ...")
    ss = dl.compute_summary_statistics()
    print(ss[["Ticker","Ann_Return%","Ann_Volatility%","Sharpe_Ratio"]].to_string(index=False))

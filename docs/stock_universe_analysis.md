# 📈 Stock Universe — Selection Rationale & Analysis Framework
# StockGro Capstone | Phase 1

## 1. UNIVERSE OVERVIEW

| # | Ticker | Company | Sector | Market Cap | NSE Index |
|---|--------|---------|--------|------------|-----------|
| 1 | HDFCBANK.NS | HDFC Bank | Banking | ₹12.5L Cr | Nifty50 |
| 2 | ICICIBANK.NS | ICICI Bank | Banking | ₹9.2L Cr | Nifty50 |
| 3 | INFY.NS | Infosys | IT | ₹6.3L Cr | Nifty50 |
| 4 | TCS.NS | Tata Consultancy | IT | ₹13.8L Cr | Nifty50 |
| 5 | SUNPHARMA.NS | Sun Pharma | Pharma | ₹4.8L Cr | Nifty50 |
| 6 | DRREDDY.NS | Dr. Reddy's Labs | Pharma | ₹1.1L Cr | Nifty50 |
| 7 | HINDUNILVR.NS | Hindustan Unilever | FMCG | ₹5.2L Cr | Nifty50 |
| 8 | ITC.NS | ITC Ltd | FMCG/Diversified | ₹5.6L Cr | Nifty50 |
| 9 | MARUTI.NS | Maruti Suzuki | Auto | ₹3.5L Cr | Nifty50 |
| 10 | TATAMOTORS.NS | Tata Motors | Auto | ₹3.2L Cr | Nifty50 |

---

## 2. SELECTION RATIONALE

### A. Sector Diversification
Selecting across 5 uncorrelated sectors reduces idiosyncratic risk:

- **Banking** (15–18% of Nifty50 weight): Cyclical, rate-sensitive
- **IT** (12–15% of Nifty50): Growth, USD-revenue hedge vs INR
- **Pharma** (5–6%): Defensive, export-driven
- **FMCG** (8–10%): Defensive, domestic consumption
- **Auto** (5–7%): Cyclical, volume-driven

### B. Why These Specific Stocks

**HDFCBANK.NS** — Largest private bank in India. Most liquid NSE stock. 
Merger with HDFC Ltd (April 2023) creates structural break — tests model adaptability.

**ICICIBANK.NS** — 2nd largest private bank. Strong ROE improvement story.
Good trend strength, daily volumes >20Mn shares.

**INFY.NS** — India's 2nd largest IT exporter. Quarterly guidance culture 
makes it an ideal candidate for seasonality modeling (quarterly earnings beats/misses).

**TCS.NS** — Most stable IT stock, highest market cap in IT sector.
Low volatility = good baseline for ARIMA (near-random-walk at daily frequency).

**SUNPHARMA.NS** — Sector leader in pharma. Regulatory risk from USFDA adds 
volatility spikes — ideal for GARCH modeling.

**DRREDDY.NS** — Premium pharma with global exposure. Different correlation 
pattern from SUNPHARMA.

**HINDUNILVR.NS** — FMCG bellwether. Defensive stock during market downturns.
Low beta — stabilizes portfolio.

**ITC.NS** — Unique conglomerate: cigarettes, FMCG, hotels, agri.
Strong dividend yield (4–5%). Diversifier in portfolio.

**MARUTI.NS** — Dominant auto company (45%+ market share in India).
GDP-linked cyclical. Good for regime change detection.

**TATAMOTORS.NS** — Cyclical with JLR exposure (global revenues).
Higher volatility than Maruti — tests model robustness.

---

## 3. DATA PERIOD ANALYSIS

### Why 2021–2025 (5 Years)?

| Period | Market Regime | Significance |
|--------|--------------|--------------|
| Jan–Dec 2021 | Post-COVID recovery bull run | Trend + momentum period |
| Jan–Jun 2022 | Global selloff (Fed hikes, Russia-Ukraine) | Volatility spike, trend break |
| Jul–Dec 2022 | Partial recovery, range-bound | Consolidation |
| 2023 | India outperformance vs global | Bull market, new highs |
| 2024 | Election year volatility + continuation | Mixed regimes |
| Jan–Jun 2025 | Training cutoff | Latest information used |
| Jul–Dec 2025 | Test / Forecast period | True out-of-sample evaluation |

**Key advantage:** 5 years captures bull, bear, and sideways markets — models 
trained on this data are regime-robust.

---

## 4. TRAIN/TEST SPLIT STRATEGY

```
FULL PERIOD: 2021-01-01 ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2025-12-31
                         │                           │
TRAIN:       2021-01-01 ━━━━━━━━━━━━━ 2025-06-30   │
TEST:                                  2025-07-01 ━━ 2025-12-31
```

- **Training period:** ~4.5 years ≈ 1,125 trading days
- **Testing period:** ~6 months ≈ 125 trading days
- **Split ratio:** ~90% train / ~10% test

**Why this split?**
- No future data leakage (strict cutoff at June 30, 2025)
- Sufficient test data for statistical evaluation (125 days)
- Training set spans multiple market cycles

---

## 5. EXPECTED STOCK CHARACTERISTICS

### Volatility Ranking (Expected — Annual)
```
HIGH   → TATAMOTORS > DRREDDY > SUNPHARMA > MARUTI
MEDIUM → ICICIBANK > INFY > ITC > TCS
LOW    → HDFCBANK > HINDUNILVR
```

### Forecastability Ranking (Expected)
```
MOST FORECASTABLE   → TCS, HINDUNILVR (stable trends, low noise)
MODERATELY          → HDFCBANK, ICICIBANK, INFY
CHALLENGING         → MARUTI, SUNPHARMA (regime changes)
MOST CHALLENGING    → TATAMOTORS, DRREDDY (event-driven)
```

### Correlation Structure (Expected)
```
HIGH CORRELATION:  HDFCBANK ↔ ICICIBANK (0.7–0.8)
HIGH CORRELATION:  INFY ↔ TCS (0.75–0.85)
LOW CORRELATION:   IT ↔ FMCG (0.2–0.35)
NEGATIVE/ZERO:     Pharma ↔ Banking during risk-off
```

---

## 6. FINAL PORTFOLIO SELECTION CRITERIA

After Phase 4–5 analysis, final 5–8 stocks will be selected based on:

| Criterion | Weight | Metric |
|-----------|--------|--------|
| Forecast accuracy | 30% | Best MAPE across models |
| Trend strength | 20% | ADX indicator > 25 |
| Low correlation to others | 20% | Max pairwise ρ < 0.7 |
| Liquidity | 15% | Avg daily volume > 1Mn |
| Risk-adjusted return | 15% | Historical Sharpe > 0.5 |

---

## 7. MODELING STRATEGY JUSTIFICATION

| Model | Stocks Best Suited | Reason |
|-------|-------------------|--------|
| ARIMA | TCS, HINDUNILVR | Near-stationary, linear trend |
| ETS | ITC, HDFCBANK | Clear trend + weekly seasonality |
| Prophet | All stocks | Holiday effects (Budget, elections, Diwali) |
| LSTM | TATAMOTORS, SUNPHARMA | Non-linear, complex patterns |
| GARCH | TATAMOTORS, DRREDDY | Volatility clustering present |

**Ensemble strategy:** Weighted average of top-3 models per stock 
(weights = inverse of test MAPE).

---

## 8. INDIAN MARKET SPECIFIC FACTORS

### Key NSE Calendar Events (2021–2025)
- Union Budget: February (major market mover)
- RBI Monetary Policy: 6 times/year (bi-monthly)
- F&O Expiry: Last Thursday of every month (volatility)
- Results Season: Apr, Jul, Oct, Jan (quarterly earnings)
- Diwali Mahurat Trading: October/November
- Indian National Holidays: ~12/year

### Prophet will be configured with:
```python
india_holidays = holidays.India()
# + Custom: Budget Day, RBI Policy dates, F&O Expiry Thursdays
```

---

## 9. PORTFOLIO CONSTRUCTION PHILOSOPHY

```
APPROACH: CORE-SATELLITE FRAMEWORK

Core Holdings (60–70% of capital):
  → HDFCBANK, TCS, INFY, HINDUNILVR
  → Stable, large-cap, low-volatility anchors
  → Provides consistent returns near market benchmark

Satellite Holdings (20–30% of capital):
  → ICICIBANK, SUNPHARMA, ITC
  → Moderate alpha opportunities
  → Adds sector diversification

Tactical Positions (5–15% of capital):
  → MARUTI or TATAMOTORS (one auto stock)
  → Higher risk, higher potential return
  → Adjusted based on model signals
```

**Optimization target:** Maximize Sharpe Ratio subject to:
- No stock > 25% weight
- No stock < 5% weight  
- Total = 100% (fully invested)
- No short selling

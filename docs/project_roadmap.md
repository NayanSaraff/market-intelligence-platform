# 🗺️ PROJECT ROADMAP & TIMELINE
# Data-Driven Stock Analysis using Time Series Models on StockGro
# ================================================================

## GANTT-STYLE TIMELINE (20 Working Days)

```
PHASE          │ Day 1 │ Day 2 │ Day 3 │ Day 4 │ Day 5 │ Day 6 │...│ Day 20
───────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───┼───────
Phase 1: Plan  │  ███  │       │       │       │       │       │   │
Phase 2: Data  │       │  ███  │  ███  │       │       │       │   │
Phase 3: Prep  │       │       │       │  ███  │       │       │   │
Phase 4: ARIMA │       │       │       │       │  ███  │  ███  │   │
Phase 4: ETS   │       │       │       │       │       │       │   │
Phase 4: Proph │       │       │       │       │       │       │   │
Phase 4: LSTM  │       │       │       │       │       │       │   │
Phase 5: GARCH │       │       │       │       │       │       │   │
Phase 6: Eval  │       │       │       │       │       │       │   │
Phase 7: Port  │       │       │       │       │       │       │   │
Phase 8: Trade │       │       │       │       │       │       │   │
Phase 9: Dash  │       │       │       │       │       │       │   │
Phase 10:Rpt   │       │       │       │       │       │       │  ███
```

---

## DETAILED PHASE BREAKDOWN

### ┌─────────────────────────────────────────────────────────────┐
### │  PHASE 1 — PROJECT PLANNING [Day 1]                        │
### └─────────────────────────────────────────────────────────────┘

**Objective:** Define complete project architecture before writing any code.

Deliverables:
- [x] Project roadmap (this document)
- [x] Complete folder structure
- [x] requirements.txt
- [x] project_config.yaml
- [x] Stock universe with selection rationale
- [x] Modeling strategy
- [x] Portfolio strategy
- [x] Submission checklist

---

### ┌─────────────────────────────────────────────────────────────┐
### │  PHASE 2 — DATA ACQUISITION & EDA [Day 2–3]               │
### └─────────────────────────────────────────────────────────────┘

**Objective:** Download clean NSE data and understand it deeply before modeling.

Sub-tasks:
1. Download data via yfinance (OHLCV + Adj Close)
2. Save raw CSVs → data/raw/
3. Quality checks: missing dates, splits, dividends
4. Compute basic statistics per stock
5. EDA visualizations:
   - Price trends with volume (2021–2025)
   - Daily return distributions (histogram + QQ-plot)
   - Rolling 30-day volatility
   - Correlation heatmap (returns)
   - Pairplot of returns
6. Stationarity tests:
   - ADF test (null: non-stationary)
   - KPSS test (null: stationary)
   - PP test (optional)
7. ACF/PACF plots (for ARIMA order selection)
8. Seasonal decomposition (trend, seasonal, residual)
9. Save processed data → data/processed/

**Expected Outputs:**
- 10 CSV files in data/raw/
- EDA notebook (00_setup_and_data.ipynb)
- 20+ EDA charts in outputs/charts/eda/
- Stationarity test results table

---

### ┌─────────────────────────────────────────────────────────────┐
### │  PHASE 3 — PREPROCESSING & FEATURES [Day 4]               │
### └─────────────────────────────────────────────────────────────┘

**Objective:** Prepare analysis-ready datasets with engineered features.

Sub-tasks:
1. Handle missing values (forward-fill for market holidays)
2. Align all stocks to common trading calendar
3. Strict train/test split (no look-ahead bias)
4. Compute returns: daily log returns, cumulative returns
5. Technical indicators:
   - Trend: SMA(20), SMA(50), SMA(200), EMA(12), EMA(26)
   - Momentum: MACD, RSI(14), Stochastic Oscillator
   - Volatility: Bollinger Bands, ATR(14)
   - Volume: OBV, Volume MA(20)
6. Lag features (t-1, t-2, t-5, t-21)
7. Save → data/features/

**Expected Outputs:**
- Preprocessed CSV per stock in data/processed/
- Feature-enriched CSV per stock in data/features/
- Preprocessing notebook (01_preprocessing.ipynb)

---

### ┌─────────────────────────────────────────────────────────────┐
### │  PHASE 4 — MODEL BUILDING [Day 5–10]                      │
### └─────────────────────────────────────────────────────────────┘

**Day 5–6: ARIMA/SARIMA**
1. Auto-ARIMA (pmdarima) for each stock on Close price (differenced)
2. Manual verification of (p,d,q) using ACF/PACF
3. Fit ARIMA on training data
4. Generate test-period predictions (rolling 1-step ahead)
5. Residual diagnostics (Ljung-Box, normality)
6. Record AIC, BIC
7. 5-day ahead forecast with confidence intervals
8. Save models, plots, metrics

**Day 6: Holt-Winters ETS**
1. Fit ExponentialSmoothing on training data
2. Test additive vs multiplicative decomposition
3. Cross-validate window size
4. Generate test predictions
5. 5-day forecast
6. Save models, plots, metrics

**Day 7: Prophet**
1. Format data to Prophet's ds/y format
2. Add Indian market holidays
3. Configure seasonality (yearly + weekly)
4. Tune changepoint_prior_scale
5. Cross-validate using Prophet's built-in CV
6. Generate test predictions with uncertainty
7. Plot components (trend, seasonality)
8. 5-day forecast
9. Save models, plots, metrics

**Day 8–10: LSTM**
1. Scale data (MinMaxScaler per stock)
2. Create sequences (lookback=60 trading days)
3. Build architecture: Input→LSTM(128)→Dropout(0.2)→LSTM(64)→Dense(1)
4. Compile: Adam optimizer, MSE loss
5. Train with EarlyStopping (patience=10)
6. Plot training/validation curves
7. Generate test predictions (inverse transform)
8. 5-day iterative forecast
9. Save model weights, scalers, metrics

---

### ┌─────────────────────────────────────────────────────────────┐
### │  PHASE 5 — VOLATILITY ANALYSIS [Day 11]                   │
### └─────────────────────────────────────────────────────────────┘

**Objective:** Model time-varying volatility for risk management.

Sub-tasks:
1. Compute log returns for all stocks
2. Test for ARCH effects (Engle's test)
3. Fit GARCH(1,1) on each stock's returns
4. Compute conditional volatility
5. Annualize volatility: σ_annual = σ_daily × √252
6. Rolling 30-day realized volatility
7. 5-day volatility forecast per stock
8. Volatility ranking and regime classification
9. Correlation between volatility and returns

**Expected Outputs:**
- GARCH models saved in models/garch/
- Conditional volatility plots per stock
- Volatility ranking table

---

### ┌─────────────────────────────────────────────────────────────┐
### │  PHASE 6 — MODEL EVALUATION [Day 12–13]                   │
### └─────────────────────────────────────────────────────────────┘

**Objective:** Rigorously compare all models, select best per stock.

Sub-tasks:
1. Compute all metrics for every model × stock combination
2. Build master comparison table (40 rows: 10 stocks × 4 models)
3. Rank models per stock
4. Identify best model per stock
5. Predicted vs Actual plots (all models, all stocks)
6. Error analysis: where do models fail?
7. Export metrics_summary.xlsx
8. Save predicted vs actual CSV template

---

### ┌─────────────────────────────────────────────────────────────┐
### │  PHASE 7 — PORTFOLIO CONSTRUCTION [Day 14]                │
### └─────────────────────────────────────────────────────────────┘

**Objective:** Build optimally allocated ₹10,00,000 portfolio.

Sub-tasks:
1. Select 5–8 stocks (based on model performance + diversification)
2. Compute historical expected returns (annualized)
3. Compute covariance matrix of returns
4. Run Mean-Variance Optimization (scipy.optimize)
5. Generate efficient frontier curve
6. Find Maximum Sharpe Ratio portfolio
7. Compare vs Equal-Weight portfolio
8. Compute portfolio metrics:
   - Expected annual return
   - Portfolio volatility
   - Sharpe ratio
   - Maximum drawdown
   - VaR₉₅, CVaR₉₅
9. Allocate ₹10,00,000 → share counts per stock
10. Export portfolio_allocation.csv

---

### ┌─────────────────────────────────────────────────────────────┐
### │  PHASE 8 — STOCKGRO INSTRUCTIONS [Day 15]                 │
### └─────────────────────────────────────────────────────────────┘

**Objective:** Generate actionable trading signals for StockGro platform.

Sub-tasks:
1. Compile 5-day forecasts for all portfolio stocks
2. Signal logic:
   - Forecast > Current Price + Threshold → BUY
   - Forecast < Current Price - Threshold → SELL
   - Otherwise → HOLD
3. Compute: Entry price, Target price, Stop-loss
4. Compute Risk-Reward Ratio (min 1:2)
5. Position sizing based on portfolio weights
6. Export stockgro_instructions.xlsx (formatted)
7. Generate instruction cards per stock

---

### ┌─────────────────────────────────────────────────────────────┐
### │  PHASE 9 — DASHBOARD [Day 16–17]                          │
### └─────────────────────────────────────────────────────────────┘

**Objective:** Interactive web dashboard summarizing all findings.

Panels:
1. Header: Project title, date, portfolio value
2. Stock selector dropdown
3. Price chart with forecast (Plotly + confidence bands)
4. Model comparison bar chart (RMSE)
5. Portfolio allocation pie chart
6. Risk metrics table
7. 5-day forecast signals table
8. Rolling volatility chart

Tech: Plotly Dash + Dash Bootstrap Components

---

### ┌─────────────────────────────────────────────────────────────┐
### │  PHASE 10 — REPORT & SUBMISSION [Day 18–20]               │
### └─────────────────────────────────────────────────────────────┘

**Objective:** Professional 10-page academic report.

Structure:
1. Abstract (250 words)
2. Introduction & Objectives
3. Data Description
4. Methodology (models)
5. Results & Analysis
6. Portfolio Construction
7. Trading Signals
8. Conclusion & Future Work
9. References
10. Appendix (optional: extra charts)

Final checks:
- Run all notebooks clean (Restart & Run All)
- Verify all outputs are saved
- Check all links in README
- Final zip: StockGro_Capstone_[Name]_[Date].zip

---

## RISK REGISTER & MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| yfinance data gaps | Medium | High | Use forward-fill, document gaps |
| LSTM not converging | Medium | Medium | Tune LR, reduce complexity |
| Prophet poor fit | Low | Medium | Tune changepoint scale |
| ARIMA non-stationarity | Low | Low | Auto-difference (auto_arima d=1,2) |
| Portfolio concentration | Low | High | Max 25% weight constraint |
| Overfitting | Medium | High | Strict train/test split, CV |
| Library version conflicts | Medium | Medium | requirements.txt pinned versions |

# ✅ CAPSTONE SUBMISSION CHECKLIST
# Data-Driven Stock Analysis using Time Series Models on StockGro
# ============================================================
# Mark each item [x] when complete

## PHASE 1 — PROJECT PLANNING
- [x] Project roadmap defined
- [x] Timeline established (20 days)
- [x] Folder structure created
- [x] requirements.txt generated
- [x] project_config.yaml created
- [x] Stock universe selected (10 stocks, 5 sectors)
- [x] Modeling strategy documented
- [x] Portfolio strategy defined
- [x] Submission checklist created

## PHASE 2 — DATA ACQUISITION & EDA
- [ ] yfinance data downloaded (2021-01-01 to 2025-12-31)
- [ ] Raw CSVs saved to data/raw/
- [ ] Missing data checked and documented
- [ ] Data quality report generated
- [ ] EDA plots: price trends, volume, returns distribution
- [ ] Correlation matrix computed and plotted
- [ ] Stationarity tests (ADF, KPSS) performed
- [ ] ACF/PACF plots generated
- [ ] Seasonal decomposition performed
- [ ] EDA notebook completed

## PHASE 3 — PREPROCESSING & FEATURE ENGINEERING
- [ ] Train/test split performed (70/30 by date)
- [ ] Missing values handled
- [ ] Returns computed (daily log returns)
- [ ] Technical indicators computed:
  - [ ] SMA 20, SMA 50, SMA 200
  - [ ] EMA 12, EMA 26
  - [ ] MACD + Signal
  - [ ] RSI (14-day)
  - [ ] Bollinger Bands
  - [ ] ATR (Average True Range)
  - [ ] OBV (On-Balance Volume)
- [ ] Rolling volatility computed (30-day)
- [ ] Processed data saved to data/processed/
- [ ] Feature data saved to data/features/

## PHASE 4 — MODEL BUILDING
### ARIMA / SARIMA
- [ ] Auto-ARIMA run on all 10 stocks
- [ ] Best (p,d,q) parameters selected per stock
- [ ] Model fitted on training data
- [ ] Residual diagnostics: Ljung-Box, ACF plots
- [ ] AIC, BIC values recorded
- [ ] Test set predictions generated
- [ ] 5-day forecast generated
- [ ] Models saved to models/arima/

### Holt-Winters (ETS)
- [ ] Model fitted with additive/multiplicative components
- [ ] Trend and seasonality components visualized
- [ ] Test set predictions generated
- [ ] 5-day forecast generated
- [ ] Models saved to models/ets/

### Prophet
- [ ] Holiday calendar added (Indian market holidays)
- [ ] Model fitted with regressor tuning
- [ ] Trend changepoints visualized
- [ ] Component decomposition plotted
- [ ] Test set predictions with uncertainty intervals
- [ ] 5-day forecast generated
- [ ] Models saved to models/prophet/

### LSTM
- [ ] Data scaled with MinMaxScaler
- [ ] Sequences created (lookback=60 days)
- [ ] Architecture defined: [128→64→Dense(1)]
- [ ] Model trained with early stopping
- [ ] Training/validation loss plotted
- [ ] Predictions inverse-transformed
- [ ] 5-day forecast generated
- [ ] Models saved to models/lstm/

## PHASE 5 — VOLATILITY ANALYSIS (GARCH)
- [ ] Daily log returns computed for all stocks
- [ ] Return distribution analysis (normality test)
- [ ] Volatility clustering visualized
- [ ] ARCH effects test (Engle's test)
- [ ] GARCH(1,1) fitted on all stocks
- [ ] Conditional volatility plotted
- [ ] 5-day volatility forecast generated
- [ ] Annualized volatility ranked by stock
- [ ] Models saved to models/garch/

## PHASE 6 — MODEL EVALUATION & COMPARISON
- [ ] RMSE calculated for all models x all stocks
- [ ] MAE calculated for all models x all stocks
- [ ] MAPE calculated for all models x all stocks
- [ ] Directional Accuracy calculated
- [ ] R² calculated
- [ ] Model ranking table generated
- [ ] Best model per stock identified
- [ ] Comparison visualizations created
- [ ] Predicted vs Actual plots generated
- [ ] metrics_summary.xlsx exported

## PHASE 7 — PORTFOLIO CONSTRUCTION
- [ ] Final 5–8 stocks selected (based on model performance + diversification)
- [ ] Expected returns computed
- [ ] Covariance matrix computed
- [ ] Efficient frontier generated
- [ ] Mean-variance optimized weights computed
- [ ] Equal-weight benchmark computed
- [ ] Portfolio metrics: Expected Return, Volatility, Sharpe Ratio
- [ ] Maximum Drawdown computed
- [ ] VaR₉₅ and CVaR₉₅ computed
- [ ] ₹10,00,000 allocated with actual share counts
- [ ] portfolio_allocation.csv exported
- [ ] Portfolio pie chart generated

## PHASE 8 — STOCKGRO TRADING INSTRUCTIONS
- [ ] 5-day forecasts compiled for all portfolio stocks
- [ ] Signal generated per stock (BUY / HOLD / SELL)
- [ ] Entry price, target price, stop-loss computed
- [ ] Risk-reward ratio computed
- [ ] Position sizing determined
- [ ] stockgro_instructions.xlsx exported
- [ ] Instructions formatted for StockGro platform

## PHASE 9 — DASHBOARD
- [ ] Dash application created
- [ ] Stock price viewer (interactive)
- [ ] Forecast comparison panel
- [ ] Portfolio allocation chart
- [ ] Risk metrics panel
- [ ] Model accuracy comparison chart
- [ ] Dashboard tested and functional

## PHASE 10 — FINAL REPORT & DOCUMENTATION
- [ ] Abstract written (250 words)
- [ ] Introduction (objectives, scope)
- [ ] Data description section
- [ ] Methodology section (all models)
- [ ] Results & findings section
- [ ] Portfolio analysis section
- [ ] Conclusion and future work
- [ ] References (15+ citations)
- [ ] All charts embedded in report
- [ ] Report formatted to 10 pages max
- [ ] Reflection section included
- [ ] README.md finalized
- [ ] All notebooks run clean (Restart & Run All)
- [ ] All outputs saved and verifiable
- [ ] Final zip/folder ready for submission

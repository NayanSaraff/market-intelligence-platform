## EXECUTION GUIDE — StockGro Capstone Project

### Project Status
- **Phases 1-11**: Code complete, ready for execution
- **Phase 12**: Template created, ready for execution
- **Phases 13-14**: Not yet started

---

## QUICK START (One-Click Full Pipeline)

```bash
cd d:\PROJECTS\IIT GUWAHTI- CAPSTONE\StockGro_Capstone
jupyter notebook notebooks/10_master_runner.ipynb
```

Then execute all cells. This will run all phases sequentially and generate all outputs.

---

## STEP-BY-STEP EXECUTION

### Step 1: Setup Environment
```bash
cd d:\PROJECTS\IIT GUWAHTI- CAPSTONE\StockGro_Capstone
pip install -r requirements.txt
```

### Step 2: Validate Project Environment
```bash
$env:PYTHONPATH = $PWD
python tests\test_environment.py
python -c "from pathlib import Path; required=['outputs/reports/all_model_metrics.csv','outputs/portfolio/final_portfolio_allocation.csv']; missing=[p for p in required if not Path(p).exists()]; print('outputs OK' if not missing else f'missing: {missing}')"
```

### Step 3: Run Individual Phases (Manual)

Execute each notebook sequentially:

```bash
# Phase 1: Project Planning & Setup
jupyter notebook notebooks/Phase1_Planning_Setup.ipynb

# Phase 2: Data Collection
jupyter notebook notebooks/00_Phase2_Data_Collection.ipynb

# Phase 3: Stock Selection
jupyter notebook notebooks/01_Phase3_Stock_Selection.ipynb

# Phase 6: Forecasting Models (Auto-tuned)
jupyter notebook notebooks/02_arima_sarima.ipynb
jupyter notebook notebooks/03_ets_holtwinters.ipynb
jupyter notebook notebooks/04_prophet.ipynb
jupyter notebook notebooks/05_lstm.ipynb

# Phase 8: Volatility Analysis
jupyter notebook notebooks/06_garch_volatility.ipynb

# Phase 11: Model Comparison
jupyter notebook notebooks/07_model_comparison.ipynb

# Phase 9: Portfolio Construction
jupyter notebook notebooks/08_portfolio.ipynb

# Phase 10: StockGro Trading Signals
jupyter notebook notebooks/09_stockgro_signals.ipynb

# Phase 12: Predicted vs Actual Validation
jupyter notebook notebooks/11_predicted_vs_actual.ipynb
```

### Step 4: Run Master Orchestrator
```bash
jupyter notebook notebooks/10_master_runner.ipynb
```

Execute all cells. This will:
- Run all phases in sequence
- Time each phase execution
- Generate comprehensive output checklist
- Report any errors

---

## COMMAND-LINE EXECUTION (No GUI)

If you prefer to run notebooks without opening Jupyter GUI:

```bash
# Convert and execute individual notebooks
jupyter nbconvert --to notebook --execute --inplace notebooks/02_arima_sarima.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_ets_holtwinters.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_prophet.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_lstm.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_garch_volatility.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/07_model_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/08_portfolio.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/09_stockgro_signals.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/11_predicted_vs_actual.ipynb

# Run master runner last
jupyter nbconvert --to notebook --execute --inplace notebooks/10_master_runner.ipynb
```

---

## EXPECTED EXECUTION TIME

- **Phase 2-3** (Data & Selection): ~5 minutes
- **Phase 4-6** (Models): ~30-45 minutes (ARIMA auto-tuning is slow)
- **Phase 8** (GARCH): ~5 minutes
- **Phase 11** (Comparison): ~2 minutes
- **Phase 9** (Portfolio): ~2 minutes
- **Phase 10** (Signals): ~2 minutes
- **Phase 12** (Validation): ~3 minutes
- **Master Runner** (Full Pipeline): ~1 hour total

---

## OUTPUT FILES GENERATED

### After Phase 6 (Models):
```
outputs/reports/
  - arima_metrics.csv
  - ets_metrics.csv
  - prophet_metrics.csv
  - lstm_metrics.csv
  - all_model_metrics.csv (combined)

outputs/predictions/
  - arima_5day_forecasts.csv
  - ets_5day_forecasts.csv
  - prophet_5day_forecasts.csv
  - lstm_5day_forecasts.csv
  - all_5day_forecasts.csv (combined)
  - *_actual_vs_predicted.csv (for each model)
```

### After Phase 8 (GARCH):
```
outputs/reports/
  - garch_volatility_report.csv
```

### After Phase 11 (Comparison):
```
outputs/reports/
  - model_comparison_summary.csv
  - best_model_per_ticker.csv

outputs/charts/comparison/
  - model_comparison_summary.png
```

### After Phase 9 (Portfolio):
```
outputs/portfolio/
  - final_portfolio_allocation.csv
  - weights.csv
  - portfolio_metrics_summary.csv
  - stockgro_trade_instructions.csv
  - StockGro_Phase8_Phase9_Report.xlsx

outputs/charts/portfolio/
  - allocation_pie_chart.png
  - risk_return_scatter.png
```

### After Phase 10 (Signals):
```
outputs/portfolio/
  - day1_execution_plan.json
  - day2_tracking_schedule.csv

outputs/charts/signals/
  - trading_signals_heatmap.png
```

### After Phase 12 (Validation):
```
outputs/reports/
  - day1_prediction_errors.csv
  - directional_accuracy_analysis.csv

outputs/charts/validation/
  - prediction_validation_summary.png
```

---

## TROUBLESHOOTING

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**: Make sure you're running from the project root directory:
```bash
cd d:\PROJECTS\IIT GUWAHTI- CAPSTONE\StockGro_Capstone
```

### Issue: "FileNotFoundError: data/raw/*.csv"
**Solution**: Ensure raw data files exist. Run Phase 2 first:
```bash
jupyter notebook notebooks/00_Phase2_Data_Collection.ipynb
```

### Issue: ARIMA is taking too long
**Solution**: This is normal. ARIMA auto-tuning uses grid search. It can take 10-15 minutes per stock. Be patient.

### Issue: "LSTM: Out of memory"
**Solution**: The LSTM surrogate (Ridge) is memory-efficient. If you get out-of-memory errors, close other applications or use the Keras pathway (see commented code in src/models/lstm_model.py).

### Issue: "Prophet: ImportError"
**Solution**: Prophet requires Pystan. Ensure you installed from requirements.txt:
```bash
pip install -r requirements.txt
```

---

## REPRODUCIBILITY

All notebooks are fully reproducible:

1. **Global Seed**: Set to 42 at package initialization (src/__init__.py)
2. **Config-Driven**: All dates, stocks, and parameters come from `configs/project_config.yaml`
3. **Deterministic Models**: All models use seed=42
4. **No Random Elements**: No randomization in feature engineering or train/test split

To verify reproducibility, run the full pipeline twice and compare outputs:
```bash
# First run
jupyter nbconvert --to notebook --execute --inplace notebooks/10_master_runner.ipynb

# Copy outputs to backup
mkdir outputs_run1
cp -r outputs/* outputs_run1/

# Clean outputs
rm -rf outputs/reports/* outputs/predictions/* outputs/portfolio/* outputs/charts/*

# Second run
jupyter nbconvert --to notebook --execute --inplace notebooks/10_master_runner.ipynb

# Compare (outputs should be identical)
diff -r outputs outputs_run1
```

---

## NEXT STEPS

1. **Execute Full Pipeline** using Master Runner
2. **Validate Outputs** by checking outputs/ directories
3. **Phase 13**: Generate Final 8–10 Page Academic Report
4. **Phase 14** (Optional): Build Interactive Dashboard

---

## SUPPORT

For issues or questions:
1. Check ASSUMPTIONS.md for design decisions
2. Review DEEP_CHECK_REPORT.md for validation results
3. Check notebook markdown cells for phase-specific instructions
4. Run test_validation.py for environment diagnostics

---

**Project**: StockGro Capstone | Data-Driven Stock Analysis using Time Series Models
**Institution**: IIT Guwahati
**Date**: May 2026
**Status**: Phases 1-12 Ready | Phases 13-14 Pending

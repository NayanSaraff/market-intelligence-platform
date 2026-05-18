# Project Assumptions and Key Decisions

This file records decisions enforced by recent code updates (May 13, 2026).

- Date-driven train/test splitting: `configs/project_config.yaml` `dates.train_end` and `dates.test_start` are used by `src/data/preprocessor.py` to create strict temporal splits. A legacy `TRAIN_ROWS` fallback remains but is discouraged.

- Adjusted close precedence: If raw CSVs contain `Adj Close` (or `Adj_Close`), the preprocessor will use it and map to `Close` for downstream modules. This prevents incorrect price adjustments.

- Volatility computation: All volatility metrics (30d, 10d) are computed on log-returns (`LogReturn`) not simple returns.

- Reproducibility: A global seed (`GLOBAL_RANDOM_SEED`) is set in `src/__init__.py` (default 42). Set `STOCKGRO_SEED` env var to override.

- LSTM: The current implementation uses a reproducible Ridge-based surrogate for academic speed and determinism. A Keras/TensorFlow implementation is included (commented) and can be enabled if GPUs are available.

- Outputs: Feature and processed CSVs are saved under `data/processed/` and `data/features/`. Models read the `Close` column and `LogReturn` where required.

If you want alternative choices (e.g., percentage returns for volatility, different seed), I can update the code accordingly.

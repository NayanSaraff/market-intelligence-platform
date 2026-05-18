"""Visualization helpers exposed as a flat public API."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(actual, predicted, title: str = "Forecast vs Actual"):
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(pd.Series(actual).index, pd.Series(actual).values, label="Actual", linewidth=2)
	ax.plot(pd.Series(predicted).index, pd.Series(predicted).values, label="Forecast", linewidth=2)
	ax.set_title(title)
	ax.set_xlabel("Date")
	ax.set_ylabel("Value")
	ax.legend()
	fig.tight_layout()
	return fig, ax


def plot_residuals(residuals, title: str = "Residuals"):
	residuals = pd.Series(residuals)
	fig, ax = plt.subplots(figsize=(10, 3.5))
	ax.plot(residuals.index, residuals.values, color="#444")
	ax.axhline(0.0, linestyle="--", linewidth=1)
	ax.set_title(title)
	ax.set_xlabel("Date")
	ax.set_ylabel("Residual")
	fig.tight_layout()
	return fig, ax


def plot_metric_bars(df: pd.DataFrame, metric: str, title: str = "Model Metric Comparison"):
	if metric not in df.columns:
		raise KeyError(f"Metric '{metric}' not found in DataFrame")
	labels = df["Model"] if "Model" in df.columns else df.index.astype(str)
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.bar(labels, df[metric].values)
	ax.set_title(title)
	ax.set_xlabel("Model")
	ax.set_ylabel(metric)
	ax.tick_params(axis="x", rotation=45)
	fig.tight_layout()
	return fig, ax


__all__ = ["plot_forecast", "plot_residuals", "plot_metric_bars"]

"""Evaluation helpers exposed as a flat public API."""

from typing import Dict

import numpy as np


def rmse(y_true, y_pred) -> float:
	y_t = np.array(y_true, dtype=float)
	y_p = np.array(y_pred, dtype=float)
	return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mae(y_true, y_pred) -> float:
	y_t = np.array(y_true, dtype=float)
	y_p = np.array(y_pred, dtype=float)
	return float(np.mean(np.abs(y_t - y_p)))


def mape(y_true, y_pred) -> float:
	y_t = np.array(y_true, dtype=float)
	y_p = np.array(y_pred, dtype=float)
	mask = y_t != 0
	if not np.any(mask):
		return 0.0
	return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100)


def directional_accuracy(y_true, y_pred) -> float:
	y_t = np.array(y_true, dtype=float)
	y_p = np.array(y_pred, dtype=float)
	if y_t.size < 2 or y_p.size < 2:
		return 0.0
	dir_t = np.diff(y_t) > 0
	dir_p = np.diff(y_p) > 0
	return float(np.mean(dir_t == dir_p) * 100)


def r2_score(y_true, y_pred) -> float:
	y_t = np.array(y_true, dtype=float)
	y_p = np.array(y_pred, dtype=float)
	ss_res = np.sum((y_t - y_p) ** 2)
	ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
	return float(1 - ss_res / (ss_tot + 1e-10))


def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
	return {
		"RMSE": round(rmse(y_true, y_pred), 4),
		"MAE": round(mae(y_true, y_pred), 4),
		"MAPE_%": round(mape(y_true, y_pred), 4),
		"DA_%": round(directional_accuracy(y_true, y_pred), 2),
		"R2": round(r2_score(y_true, y_pred), 4),
	}


__all__ = [
	"rmse",
	"mae",
	"mape",
	"directional_accuracy",
	"r2_score",
	"evaluate_forecast",
]

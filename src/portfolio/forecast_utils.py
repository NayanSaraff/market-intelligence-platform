"""Utilities for handling model forecasts in portfolio construction.

Rationale for negative-forecast handling
--------------------------------------
Some forecasters (especially returns-based or mean-difference models)
can produce negative point forecasts. For price forecasts, negative
values are non-sensical and should be handled before portfolio sizing.

Policy implemented here:
- If `allow_negative` is False, forecasts are clipped at `min_value` (default 0.0).
- If `allow_negative` is True, values are left untouched.
- Optionally, a `method='rescale'` will shift the series additively so that
  the minimum equals `min_value`, preserving relative distances.

These choices are conservative and transparent: clipping avoids assigning
negative weights or positions that would imply negative prices, while
rescaling preserves forecast shape when small negative biases are present.
"""
from typing import Optional
import pandas as pd


def sanitize_forecasts(forecasts: pd.Series,
                       allow_negative: bool = False,
                       min_value: float = 0.0,
                       method: str = "clip") -> pd.Series:
    """Return a sanitized copy of `forecasts` according to policy.

    Args:
        forecasts: Series of numeric forecasts (indexed by date or ticker).
        allow_negative: If True, return `forecasts` unchanged.
        min_value: Minimum allowed value when not allowing negatives.
        method: 'clip' (default) or 'rescale'.

    Returns:
        A new pandas Series with adjusted forecasts.
    """
    if allow_negative:
        return forecasts.copy()

    s = forecasts.copy().astype(float)
    if method == "clip":
        return s.clip(lower=min_value)
    elif method == "rescale":
        cur_min = s.min()
        if cur_min >= min_value:
            return s
        shift = min_value - cur_min
        return s + shift
    else:
        raise ValueError("Unknown method: choose 'clip' or 'rescale'")


__all__ = ["sanitize_forecasts"]

"""
src/utils/config_loader.py
──────────────────────────
Utility to load and access project configuration from YAML.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path: str = "configs/project_config.yaml") -> dict:
    """
    Load project configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file (relative to project root).

    Returns
    -------
    dict
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Ensure you are running from the project root directory."
        )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_tickers(config: dict) -> list:
    """Extract all ticker symbols from config."""
    tickers = []
    for sector_stocks in config['stocks'].values():
        for stock in sector_stocks:
            tickers.append(stock['symbol'])
    return tickers


def get_stock_info(config: dict) -> dict:
    """Get {ticker: {name, sector, role}} mapping."""
    info = {}
    for sector, sector_stocks in config['stocks'].items():
        for stock in sector_stocks:
            info[stock['symbol']] = {
                'name':   stock['name'],
                'sector': sector,
                'role':   stock['weight_class'],
            }
    return info


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully.")
    print(f"Tickers: {get_tickers(cfg)}")

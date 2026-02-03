"""
Real-Time Macro Indicators for Economic Regime Features

This module downloads real-time macro indicators that can be used
for regime-dependent revision prediction. These are publicly available
via FRED and are known AT THE TIME of the economic release.

Indicators:
- T10Y2Y: 10-Year minus 2-Year Treasury yield spread (yield curve)
- VIXCLS: CBOE Volatility Index (VIX)
- SAHMREALTIME: Sahm Rule Recession Indicator

IMPORTANT: These are REAL-TIME indicators - they're published daily/weekly
and were known to market participants at the time. This is not the same
as NBER recession dates, which are announced with significant lag.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import FRED_API_KEY, RAW_DATA_DIR, logger
from data.scripts.fred_client import FREDClient


MACRO_INDICATORS = {
    "T10Y2Y": {
        "name": "10Y-2Y Treasury Spread",
        "column": "yield_curve_spread",
        "description": "Yield curve slope, negative = inverted = recession signal",
    },
    "VIXCLS": {
        "name": "VIX Volatility Index",
        "column": "vix",
        "description": "Market uncertainty, >25 = elevated",
    },
    "SAHMREALTIME": {
        "name": "Sahm Rule Recession Indicator",
        "column": "sahm_indicator",
        "description": "Real-time recession indicator, >=0.5 = recession signal",
    },
}


def download_macro_indicator(
    series_id: str,
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download a macro indicator from FRED.

    Unlike revision data, we just need current values here since
    these are market prices/indices that are known in real-time.
    """
    import requests

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    observations = data.get("observations", [])

    if not observations:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(observations)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df[["date", "value"]].dropna()


def download_all_macro_indicators(
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download all macro indicators and merge into single DataFrame.

    Returns DataFrame with columns:
    - date
    - yield_curve_spread
    - vix
    - sahm_indicator
    - is_contraction (derived from sahm_indicator >= 0.5)
    """
    output_dir = output_dir or RAW_DATA_DIR
    all_data = []

    for series_id, config in MACRO_INDICATORS.items():
        logger.info(f"Downloading {series_id} ({config['name']})")

        try:
            df = download_macro_indicator(series_id, start_date, end_date)
            df = df.rename(columns={"value": config["column"]})
            all_data.append(df)
            logger.info(f"  Downloaded {len(df)} observations")
        except Exception as e:
            logger.error(f"Failed to download {series_id}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    # Merge all indicators on date
    result = all_data[0]
    for df in all_data[1:]:
        result = pd.merge(result, df, on="date", how="outer")

    # Sort by date and forward-fill missing values
    # (e.g., VIX is daily, but we might have gaps)
    result = result.sort_values("date")
    result = result.ffill()

    # Add derived features
    if "sahm_indicator" in result.columns:
        result["is_contraction"] = (result["sahm_indicator"] >= 0.5).astype(int)
    else:
        result["is_contraction"] = 0

    # Save to disk
    output_path = output_dir / "macro_indicators.parquet"
    result.to_parquet(output_path, index=False)
    logger.info(f"Saved macro indicators to {output_path}")

    return result


def load_macro_indicators(path: Optional[Path] = None) -> pd.DataFrame:
    """Load previously downloaded macro indicators."""
    path = path or RAW_DATA_DIR / "macro_indicators.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


if __name__ == "__main__":
    print("Downloading macro indicators...")
    df = download_all_macro_indicators()
    print(f"\nDownloaded {len(df)} observations")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample:\n{df.tail()}")

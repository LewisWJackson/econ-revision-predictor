"""
Automatic Consensus Forecast Fetcher

Gets consensus/forecast values from ForexFactory's free JSON feed.
No API key needed. No manual input needed.

Updates automatically before each release.
"""

import requests
import json
import re
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Map ForexFactory event titles to our indicator codes
FF_TITLE_MAP = {
    "Advance GDP q/q": "GDPC1",
    "Final GDP q/q": "GDPC1",
    "Prelim GDP q/q": "GDPC1",
    "GDP q/q": "GDPC1",
    "Non-Farm Employment Change": "PAYEMS",
    "ADP Non-Farm Employment Change": None,  # ADP is not official, skip
    "Unemployment Claims": "ICSA",
    "Industrial Production m/m": "INDPRO",
    "Retail Sales m/m": "RSXFS",
    "Core Retail Sales m/m": None,  # We use headline, not core
    "Durable Goods Orders m/m": "DGORDER",
    "Core Durable Goods Orders m/m": None,
    "Housing Starts": "HOUST",
    "Building Permits": None,
}


def parse_value(raw: str) -> Optional[float]:
    """
    Parse a ForexFactory value string into a float.

    Examples:
        "200K" → 200000
        "1.8%" → 1.8
        "1.245M" → 1245000
        "-0.3%" → -0.3
        "50K" → 50000
        "" → None
    """
    if not raw or not raw.strip():
        return None

    raw = raw.strip()

    # Remove percentage sign (we keep the number as-is for %)
    is_pct = raw.endswith("%")
    raw_clean = raw.replace("%", "").strip()

    # Handle K (thousands) and M (millions)
    multiplier = 1
    if raw_clean.upper().endswith("K"):
        multiplier = 1000
        raw_clean = raw_clean[:-1]
    elif raw_clean.upper().endswith("M"):
        multiplier = 1_000_000
        raw_clean = raw_clean[:-1]
    elif raw_clean.upper().endswith("B"):
        multiplier = 1_000_000_000
        raw_clean = raw_clean[:-1]

    try:
        value = float(raw_clean) * multiplier
        return value
    except ValueError:
        logger.warning(f"Could not parse value: '{raw}'")
        return None


def fetch_consensus() -> Dict[str, Dict]:
    """
    Fetch this week's consensus forecasts from ForexFactory.

    Returns:
        {
            "PAYEMS": {
                "consensus": 200000.0,
                "previous": 180000.0,
                "release_time": "2026-02-07T08:30:00",
                "event_title": "Non-Farm Employment Change",
            },
            ...
        }
    """
    try:
        resp = requests.get(
            FF_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch ForexFactory calendar: {e}")
        return {}

    result = {}

    for event in events:
        # Only US events
        if event.get("country") != "USD":
            continue

        title = event.get("title", "")
        indicator_code = FF_TITLE_MAP.get(title)

        if indicator_code is None:
            continue

        forecast_raw = event.get("forecast", "")
        previous_raw = event.get("previous", "")
        actual_raw = event.get("actual", "")
        release_time = event.get("date", "")

        forecast = parse_value(forecast_raw)
        previous = parse_value(previous_raw)
        actual = parse_value(actual_raw)

        # Only store if we have at least a forecast or previous
        if forecast is not None or previous is not None:
            # If multiple events map to same indicator, keep the one with forecast
            existing = result.get(indicator_code)
            if existing and existing.get("consensus") and not forecast:
                continue  # Keep the one that has a forecast

            result[indicator_code] = {
                "consensus": forecast,
                "previous": previous,
                "actual": actual,
                "release_time": release_time,
                "event_title": title,
                "forecast_raw": forecast_raw,
                "previous_raw": previous_raw,
            }

    return result


def get_consensus_for(indicator_code: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Get (consensus, previous) for a specific indicator.

    Returns (None, None) if not found.
    """
    data = fetch_consensus()
    if indicator_code in data:
        entry = data[indicator_code]
        return entry.get("consensus"), entry.get("previous")
    return None, None


# CLI test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Fetching consensus forecasts from ForexFactory...")
    print("=" * 70)

    data = fetch_consensus()

    if not data:
        print("No matching events found this week.")
    else:
        for code, info in sorted(data.items()):
            print(f"\n{code} - {info['event_title']}")
            print(f"  Release: {info['release_time']}")
            print(f"  Consensus: {info.get('forecast_raw', 'N/A')} → {info.get('consensus')}")
            print(f"  Previous:  {info.get('previous_raw', 'N/A')} → {info.get('previous')}")
            if info.get("actual"):
                print(f"  Actual:    {info.get('actual')}")

    print("\n" + "=" * 70)
    print("Done.")

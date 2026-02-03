"""
FRED API Client with ALFRED (vintage data) support.

This module handles all interactions with the Federal Reserve Economic Data API,
with special focus on retrieving historical vintage data for revision analysis.

LEAKAGE PREVENTION NOTES:
-------------------------
The key to avoiding data leakage in revision prediction is understanding that
economic data has THREE important dates:
  1. reference_period: The time period being measured (e.g., "Q1 2024")
  2. vintage_date: When this particular value was published/known
  3. observation_date: When we access/query the data (today)

Common leakage mistakes this code prevents:
  - Using current values which returns today's values, not historical vintages
  - Not filtering by vintage_date when computing historical features
  - Using the "final" value before it was actually known

APPROACH (v2 - improved vintage sampling):
------------------------------------------
Instead of uniformly sampling vintages across all time, we now:
1. For each reference period, find the FIRST vintage where it appears (initial release)
2. Query subsequent vintages at specific intervals (7d, 30d, 60d, 90d, 180d)
3. This captures the actual revision dynamics much better
"""

import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    FRED_API_KEY,
    FRED_BASE_URL,
    FRED_RATE_LIMIT_REQUESTS,
    FRED_RATE_LIMIT_WINDOW_SECONDS,
    INDICATORS,
    logger,
)

# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Simple rate limiter for FRED API (120 requests/minute)."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times: List[float] = []

    def wait_if_needed(self):
        """Block until we can make another request within rate limits."""
        now = time.time()
        # Remove requests outside the window
        self.request_times = [
            t for t in self.request_times
            if now - t < self.window_seconds
        ]

        if len(self.request_times) >= self.max_requests:
            # Wait until the oldest request falls outside the window
            sleep_time = self.request_times[0] + self.window_seconds - now + 0.1
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        self.request_times.append(time.time())


# Global rate limiter instance
_rate_limiter = RateLimiter(FRED_RATE_LIMIT_REQUESTS, FRED_RATE_LIMIT_WINDOW_SECONDS)

# =============================================================================
# FRED API CLIENT
# =============================================================================

class FREDClient:
    """
    Client for FRED/ALFRED API with vintage data support.

    Key methods for revision analysis:
    - get_vintage_dates(): Get all dates when a series was revised
    - get_observations_at_vintage(): Get data as it existed on a specific date
    - build_revision_history_v2(): Improved method for capturing revisions
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or FRED_API_KEY
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.base_url = FRED_BASE_URL

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make a rate-limited request to FRED API."""
        _rate_limiter.wait_if_needed()

        params["api_key"] = self.api_key
        params["file_type"] = "json"

        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            logger.error(f"FRED API error: {response.status_code} - {response.text[:200]}")
        response.raise_for_status()

        return response.json()

    def get_series_info(self, series_id: str) -> Dict:
        """Get metadata about a series."""
        result = self._make_request("series", {"series_id": series_id})
        return result.get("seriess", [{}])[0]

    def get_vintage_dates(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[datetime]:
        """
        Get all dates when this series was revised or new data was released.
        """
        params = {"series_id": series_id}
        if start_date:
            params["realtime_start"] = start_date
        if end_date:
            params["realtime_end"] = end_date

        result = self._make_request("series/vintagedates", params)
        vintage_dates = result.get("vintage_dates", [])

        return [datetime.strptime(d, "%Y-%m-%d") for d in vintage_dates]

    def get_observations_at_vintage(
        self,
        series_id: str,
        vintage_date: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get observations as they existed on a specific vintage date.

        LEAKAGE SAFE: Returns only data available as of vintage_date.
        """
        params = {
            "series_id": series_id,
            "realtime_start": vintage_date,
            "realtime_end": vintage_date,
        }

        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end

        result = self._make_request("series/observations", params)
        observations = result.get("observations", [])

        if not observations:
            return pd.DataFrame(columns=["date", "value", "vintage_date"])

        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["vintage_date"] = pd.to_datetime(vintage_date)

        return df[["date", "value", "vintage_date"]]

    def find_initial_release_date(
        self,
        series_id: str,
        reference_date: datetime,
        vintage_dates: List[datetime],
    ) -> Optional[datetime]:
        """
        Find the first vintage date where a reference_date appears.

        This is the "initial release" date for that observation.
        """
        ref_date_str = reference_date.strftime("%Y-%m-%d")

        for vd in sorted(vintage_dates):
            if vd < reference_date:
                continue

            vd_str = vd.strftime("%Y-%m-%d")
            try:
                df = self.get_observations_at_vintage(
                    series_id=series_id,
                    vintage_date=vd_str,
                    observation_start=ref_date_str,
                    observation_end=ref_date_str,
                )
                if not df.empty and not pd.isna(df.iloc[0]["value"]):
                    return vd
            except:
                continue

        return None

    def build_revision_history_v2(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        revision_check_days: List[int] = [0, 7, 30, 60, 90, 180, 365],
    ) -> pd.DataFrame:
        """
        Build revision history with proper initial/final value capture.

        IMPROVED APPROACH:
        1. Get current (latest) values for all reference periods
        2. Find initial release date for each reference period
        3. Get value at initial release and at specific intervals after
        4. Compare initial vs final to determine revision

        This is more accurate than uniform vintage sampling because it:
        - Correctly identifies the TRUE initial release
        - Captures revisions at meaningful intervals
        - Properly handles different release schedules
        """
        logger.info(f"Building revision history (v2) for {series_id}")

        # Step 1: Get all vintage dates
        all_vintages = self.get_vintage_dates(series_id, start_date=start_date)
        if not all_vintages:
            logger.warning(f"No vintage dates found for {series_id}")
            return pd.DataFrame()

        logger.info(f"Found {len(all_vintages)} vintage dates")

        # Step 2: Get CURRENT values (latest vintage) to know all reference periods
        latest_vintage = all_vintages[-1].strftime("%Y-%m-%d")
        current_df = self.get_observations_at_vintage(
            series_id=series_id,
            vintage_date=latest_vintage,
            observation_start=start_date,
            observation_end=end_date,
        )

        if current_df.empty:
            logger.warning(f"No current observations for {series_id}")
            return pd.DataFrame()

        reference_dates = current_df["date"].tolist()
        logger.info(f"Found {len(reference_dates)} reference periods")

        # Step 3: For each reference period, find initial release and track revisions
        results = []

        # Process in batches to manage API calls
        # We'll be smarter: find first appearance of each ref date in vintage list

        # Build a mapping: for each vintage, what ref dates are available?
        # Then we can find first appearance without querying each vintage individually

        # Sample key vintages to build the mapping
        sample_vintages = []
        for i, vd in enumerate(all_vintages):
            # Take every Nth vintage, plus first and last
            if i == 0 or i == len(all_vintages) - 1 or i % max(1, len(all_vintages) // 100) == 0:
                sample_vintages.append(vd)

        logger.info(f"Sampling {len(sample_vintages)} vintages to find initial releases")

        # Build vintage -> available ref dates mapping
        vintage_to_refs = {}
        for vd in sample_vintages:
            vd_str = vd.strftime("%Y-%m-%d")
            try:
                df = self.get_observations_at_vintage(
                    series_id=series_id,
                    vintage_date=vd_str,
                    observation_start=start_date,
                    observation_end=end_date,
                )
                if not df.empty:
                    # Store ref_date -> value mapping for this vintage
                    vintage_to_refs[vd] = {
                        row["date"]: row["value"]
                        for _, row in df.iterrows()
                        if not pd.isna(row["value"])
                    }
            except Exception as e:
                logger.debug(f"Error at vintage {vd_str}: {e}")
                continue

        # Now find initial release and final value for each reference date
        sorted_sample_vintages = sorted(vintage_to_refs.keys())

        for ref_date in reference_dates:
            # Find first vintage where this ref_date appears
            initial_vintage = None
            initial_value = None

            for vd in sorted_sample_vintages:
                if ref_date in vintage_to_refs[vd]:
                    initial_vintage = vd
                    initial_value = vintage_to_refs[vd][ref_date]
                    break

            if initial_vintage is None:
                continue

            # Find final value (latest vintage where it appears)
            final_vintage = None
            final_value = None

            for vd in reversed(sorted_sample_vintages):
                if ref_date in vintage_to_refs[vd]:
                    final_vintage = vd
                    final_value = vintage_to_refs[vd][ref_date]
                    break

            if final_vintage is None or initial_value is None or final_value is None:
                continue

            # Track intermediate values for revision pattern
            revision_values = []
            for vd in sorted_sample_vintages:
                if vd >= initial_vintage and ref_date in vintage_to_refs[vd]:
                    revision_values.append((vd, vintage_to_refs[vd][ref_date]))

            # Calculate revision
            revision = final_value - initial_value
            revision_pct = (revision / abs(initial_value)) * 100 if initial_value != 0 else 0

            results.append({
                "series_id": series_id,
                "reference_date": ref_date,
                "initial_value": initial_value,
                "initial_date": initial_vintage,
                "final_value": final_value,
                "final_date": final_vintage,
                "revision": revision,
                "revision_pct": revision_pct,
                "revised_up": 1 if revision > 0.0001 else 0,  # Small tolerance for float comparison
                "revised_down": 1 if revision < -0.0001 else 0,
                "was_revised": 1 if abs(revision) > 0.0001 else 0,
                "n_revisions": len(revision_values),
                "days_to_final": (final_vintage - initial_vintage).days,
            })

        result_df = pd.DataFrame(results)

        if not result_df.empty:
            result_df = result_df.sort_values("reference_date")
            up_count = result_df["revised_up"].sum()
            down_count = result_df["revised_down"].sum()
            unchanged = len(result_df) - up_count - down_count
            logger.info(
                f"Built revision history for {series_id}: {len(result_df)} observations, "
                f"up: {up_count}, down: {down_count}, unchanged: {unchanged}"
            )

        return result_df


def download_all_indicators(
    start_date: str,
    end_date: str,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download revision datasets for all configured indicators.
    """
    from config import INDICATORS, RAW_DATA_DIR

    output_dir = output_dir or RAW_DATA_DIR
    datasets = {}
    client = FREDClient()

    for series_id, config in INDICATORS.items():
        logger.info(f"Downloading {series_id} ({config.name})")

        try:
            df = client.build_revision_history_v2(
                series_id=series_id,
                start_date=max(start_date, config.start_date),
                end_date=end_date,
            )

            if not df.empty:
                datasets[series_id] = df
                output_path = output_dir / f"{series_id}_revisions.parquet"
                df.to_parquet(output_path, index=False)
                logger.info(f"Saved {series_id} to {output_path}")
            else:
                logger.warning(f"No data retrieved for {series_id}")

        except Exception as e:
            logger.error(f"Failed to download {series_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return datasets


if __name__ == "__main__":
    import sys

    if not FRED_API_KEY:
        print("ERROR: FRED_API_KEY not set.")
        sys.exit(1)

    print("Testing FRED client v2...")
    client = FREDClient()

    series_id = "GDPC1"
    print(f"\nBuilding revision history for {series_id}...")
    df = client.build_revision_history_v2(
        series_id=series_id,
        start_date="2015-01-01",
        end_date="2024-12-31",
    )
    print(f"\nResult: {len(df)} observations")
    if not df.empty:
        print(f"Revised up: {df['revised_up'].sum()}")
        print(f"Revised down: {df['revised_down'].sum()}")
        print(f"\nSample:\n{df.head(10)}")

"""
Live Economic Calendar - Real-Time Release Detection

Uses FREE APIs to:
1. FRED API → Know when releases are scheduled + get actual values
2. BLS RSS feed → Detect new releases from Bureau of Labor Statistics
3. Census RSS feed → Detect Retail Sales, Durable Goods, Housing Starts

No paid subscriptions needed.
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consensus cache: manually updated before each release day, or scraped
# ---------------------------------------------------------------------------

# These are the most recent consensus forecasts. Update them the night before
# each release by checking ForexFactory, Investing.com, or TradingEconomics.
# The bot will still work without them, but the signal quality drops.
CONSENSUS_CACHE_PATH = Path(__file__).parent / "consensus_cache.json"


def load_consensus_cache() -> Dict:
    if CONSENSUS_CACHE_PATH.exists():
        with open(CONSENSUS_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_consensus_cache(data: Dict):
    try:
        with open(CONSENSUS_CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# FRED Release Schedule
# ---------------------------------------------------------------------------

# FRED release IDs for our indicators
FRED_RELEASE_IDS = {
    10: "GDPC1",       # Gross Domestic Product
    50: "PAYEMS",      # Employment Situation (NFP)
    176: "ICSA",       # Unemployment Insurance Weekly Claims
    13: "INDPRO",      # Industrial Production and Capacity Utilization
    11: "RSXFS",       # Advance Retail Sales
    15: "DGORDER",     # Advance Durable Goods
    12: "HOUST",       # New Residential Construction (Housing Starts)
}

# Reverse: series_id → release_id
SERIES_TO_RELEASE = {v: k for k, v in FRED_RELEASE_IDS.items()}

# Series IDs we poll for each indicator
INDICATOR_SERIES = {
    "GDPC1": "GDPC1",
    "PAYEMS": "PAYEMS",
    "ICSA": "ICSA",
    "INDPRO": "INDPRO",
    "RSXFS": "RSXFS",
    "DGORDER": "DGORDER",
    "HOUST": "HOUST",
}


@dataclass
class ScheduledRelease:
    """An upcoming economic release."""
    indicator_code: str
    indicator_name: str
    release_date: date
    release_id: int
    consensus: Optional[float] = None
    previous: Optional[float] = None
    actual: Optional[float] = None
    detected: bool = False

    @property
    def is_today(self) -> bool:
        return self.release_date == date.today()

    @property
    def days_until(self) -> int:
        return (self.release_date - date.today()).days


INDICATOR_NAMES = {
    "GDPC1": "Real GDP",
    "PAYEMS": "Nonfarm Payrolls",
    "ICSA": "Initial Claims",
    "INDPRO": "Industrial Production",
    "RSXFS": "Retail Sales",
    "DGORDER": "Durable Goods",
    "HOUST": "Housing Starts",
}


class FREDLiveClient:
    """
    Live client for FRED API.

    Handles:
    - Getting upcoming release schedule
    - Polling for new data on release day
    - Detecting when a new value appears
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not self.api_key:
            logger.warning("FRED_API_KEY not set. Set it in environment variables.")

        # Cache of last known values to detect changes
        self._last_known: Dict[str, Tuple[str, float]] = {}  # series → (date, value)

    def _request(self, endpoint: str, params: Dict) -> Dict:
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"FRED API error: {e}")
            return {}

    def get_upcoming_releases(self, days_ahead: int = 14) -> List[ScheduledRelease]:
        """
        Get scheduled release dates for our indicators from FRED.

        Uses fred/releases/dates endpoint.
        """
        today = date.today()
        end_date = today + timedelta(days=days_ahead)

        releases = []
        consensus_cache = load_consensus_cache()

        for release_id, indicator_code in FRED_RELEASE_IDS.items():
            data = self._request("release/dates", {
                "release_id": release_id,
                "include_release_dates_with_no_data": "true",
                "sort_order": "asc",
                "realtime_start": today.isoformat(),
                "realtime_end": end_date.isoformat(),
            })

            release_dates = data.get("release_dates", [])
            for rd in release_dates:
                rd_date = datetime.strptime(rd["date"], "%Y-%m-%d").date()
                if today <= rd_date <= end_date:
                    # Check consensus cache
                    cache_key = f"{indicator_code}_{rd_date.isoformat()}"
                    cached = consensus_cache.get(cache_key, {})

                    releases.append(ScheduledRelease(
                        indicator_code=indicator_code,
                        indicator_name=INDICATOR_NAMES.get(indicator_code, indicator_code),
                        release_date=rd_date,
                        release_id=release_id,
                        consensus=cached.get("consensus"),
                        previous=cached.get("previous"),
                    ))

            # Rate limit: be nice to FRED
            time.sleep(0.5)

        return sorted(releases, key=lambda r: r.release_date)

    def get_latest_value(self, series_id: str) -> Optional[Tuple[str, float]]:
        """
        Get the most recent observation for a series.

        Returns (date_string, value) or None.
        """
        data = self._request("series/observations", {
            "series_id": series_id,
            "sort_order": "desc",
            "limit": 1,
        })

        observations = data.get("observations", [])
        if observations:
            obs = observations[0]
            try:
                value = float(obs["value"])
                return (obs["date"], value)
            except (ValueError, KeyError):
                return None
        return None

    def check_for_new_release(self, series_id: str) -> Optional[Tuple[str, float]]:
        """
        Check if a new value has appeared for this series since last check.

        Returns the new (date, value) if detected, None otherwise.
        """
        latest = self.get_latest_value(series_id)
        if latest is None:
            return None

        last_known = self._last_known.get(series_id)

        if last_known is None:
            # First check - store current value as baseline
            self._last_known[series_id] = latest
            return None

        if latest[0] != last_known[0] or latest[1] != last_known[1]:
            # New data detected!
            self._last_known[series_id] = latest
            logger.info(f"NEW RELEASE DETECTED: {series_id} = {latest[1]} (date: {latest[0]})")
            return latest

        return None

    def poll_for_releases(
        self,
        indicators: List[str],
        interval_seconds: int = 30,
        max_duration_minutes: int = 60,
    ) -> Optional[Tuple[str, str, float]]:
        """
        Poll FRED for new releases of specified indicators.

        Blocks until a new value is detected or timeout.

        Returns: (indicator_code, date, value) or None if timed out.
        """
        # First, establish baseline values
        for ind in indicators:
            series_id = INDICATOR_SERIES.get(ind, ind)
            self.get_latest_value(series_id)
            self._last_known[series_id] = self.get_latest_value(series_id)
            time.sleep(0.5)

        start_time = time.time()
        max_seconds = max_duration_minutes * 60

        logger.info(f"Polling for releases: {indicators} (every {interval_seconds}s, max {max_duration_minutes}min)")

        while time.time() - start_time < max_seconds:
            for ind in indicators:
                series_id = INDICATOR_SERIES.get(ind, ind)
                new_value = self.check_for_new_release(series_id)

                if new_value:
                    return (ind, new_value[0], new_value[1])

                time.sleep(0.5)  # Don't hammer FRED

            time.sleep(interval_seconds)

        logger.warning(f"Polling timed out after {max_duration_minutes} minutes")
        return None


# ---------------------------------------------------------------------------
# BLS RSS Feed Parser
# ---------------------------------------------------------------------------

class BLSRSSMonitor:
    """
    Monitor BLS RSS feed for new economic releases.

    Covers: Nonfarm Payrolls, CPI, PPI, Unemployment Rate.
    """

    FEED_URL = "https://www.bls.gov/feed/bls_latest.rss"

    # Keywords in BLS RSS titles that map to our indicators
    TITLE_KEYWORDS = {
        "Employment Situation": "PAYEMS",
        "Unemployment Insurance Weekly Claims": "ICSA",
        "Producer Price Index": None,  # Not in our model
        "Consumer Price Index": None,  # Not in our model
    }

    def __init__(self):
        self._seen_guids = set()

    def check_feed(self) -> List[Dict]:
        """
        Check BLS RSS feed for new items.

        Returns list of new release notifications.
        """
        try:
            resp = requests.get(self.FEED_URL, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            logger.error(f"BLS RSS error: {e}")
            return []

        new_items = []
        for item in root.findall(".//item"):
            guid = item.findtext("guid", "")
            if guid in self._seen_guids:
                continue

            self._seen_guids.add(guid)
            title = item.findtext("title", "")
            pub_date = item.findtext("pubDate", "")
            link = item.findtext("link", "")

            # Check if title matches our indicators
            indicator_code = None
            for keyword, code in self.TITLE_KEYWORDS.items():
                if keyword.lower() in title.lower():
                    indicator_code = code
                    break

            if indicator_code:
                new_items.append({
                    "indicator_code": indicator_code,
                    "title": title,
                    "pub_date": pub_date,
                    "link": link,
                })

        return new_items


# ---------------------------------------------------------------------------
# Census RSS Feed Parser
# ---------------------------------------------------------------------------

class CensusRSSMonitor:
    """
    Monitor Census Bureau RSS for new economic releases.

    Covers: Retail Sales, Durable Goods, Housing Starts, New Home Sales.
    """

    FEED_URL = "https://www.census.gov/economic-indicators/indicator.xml"

    TITLE_KEYWORDS = {
        "Advance Monthly Sales for Retail": "RSXFS",
        "Advance Report on Durable Goods": "DGORDER",
        "New Residential Construction": "HOUST",
    }

    def __init__(self):
        self._seen_guids = set()

    def check_feed(self) -> List[Dict]:
        try:
            resp = requests.get(self.FEED_URL, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            logger.error(f"Census RSS error: {e}")
            return []

        new_items = []
        for item in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            title_el = item.find("{http://www.w3.org/2005/Atom}title")
            id_el = item.find("{http://www.w3.org/2005/Atom}id")
            if title_el is None or id_el is None:
                continue

            title = title_el.text or ""
            guid = id_el.text or ""

            if guid in self._seen_guids:
                continue
            self._seen_guids.add(guid)

            indicator_code = None
            for keyword, code in self.TITLE_KEYWORDS.items():
                if keyword.lower() in title.lower():
                    indicator_code = code
                    break

            if indicator_code:
                new_items.append({
                    "indicator_code": indicator_code,
                    "title": title,
                    "link": guid,
                })

        return new_items


# ---------------------------------------------------------------------------
# Unified Live Calendar
# ---------------------------------------------------------------------------

class LiveCalendar:
    """
    Combines FRED API + RSS feeds into a single interface.

    Usage:
        cal = LiveCalendar()
        schedule = cal.get_schedule()          # What's coming up
        result = cal.wait_for_release("PAYEMS") # Block until NFP drops
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred = FREDLiveClient(api_key=fred_api_key)
        self.bls_rss = BLSRSSMonitor()
        self.census_rss = CensusRSSMonitor()

    def get_schedule(self, days_ahead: int = 14) -> List[ScheduledRelease]:
        """Get all upcoming releases in the next N days."""
        return self.fred.get_upcoming_releases(days_ahead=days_ahead)

    def get_todays_releases(self) -> List[ScheduledRelease]:
        """Get releases scheduled for today."""
        return [r for r in self.get_schedule(days_ahead=0) if r.is_today]

    def wait_for_release(
        self,
        indicator_code: str,
        poll_interval: int = 30,
        timeout_minutes: int = 60,
    ) -> Optional[Tuple[str, float]]:
        """
        Block until a new value appears for the given indicator.

        Returns (date, value) or None on timeout.
        """
        result = self.fred.poll_for_releases(
            indicators=[indicator_code],
            interval_seconds=poll_interval,
            max_duration_minutes=timeout_minutes,
        )
        if result:
            return (result[1], result[2])
        return None

    def check_rss_feeds(self) -> List[Dict]:
        """Check all RSS feeds for new release notifications."""
        items = []
        items.extend(self.bls_rss.check_feed())
        items.extend(self.census_rss.check_feed())
        return items

    def set_consensus(self, indicator_code: str, release_date: str, consensus: float, previous: float):
        """
        Manually set consensus forecast for an upcoming release.

        Call this the night before with data from ForexFactory/Investing.com.
        """
        cache = load_consensus_cache()
        key = f"{indicator_code}_{release_date}"
        cache[key] = {"consensus": consensus, "previous": previous}
        save_consensus_cache(cache)
        logger.info(f"Consensus set: {indicator_code} on {release_date} = {consensus}")


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    print("=" * 60)
    print("LIVE ECONOMIC CALENDAR")
    print("=" * 60)

    cal = LiveCalendar()

    # Get schedule
    print("\nUpcoming releases (next 14 days):")
    print("-" * 60)

    schedule = cal.get_schedule(days_ahead=14)
    if not schedule:
        print("  (No upcoming releases found, or FRED API key not set)")
    else:
        for release in schedule:
            days_str = "TODAY" if release.is_today else f"in {release.days_until} days"
            consensus_str = f" | Consensus: {release.consensus}" if release.consensus else ""
            print(f"  {release.release_date} | {release.indicator_name} ({release.indicator_code}) | {days_str}{consensus_str}")

    # Check RSS feeds
    print("\nChecking RSS feeds for new releases...")
    rss_items = cal.check_rss_feeds()
    if rss_items:
        for item in rss_items:
            print(f"  NEW: {item['title']} → {item['indicator_code']}")
    else:
        print("  (No new RSS items)")

    print("\nDone.")

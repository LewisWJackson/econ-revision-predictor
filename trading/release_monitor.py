"""
Economic Release Monitor

Real-time monitoring of economic releases with automated alerts.
Watches for releases that match our revision-bias trading criteria.

FEATURES:
- Monitors upcoming economic releases from multiple sources
- Sends alerts when release is imminent (1 hour, 15 min, 5 min)
- Captures actual vs consensus values immediately on release
- Triggers trading signals based on revision model
- Supports Telegram, Discord, Email notifications

DATA SOURCES:
- Investing.com economic calendar (web scraping)
- Trading Economics API (if available)
- FRED release schedule
"""

import os
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import logging
import threading
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.alerts import AlertManager
from trading.economic_calendar import EconomicCalendar, EconomicEvent, REVISION_BIASES

logger = logging.getLogger(__name__)


@dataclass
class ReleaseAlert:
    """Configuration for an alert on a specific release."""
    indicator_code: str
    indicator_name: str
    release_time: datetime
    threshold: Optional[float] = None  # Alert if actual crosses this
    consensus: Optional[float] = None
    previous: Optional[float] = None
    importance: str = "high"  # high, medium, low
    alerted_1h: bool = False
    alerted_15m: bool = False
    alerted_5m: bool = False
    alerted_release: bool = False


class ReleaseMonitor:
    """
    Monitors economic releases and sends alerts based on criteria.

    Usage:
        monitor = ReleaseMonitor()
        monitor.add_telegram_alerts(bot_token, chat_id)
        monitor.start()  # Runs in background
    """

    def __init__(
        self,
        check_interval: int = 60,  # seconds
        alert_minutes: List[int] = [60, 15, 5],  # Alert at these minutes before
    ):
        self.check_interval = check_interval
        self.alert_minutes = sorted(alert_minutes, reverse=True)

        self.calendar = EconomicCalendar()
        self.alerts = AlertManager()

        # Tracked releases
        self.tracked_releases: Dict[str, ReleaseAlert] = {}

        # Callbacks for release events
        self.on_release_callbacks: List[Callable] = []

        # Background thread
        self._running = False
        self._thread = None

    def configure_alerts(self, channels: List[str]):
        """Configure which alert channels to use."""
        self.alerts.configure(channels)

    def add_telegram_alerts(self, bot_token: str, chat_id: str):
        """Add Telegram as an alert channel."""
        os.environ["TELEGRAM_BOT_TOKEN"] = bot_token
        os.environ["TELEGRAM_CHAT_ID"] = chat_id
        self.alerts.configure(["telegram", "console"])

    def add_discord_alerts(self, webhook_url: str):
        """Add Discord as an alert channel."""
        os.environ["DISCORD_WEBHOOK_URL"] = webhook_url
        self.alerts.configure(["discord", "console"])

    def on_release(self, callback: Callable):
        """
        Register a callback to be called when a release happens.

        Callback signature:
            def my_callback(indicator_code: str, actual: float, consensus: float, previous: float):
                ...
        """
        self.on_release_callbacks.append(callback)

    def start(self):
        """Start monitoring in background thread."""
        if self._running:
            logger.warning("Monitor already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Release monitor started")

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Release monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_upcoming_releases()
                self._check_recent_releases()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(10)

    def _check_upcoming_releases(self):
        """Check for upcoming releases and send alerts."""
        events = self.calendar.get_upcoming_events(days_ahead=1)
        now = datetime.utcnow()

        for event in events:
            # Only track indicators we have revision data for
            if event.indicator_code not in REVISION_BIASES:
                continue

            # Get or create alert tracking
            key = f"{event.indicator_code}_{event.datetime_utc.date()}"
            if key not in self.tracked_releases:
                self.tracked_releases[key] = ReleaseAlert(
                    indicator_code=event.indicator_code,
                    indicator_name=event.name,
                    release_time=event.datetime_utc,
                    consensus=event.consensus,
                    previous=event.previous,
                    importance=event.importance,
                )

            alert = self.tracked_releases[key]
            minutes_until = (alert.release_time - now).total_seconds() / 60

            # Send alerts at configured intervals
            if minutes_until <= 60 and minutes_until > 55 and not alert.alerted_1h:
                self._send_imminent_alert(alert, 60)
                alert.alerted_1h = True

            elif minutes_until <= 15 and minutes_until > 10 and not alert.alerted_15m:
                self._send_imminent_alert(alert, 15)
                alert.alerted_15m = True

            elif minutes_until <= 5 and minutes_until > 2 and not alert.alerted_5m:
                self._send_imminent_alert(alert, 5)
                alert.alerted_5m = True

    def _send_imminent_alert(self, alert: ReleaseAlert, minutes: int):
        """Send an alert for an upcoming release."""
        bias = REVISION_BIASES.get(alert.indicator_code, {})
        bias_str = f"{bias.get('direction', 'unknown').upper()} ({bias.get('probability', 0):.0%})"

        self.alerts.release_imminent(
            event_name=alert.indicator_name,
            minutes_until=minutes,
            revision_bias=bias_str,
        )

        logger.info(f"Alert sent: {alert.indicator_name} in {minutes} minutes")

    def _check_recent_releases(self):
        """
        Check if any tracked releases have happened.

        In production, this would poll a real-time data source.
        For now, we simulate by checking if release time has passed.
        """
        now = datetime.utcnow()

        for key, alert in list(self.tracked_releases.items()):
            # Check if release time has passed
            if alert.release_time < now and not alert.alerted_release:
                # In production: fetch actual value from data feed
                # For now, we mark it as released and wait for manual input
                # or integration with a real-time data source

                minutes_since = (now - alert.release_time).total_seconds() / 60
                if minutes_since > 5:  # Give 5 minutes for data to appear
                    alert.alerted_release = True
                    logger.info(f"Release detected: {alert.indicator_name}")
                    # Would trigger on_release_callbacks here with actual data

            # Clean up old alerts (more than 1 day old)
            if (now - alert.release_time).days > 1:
                del self.tracked_releases[key]

    def process_release(
        self,
        indicator_code: str,
        actual: float,
        consensus: Optional[float] = None,
        previous: Optional[float] = None,
    ):
        """
        Process a release and trigger callbacks.

        Call this manually or from a real-time data feed integration.
        """
        logger.info(f"Processing release: {indicator_code} = {actual}")

        # Use stored consensus/previous if not provided
        key = None
        for k, alert in self.tracked_releases.items():
            if alert.indicator_code == indicator_code and not alert.alerted_release:
                key = k
                if consensus is None:
                    consensus = alert.consensus
                if previous is None:
                    previous = alert.previous
                alert.alerted_release = True
                break

        # Trigger callbacks
        for callback in self.on_release_callbacks:
            try:
                callback(indicator_code, actual, consensus, previous)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class InvestingComScraper:
    """
    Scraper for Investing.com economic calendar.

    Note: Web scraping may violate ToS. Consider using official APIs
    or paid data services for production use.
    """

    BASE_URL = "https://www.investing.com/economic-calendar/"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def get_calendar(self, date: datetime) -> List[Dict]:
        """
        Get economic calendar for a specific date.

        Returns list of events with:
        - time: Release time (UTC)
        - country: Country code
        - indicator: Indicator name
        - importance: 1-3 stars
        - actual: Actual value (if released)
        - forecast: Consensus forecast
        - previous: Previous value
        """
        # Implementation would parse the Investing.com calendar page
        # This is a placeholder - actual implementation requires HTML parsing
        logger.warning("InvestingComScraper not implemented - use official APIs")
        return []


class TradingEconomicsClient:
    """
    Client for Trading Economics API (requires subscription).

    https://tradingeconomics.com/api/
    """

    BASE_URL = "https://api.tradingeconomics.com"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TRADING_ECONOMICS_API_KEY")

    def get_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        country: str = "united states",
    ) -> List[Dict]:
        """Get economic calendar from Trading Economics."""
        if not self.api_key:
            logger.warning("Trading Economics API key not set")
            return []

        try:
            url = f"{self.BASE_URL}/calendar/country/{country}"
            params = {
                "c": self.api_key,
                "f": "json",
                "d1": start_date.strftime("%Y-%m-%d"),
                "d2": end_date.strftime("%Y-%m-%d"),
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Trading Economics API error: {e}")
            return []


# Quick setup function
def create_monitor_with_telegram(bot_token: str, chat_id: str) -> ReleaseMonitor:
    """Create a monitor with Telegram alerts configured."""
    monitor = ReleaseMonitor()
    monitor.add_telegram_alerts(bot_token, chat_id)
    return monitor


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Economic Release Monitor")
    print("=" * 60)

    # Create monitor
    monitor = ReleaseMonitor()
    monitor.configure_alerts(["console"])

    # Add callback for when releases happen
    def on_release(indicator, actual, consensus, previous):
        print(f"\n{'='*60}")
        print(f"RELEASE: {indicator}")
        print(f"Actual: {actual} | Consensus: {consensus} | Previous: {previous}")

        bias = REVISION_BIASES.get(indicator, {})
        if bias:
            direction = bias["direction"]
            prob = bias["probability"]
            print(f"Revision Bias: {direction.upper()} ({prob:.0%})")

            # Determine if there's edge
            if consensus and actual:
                missed = actual < consensus
                beat = actual > consensus

                if missed and direction == "up":
                    print("SIGNAL: BUY - Miss will likely revise up!")
                elif beat and direction == "down":
                    print("SIGNAL: SELL - Beat will likely revise down!")
        print("=" * 60)

    monitor.on_release(on_release)

    print("\nTracked Indicators (with revision bias):")
    for code, bias in REVISION_BIASES.items():
        print(f"  {code}: {bias['direction'].upper()} ({bias['probability']:.0%})")

    print("\nTo start monitoring:")
    print("  monitor.start()")
    print("\nTo simulate a release:")
    print("  monitor.process_release('PAYEMS', 150000, 200000, 180000)")

    # Simulate a release for demo
    print("\n" + "=" * 60)
    print("SIMULATING NFP RELEASE (miss)")
    print("=" * 60)
    monitor.process_release("PAYEMS", 150000, 200000, 180000)

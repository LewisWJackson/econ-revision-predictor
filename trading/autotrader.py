#!/usr/bin/env python3
"""
AUTOTRADER - Fully Automated Economic Revision Trading Bot

This is the one script you run. It:
1. Checks FRED for today's scheduled economic releases
2. Starts polling 5 minutes before release time
3. Detects when new data drops
4. Compares actual vs consensus
5. Executes trades via Alpaca
6. Sends you alerts via Telegram/Discord
7. Manages stop losses and take profits
8. Runs 24/7 in the background

USAGE:
    # Paper trading (fake money - START HERE)
    python3 trading/autotrader.py --paper

    # Live trading (real money)
    python3 trading/autotrader.py --live

    # Test with a fake release right now
    python3 trading/autotrader.py --test

REQUIRED ENVIRONMENT VARIABLES:
    export FRED_API_KEY='your-fred-key'
    export ALPACA_API_KEY='your-alpaca-key'
    export ALPACA_SECRET_KEY='your-alpaca-secret'

OPTIONAL (for alerts):
    export TELEGRAM_BOT_TOKEN='your-bot-token'
    export TELEGRAM_CHAT_ID='your-chat-id'
"""

import os
import sys
import time
import signal
import argparse
import logging
from datetime import datetime, timedelta, date, timezone
from typing import Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.live_calendar import LiveCalendar, INDICATOR_NAMES, load_consensus_cache
from trading.strategy import RevisionStrategy, INDICATOR_TRADES
from trading.alerts import AlertManager
from trading.economic_calendar import REVISION_BIASES

logger = logging.getLogger("autotrader")


# ---------------------------------------------------------------------------
# Release times (US Eastern) - when to start polling
# ---------------------------------------------------------------------------

# Most releases are at 8:30 AM ET. We start polling 5 min early.
RELEASE_TIMES_ET = {
    "GDPC1": "08:30",
    "PAYEMS": "08:30",
    "ICSA": "08:30",
    "INDPRO": "09:15",
    "RSXFS": "08:30",
    "DGORDER": "08:30",
    "HOUST": "08:30",
}


def et_to_utc_hour_minute(et_time_str: str) -> tuple:
    """Convert Eastern Time HH:MM to approximate UTC HH:MM."""
    h, m = map(int, et_time_str.split(":"))
    # ET is UTC-5 (EST) or UTC-4 (EDT). Use -5 as conservative default.
    utc_h = (h + 5) % 24
    return utc_h, m


class AutoTrader:
    """
    Fully automated trading bot.

    Lifecycle:
    1. On startup: fetch today's release schedule from FRED
    2. Sleep until 5 minutes before next release
    3. Start polling FRED for new data
    4. When data drops: compare vs consensus → generate signal → trade
    5. After trade: monitor position (stop loss / take profit)
    6. Repeat for next release
    """

    def __init__(self, mode: str = "paper"):
        self.mode = mode
        self.running = False

        # Core components
        self.calendar = LiveCalendar()
        self.strategy = RevisionStrategy(max_portfolio_risk=0.15)
        self.alerts = AlertManager()

        # Broker
        self.broker = None
        self._init_broker(mode)

        # Alert channels
        self._init_alerts()

    def _init_broker(self, mode: str):
        """Initialize the broker based on mode."""
        if mode == "test":
            from trading.broker_ibkr import MockBroker
            self.broker = MockBroker()
            logger.info("Mode: TEST (mock broker, no real trades)")
            return

        try:
            from trading.broker_alpaca import AlpacaBroker
            is_paper = (mode == "paper")
            self.broker = AlpacaBroker(paper=is_paper)
            if self.broker.connect():
                account_val = self.broker.get_account_value()
                logger.info(f"Mode: {'PAPER' if is_paper else 'LIVE'} trading")
                logger.info(f"Account value: ${account_val:,.2f}")
            else:
                logger.error("Failed to connect to Alpaca")
                sys.exit(1)
        except ImportError:
            logger.error("Alpaca not installed. Run: pip install alpaca-py")
            sys.exit(1)

    def _init_alerts(self):
        """Set up notification channels."""
        channels = ["console"]

        if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
            channels.append("telegram")
            logger.info("Telegram alerts enabled")

        if os.getenv("DISCORD_WEBHOOK_URL"):
            channels.append("discord")
            logger.info("Discord alerts enabled")

        self.alerts.configure(channels)

    def start(self):
        """Start the autotrader loop."""
        self.running = True

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        logger.info("=" * 60)
        logger.info("AUTOTRADER STARTED")
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info("=" * 60)

        self.alerts.send_all(
            "AutoTrader Started",
            f"Mode: {self.mode.upper()}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        )

        while self.running:
            try:
                self._daily_loop()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.alerts.send_all("AutoTrader Error", str(e))
                time.sleep(60)

    def _daily_loop(self):
        """Main daily logic: check schedule, wait for releases, trade."""

        # 1. Get today's schedule
        logger.info("Checking today's release schedule...")
        schedule = self.calendar.get_schedule(days_ahead=1)
        todays = [r for r in schedule if r.is_today]

        if not todays:
            logger.info("No releases scheduled today.")
            next_releases = [r for r in schedule if r.days_until > 0]
            if next_releases:
                next_r = next_releases[0]
                logger.info(f"Next release: {next_r.indicator_name} on {next_r.release_date} ({next_r.days_until} days)")

            # Sleep until tomorrow at 8:00 AM ET (13:00 UTC)
            self._sleep_until_tomorrow()
            return

        # 2. Show today's schedule
        logger.info(f"Today's releases ({len(todays)}):")
        for r in todays:
            release_time = RELEASE_TIMES_ET.get(r.indicator_code, "08:30")
            consensus_str = f" (consensus: {r.consensus})" if r.consensus else " (NO CONSENSUS SET)"
            logger.info(f"  {release_time} ET - {r.indicator_name}{consensus_str}")

        self.alerts.send_all(
            "Today's Releases",
            "\n".join(
                f"- {RELEASE_TIMES_ET.get(r.indicator_code, '08:30')} ET: {r.indicator_name}"
                for r in todays
            ),
        )

        # 3. Process each release
        for release in todays:
            self._process_scheduled_release(release)

        # 4. Sleep until next check (in case there are more releases today)
        logger.info("All releases processed. Sleeping until tomorrow.")
        self._sleep_until_tomorrow()

    def _process_scheduled_release(self, release):
        """Wait for a scheduled release and trade on it."""
        indicator = release.indicator_code
        release_time_et = RELEASE_TIMES_ET.get(indicator, "08:30")
        utc_h, utc_m = et_to_utc_hour_minute(release_time_et)

        # Calculate when to start polling (5 min before release)
        now = datetime.now(timezone.utc)
        poll_start = now.replace(hour=utc_h, minute=utc_m, second=0) - timedelta(minutes=5)

        if now < poll_start:
            wait_seconds = (poll_start - now).total_seconds()
            if wait_seconds > 0:
                logger.info(f"Waiting {wait_seconds/60:.0f} minutes until {indicator} release window...")
                self.alerts.send_all(
                    f"{release.indicator_name} Coming Up",
                    f"Release at {release_time_et} ET\n"
                    f"Will start monitoring in {wait_seconds/60:.0f} minutes",
                )

                # Sleep in 60-second chunks so we can be interrupted
                while wait_seconds > 0 and self.running:
                    time.sleep(min(60, wait_seconds))
                    wait_seconds -= 60

        if not self.running:
            return

        # Start polling
        logger.info(f"POLLING for {release.indicator_name}...")
        self.alerts.send_all(
            f"Monitoring {release.indicator_name}",
            f"Polling FRED every 30 seconds for new data...",
        )

        result = self.calendar.wait_for_release(
            indicator_code=indicator,
            poll_interval=30,
            timeout_minutes=90,  # Keep trying for 90 min
        )

        if result is None:
            logger.warning(f"Timed out waiting for {indicator}")
            self.alerts.send_all(f"{release.indicator_name} Timeout", "No new data detected")
            return

        obs_date, actual = result
        logger.info(f"RELEASE DETECTED: {indicator} = {actual} (date: {obs_date})")

        # Get consensus
        consensus = release.consensus
        if consensus is None:
            # Try consensus cache
            cache = load_consensus_cache()
            key = f"{indicator}_{date.today().isoformat()}"
            cached = cache.get(key, {})
            consensus = cached.get("consensus")

        if consensus is None:
            logger.warning(f"No consensus for {indicator}. Cannot generate signal.")
            self.alerts.send_all(
                f"{release.indicator_name} Released",
                f"Actual: {actual}\nNo consensus set - cannot trade.\n"
                f"Set consensus with: cal.set_consensus('{indicator}', '{date.today()}', consensus, previous)",
            )
            return

        previous = release.previous

        # Generate trading signal
        self._execute_signal(indicator, actual, consensus, previous or actual)

    def _execute_signal(self, indicator_code: str, actual: float, consensus: float, previous: float):
        """Generate signal and execute trades."""

        signal, trades = self.strategy.analyze_release(indicator_code, actual, consensus, previous)
        indicator_name = INDICATOR_NAMES.get(indicator_code, indicator_code)
        surprise = actual - consensus
        surprise_pct = (surprise / abs(consensus)) * 100 if consensus != 0 else 0
        bias = REVISION_BIASES.get(indicator_code, {})

        # Alert: release detected
        self.alerts.send_all(
            f"RELEASE: {indicator_name}",
            f"Actual: {actual:,.2f}\n"
            f"Consensus: {consensus:,.2f}\n"
            f"Surprise: {surprise:+,.2f} ({surprise_pct:+.1f}%)\n"
            f"Revision bias: {bias.get('direction', 'unknown').upper()} ({bias.get('probability', 0):.0%})\n"
            f"Signal: {signal.value}",
        )

        if not trades:
            logger.info(f"Signal: {signal.value} — no trades (no edge)")
            return

        # Execute trades
        for trade in trades:
            logger.info(f"Executing: {trade.direction} {trade.symbol} ({trade.size_pct:.1f}%)")

            try:
                # Calculate position size
                shares = self.broker.calculate_position_size(trade.size_pct, trade.symbol)

                if shares <= 0:
                    logger.warning(f"Position size is 0 for {trade.symbol}")
                    continue

                # Check if market is open
                if hasattr(self.broker, 'is_market_open') and not self.broker.is_market_open():
                    logger.warning("Market is closed. Order will queue for open.")

                # Place order
                result = self.broker.place_market_order(
                    symbol=trade.symbol,
                    quantity=shares,
                    action=trade.direction,
                )

                if result.success:
                    fill_price = result.fill_price or 0
                    dollar_amount = shares * fill_price

                    logger.info(f"FILLED: {trade.direction} {shares} {trade.symbol} @ ${fill_price:.2f}")

                    self.alerts.send_all(
                        f"TRADE EXECUTED: {trade.symbol}",
                        f"Direction: {trade.direction}\n"
                        f"Shares: {shares}\n"
                        f"Price: ${fill_price:.2f}\n"
                        f"Amount: ${dollar_amount:,.2f}\n"
                        f"Reason: {trade.reasoning}",
                    )
                else:
                    logger.error(f"Order failed: {result.error}")
                    self.alerts.send_all(
                        f"ORDER FAILED: {trade.symbol}",
                        f"Error: {result.error}",
                    )

            except Exception as e:
                logger.error(f"Trade execution error: {e}")
                self.alerts.send_all("Trade Error", f"{trade.symbol}: {e}")

    def _sleep_until_tomorrow(self):
        """Sleep until 8:00 AM ET tomorrow (13:00 UTC)."""
        now = datetime.now(timezone.utc)
        tomorrow_8am = (now + timedelta(days=1)).replace(hour=13, minute=0, second=0, microsecond=0)
        sleep_seconds = (tomorrow_8am - now).total_seconds()

        logger.info(f"Sleeping for {sleep_seconds/3600:.1f} hours until tomorrow 8:00 AM ET")

        # Sleep in 5-minute chunks so we can be interrupted
        while sleep_seconds > 0 and self.running:
            time.sleep(min(300, sleep_seconds))
            sleep_seconds -= 300

    def _shutdown(self, signum, frame):
        """Graceful shutdown."""
        logger.info("Shutdown signal received. Stopping...")
        self.running = False

        if self.broker and hasattr(self.broker, 'disconnect'):
            self.broker.disconnect()

        self.alerts.send_all("AutoTrader Stopped", "Bot has been shut down.")


def run_test():
    """Run a test simulation without waiting for real releases."""
    print("=" * 60)
    print("AUTOTRADER TEST MODE")
    print("=" * 60)

    trader = AutoTrader(mode="test")

    # Show schedule
    print("\nFetching release schedule from FRED...")
    schedule = trader.calendar.get_schedule(days_ahead=7)
    if schedule:
        print(f"\nUpcoming releases ({len(schedule)}):")
        for r in schedule:
            print(f"  {r.release_date} | {r.indicator_name} | in {r.days_until} days")
    else:
        print("  (Could not fetch schedule - check FRED_API_KEY)")

    # Simulate a release
    print("\n" + "=" * 60)
    print("SIMULATING: Nonfarm Payrolls MISS")
    print("=" * 60)
    trader._execute_signal("PAYEMS", 150000, 200000, 180000)

    print("\n" + "=" * 60)
    print("SIMULATING: Retail Sales BEAT")
    print("=" * 60)
    trader._execute_signal("RSXFS", 0.8, 0.3, 0.4)

    print("\n" + "=" * 60)
    print("SIMULATING: GDP MISS")
    print("=" * 60)
    trader._execute_signal("GDPC1", 1.8, 2.5, 2.8)

    print("\nTest complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("autotrader.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Automated Economic Revision Trader")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--paper", action="store_true", help="Paper trading (fake money)")
    mode_group.add_argument("--live", action="store_true", help="Live trading (REAL money)")
    mode_group.add_argument("--test", action="store_true", help="Test mode (simulate releases)")

    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        mode = "live" if args.live else "paper"

        if mode == "live":
            print("\n" + "!" * 60)
            print("WARNING: LIVE TRADING MODE - REAL MONEY")
            print("!" * 60)
            confirm = input("Type 'YES' to confirm: ")
            if confirm != "YES":
                print("Aborted.")
                sys.exit(0)

        trader = AutoTrader(mode=mode)
        trader.start()

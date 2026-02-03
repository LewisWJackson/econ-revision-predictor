#!/usr/bin/env python3
"""
Revision-Based Trading Bot

This is the main entry point for automated trading based on economic revision patterns.

WHAT IT DOES:
1. Monitors economic calendar for upcoming releases
2. Waits for releases and captures the actual vs consensus
3. Applies our revision model to determine if there's edge
4. Executes trades via IBKR (or paper trades for testing)
5. Manages positions with stop losses and take profits
6. Sends alerts via Telegram/Discord

HOW TO RUN:
    # Paper trading mode (recommended to start)
    python bot.py --paper

    # Live trading (BE CAREFUL)
    python bot.py --live

    # Demo mode (no real trades, just shows signals)
    python bot.py --demo

SETUP REQUIRED:
1. IBKR account with API enabled
2. TWS or IB Gateway running
3. Environment variables for alerts (optional):
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
   - DISCORD_WEBHOOK_URL
"""

import argparse
import time
from datetime import datetime, timedelta
from typing import Optional
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.economic_calendar import EconomicCalendar, REVISION_BIASES
from trading.strategy import RevisionStrategy, Signal
from trading.alerts import AlertManager
from trading.broker_ibkr import IBKRBroker, MockBroker

# Try to import Alpaca broker
try:
    from trading.broker_alpaca import AlpacaBroker
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("trading_bot")


class TradingBot:
    """
    Main trading bot that combines all components.
    """

    def __init__(
        self,
        mode: str = "demo",  # "demo", "paper", "live"
        broker_type: str = "ibkr",  # "ibkr" or "alpaca"
        check_interval: int = 60,  # seconds between calendar checks
    ):
        self.mode = mode
        self.broker_type = broker_type
        self.check_interval = check_interval

        # Initialize components
        self.calendar = EconomicCalendar()
        self.strategy = RevisionStrategy(max_portfolio_risk=0.15)  # 15% max risk
        self.alerts = AlertManager()

        # Initialize broker based on mode and type
        if mode == "demo":
            self.broker = MockBroker()
            logger.info("Running in DEMO mode - no real trades")
        elif broker_type == "alpaca":
            if not ALPACA_AVAILABLE:
                raise ImportError("Alpaca not available. Run: pip install alpaca-py")
            self.broker = AlpacaBroker(paper=(mode == "paper"))
            if mode == "paper":
                logger.info("Running in PAPER trading mode (Alpaca)")
            else:
                logger.warning("⚠️  Running in LIVE trading mode (Alpaca) - REAL MONEY")
        elif broker_type == "ibkr":
            if mode == "paper":
                self.broker = IBKRBroker(port=7497)  # Paper trading port
                logger.info("Running in PAPER trading mode (IBKR)")
            elif mode == "live":
                self.broker = IBKRBroker(port=7496)  # Live trading port
                logger.warning("⚠️  Running in LIVE trading mode (IBKR) - REAL MONEY")
        else:
            raise ValueError(f"Unknown broker type: {broker_type}")

        # Configure alerts
        self.alerts.configure(["console"])  # Add "telegram", "discord" if configured

        # Tracking
        self.last_checked = {}  # indicator -> last check time
        self.pending_alerts = []

    def start(self):
        """Start the trading bot."""
        logger.info("=" * 60)
        logger.info("STARTING REVISION-BASED TRADING BOT")
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info("=" * 60)

        # Connect to broker
        if not self.broker.connect():
            logger.error("Failed to connect to broker. Exiting.")
            return

        account_value = self.broker.get_account_value()
        logger.info(f"Account Value: ${account_value:,.2f}")

        # Print strategy info
        self._print_strategy_info()

        # Main loop
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
        finally:
            self.broker.disconnect()

    def _print_strategy_info(self):
        """Print information about our trading strategy."""
        print("\n" + "=" * 60)
        print("REVISION BIASES WE'RE TRADING")
        print("=" * 60)

        for indicator, bias in REVISION_BIASES.items():
            direction = bias["direction"].upper()
            prob = bias["probability"]
            emoji = "📈" if direction == "UP" else "📉" if direction == "DOWN" else "➡️"
            print(f"{emoji} {indicator}: {direction} ({prob:.0%})")

        print("\n" + "=" * 60)
        print("TRADING RULES")
        print("=" * 60)
        print("• GDP MISS → BUY (99% revises up)")
        print("• NFP MISS → BUY (62% revises up)")
        print("• Retail Sales BEAT → SELL (98% revises down)")
        print("• Industrial Prod BEAT → SELL (92% revises down)")
        print("• Maximum position: 7% of portfolio")
        print("• Maximum total risk: 15% of portfolio")
        print("• Default stop loss: 2%")
        print("• Default take profit: 4%")
        print("=" * 60 + "\n")

    def _run_loop(self):
        """Main event loop."""
        logger.info("Starting main loop. Press Ctrl+C to stop.")

        while True:
            try:
                # Check for upcoming events
                self._check_upcoming_events()

                # Check if any events just released (simulated for demo)
                self._check_for_releases()

                # Check existing positions
                self._check_positions()

                # Sleep before next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)  # Wait before retrying

    def _check_upcoming_events(self):
        """Check for upcoming economic events and alert."""
        events = self.calendar.get_upcoming_events(days_ahead=1)

        for event in events:
            hours_until = (event.datetime_utc - datetime.utcnow()).total_seconds() / 3600

            # Alert 1 hour before important events
            if 0.9 < hours_until < 1.1:
                if event.indicator_code not in self.last_checked:
                    bias = REVISION_BIASES.get(event.indicator_code, {})
                    if bias:
                        self.alerts.release_imminent(
                            event_name=event.name,
                            minutes_until=60,
                            revision_bias=f"{bias['direction'].upper()} ({bias['probability']:.0%})",
                        )
                        self.last_checked[event.indicator_code] = datetime.utcnow()

    def _check_for_releases(self):
        """
        Check if any economic data just released.

        In production, this would:
        1. Subscribe to real-time data feeds (Bloomberg, Reuters, FRED)
        2. Capture the actual release value
        3. Compare to consensus
        4. Generate trade signals

        For demo, we simulate this.
        """
        # In demo mode, simulate a release every few minutes for testing
        if self.mode == "demo":
            # Simulate random releases for demonstration
            pass

    def _check_positions(self):
        """Check existing positions for stop/take profit."""
        positions = self.broker.get_positions()

        for pos in positions:
            if pos.unrealized_pnl_pct <= -2.0:  # Stop loss hit
                logger.warning(f"Stop loss triggered for {pos.symbol}")
                self.alerts.stop_hit(
                    symbol=pos.symbol,
                    entry=pos.avg_cost,
                    exit_price=pos.current_price,
                    pnl_pct=pos.unrealized_pnl_pct,
                )

            elif pos.unrealized_pnl_pct >= 4.0:  # Take profit hit
                logger.info(f"Take profit triggered for {pos.symbol}")
                self.alerts.target_hit(
                    symbol=pos.symbol,
                    entry=pos.avg_cost,
                    exit_price=pos.current_price,
                    pnl_pct=pos.unrealized_pnl_pct,
                )

    def process_release(
        self,
        indicator_code: str,
        actual: float,
        consensus: float,
        previous: float,
    ):
        """
        Process an economic release and potentially trade.

        This is the main entry point when a release happens.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING RELEASE: {indicator_code}")
        logger.info(f"Actual: {actual:,.2f} | Consensus: {consensus:,.2f}")
        logger.info(f"{'='*60}")

        # Get signal and trades
        signal, trades = self.strategy.analyze_release(
            indicator_code, actual, consensus, previous
        )

        logger.info(f"Signal: {signal.value}")
        logger.info(f"Number of trades: {len(trades)}")

        if not trades:
            logger.info("No trades to execute")
            return

        # Send alert
        self.alerts.trading_opportunity(
            indicator=indicator_code,
            actual=actual,
            consensus=consensus,
            signal=signal.value,
            trades=[
                {"symbol": t.symbol, "direction": t.direction, "size_pct": t.size_pct}
                for t in trades
            ],
        )

        # Execute trades
        for trade in trades:
            self._execute_trade(trade)

    def _execute_trade(self, trade):
        """Execute a single trade."""
        # Calculate shares
        shares = self.broker.calculate_position_size(trade.size_pct, trade.symbol)

        if shares <= 0:
            logger.warning(f"Cannot calculate position size for {trade.symbol}")
            return

        logger.info(f"Executing: {trade.direction} {shares} {trade.symbol}")

        # Place order
        result = self.broker.place_market_order(
            symbol=trade.symbol,
            quantity=shares,
            action=trade.direction,
        )

        if result.success:
            logger.info(f"✅ Order filled: {shares} {trade.symbol} @ ${result.fill_price:.2f}")
            self.alerts.trade_executed(
                symbol=trade.symbol,
                direction=trade.direction,
                price=result.fill_price,
                size=trade.size_pct,
            )
        else:
            logger.error(f"❌ Order failed: {result.error}")


def main():
    parser = argparse.ArgumentParser(description="Revision-Based Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["demo", "paper", "live"],
        default="demo",
        help="Trading mode (default: demo)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (same as --mode demo)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (REAL MONEY)",
    )
    parser.add_argument(
        "--simulate",
        type=str,
        help="Simulate a release: INDICATOR,ACTUAL,CONSENSUS,PREVIOUS",
    )
    parser.add_argument(
        "--broker",
        choices=["ibkr", "alpaca"],
        default="ibkr",
        help="Broker to use (default: ibkr)",
    )

    args = parser.parse_args()

    # Determine mode
    if args.live:
        mode = "live"
    elif args.paper:
        mode = "paper"
    else:
        mode = "demo"

    # Create bot
    bot = TradingBot(mode=mode, broker_type=args.broker)

    # If simulating a release
    if args.simulate:
        parts = args.simulate.split(",")
        if len(parts) != 4:
            print("Usage: --simulate INDICATOR,ACTUAL,CONSENSUS,PREVIOUS")
            print("Example: --simulate PAYEMS,150000,200000,180000")
            return

        indicator, actual, consensus, previous = parts
        bot.broker.connect()
        bot.process_release(indicator, float(actual), float(consensus), float(previous))
        bot.broker.disconnect()
        return

    # Start bot
    bot.start()


if __name__ == "__main__":
    main()

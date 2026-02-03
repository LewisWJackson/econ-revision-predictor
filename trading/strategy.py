"""
Revision-Based Trading Strategy

This module implements the core trading logic:
1. Wait for economic release
2. Analyze vs consensus and our revision model
3. Generate trade signal
4. Execute via broker API

INSTRUMENTS TO TRADE:
- SPY/SPX (S&P 500) - General risk sentiment
- QQQ (Nasdaq) - Tech/growth
- TLT/IEF (Bonds) - Rate sensitive
- XLF (Financials) - Jobs sensitive
- XLI (Industrials) - Industrial production sensitive
- XLY (Consumer Discretionary) - Retail sales sensitive
- DIA (Dow) - Industrial proxy
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Signal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class Trade:
    """Represents a trade to execute."""
    symbol: str
    direction: str  # "BUY" or "SELL"
    size_pct: float  # % of portfolio
    entry_price: Optional[float] = None
    stop_loss_pct: float = 2.0  # Default 2% stop
    take_profit_pct: float = 4.0  # Default 4% take profit
    hold_days: int = 5  # Default hold period
    reasoning: str = ""


# What to trade for each economic indicator
INDICATOR_TRADES = {
    "GDPC1": {  # GDP
        "miss": [
            Trade("SPY", "BUY", 5.0, reasoning="GDP miss but will revise up"),
            Trade("QQQ", "BUY", 3.0, reasoning="GDP miss but will revise up"),
        ],
        "beat": [],  # No trade on beats - already priced
    },
    "PAYEMS": {  # Nonfarm Payrolls
        "miss": [
            Trade("SPY", "BUY", 4.0, reasoning="Jobs miss but 62% revise up"),
            Trade("XLF", "BUY", 3.0, reasoning="Financials oversold on jobs miss"),
        ],
        "beat": [
            Trade("TLT", "BUY", 2.0, reasoning="Strong jobs may not last - bonds oversold"),
        ],
    },
    "ICSA": {  # Initial Claims
        "miss": [],  # Neutral revision bias
        "beat": [],
    },
    "INDPRO": {  # Industrial Production
        "miss": [],  # Already expects downward revision
        "beat": [
            Trade("XLI", "SELL", 3.0, reasoning="Industrial beat will revise down 92%"),
            Trade("SPY", "SELL", 2.0, reasoning="Industrial strength overstated"),
        ],
    },
    "RSXFS": {  # Retail Sales
        "miss": [],  # Already expects downward revision
        "beat": [
            Trade("XLY", "SELL", 3.0, reasoning="Retail beat will revise down 98%"),
            Trade("SPY", "SELL", 2.0, reasoning="Consumer strength overstated"),
        ],
    },
    "DGORDER": {  # Durable Goods
        "miss": [],
        "beat": [
            Trade("XLI", "SELL", 2.0, reasoning="Durable goods will revise down 69%"),
        ],
    },
    "HOUST": {  # Housing Starts
        "miss": [],  # Neutral - 50/50
        "beat": [],
    },
}


class RevisionStrategy:
    """
    Main strategy class that generates trades based on economic releases.
    """

    def __init__(self, max_portfolio_risk: float = 0.10):
        """
        Args:
            max_portfolio_risk: Maximum % of portfolio at risk at any time
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.active_trades: List[Trade] = []

    def analyze_release(
        self,
        indicator_code: str,
        actual: float,
        consensus: float,
        previous: float,
    ) -> Tuple[Signal, List[Trade]]:
        """
        Analyze an economic release and generate trades.

        Args:
            indicator_code: Our indicator code (GDPC1, PAYEMS, etc.)
            actual: The released value
            consensus: Market consensus expectation
            previous: Previous period value

        Returns:
            Tuple of (Signal, List of Trades to execute)
        """
        if indicator_code not in INDICATOR_TRADES:
            return Signal.HOLD, []

        # Determine beat or miss
        surprise = actual - consensus
        surprise_pct = (surprise / abs(consensus)) * 100 if consensus != 0 else 0

        # Determine magnitude
        is_big_surprise = abs(surprise_pct) > 5  # More than 5% surprise

        # Get trades for this scenario
        trades_config = INDICATOR_TRADES[indicator_code]

        if surprise < 0:  # MISS
            trades = trades_config.get("miss", [])
            if is_big_surprise and trades:
                # Big miss - increase position sizes
                trades = [
                    Trade(
                        t.symbol,
                        t.direction,
                        min(t.size_pct * 1.5, 7.0),  # 50% bigger, max 7%
                        reasoning=t.reasoning + f" (BIG MISS: {surprise_pct:.1f}%)"
                    )
                    for t in trades
                ]
                signal = Signal.STRONG_BUY if trades[0].direction == "BUY" else Signal.STRONG_SELL
            else:
                signal = Signal.BUY if trades and trades[0].direction == "BUY" else Signal.HOLD

        else:  # BEAT
            trades = trades_config.get("beat", [])
            if is_big_surprise and trades:
                trades = [
                    Trade(
                        t.symbol,
                        t.direction,
                        min(t.size_pct * 1.5, 7.0),
                        reasoning=t.reasoning + f" (BIG BEAT: {surprise_pct:+.1f}%)"
                    )
                    for t in trades
                ]
                signal = Signal.STRONG_SELL if trades[0].direction == "SELL" else Signal.STRONG_BUY
            else:
                signal = Signal.SELL if trades and trades[0].direction == "SELL" else Signal.HOLD

        # Check portfolio risk limits
        current_risk = sum(t.size_pct for t in self.active_trades)
        new_risk = sum(t.size_pct for t in trades)

        if current_risk + new_risk > self.max_portfolio_risk * 100:
            # Scale down trades to fit risk budget
            available_risk = (self.max_portfolio_risk * 100) - current_risk
            if available_risk > 0:
                scale_factor = available_risk / new_risk
                trades = [
                    Trade(
                        t.symbol,
                        t.direction,
                        t.size_pct * scale_factor,
                        reasoning=t.reasoning + " (scaled for risk)"
                    )
                    for t in trades
                ]
            else:
                logger.warning("Risk budget exhausted - no new trades")
                trades = []

        return signal, trades

    def get_trade_plan(
        self,
        indicator_code: str,
        actual: float,
        consensus: float,
        previous: float,
    ) -> str:
        """
        Get a human-readable trade plan for a release.
        """
        signal, trades = self.analyze_release(indicator_code, actual, consensus, previous)

        lines = []
        lines.append("=" * 60)
        lines.append(f"TRADE PLAN: {indicator_code}")
        lines.append("=" * 60)
        lines.append(f"Actual: {actual:,.2f}")
        lines.append(f"Consensus: {consensus:,.2f}")
        lines.append(f"Surprise: {actual - consensus:+,.2f} ({(actual-consensus)/consensus*100:+.1f}%)")
        lines.append(f"\nSignal: {signal.value}")
        lines.append(f"Number of trades: {len(trades)}")

        if trades:
            lines.append("\nTRADES TO EXECUTE:")
            for i, trade in enumerate(trades, 1):
                lines.append(f"\n  Trade {i}:")
                lines.append(f"    Symbol: {trade.symbol}")
                lines.append(f"    Direction: {trade.direction}")
                lines.append(f"    Size: {trade.size_pct:.1f}% of portfolio")
                lines.append(f"    Stop Loss: {trade.stop_loss_pct}%")
                lines.append(f"    Take Profit: {trade.take_profit_pct}%")
                lines.append(f"    Hold Period: {trade.hold_days} days")
                lines.append(f"    Reasoning: {trade.reasoning}")
        else:
            lines.append("\nNo trades - either no edge or risk limit reached")

        return "\n".join(lines)


def simulate_releases():
    """Simulate various economic releases to show strategy behavior."""
    strategy = RevisionStrategy()

    scenarios = [
        ("GDPC1", 1.8, 2.0, 2.3, "GDP misses expectations"),
        ("GDPC1", 2.5, 2.0, 2.3, "GDP beats expectations"),
        ("PAYEMS", 150000, 200000, 180000, "Jobs big miss"),
        ("PAYEMS", 250000, 200000, 180000, "Jobs big beat"),
        ("RSXFS", 0.8, 0.3, 0.4, "Retail sales big beat"),
        ("INDPRO", 0.5, 0.1, -0.1, "Industrial production beats"),
    ]

    print("\n" + "=" * 70)
    print("STRATEGY SIMULATION - VARIOUS ECONOMIC RELEASES")
    print("=" * 70)

    for indicator, actual, consensus, previous, description in scenarios:
        print(f"\n{'─' * 70}")
        print(f"SCENARIO: {description}")
        print(f"{'─' * 70}")
        print(strategy.get_trade_plan(indicator, actual, consensus, previous))


if __name__ == "__main__":
    simulate_releases()

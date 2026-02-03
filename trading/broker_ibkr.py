"""
Interactive Brokers Integration

This module handles trade execution via Interactive Brokers API.
IBKR is available to UK residents and offers:
- Access to US markets (stocks, ETFs, futures)
- Low commissions
- Excellent API (ib_insync library)

SETUP:
1. Open IBKR account (ibkr.com)
2. Download Trader Workstation (TWS) or IB Gateway
3. Enable API connections in TWS settings
4. pip install ib_insync

IMPORTANT:
- Run TWS/Gateway before running this code
- Paper trading account recommended for testing
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

# Try to import ib_insync, but don't fail if not installed
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning("ib_insync not installed. Run: pip install ib_insync")


@dataclass
class Position:
    """Represents a position in the portfolio."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class OrderResult:
    """Result of an order submission."""
    success: bool
    order_id: Optional[int] = None
    fill_price: Optional[float] = None
    filled_quantity: Optional[int] = None
    error: Optional[str] = None


class IBKRBroker:
    """
    Interactive Brokers API wrapper for automated trading.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497 for paper, 7496 for live
        client_id: int = 1,
    ):
        """
        Initialize IBKR connection.

        Args:
            host: TWS/Gateway host (usually localhost)
            port: TWS port (7497=paper, 7496=live)
            client_id: Unique client ID for this connection
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to TWS/Gateway."""
        if not IB_AVAILABLE:
            logger.error("ib_insync not installed")
            return False

        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info(f"Connected to IBKR at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")

    def get_account_value(self) -> float:
        """Get total account value."""
        if not self.connected:
            return 0.0

        account_values = self.ib.accountValues()
        for av in account_values:
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        return 0.0

    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        if not self.connected:
            return []

        positions = []
        for pos in self.ib.positions():
            contract = pos.contract
            quantity = pos.position
            avg_cost = pos.avgCost

            # Get current price
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data

            current_price = ticker.last if ticker.last else ticker.close
            if current_price:
                unrealized_pnl = (current_price - avg_cost) * quantity
                unrealized_pnl_pct = ((current_price / avg_cost) - 1) * 100
            else:
                unrealized_pnl = 0
                unrealized_pnl_pct = 0

            positions.append(Position(
                symbol=contract.symbol,
                quantity=int(quantity),
                avg_cost=avg_cost,
                current_price=current_price or 0,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
            ))

        return positions

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        if not self.connected:
            return None

        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(1)

        return ticker.last if ticker.last else ticker.close

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        action: str,  # "BUY" or "SELL"
    ) -> OrderResult:
        """
        Place a market order.

        Args:
            symbol: Stock symbol (e.g., "SPY")
            quantity: Number of shares
            action: "BUY" or "SELL"
        """
        if not self.connected:
            return OrderResult(success=False, error="Not connected")

        try:
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(contract, order)

            # Wait for fill
            timeout = 30  # seconds
            start = time.time()
            while not trade.isDone() and (time.time() - start) < timeout:
                self.ib.sleep(0.5)

            if trade.orderStatus.status == "Filled":
                return OrderResult(
                    success=True,
                    order_id=trade.order.orderId,
                    fill_price=trade.orderStatus.avgFillPrice,
                    filled_quantity=int(trade.orderStatus.filled),
                )
            else:
                return OrderResult(
                    success=False,
                    error=f"Order not filled: {trade.orderStatus.status}",
                )

        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def place_bracket_order(
        self,
        symbol: str,
        quantity: int,
        action: str,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> OrderResult:
        """
        Place a bracket order (entry + stop loss + take profit).

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            action: "BUY" or "SELL"
            stop_loss_pct: Stop loss percentage (e.g., 2.0 for 2%)
            take_profit_pct: Take profit percentage (e.g., 4.0 for 4%)
        """
        if not self.connected:
            return OrderResult(success=False, error="Not connected")

        try:
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            # Get current price for stop/limit calculations
            current_price = self.get_price(symbol)
            if not current_price:
                return OrderResult(success=False, error="Could not get current price")

            # Calculate stop and limit prices
            if action == "BUY":
                stop_price = current_price * (1 - stop_loss_pct / 100)
                limit_price = current_price * (1 + take_profit_pct / 100)
            else:  # SELL (short)
                stop_price = current_price * (1 + stop_loss_pct / 100)
                limit_price = current_price * (1 - take_profit_pct / 100)

            # Create bracket order
            bracket = self.ib.bracketOrder(
                action=action,
                quantity=quantity,
                limitPrice=current_price,  # Entry at market (or slight limit)
                takeProfitPrice=round(limit_price, 2),
                stopLossPrice=round(stop_price, 2),
            )

            # Place all three orders
            for order in bracket:
                self.ib.placeOrder(contract, order)

            # Wait for parent fill
            self.ib.sleep(2)

            return OrderResult(
                success=True,
                order_id=bracket[0].orderId,
                fill_price=current_price,
                filled_quantity=quantity,
            )

        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def calculate_position_size(
        self,
        portfolio_pct: float,
        symbol: str,
    ) -> int:
        """
        Calculate number of shares for a given portfolio percentage.

        Args:
            portfolio_pct: Target percentage of portfolio (e.g., 5.0 for 5%)
            symbol: Stock symbol

        Returns:
            Number of shares to buy/sell
        """
        account_value = self.get_account_value()
        if account_value <= 0:
            return 0

        target_value = account_value * (portfolio_pct / 100)
        price = self.get_price(symbol)

        if not price or price <= 0:
            return 0

        return int(target_value / price)


class MockBroker:
    """
    Mock broker for testing without real connection.
    Simulates IBKR behavior.
    """

    def __init__(self):
        self.account_value = 100000.0  # $100k paper account
        self.positions: Dict[str, Position] = {}
        self.connected = True
        self.prices = {
            "SPY": 480.0,
            "QQQ": 420.0,
            "TLT": 95.0,
            "XLF": 42.0,
            "XLI": 125.0,
            "XLY": 185.0,
            "DIA": 390.0,
        }

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False

    def get_account_value(self) -> float:
        return self.account_value

    def get_price(self, symbol: str) -> Optional[float]:
        return self.prices.get(symbol)

    def get_positions(self) -> List[Position]:
        return list(self.positions.values())

    def place_market_order(self, symbol: str, quantity: int, action: str) -> OrderResult:
        price = self.get_price(symbol)
        if not price:
            return OrderResult(success=False, error=f"Unknown symbol: {symbol}")

        logger.info(f"[MOCK] {action} {quantity} {symbol} @ ${price:.2f}")

        # Update position
        if symbol in self.positions:
            pos = self.positions[symbol]
            if action == "BUY":
                new_qty = pos.quantity + quantity
                pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / new_qty
                pos.quantity = new_qty
            else:
                pos.quantity -= quantity
                if pos.quantity <= 0:
                    del self.positions[symbol]
        else:
            if action == "BUY":
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=price,
                    current_price=price,
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0,
                )

        return OrderResult(
            success=True,
            order_id=12345,
            fill_price=price,
            filled_quantity=quantity,
        )

    def calculate_position_size(self, portfolio_pct: float, symbol: str) -> int:
        price = self.get_price(symbol)
        if not price:
            return 0
        target_value = self.account_value * (portfolio_pct / 100)
        return int(target_value / price)


def demo_broker():
    """Demonstrate broker functionality with mock."""
    print("\n" + "=" * 60)
    print("BROKER DEMO (Mock Mode)")
    print("=" * 60)

    broker = MockBroker()
    broker.connect()

    print(f"\nAccount Value: ${broker.get_account_value():,.2f}")

    # Calculate position size
    symbol = "SPY"
    pct = 5.0
    shares = broker.calculate_position_size(pct, symbol)
    print(f"\n{pct}% position in {symbol} = {shares} shares")

    # Place order
    print(f"\nPlacing order: BUY {shares} {symbol}")
    result = broker.place_market_order(symbol, shares, "BUY")
    print(f"Result: {result}")

    # Show positions
    print("\nCurrent Positions:")
    for pos in broker.get_positions():
        print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f}")


if __name__ == "__main__":
    demo_broker()

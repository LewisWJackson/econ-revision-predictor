"""
Alpaca Broker Implementation

Alpaca is commission-free and has excellent API support.
Available to UK residents (190+ countries supported).

SETUP:
1. Create account at https://alpaca.markets/
   - Sign up and verify identity (passport/driving licence)
   - UK residents: GBP, EUR, USD currencies available

2. Get API keys from dashboard:
   - Go to https://app.alpaca.markets/paper/dashboard/overview
   - Click "View" under "API Keys" on the right side
   - You'll get APCA-API-KEY-ID and APCA-API-SECRET-KEY

3. Set environment variables:
   export ALPACA_API_KEY='PKXXXXXXXXXXXXXXXX'      # Your Key ID
   export ALPACA_SECRET_KEY='XXXXXXXXXXXXXXXX'     # Your Secret Key

4. Test with curl (should return 200, not 403):
   curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \\
        -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \\
        https://paper-api.alpaca.markets/v2/account

NOTE: The 403 error you saw was because the curl request had NO authentication.
      Alpaca requires API key headers on every request.
"""

import os
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Check if alpaca-trade-api is installed
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. Run: pip install alpaca-py")


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    filled_qty: Optional[int] = None
    error: Optional[str] = None


class AlpacaBroker:
    """
    Alpaca broker implementation for automated trading.

    Features:
    - Commission-free US stocks and ETFs
    - Paper trading for testing
    - Real-time and historical data
    - Simple REST API
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is required. Install with: pip install alpaca-py"
            )

        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.paper = paper

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables."
            )

        self.trading_client = None
        self.data_client = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )

            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )

            # Test connection by getting account
            account = self.trading_client.get_account()
            logger.info(f"Connected to Alpaca ({'Paper' if self.paper else 'Live'})")
            logger.info(f"Account status: {account.status}")

            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def disconnect(self):
        """Disconnect from Alpaca API."""
        self.trading_client = None
        self.data_client = None
        self.connected = False
        logger.info("Disconnected from Alpaca")

    def get_account_value(self) -> float:
        """Get total account value (cash + positions)."""
        if not self.connected:
            return 0.0

        account = self.trading_client.get_account()
        return float(account.portfolio_value)

    def get_buying_power(self) -> float:
        """Get available buying power."""
        if not self.connected:
            return 0.0

        account = self.trading_client.get_account()
        return float(account.buying_power)

    def get_cash(self) -> float:
        """Get cash balance."""
        if not self.connected:
            return 0.0

        account = self.trading_client.get_account()
        return float(account.cash)

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if not self.connected:
            return []

        positions = self.trading_client.get_all_positions()

        return [
            Position(
                symbol=pos.symbol,
                quantity=int(pos.qty),
                avg_cost=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
            )
            for pos in positions
        ]

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        if not self.connected:
            return None

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)

            if symbol in quotes:
                # Use midpoint of bid/ask
                quote = quotes[symbol]
                return (float(quote.bid_price) + float(quote.ask_price)) / 2
            return None

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def calculate_position_size(
        self,
        size_pct: float,
        symbol: str,
    ) -> int:
        """Calculate number of shares for a given portfolio percentage."""
        account_value = self.get_account_value()
        price = self.get_current_price(symbol)

        if not price or price <= 0:
            return 0

        target_value = account_value * (size_pct / 100)
        shares = int(target_value / price)

        return max(0, shares)

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        action: str,  # "BUY" or "SELL"
    ) -> OrderResult:
        """Place a market order."""
        if not self.connected:
            return OrderResult(success=False, error="Not connected")

        if quantity <= 0:
            return OrderResult(success=False, error="Invalid quantity")

        try:
            side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
            )

            order = self.trading_client.submit_order(order_data)

            logger.info(f"Order submitted: {action} {quantity} {symbol}")

            # For market orders, get the fill price
            # Note: May need to poll for fill in production
            fill_price = self.get_current_price(symbol)

            return OrderResult(
                success=True,
                order_id=str(order.id),
                fill_price=fill_price,
                filled_qty=quantity,
            )

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return OrderResult(success=False, error=str(e))

    def place_bracket_order(
        self,
        symbol: str,
        quantity: int,
        action: str,
        take_profit_pct: float = 4.0,
        stop_loss_pct: float = 2.0,
    ) -> OrderResult:
        """
        Place a bracket order with take profit and stop loss.

        This is useful for our strategy to automatically manage risk.
        """
        if not self.connected:
            return OrderResult(success=False, error="Not connected")

        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                TakeProfitRequest,
                StopLossRequest,
            )

            side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL
            current_price = self.get_current_price(symbol)

            if not current_price:
                return OrderResult(success=False, error="Could not get price")

            # Calculate take profit and stop loss prices
            if side == OrderSide.BUY:
                take_profit_price = current_price * (1 + take_profit_pct / 100)
                stop_loss_price = current_price * (1 - stop_loss_pct / 100)
            else:
                take_profit_price = current_price * (1 - take_profit_pct / 100)
                stop_loss_price = current_price * (1 + stop_loss_pct / 100)

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC,
                order_class="bracket",
                take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2)),
            )

            order = self.trading_client.submit_order(order_data)

            logger.info(
                f"Bracket order submitted: {action} {quantity} {symbol} "
                f"(TP: ${take_profit_price:.2f}, SL: ${stop_loss_price:.2f})"
            )

            return OrderResult(
                success=True,
                order_id=str(order.id),
                fill_price=current_price,
                filled_qty=quantity,
            )

        except Exception as e:
            logger.error(f"Bracket order failed: {e}")
            return OrderResult(success=False, error=str(e))

    def close_position(self, symbol: str) -> OrderResult:
        """Close an entire position."""
        if not self.connected:
            return OrderResult(success=False, error="Not connected")

        try:
            self.trading_client.close_position(symbol)
            logger.info(f"Closed position in {symbol}")
            return OrderResult(success=True)

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return OrderResult(success=False, error=str(e))

    def close_all_positions(self) -> bool:
        """Close all open positions."""
        if not self.connected:
            return False

        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            logger.info("Closed all positions")
            return True

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        if not self.connected:
            return False

        clock = self.trading_client.get_clock()
        return clock.is_open


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Alpaca Broker...")
    print("=" * 60)

    # Check if credentials are set
    if not os.getenv("ALPACA_API_KEY"):
        print("\nTo use Alpaca:")
        print("1. Create account at https://alpaca.markets/")
        print("2. Get API keys from the dashboard")
        print("3. Set environment variables:")
        print("   export ALPACA_API_KEY='your-api-key'")
        print("   export ALPACA_SECRET_KEY='your-secret-key'")
        print("\nFor paper trading, use paper trading keys (safer for testing)")
    else:
        broker = AlpacaBroker(paper=True)
        if broker.connect():
            print(f"\nAccount Value: ${broker.get_account_value():,.2f}")
            print(f"Buying Power: ${broker.get_buying_power():,.2f}")
            print(f"Market Open: {broker.is_market_open()}")

            positions = broker.get_positions()
            if positions:
                print(f"\nOpen Positions:")
                for pos in positions:
                    print(f"  {pos.symbol}: {pos.quantity} @ ${pos.avg_cost:.2f}")

            broker.disconnect()

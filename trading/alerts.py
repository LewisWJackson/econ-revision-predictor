"""
Alert System for Economic Releases

Sends alerts via:
- Telegram
- Email
- Desktop notification
- Discord webhook

When:
1. Economic release is imminent (1 hour before)
2. Release just happened with trading opportunity
3. Our active trades hit stop/target
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Represents an alert to send."""
    title: str
    message: str
    priority: str = "normal"  # low, normal, high, urgent
    data: Optional[Dict] = None


class TelegramNotifier:
    """Send alerts via Telegram bot."""

    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    def send(self, alert: Alert) -> bool:
        """Send alert via Telegram."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            return False

        # Format message with emoji based on priority
        emoji = {
            "low": "ℹ️",
            "normal": "📊",
            "high": "⚠️",
            "urgent": "🚨",
        }.get(alert.priority, "📊")

        message = f"{emoji} *{alert.title}*\n\n{alert.message}"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False


class DiscordNotifier:
    """Send alerts via Discord webhook."""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")

    def send(self, alert: Alert) -> bool:
        """Send alert via Discord."""
        if not self.webhook_url:
            logger.warning("Discord webhook not configured")
            return False

        # Color based on priority
        color = {
            "low": 0x808080,     # Gray
            "normal": 0x0099ff,  # Blue
            "high": 0xff9900,    # Orange
            "urgent": 0xff0000,  # Red
        }.get(alert.priority, 0x0099ff)

        payload = {
            "embeds": [{
                "title": alert.title,
                "description": alert.message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
            }]
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False


class EmailNotifier:
    """Send alerts via email (using SendGrid or similar)."""

    def __init__(self, api_key: str = None, from_email: str = None, to_email: str = None):
        self.api_key = api_key or os.environ.get("SENDGRID_API_KEY")
        self.from_email = from_email or os.environ.get("ALERT_FROM_EMAIL")
        self.to_email = to_email or os.environ.get("ALERT_TO_EMAIL")

    def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not all([self.api_key, self.from_email, self.to_email]):
            logger.warning("Email credentials not configured")
            return False

        # Using SendGrid API
        url = "https://api.sendgrid.com/v3/mail/send"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "personalizations": [{"to": [{"email": self.to_email}]}],
            "from": {"email": self.from_email},
            "subject": f"[{alert.priority.upper()}] {alert.title}",
            "content": [{"type": "text/plain", "value": alert.message}],
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            return response.status_code in [200, 202]
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False


class ConsoleNotifier:
    """Print alerts to console (always works, good for testing)."""

    def send(self, alert: Alert) -> bool:
        """Print alert to console."""
        border = "=" * 60
        priority_indicator = {
            "low": "[INFO]",
            "normal": "[ALERT]",
            "high": "[WARNING]",
            "urgent": "[URGENT]",
        }.get(alert.priority, "[ALERT]")

        print(f"\n{border}")
        print(f"{priority_indicator} {alert.title}")
        print(border)
        print(alert.message)
        print(f"{border}\n")
        return True


class AlertManager:
    """
    Manages all alert channels and sends notifications.
    """

    def __init__(self):
        self.notifiers = {
            "console": ConsoleNotifier(),
            "telegram": TelegramNotifier(),
            "discord": DiscordNotifier(),
            "email": EmailNotifier(),
        }
        # Default to console only
        self.active_channels = ["console"]

    def configure(self, channels: List[str]):
        """Set which channels to use."""
        self.active_channels = [c for c in channels if c in self.notifiers]

    def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """Send alert to all active channels."""
        results = {}
        for channel in self.active_channels:
            notifier = self.notifiers.get(channel)
            if notifier:
                results[channel] = notifier.send(alert)
        return results

    def release_imminent(self, event_name: str, minutes_until: int, revision_bias: str):
        """Alert that a release is coming soon."""
        alert = Alert(
            title=f"📅 {event_name} in {minutes_until} minutes",
            message=f"Economic release imminent!\n\n"
                    f"Event: {event_name}\n"
                    f"Time: {minutes_until} minutes\n"
                    f"Revision Bias: {revision_bias}\n\n"
                    f"Prepare to analyze the release.",
            priority="normal",
        )
        return self.send_alert(alert)

    def trading_opportunity(
        self,
        indicator: str,
        actual: float,
        consensus: float,
        signal: str,
        trades: List[Dict],
    ):
        """Alert that there's a trading opportunity."""
        surprise = actual - consensus
        surprise_pct = (surprise / abs(consensus)) * 100 if consensus else 0

        trade_details = "\n".join([
            f"  • {t['symbol']} {t['direction']} {t['size_pct']}%"
            for t in trades
        ])

        alert = Alert(
            title=f"🎯 TRADE SIGNAL: {indicator}",
            message=f"Trading opportunity detected!\n\n"
                    f"Actual: {actual:,.2f}\n"
                    f"Consensus: {consensus:,.2f}\n"
                    f"Surprise: {surprise:+,.2f} ({surprise_pct:+.1f}%)\n\n"
                    f"Signal: {signal}\n\n"
                    f"Trades:\n{trade_details}",
            priority="high" if "STRONG" in signal else "normal",
        )
        return self.send_alert(alert)

    def trade_executed(self, symbol: str, direction: str, price: float, size: float):
        """Alert that a trade was executed."""
        alert = Alert(
            title=f"✅ Trade Executed: {symbol}",
            message=f"Trade filled!\n\n"
                    f"Symbol: {symbol}\n"
                    f"Direction: {direction}\n"
                    f"Price: ${price:.2f}\n"
                    f"Size: {size:.1f}%",
            priority="normal",
        )
        return self.send_alert(alert)

    def stop_hit(self, symbol: str, entry: float, exit_price: float, pnl_pct: float):
        """Alert that stop loss was hit."""
        alert = Alert(
            title=f"🛑 Stop Hit: {symbol}",
            message=f"Stop loss triggered!\n\n"
                    f"Symbol: {symbol}\n"
                    f"Entry: ${entry:.2f}\n"
                    f"Exit: ${exit_price:.2f}\n"
                    f"P&L: {pnl_pct:+.1f}%",
            priority="high",
        )
        return self.send_alert(alert)

    def target_hit(self, symbol: str, entry: float, exit_price: float, pnl_pct: float):
        """Alert that take profit was hit."""
        alert = Alert(
            title=f"🎉 Target Hit: {symbol}",
            message=f"Take profit triggered!\n\n"
                    f"Symbol: {symbol}\n"
                    f"Entry: ${entry:.2f}\n"
                    f"Exit: ${exit_price:.2f}\n"
                    f"P&L: {pnl_pct:+.1f}%",
            priority="normal",
        )
        return self.send_alert(alert)


# Convenience function
def demo_alerts():
    """Demonstrate the alert system."""
    manager = AlertManager()

    print("Testing Console Alerts...")
    print("-" * 40)

    # Test release imminent
    manager.release_imminent(
        event_name="Nonfarm Payrolls",
        minutes_until=60,
        revision_bias="UP (62%)",
    )

    # Test trading opportunity
    manager.trading_opportunity(
        indicator="PAYEMS",
        actual=150000,
        consensus=200000,
        signal="STRONG_BUY",
        trades=[
            {"symbol": "SPY", "direction": "BUY", "size_pct": 5.0},
            {"symbol": "XLF", "direction": "BUY", "size_pct": 3.0},
        ],
    )

    # Test trade executed
    manager.trade_executed(
        symbol="SPY",
        direction="BUY",
        price=475.50,
        size=5.0,
    )


if __name__ == "__main__":
    demo_alerts()

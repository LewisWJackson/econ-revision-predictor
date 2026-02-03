"""
Economic Calendar Monitor

Monitors upcoming economic releases and sends alerts when:
1. A release is imminent
2. A release just happened and is near a threshold
3. Our model sees edge

Data sources:
- FRED for release schedules
- Investing.com calendar (scraped)
- BLS/BEA release schedules
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EconomicEvent:
    """Represents an economic data release."""
    name: str
    datetime_utc: datetime
    indicator_code: str  # Our internal code (GDPC1, PAYEMS, etc.)
    consensus: Optional[float]
    previous: Optional[float]
    actual: Optional[float] = None
    revision_bias: Optional[str] = None  # "up", "down", "neutral"
    revision_probability: Optional[float] = None


# Our revision biases from the model
REVISION_BIASES = {
    "GDPC1": {"direction": "up", "probability": 0.993, "avg_revision_pct": 0.15},
    "PAYEMS": {"direction": "up", "probability": 0.618, "avg_revision": 190},
    "ICSA": {"direction": "neutral", "probability": 0.5, "avg_revision": -2863},
    "INDPRO": {"direction": "down", "probability": 0.921, "avg_revision_pct": -0.02},
    "RSXFS": {"direction": "down", "probability": 0.983, "avg_revision_pct": -0.03},
    "DGORDER": {"direction": "down", "probability": 0.686, "avg_revision_pct": -0.01},
    "HOUST": {"direction": "neutral", "probability": 0.507, "avg_revision": 0},
    "UNRATE": {"direction": "neutral", "probability": 0.5, "avg_revision": 0},
    "CPI": {"direction": "neutral", "probability": 0.5, "avg_revision": 0},
}

# Map common names to our codes
EVENT_NAME_MAP = {
    "GDP": "GDPC1",
    "Gross Domestic Product": "GDPC1",
    "Nonfarm Payrolls": "PAYEMS",
    "Non-Farm Payrolls": "PAYEMS",
    "NFP": "PAYEMS",
    "Employment Change": "PAYEMS",
    "Initial Jobless Claims": "ICSA",
    "Jobless Claims": "ICSA",
    "Industrial Production": "INDPRO",
    "Retail Sales": "RSXFS",
    "Durable Goods Orders": "DGORDER",
    "Housing Starts": "HOUST",
    "Unemployment Rate": "UNRATE",
    "CPI": "CPI",
    "Consumer Price Index": "CPI",
}


class EconomicCalendar:
    """
    Fetches and monitors economic calendar events.
    """

    def __init__(self):
        self.events: List[EconomicEvent] = []

    def get_upcoming_events(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """
        Get upcoming economic events for the next N days.

        In production, this would scrape/API from:
        - Investing.com
        - ForexFactory
        - Bloomberg
        - FRED release calendar
        """
        # For now, return hardcoded upcoming events
        # In production, implement actual scraping

        now = datetime.utcnow()
        events = []

        # Example upcoming events (you'd fetch these dynamically)
        upcoming = [
            {
                "name": "Initial Jobless Claims",
                "days_from_now": 1,
                "consensus": 220000,
                "previous": 215000,
            },
            {
                "name": "Nonfarm Payrolls",
                "days_from_now": 5,
                "consensus": 180000,
                "previous": 256000,
            },
            {
                "name": "GDP",
                "days_from_now": 14,
                "consensus": 2.1,
                "previous": 2.8,
            },
            {
                "name": "Retail Sales",
                "days_from_now": 10,
                "consensus": 0.3,
                "previous": 0.4,
            },
            {
                "name": "Industrial Production",
                "days_from_now": 12,
                "consensus": 0.2,
                "previous": -0.1,
            },
        ]

        for event_data in upcoming:
            if event_data["days_from_now"] <= days_ahead:
                indicator_code = EVENT_NAME_MAP.get(event_data["name"], "UNKNOWN")
                bias_info = REVISION_BIASES.get(indicator_code, {})

                event = EconomicEvent(
                    name=event_data["name"],
                    datetime_utc=now + timedelta(days=event_data["days_from_now"]),
                    indicator_code=indicator_code,
                    consensus=event_data["consensus"],
                    previous=event_data["previous"],
                    revision_bias=bias_info.get("direction"),
                    revision_probability=bias_info.get("probability"),
                )
                events.append(event)

        return sorted(events, key=lambda x: x.datetime_utc)

    def analyze_release(
        self,
        indicator_code: str,
        actual: float,
        consensus: float,
        thresholds: List[float] = None,
    ) -> Dict:
        """
        Analyze a just-released economic number.

        Returns trading signal based on:
        1. Beat/miss vs consensus
        2. Our revision probability
        3. Proximity to thresholds
        """
        bias_info = REVISION_BIASES.get(indicator_code, {})

        # Beat or miss?
        surprise = actual - consensus
        surprise_pct = (surprise / abs(consensus)) * 100 if consensus != 0 else 0

        # What do we expect the revision to do?
        revision_direction = bias_info.get("direction", "neutral")
        revision_prob = bias_info.get("probability", 0.5)

        # Determine signal
        signal = "HOLD"
        confidence = "low"
        reasoning = []

        if revision_direction == "up":
            if surprise < 0:  # MISS
                signal = "BUY"
                confidence = "high" if revision_prob > 0.8 else "medium"
                reasoning.append(f"Release missed by {abs(surprise_pct):.1f}%")
                reasoning.append(f"But {revision_prob:.0%} chance of upward revision")
                reasoning.append("Market overreacting to the downside")
            else:  # BEAT
                signal = "HOLD"
                reasoning.append(f"Release beat by {surprise_pct:.1f}%")
                reasoning.append("Upward revision already priced in")

        elif revision_direction == "down":
            if surprise > 0:  # BEAT
                signal = "SELL"
                confidence = "high" if revision_prob > 0.8 else "medium"
                reasoning.append(f"Release beat by {surprise_pct:.1f}%")
                reasoning.append(f"But {revision_prob:.0%} chance of downward revision")
                reasoning.append("Market overreacting to the upside")
            else:  # MISS
                signal = "HOLD"
                reasoning.append(f"Release missed by {abs(surprise_pct):.1f}%")
                reasoning.append("Downward revision already expected")

        else:  # Neutral
            signal = "HOLD"
            confidence = "low"
            reasoning.append("No systematic revision bias for this indicator")

        # Check threshold proximity
        if thresholds:
            for threshold in thresholds:
                distance_pct = abs(actual - threshold) / abs(threshold) * 100
                if distance_pct < 5:  # Within 5% of threshold
                    reasoning.append(f"CLOSE TO THRESHOLD {threshold} ({distance_pct:.1f}% away)")
                    if revision_direction == "up" and actual < threshold:
                        reasoning.append(f"Likely to cross ABOVE {threshold} after revision")
                        signal = "BUY"
                        confidence = "high"
                    elif revision_direction == "down" and actual > threshold:
                        reasoning.append(f"Likely to cross BELOW {threshold} after revision")
                        signal = "SELL"
                        confidence = "high"

        return {
            "signal": signal,
            "confidence": confidence,
            "actual": actual,
            "consensus": consensus,
            "surprise": surprise,
            "surprise_pct": surprise_pct,
            "revision_direction": revision_direction,
            "revision_probability": revision_prob,
            "reasoning": reasoning,
        }


def print_upcoming_calendar():
    """Print upcoming economic events with revision analysis."""
    calendar = EconomicCalendar()
    events = calendar.get_upcoming_events(days_ahead=14)

    print("\n" + "=" * 75)
    print("UPCOMING ECONOMIC RELEASES - REVISION ANALYSIS")
    print("=" * 75)

    for event in events:
        days_until = (event.datetime_utc - datetime.utcnow()).days

        print(f"\n📅 {event.name}")
        print(f"   When: {days_until} days from now")
        print(f"   Consensus: {event.consensus}")
        print(f"   Previous: {event.previous}")

        if event.revision_bias:
            bias_emoji = "📈" if event.revision_bias == "up" else "📉" if event.revision_bias == "down" else "➡️"
            print(f"   Revision Bias: {bias_emoji} {event.revision_bias.upper()} ({event.revision_probability:.0%})")

            if event.revision_bias == "up":
                print(f"   💡 Strategy: If MISSES consensus → BUY the dip (will likely revise up)")
            elif event.revision_bias == "down":
                print(f"   💡 Strategy: If BEATS consensus → FADE the rally (will likely revise down)")


if __name__ == "__main__":
    print_upcoming_calendar()

    # Example: Analyze a release
    print("\n" + "=" * 75)
    print("EXAMPLE: NFP Release Analysis")
    print("=" * 75)

    calendar = EconomicCalendar()
    analysis = calendar.analyze_release(
        indicator_code="PAYEMS",
        actual=175000,  # Released at 175k
        consensus=200000,  # Expected 200k
        thresholds=[150000, 200000, 250000],
    )

    print(f"\nSignal: {analysis['signal']}")
    print(f"Confidence: {analysis['confidence']}")
    print(f"Surprise: {analysis['surprise']:+,.0f} ({analysis['surprise_pct']:+.1f}%)")
    print(f"Revision Direction: {analysis['revision_direction']}")
    print(f"Revision Probability: {analysis['revision_probability']:.0%}")
    print("\nReasoning:")
    for reason in analysis["reasoning"]:
        print(f"  • {reason}")

"""
Initial Release Predictor

This module adapts our revision prediction framework to predict INITIAL economic
releases - which is what Polymarket/Kalshi actually trade.

KEY INSIGHT:
Our revision model has discovered structural patterns in how economic data behaves.
We can leverage this to predict initial releases by:
1. Using the same macro regime features (VIX, yield curve, Sahm indicator)
2. Using lagged indicator relationships
3. Incorporating seasonal patterns

TARGET: Predict whether initial release will exceed/miss a threshold
(e.g., "Will January NFP be > 200k?")
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import logger, RANDOM_SEED
from data.scripts.fred_client import FREDClient
from data.scripts.macro_indicators import load_macro_indicators


class InitialReleasePredictor:
    """
    Predicts whether an initial economic release will exceed a given threshold.

    This is useful for Polymarket/Kalshi markets like:
    - "Will January NFP be > 200k?"
    - "Will Q4 GDP growth be > 2%?"
    - "Will unemployment rate be < 4%?"
    """

    def __init__(self):
        self.client = FREDClient()
        self.macro_data = load_macro_indicators()
        self.historical_data = {}

    def load_indicator_history(self, series_id: str, start_date: str = "2000-01-01") -> pd.DataFrame:
        """Load historical initial release values for an indicator."""
        if series_id in self.historical_data:
            return self.historical_data[series_id]

        # Get current values (we'll use these as proxy for initial releases)
        # In production, you'd want actual vintage data
        vintages = self.client.get_vintage_dates(series_id, start_date=start_date)
        if not vintages:
            return pd.DataFrame()

        latest = vintages[-1].strftime("%Y-%m-%d")
        df = self.client.get_observations_at_vintage(
            series_id=series_id,
            vintage_date=latest,
            observation_start=start_date,
        )

        if not df.empty:
            df = df.rename(columns={"date": "reference_date"})
            self.historical_data[series_id] = df

        return df

    def compute_features_for_prediction(
        self,
        series_id: str,
        prediction_date: datetime,
        reference_period: datetime,
    ) -> Dict[str, float]:
        """
        Compute features for predicting an initial release.

        Features include:
        - Recent values and trends
        - Seasonal patterns
        - Macro regime indicators
        - Cross-indicator signals
        """
        features = {}

        # Load historical data
        hist = self.load_indicator_history(series_id)
        if hist.empty:
            return features

        # Filter to data available before prediction
        hist = hist[hist["reference_date"] < prediction_date].copy()
        hist = hist.sort_values("reference_date")

        if len(hist) < 12:
            return features

        recent = hist.tail(12)

        # Recent level and trends
        features["last_value"] = recent.iloc[-1]["value"]
        features["last_3_mean"] = recent.tail(3)["value"].mean()
        features["last_6_mean"] = recent.tail(6)["value"].mean()
        features["last_12_mean"] = recent["value"].mean()

        # Momentum
        features["mom_1m"] = recent.iloc[-1]["value"] - recent.iloc[-2]["value"]
        features["mom_3m"] = recent.iloc[-1]["value"] - recent.iloc[-4]["value"]
        features["mom_6m"] = recent.iloc[-1]["value"] - recent.iloc[-7]["value"]

        # Volatility
        features["volatility_6m"] = recent.tail(6)["value"].std()
        features["volatility_12m"] = recent["value"].std()

        # Seasonal patterns (same month last year)
        ref_month = reference_period.month
        same_month = hist[hist["reference_date"].dt.month == ref_month]
        if len(same_month) >= 3:
            features["seasonal_mean"] = same_month.tail(3)["value"].mean()
            features["seasonal_vs_recent"] = features["seasonal_mean"] - features["last_3_mean"]

        # Macro regime features
        if not self.macro_data.empty:
            macro_available = self.macro_data[self.macro_data["date"] <= prediction_date]
            if not macro_available.empty:
                latest_macro = macro_available.iloc[-1]
                features["vix"] = latest_macro.get("vix", np.nan)
                features["yield_curve"] = latest_macro.get("yield_curve_spread", np.nan)
                features["sahm_indicator"] = latest_macro.get("sahm_indicator", np.nan)
                features["is_recession"] = 1 if latest_macro.get("sahm_indicator", 0) >= 0.5 else 0

        return features

    def predict_threshold_probability(
        self,
        series_id: str,
        reference_period: datetime,
        threshold: float,
        direction: str = "above",  # "above" or "below"
        prediction_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Predict probability that initial release will be above/below threshold.

        Uses a simple approach:
        1. Look at historical distribution
        2. Adjust based on recent trends
        3. Adjust based on macro regime

        Returns dict with:
        - probability: estimated P(release > threshold) or P(release < threshold)
        - confidence: how confident we are (based on data quality)
        - reasoning: explanation of the prediction
        """
        if prediction_date is None:
            prediction_date = datetime.now()

        features = self.compute_features_for_prediction(
            series_id, prediction_date, reference_period
        )

        if not features:
            return {
                "probability": 0.5,
                "confidence": "low",
                "reasoning": "Insufficient historical data"
            }

        hist = self.load_indicator_history(series_id)
        hist = hist[hist["reference_date"] < prediction_date]

        # Base rate: what % of historical values exceed threshold?
        if direction == "above":
            base_rate = (hist["value"] > threshold).mean()
        else:
            base_rate = (hist["value"] < threshold).mean()

        # Trend adjustment
        trend_signal = 0
        if "mom_3m" in features and "volatility_6m" in features:
            # Standardized momentum
            if features["volatility_6m"] > 0:
                z_score = features["mom_3m"] / features["volatility_6m"]
                trend_signal = np.clip(z_score * 0.1, -0.2, 0.2)

        # Recent level vs threshold
        level_signal = 0
        if "last_value" in features and "volatility_6m" in features:
            distance = features["last_value"] - threshold
            if features["volatility_6m"] > 0:
                z_distance = distance / features["volatility_6m"]
                level_signal = np.clip(z_distance * 0.15, -0.3, 0.3)

        # Combine signals
        if direction == "above":
            probability = base_rate + trend_signal + level_signal
        else:
            probability = base_rate - trend_signal - level_signal

        probability = np.clip(probability, 0.05, 0.95)

        # Confidence based on data quality
        n_obs = len(hist)
        if n_obs > 100:
            confidence = "high"
        elif n_obs > 50:
            confidence = "medium"
        else:
            confidence = "low"

        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Historical base rate: {base_rate:.1%} of values {'exceed' if direction == 'above' else 'below'} {threshold}")

        if "last_value" in features:
            reasoning_parts.append(f"Last value: {features['last_value']:.1f}")

        if trend_signal != 0:
            trend_dir = "upward" if trend_signal > 0 else "downward"
            reasoning_parts.append(f"Recent trend: {trend_dir} ({trend_signal:+.1%} adjustment)")

        if "vix" in features and not np.isnan(features["vix"]):
            vix_status = "elevated" if features["vix"] > 25 else "normal"
            reasoning_parts.append(f"VIX: {features['vix']:.1f} ({vix_status})")

        return {
            "probability": probability,
            "confidence": confidence,
            "reasoning": " | ".join(reasoning_parts),
            "features": features,
        }


def analyze_polymarket_opportunities(predictor: InitialReleasePredictor) -> pd.DataFrame:
    """
    Analyze current Polymarket-style opportunities based on our model.
    """
    results = []
    now = datetime.now()

    # January 2026 NFP (jobs added)
    # Market shows: <0 at 3%, 0-25k at 18%
    # This implies market expects positive job growth
    nfp_pred = predictor.predict_threshold_probability(
        series_id="PAYEMS",
        reference_period=datetime(2026, 1, 1),
        threshold=0,  # Change from prior month
        direction="above",
        prediction_date=now,
    )
    results.append({
        "market": "January NFP > 0",
        "our_probability": nfp_pred["probability"],
        "market_implied": 0.97,  # 100% - 3%
        "edge": nfp_pred["probability"] - 0.97,
        "confidence": nfp_pred["confidence"],
        "reasoning": nfp_pred["reasoning"],
    })

    # Unemployment rate
    unemp_pred = predictor.predict_threshold_probability(
        series_id="UNRATE",
        reference_period=datetime(2026, 1, 1),
        threshold=4.2,
        direction="below",
        prediction_date=now,
    )
    results.append({
        "market": "January Unemployment ≤4.2%",
        "our_probability": unemp_pred["probability"],
        "market_implied": 0.06,  # From screenshot
        "edge": unemp_pred["probability"] - 0.06,
        "confidence": unemp_pred["confidence"],
        "reasoning": unemp_pred["reasoning"],
    })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Testing Initial Release Predictor...")

    predictor = InitialReleasePredictor()

    # Test prediction
    result = predictor.predict_threshold_probability(
        series_id="PAYEMS",
        reference_period=datetime(2026, 2, 1),
        threshold=150000,  # 150k jobs
        direction="above",
    )

    print(f"\nPrediction: P(Feb 2026 NFP > 150k)")
    print(f"  Probability: {result['probability']:.1%}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Reasoning: {result['reasoning']}")

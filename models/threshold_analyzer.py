"""
Threshold Analyzer for Revision-Based Trading Edge

This module identifies trading opportunities where our revision predictions
can give edge on threshold-based markets.

KEY INSIGHT:
If a market asks "Will Q4 GDP be > 2.0%?" and:
- The initial release comes in at 1.9%
- Our model says 95% chance of upward revision
- Then the FINAL value likely exceeds 2.0%

This creates edge if:
1. Markets settle on FINAL values (rare but some do)
2. You can trade related markets after initial release
3. You understand the revision dynamics better than the market
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import logger, RAW_DATA_DIR, PROCESSED_DATA_DIR


class ThresholdAnalyzer:
    """
    Analyzes how revisions affect threshold-crossing probabilities.

    Given:
    - An initial release value
    - A threshold (e.g., 2.0% GDP growth)
    - Our revision prediction

    Calculates:
    - P(final value crosses threshold | initial value, revision prediction)
    """

    def __init__(self):
        self.revision_data = self._load_revision_data()
        self.revision_stats = self._compute_revision_statistics()

    def _load_revision_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical revision data for all indicators."""
        data = {}
        for parquet_file in RAW_DATA_DIR.glob("*_revisions.parquet"):
            series_id = parquet_file.stem.replace("_revisions", "")
            data[series_id] = pd.read_parquet(parquet_file)
        return data

    def _compute_revision_statistics(self) -> Dict[str, Dict]:
        """Compute revision statistics for each indicator."""
        stats = {}

        for series_id, df in self.revision_data.items():
            if df.empty:
                continue

            # Compute revision distribution
            revisions = df["revision"].dropna()
            revision_pct = df["revision_pct"].dropna()

            stats[series_id] = {
                "n_observations": len(df),
                "revision_mean": revisions.mean(),
                "revision_std": revisions.std(),
                "revision_pct_mean": revision_pct.mean(),
                "revision_pct_std": revision_pct.std(),
                "p_revised_up": df["revised_up"].mean(),
                "p_revised_down": df["revised_down"].mean(),
                "p_unchanged": 1 - df["revised_up"].mean() - df["revised_down"].mean(),
                # Percentiles for revision magnitude
                "revision_p10": revisions.quantile(0.10),
                "revision_p25": revisions.quantile(0.25),
                "revision_p50": revisions.quantile(0.50),
                "revision_p75": revisions.quantile(0.75),
                "revision_p90": revisions.quantile(0.90),
            }

        return stats

    def probability_crosses_threshold(
        self,
        series_id: str,
        initial_value: float,
        threshold: float,
        direction: str = "above",  # Does initial need to cross "above" or "below" threshold?
        revision_probability_up: Optional[float] = None,
    ) -> Dict:
        """
        Calculate probability that final value crosses threshold.

        Args:
            series_id: The indicator (e.g., "GDPC1")
            initial_value: The initial release value
            threshold: The threshold to cross
            direction: "above" if we want final > threshold, "below" if final < threshold
            revision_probability_up: Our model's P(revised up), or None to use historical

        Returns:
            Dict with probability and analysis
        """
        if series_id not in self.revision_stats:
            return {"probability": 0.5, "error": "Unknown indicator"}

        stats = self.revision_stats[series_id]
        df = self.revision_data[series_id]

        # Use model prediction or historical base rate
        if revision_probability_up is None:
            p_up = stats["p_revised_up"]
        else:
            p_up = revision_probability_up

        p_down = 1 - p_up  # Simplified: ignore unchanged

        # Distance from threshold
        distance = initial_value - threshold
        distance_pct = (distance / abs(threshold)) * 100 if threshold != 0 else 0

        # Already crossed?
        if direction == "above":
            already_crossed = initial_value > threshold
        else:
            already_crossed = initial_value < threshold

        if already_crossed:
            # Initial already crossed - what's probability it stays crossed?
            # Need to check if revision could push it back
            if direction == "above":
                # Initial > threshold, need to stay above
                # Fails if revised down enough to go below threshold
                revision_needed_to_fail = initial_value - threshold

                # P(revision < -revision_needed_to_fail)
                # Use historical distribution
                p_fail = (df["revision"] < -revision_needed_to_fail).mean()
                probability = 1 - p_fail
            else:
                # Initial < threshold, need to stay below
                revision_needed_to_fail = threshold - initial_value
                p_fail = (df["revision"] > revision_needed_to_fail).mean()
                probability = 1 - p_fail
        else:
            # Initial hasn't crossed - what's probability revision pushes it over?
            if direction == "above":
                # Initial < threshold, need upward revision
                revision_needed = threshold - initial_value
                # P(revision > revision_needed)
                probability = (df["revision"] > revision_needed).mean()

                # Adjust by our model's revision prediction
                # If model says high P(up), increase probability
                historical_p_up = stats["p_revised_up"]
                if revision_probability_up is not None:
                    adjustment = (revision_probability_up - historical_p_up) * 0.5
                    probability = np.clip(probability + adjustment, 0.01, 0.99)
            else:
                # Initial > threshold, need downward revision
                revision_needed = initial_value - threshold
                probability = (df["revision"] < -revision_needed).mean()

                historical_p_down = stats["p_revised_down"]
                if revision_probability_up is not None:
                    # Lower P(up) means higher P(down)
                    p_down_model = 1 - revision_probability_up
                    adjustment = (p_down_model - historical_p_down) * 0.5
                    probability = np.clip(probability + adjustment, 0.01, 0.99)

        # Build analysis
        analysis = {
            "probability": probability,
            "initial_value": initial_value,
            "threshold": threshold,
            "distance": distance,
            "distance_pct": distance_pct,
            "already_crossed": already_crossed,
            "direction": direction,
            "historical_p_up": stats["p_revised_up"],
            "historical_p_down": stats["p_revised_down"],
            "model_p_up": revision_probability_up,
            "revision_mean": stats["revision_mean"],
            "revision_std": stats["revision_std"],
        }

        return analysis

    def find_edge_opportunities(
        self,
        our_predictions: Dict[str, float],  # series_id -> P(revised up)
    ) -> pd.DataFrame:
        """
        Find opportunities where our revision predictions create edge.

        Looks for cases where:
        1. Initial release is close to a threshold
        2. Our revision prediction differs significantly from historical
        3. This creates a trading opportunity
        """
        opportunities = []

        for series_id, p_up in our_predictions.items():
            if series_id not in self.revision_stats:
                continue

            stats = self.revision_stats[series_id]
            df = self.revision_data[series_id]

            # Get recent initial value
            recent = df.sort_values("reference_date").iloc[-1]
            initial_value = recent["initial_value"]

            # Check various threshold distances
            # These would be actual market thresholds in production
            test_thresholds = [
                initial_value * 0.99,  # 1% below
                initial_value * 1.01,  # 1% above
                initial_value * 0.98,  # 2% below
                initial_value * 1.02,  # 2% above
            ]

            for threshold in test_thresholds:
                for direction in ["above", "below"]:
                    result = self.probability_crosses_threshold(
                        series_id=series_id,
                        initial_value=initial_value,
                        threshold=threshold,
                        direction=direction,
                        revision_probability_up=p_up,
                    )

                    # Calculate edge vs historical
                    historical_result = self.probability_crosses_threshold(
                        series_id=series_id,
                        initial_value=initial_value,
                        threshold=threshold,
                        direction=direction,
                        revision_probability_up=None,  # Use historical
                    )

                    edge = result["probability"] - historical_result["probability"]

                    if abs(edge) > 0.05:  # 5% edge threshold
                        opportunities.append({
                            "series_id": series_id,
                            "initial_value": initial_value,
                            "threshold": threshold,
                            "direction": direction,
                            "our_probability": result["probability"],
                            "historical_probability": historical_result["probability"],
                            "edge": edge,
                            "model_p_up": p_up,
                            "historical_p_up": stats["p_revised_up"],
                        })

        return pd.DataFrame(opportunities).sort_values("edge", ascending=False)


def analyze_current_markets():
    """
    Analyze current Polymarket markets using our revision predictions.
    """
    analyzer = ThresholdAnalyzer()

    print("=" * 70)
    print("THRESHOLD ANALYZER - REVISION-BASED TRADING EDGE")
    print("=" * 70)

    # Print revision statistics
    print("\n📊 REVISION STATISTICS BY INDICATOR:")
    print("-" * 70)

    for series_id, stats in analyzer.revision_stats.items():
        print(f"\n{series_id}:")
        print(f"  Historical P(revise up): {stats['p_revised_up']:.1%}")
        print(f"  Historical P(revise down): {stats['p_revised_down']:.1%}")
        print(f"  Mean revision: {stats['revision_mean']:.2f}")
        print(f"  Revision std: {stats['revision_std']:.2f}")
        print(f"  Revision range (10th-90th): [{stats['revision_p10']:.2f}, {stats['revision_p90']:.2f}]")

    # Load our model predictions
    print("\n\n📈 OUR MODEL'S CURRENT PREDICTIONS:")
    print("-" * 70)

    # These are from our latest model run
    our_predictions = {
        "GDPC1": 0.955,    # 95.5% up
        "PAYEMS": 0.695,   # 69.5% up
        "ICSA": 0.538,     # 53.8% up
        "HOUST": 0.450,    # 45.0% up
        "DGORDER": 0.400,  # 40.0% up
        "INDPRO": 0.097,   # 9.7% up
        "RSXFS": 0.071,    # 7.1% up
    }

    for series_id, p_up in our_predictions.items():
        if series_id in analyzer.revision_stats:
            hist = analyzer.revision_stats[series_id]["p_revised_up"]
            diff = p_up - hist
            print(f"{series_id:10s}: Model={p_up:.1%}, Historical={hist:.1%}, Diff={diff:+.1%}")

    # Find edge opportunities
    print("\n\n🎯 EDGE OPPORTUNITIES:")
    print("-" * 70)

    opportunities = analyzer.find_edge_opportunities(our_predictions)
    if not opportunities.empty:
        print(opportunities.head(10).to_string(index=False))
    else:
        print("No significant edge opportunities found.")

    # Specific market analysis
    print("\n\n📊 SPECIFIC MARKET ANALYSIS:")
    print("-" * 70)

    # GDP example
    # If Q4 GDP initial comes in at 1.9%, what's P(final > 2.0%)?
    gdp_analysis = analyzer.probability_crosses_threshold(
        series_id="GDPC1",
        initial_value=1.9,  # Hypothetical initial
        threshold=2.0,
        direction="above",
        revision_probability_up=0.955,  # Our model
    )
    print(f"\nGDP Example: Initial=1.9%, Threshold=2.0%")
    print(f"  P(final > 2.0%): {gdp_analysis['probability']:.1%}")
    print(f"  Our model P(revise up): {gdp_analysis['model_p_up']:.1%}")

    # Jobs example
    jobs_analysis = analyzer.probability_crosses_threshold(
        series_id="PAYEMS",
        initial_value=180000,  # Hypothetical: 180k jobs
        threshold=200000,      # Threshold: 200k
        direction="above",
        revision_probability_up=0.695,
    )
    print(f"\nNFP Example: Initial=180k, Threshold=200k")
    print(f"  P(final > 200k): {jobs_analysis['probability']:.1%}")
    print(f"  Our model P(revise up): {jobs_analysis['model_p_up']:.1%}")

    return analyzer


if __name__ == "__main__":
    analyze_current_markets()

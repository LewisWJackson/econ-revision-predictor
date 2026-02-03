"""
Feature Engineering Pipeline for Economic Revision Prediction

This module computes features for predicting whether economic indicators
will be revised upward from their initial release.

TEMPORAL INTEGRITY IS PARAMOUNT
===============================
Every feature computed here must satisfy this constraint:
    For a prediction made at time T about indicator X,
    we can ONLY use information available at time T.

This means:
- Historical revision patterns use ONLY past revisions (already finalized)
- Cross-indicator signals use ONLY past data
- Economic regime features use real-time indicators, not NBER dates

COMMON LEAKAGE VECTORS (and how we prevent them):
------------------------------------------------
1. Using "final" values before they're known
   -> We compute historical bias using only revisions that have completed

2. Using future indicator releases to predict current revisions
   -> All cross-indicator features are lagged by at least one period

3. Using recession indicators dated after the fact
   -> We use real-time indicators (yield curve, Sahm rule) only

4. Train/test contamination through random splits
   -> All splits are temporal, never random
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    INDICATORS,
    ROLLING_WINDOWS,
    MIN_OBSERVATIONS_FOR_ROLLING,
    LEAKAGE_BUFFER_DAYS,
    STRICT_LEAKAGE_CHECKS,
    LOG_LEAKAGE_CHECKS,
    logger,
)

# =============================================================================
# LEAKAGE VERIFICATION
# =============================================================================

@dataclass
class LeakageCheckResult:
    """Result of a leakage verification check."""
    passed: bool
    check_name: str
    details: str
    prediction_date: datetime
    data_dates_used: List[datetime]


def verify_no_future_data(
    prediction_date: datetime,
    data_dates: List[datetime],
    check_name: str,
) -> LeakageCheckResult:
    """
    Verify that no data from after prediction_date is used.

    This is the core leakage check. Every feature computation should
    call this to verify temporal integrity.
    """
    future_dates = [d for d in data_dates if d > prediction_date]

    result = LeakageCheckResult(
        passed=len(future_dates) == 0,
        check_name=check_name,
        details=f"Found {len(future_dates)} future dates" if future_dates else "OK",
        prediction_date=prediction_date,
        data_dates_used=data_dates,
    )

    if LOG_LEAKAGE_CHECKS:
        if result.passed:
            logger.debug(f"Leakage check '{check_name}' PASSED for {prediction_date}")
        else:
            logger.error(
                f"Leakage check '{check_name}' FAILED for {prediction_date}: "
                f"Future dates used: {future_dates[:5]}..."
            )

    if STRICT_LEAKAGE_CHECKS and not result.passed:
        raise ValueError(
            f"LEAKAGE DETECTED in '{check_name}': "
            f"Prediction date {prediction_date}, but used data from {future_dates[:3]}"
        )

    return result


# =============================================================================
# SINGLE-INDICATOR HISTORICAL FEATURES
# =============================================================================

def compute_historical_revision_bias(
    revision_history: pd.DataFrame,
    as_of_date: datetime,
    window_periods: int,
) -> Dict[str, float]:
    """
    Compute historical revision bias from past revisions.

    For a prediction made on as_of_date, this looks at revisions that
    COMPLETED before as_of_date and computes statistics.

    LEAKAGE SAFE: Only uses revisions where final_date < as_of_date

    Args:
        revision_history: DataFrame with columns [reference_date, initial_date,
                         final_date, revised_up, revision_pct, ...]
        as_of_date: The prediction time
        window_periods: Number of past periods to include

    Returns:
        Dictionary of features
    """
    # CRITICAL: Filter to revisions that completed BEFORE prediction time
    # A revision is "complete" when we've seen the final value
    completed = revision_history[
        revision_history["final_date"] < as_of_date - timedelta(days=LEAKAGE_BUFFER_DAYS)
    ].copy()

    # Sort by reference date and take most recent window_periods
    completed = completed.sort_values("reference_date", ascending=False)
    completed = completed.head(window_periods)

    # Leakage verification
    if not completed.empty:
        verify_no_future_data(
            prediction_date=as_of_date,
            data_dates=completed["final_date"].tolist(),
            check_name=f"historical_revision_bias_window_{window_periods}",
        )

    if len(completed) < MIN_OBSERVATIONS_FOR_ROLLING:
        return {
            f"revision_up_rate_w{window_periods}": np.nan,
            f"revision_magnitude_mean_w{window_periods}": np.nan,
            f"revision_magnitude_std_w{window_periods}": np.nan,
            f"revision_pct_mean_w{window_periods}": np.nan,
            f"n_observations_w{window_periods}": len(completed),
        }

    return {
        f"revision_up_rate_w{window_periods}": completed["revised_up"].mean(),
        f"revision_magnitude_mean_w{window_periods}": completed["revision"].abs().mean(),
        f"revision_magnitude_std_w{window_periods}": completed["revision"].abs().std(),
        f"revision_pct_mean_w{window_periods}": completed["revision_pct"].mean(),
        f"n_observations_w{window_periods}": len(completed),
    }


def compute_regime_specific_bias(
    revision_history: pd.DataFrame,
    regime_indicator: pd.DataFrame,
    as_of_date: datetime,
    window_periods: int,
) -> Dict[str, float]:
    """
    Compute revision bias separately for expansion vs contraction regimes.

    Uses a real-time regime indicator (e.g., yield curve or Sahm rule)
    that was available at each historical point.

    LEAKAGE SAFE: Only uses regime data available at each historical point,
    not NBER recession dates which are announced with significant lag.
    """
    # Filter completed revisions
    completed = revision_history[
        revision_history["final_date"] < as_of_date - timedelta(days=LEAKAGE_BUFFER_DAYS)
    ].copy()

    if completed.empty:
        return {
            f"revision_up_rate_expansion_w{window_periods}": np.nan,
            f"revision_up_rate_contraction_w{window_periods}": np.nan,
        }

    # Merge with regime indicator
    # The regime indicator value should be from the initial_date (when prediction would be made)
    completed = completed.sort_values("reference_date", ascending=False).head(window_periods)

    # For simplicity, merge on the nearest available regime date
    if regime_indicator.empty:
        return {
            f"revision_up_rate_expansion_w{window_periods}": np.nan,
            f"revision_up_rate_contraction_w{window_periods}": np.nan,
        }

    # Merge regime data (as of initial_date for each observation)
    completed = pd.merge_asof(
        completed.sort_values("initial_date"),
        regime_indicator.sort_values("date")[["date", "is_contraction"]],
        left_on="initial_date",
        right_on="date",
        direction="backward",
    )

    expansion = completed[completed["is_contraction"] == 0]
    contraction = completed[completed["is_contraction"] == 1]

    return {
        f"revision_up_rate_expansion_w{window_periods}": (
            expansion["revised_up"].mean() if len(expansion) >= 2 else np.nan
        ),
        f"revision_up_rate_contraction_w{window_periods}": (
            contraction["revised_up"].mean() if len(contraction) >= 2 else np.nan
        ),
        f"n_expansion_obs_w{window_periods}": len(expansion),
        f"n_contraction_obs_w{window_periods}": len(contraction),
    }


def compute_seasonal_patterns(
    revision_history: pd.DataFrame,
    as_of_date: datetime,
    reference_date: datetime,
) -> Dict[str, float]:
    """
    Compute seasonal/calendar features for revision prediction.

    Some indicators have predictable revision patterns by quarter or month.
    E.g., Q4 GDP estimates may be revised differently than Q1.

    LEAKAGE SAFE: Uses only historical data and calendar information
    from the reference period itself.
    """
    # Filter completed revisions before prediction time
    completed = revision_history[
        revision_history["final_date"] < as_of_date - timedelta(days=LEAKAGE_BUFFER_DAYS)
    ].copy()

    # Calendar features for the current prediction
    month = reference_date.month
    quarter = (reference_date.month - 1) // 3 + 1
    day_of_week = reference_date.weekday()

    features = {
        "month": month,
        "quarter": quarter,
        "is_q4": int(quarter == 4),
        "is_q1": int(quarter == 1),
        "is_year_end": int(month in [11, 12, 1]),
    }

    # Historical revision rate by quarter
    if not completed.empty:
        completed["ref_quarter"] = completed["reference_date"].dt.quarter

        for q in [1, 2, 3, 4]:
            q_data = completed[completed["ref_quarter"] == q]
            features[f"historical_revision_up_rate_q{q}"] = (
                q_data["revised_up"].mean() if len(q_data) >= 3 else np.nan
            )
    else:
        for q in [1, 2, 3, 4]:
            features[f"historical_revision_up_rate_q{q}"] = np.nan

    return features


# =============================================================================
# CROSS-INDICATOR FEATURES
# =============================================================================

def compute_cross_indicator_signals(
    target_series_id: str,
    target_initial_date: datetime,
    all_revisions: Dict[str, pd.DataFrame],
    lag_periods: int = 1,
) -> Dict[str, float]:
    """
    Compute signals from other indicators' revisions.

    The hypothesis is that revisions in correlated indicators may
    predict revisions in the target indicator.

    LEAKAGE SAFE: We use LAGGED revisions from other indicators.
    The lag ensures we only use information available before the
    target's initial release.

    Args:
        target_series_id: The indicator we're predicting
        target_initial_date: When the target was initially released
        all_revisions: Dict mapping series_id -> revision DataFrame
        lag_periods: How many periods to lag (default 1)

    Returns:
        Cross-indicator features
    """
    features = {}

    for series_id, rev_df in all_revisions.items():
        if series_id == target_series_id:
            continue

        if rev_df.empty:
            features[f"{series_id}_lag{lag_periods}_revised_up"] = np.nan
            features[f"{series_id}_lag{lag_periods}_revision_pct"] = np.nan
            continue

        # Get revisions that completed BEFORE the target's initial release
        # with additional lag for safety
        cutoff = target_initial_date - timedelta(days=LEAKAGE_BUFFER_DAYS)
        available = rev_df[rev_df["final_date"] < cutoff].copy()

        if available.empty:
            features[f"{series_id}_lag{lag_periods}_revised_up"] = np.nan
            features[f"{series_id}_lag{lag_periods}_revision_pct"] = np.nan
            continue

        # Get the most recent completed revision (lagged)
        available = available.sort_values("reference_date", ascending=False)

        if len(available) < lag_periods:
            features[f"{series_id}_lag{lag_periods}_revised_up"] = np.nan
            features[f"{series_id}_lag{lag_periods}_revision_pct"] = np.nan
            continue

        lagged_obs = available.iloc[lag_periods - 1]

        # Verify no leakage
        verify_no_future_data(
            prediction_date=target_initial_date,
            data_dates=[lagged_obs["final_date"]],
            check_name=f"cross_indicator_{series_id}_lag{lag_periods}",
        )

        features[f"{series_id}_lag{lag_periods}_revised_up"] = lagged_obs["revised_up"]
        features[f"{series_id}_lag{lag_periods}_revision_pct"] = lagged_obs["revision_pct"]

    return features


def compute_revision_correlation_features(
    target_series_id: str,
    as_of_date: datetime,
    all_revisions: Dict[str, pd.DataFrame],
    window_periods: int = 12,
) -> Dict[str, float]:
    """
    Compute rolling correlation of revision directions between indicators.

    If GDP revisions have been correlated with employment revisions,
    this might predict future patterns.

    LEAKAGE SAFE: Correlation computed only on completed historical revisions.
    """
    features = {}

    target_df = all_revisions.get(target_series_id)
    if target_df is None or target_df.empty:
        return features

    # Get target's completed revisions
    target_completed = target_df[
        target_df["final_date"] < as_of_date - timedelta(days=LEAKAGE_BUFFER_DAYS)
    ].sort_values("reference_date").tail(window_periods)

    if len(target_completed) < MIN_OBSERVATIONS_FOR_ROLLING:
        return features

    for series_id, rev_df in all_revisions.items():
        if series_id == target_series_id:
            continue

        if rev_df.empty:
            features[f"corr_{series_id}_w{window_periods}"] = np.nan
            continue

        # Get other indicator's completed revisions
        other_completed = rev_df[
            rev_df["final_date"] < as_of_date - timedelta(days=LEAKAGE_BUFFER_DAYS)
        ].sort_values("reference_date")

        if len(other_completed) < MIN_OBSERVATIONS_FOR_ROLLING:
            features[f"corr_{series_id}_w{window_periods}"] = np.nan
            continue

        # Merge on overlapping reference dates
        merged = pd.merge(
            target_completed[["reference_date", "revised_up"]],
            other_completed[["reference_date", "revised_up"]],
            on="reference_date",
            suffixes=("_target", "_other"),
        )

        if len(merged) < MIN_OBSERVATIONS_FOR_ROLLING:
            features[f"corr_{series_id}_w{window_periods}"] = np.nan
            continue

        # Compute correlation
        corr = merged["revised_up_target"].corr(merged["revised_up_other"])
        features[f"corr_{series_id}_w{window_periods}"] = corr

    return features


# =============================================================================
# ECONOMIC REGIME FEATURES (REAL-TIME ONLY)
# =============================================================================

def compute_realtime_regime_features(
    as_of_date: datetime,
    macro_data: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Compute economic regime features using ONLY real-time indicators.

    We deliberately avoid NBER recession dates because they are announced
    with significant lag (sometimes 6-12 months after the recession starts).

    Instead we use:
    - Yield curve slope (10Y - 2Y Treasury spread)
    - Sahm Rule indicator (real-time recession signal)
    - VIX level (market uncertainty)

    LEAKAGE SAFE: All inputs are real-time market data.

    Args:
        as_of_date: The prediction date
        macro_data: DataFrame with real-time macro indicators
                   Columns: date, yield_curve_spread, sahm_indicator, vix

    Returns:
        Regime-related features
    """
    if macro_data is None or macro_data.empty:
        return {
            "yield_curve_spread": np.nan,
            "yield_curve_inverted": np.nan,
            "sahm_indicator": np.nan,
            "sahm_recession_signal": np.nan,
            "vix_level": np.nan,
            "vix_elevated": np.nan,
        }

    # Get most recent data available as of prediction date
    available = macro_data[macro_data["date"] <= as_of_date].copy()

    if available.empty:
        return {
            "yield_curve_spread": np.nan,
            "yield_curve_inverted": np.nan,
            "sahm_indicator": np.nan,
            "sahm_recession_signal": np.nan,
            "vix_level": np.nan,
            "vix_elevated": np.nan,
        }

    latest = available.sort_values("date").iloc[-1]

    # Verify no leakage
    verify_no_future_data(
        prediction_date=as_of_date,
        data_dates=[latest["date"]],
        check_name="realtime_regime_features",
    )

    features = {}

    # Yield curve
    if "yield_curve_spread" in latest:
        spread = latest["yield_curve_spread"]
        features["yield_curve_spread"] = spread
        features["yield_curve_inverted"] = int(spread < 0)
    else:
        features["yield_curve_spread"] = np.nan
        features["yield_curve_inverted"] = np.nan

    # Sahm Rule (3-month moving average of unemployment rate rises 0.5pp+ above 12-month low)
    if "sahm_indicator" in latest:
        sahm = latest["sahm_indicator"]
        features["sahm_indicator"] = sahm
        features["sahm_recession_signal"] = int(sahm >= 0.5)
    else:
        features["sahm_indicator"] = np.nan
        features["sahm_recession_signal"] = np.nan

    # VIX
    if "vix" in latest:
        vix = latest["vix"]
        features["vix_level"] = vix
        features["vix_elevated"] = int(vix > 25)  # Elevated uncertainty threshold
    else:
        features["vix_level"] = np.nan
        features["vix_elevated"] = np.nan

    return features


# =============================================================================
# INITIAL RELEASE FEATURES
# =============================================================================

def compute_initial_release_features(
    initial_value: float,
    initial_date: datetime,
    historical_values: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute features based on the initial release value itself.

    - Distance from historical trend
    - Surprise vs naive forecast (previous period)
    - Magnitude relative to historical volatility

    LEAKAGE SAFE: Uses only the initial value and prior history.
    """
    features = {}

    if historical_values.empty:
        return {
            "initial_vs_prior": np.nan,
            "initial_vs_trend": np.nan,
            "initial_zscore": np.nan,
            "initial_surprise_magnitude": np.nan,
        }

    # Filter to data available before initial release
    prior = historical_values[
        historical_values["date"] < initial_date - timedelta(days=LEAKAGE_BUFFER_DAYS)
    ].sort_values("date")

    if prior.empty:
        return {
            "initial_vs_prior": np.nan,
            "initial_vs_trend": np.nan,
            "initial_zscore": np.nan,
            "initial_surprise_magnitude": np.nan,
        }

    verify_no_future_data(
        prediction_date=initial_date,
        data_dates=prior["date"].tolist(),
        check_name="initial_release_features",
    )

    # Most recent prior value
    prior_value = prior.iloc[-1]["value"]

    # Change from prior
    features["initial_vs_prior"] = initial_value - prior_value
    features["initial_pct_change"] = (
        (initial_value - prior_value) / abs(prior_value) * 100
        if prior_value != 0 else 0
    )

    # Historical statistics
    if len(prior) >= MIN_OBSERVATIONS_FOR_ROLLING:
        hist_mean = prior["value"].mean()
        hist_std = prior["value"].std()

        features["initial_vs_trend"] = initial_value - hist_mean
        features["initial_zscore"] = (
            (initial_value - hist_mean) / hist_std if hist_std > 0 else 0
        )

        # Typical change magnitude
        changes = prior["value"].diff().dropna()
        if len(changes) > 0:
            typical_change = changes.abs().mean()
            actual_change = abs(initial_value - prior_value)
            features["initial_surprise_magnitude"] = (
                actual_change / typical_change if typical_change > 0 else 1
            )
        else:
            features["initial_surprise_magnitude"] = np.nan
    else:
        features["initial_vs_trend"] = np.nan
        features["initial_zscore"] = np.nan
        features["initial_surprise_magnitude"] = np.nan

    return features


def compute_naive_forecast_surprise(
    initial_value: float,
    revision_history: pd.DataFrame,
    as_of_date: datetime,
    window_periods: int = 4,
) -> Dict[str, float]:
    """
    Compute surprise vs naive forecast as proxy for consensus surprise.

    RATIONALE:
    When we don't have consensus estimates, we can compute a "naive forecast"
    based on recent trends. The surprise (initial - naive forecast) may predict
    whether the value will be revised:
    - If initial >> naive forecast (big positive surprise), may revert down
    - If initial << naive forecast (big negative surprise), may revert up

    This is a proxy for the consensus surprise feature that would use
    actual analyst estimates if available.

    LEAKAGE SAFE: Uses only historical data before prediction time.
    """
    features = {}

    # Get completed revisions before this prediction
    completed = revision_history[
        revision_history["final_date"] < as_of_date - timedelta(days=LEAKAGE_BUFFER_DAYS)
    ].sort_values("reference_date")

    if len(completed) < window_periods:
        return {
            "naive_forecast": np.nan,
            "naive_forecast_surprise": np.nan,
            "naive_forecast_surprise_pct": np.nan,
            "surprise_vs_revision_corr": np.nan,
        }

    # Naive forecast: use the most recent final values to predict
    recent = completed.tail(window_periods)
    recent_finals = recent["final_value"].values

    # Simple forecast: last value (random walk)
    naive_forecast_rw = recent_finals[-1]

    # Trend forecast: last value + average change
    if len(recent_finals) >= 2:
        avg_change = np.diff(recent_finals).mean()
        naive_forecast_trend = recent_finals[-1] + avg_change
    else:
        naive_forecast_trend = naive_forecast_rw

    # Use average of random walk and trend
    naive_forecast = (naive_forecast_rw + naive_forecast_trend) / 2

    # Surprise
    surprise = initial_value - naive_forecast
    surprise_pct = (surprise / abs(naive_forecast) * 100) if naive_forecast != 0 else 0

    features["naive_forecast"] = naive_forecast
    features["naive_forecast_surprise"] = surprise
    features["naive_forecast_surprise_pct"] = surprise_pct

    # Historical correlation: do big surprises predict revisions?
    # This helps the model learn if surprises tend to revert
    if len(completed) >= 8:
        # For past observations, compute surprise vs actual revision
        past_surprises = []
        past_revisions = []

        for i in range(window_periods, len(completed)):
            past_window = completed.iloc[i-window_periods:i]
            past_finals = past_window["final_value"].values

            if len(past_finals) < 2:
                continue

            past_naive = past_finals[-1]
            past_initial = completed.iloc[i]["initial_value"]
            past_revision = completed.iloc[i]["revision"]

            past_surprises.append(past_initial - past_naive)
            past_revisions.append(past_revision)

        if len(past_surprises) >= 4:
            # Negative correlation means surprises revert (what we expect)
            corr = np.corrcoef(past_surprises, past_revisions)[0, 1]
            features["surprise_vs_revision_corr"] = corr if not np.isnan(corr) else 0
        else:
            features["surprise_vs_revision_corr"] = np.nan
    else:
        features["surprise_vs_revision_corr"] = np.nan

    return features


def compute_indicator_specific_features(
    series_id: str,
    reference_date: datetime,
    initial_date: datetime,
    revision_history: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute indicator-specific features.

    Different indicators have different revision patterns:
    - GDP: Tends to revise up during expansions
    - PAYEMS: Large benchmark revisions annually
    - ICSA: Weekly noise, less systematic revision
    - INDPRO: Industrial production has 5-month revision cycle

    We encode these structural differences as features.
    """
    features = {}

    # Indicator type encoding (one-hot)
    indicator_types = ["GDPC1", "PAYEMS", "ICSA", "INDPRO", "DGORDER", "RSXFS", "HOUST"]
    for ind_type in indicator_types:
        features[f"is_{ind_type}"] = 1 if series_id == ind_type else 0

    # Indicator category
    categories = {
        "GDPC1": "gdp",
        "PAYEMS": "employment",
        "ICSA": "employment",
        "INDPRO": "production",
        "DGORDER": "manufacturing",
        "RSXFS": "consumption",
        "HOUST": "housing",
    }
    cat = categories.get(series_id, "other")
    features["category_employment"] = 1 if cat == "employment" else 0
    features["category_production"] = 1 if cat in ["production", "manufacturing"] else 0
    features["category_demand"] = 1 if cat in ["consumption", "housing", "gdp"] else 0

    # Indicator-specific base rate (historical revision up rate for THIS indicator)
    if not revision_history.empty:
        completed = revision_history[revision_history["final_date"] < initial_date]
        if len(completed) >= 10:
            features["indicator_base_rate"] = completed["revised_up"].mean()
        else:
            features["indicator_base_rate"] = 0.5  # Uninformative prior
    else:
        features["indicator_base_rate"] = 0.5

    # Days since quarter/year end (timing relative to fiscal calendar)
    features["days_from_quarter_end"] = (reference_date.day +
        (reference_date.month - 1) % 3 * 30)  # Rough approximation
    features["is_quarter_end_month"] = 1 if reference_date.month in [3, 6, 9, 12] else 0

    return features


# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================

def compute_all_features(
    series_id: str,
    reference_date: datetime,
    initial_value: float,
    initial_date: datetime,
    all_revisions: Dict[str, pd.DataFrame],
    historical_values: Optional[pd.DataFrame] = None,
    macro_data: Optional[pd.DataFrame] = None,
    regime_indicator: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Compute all features for a single prediction point.

    This is the main entry point for feature engineering. It computes
    ALL features needed for prediction, ensuring temporal integrity.

    The prediction is: "Will {series_id} for {reference_date} be revised up?"
    The prediction is made at: {initial_date} (when initial value released)

    LEAKAGE VERIFICATION:
    - All historical bias features use only completed revisions
    - Cross-indicator features are lagged
    - Regime features use only real-time indicators
    - All date filtering includes LEAKAGE_BUFFER_DAYS safety margin

    Args:
        series_id: Indicator being predicted
        reference_date: The period being measured (e.g., 2024-Q1)
        initial_value: First released value
        initial_date: When initial value was published (PREDICTION TIME)
        all_revisions: Historical revisions for all indicators
        historical_values: Time series of initial values for this indicator
        macro_data: Real-time macro indicators (yield curve, VIX, etc.)
        regime_indicator: Real-time expansion/contraction indicator

    Returns:
        Dictionary of all computed features
    """
    features = {
        "series_id": series_id,
        "reference_date": reference_date,
        "initial_date": initial_date,
        "initial_value": initial_value,
    }

    # Get this indicator's revision history
    revision_history = all_revisions.get(series_id, pd.DataFrame())

    # 1. Historical revision bias features (multiple windows)
    for window in ROLLING_WINDOWS:
        bias_features = compute_historical_revision_bias(
            revision_history=revision_history,
            as_of_date=initial_date,
            window_periods=window,
        )
        features.update(bias_features)

    # 2. Regime-specific revision patterns
    for window in [8, 20]:  # Longer windows for regime analysis
        regime_features = compute_regime_specific_bias(
            revision_history=revision_history,
            regime_indicator=regime_indicator if regime_indicator is not None else pd.DataFrame(),
            as_of_date=initial_date,
            window_periods=window,
        )
        features.update(regime_features)

    # 3. Seasonal/calendar features
    seasonal_features = compute_seasonal_patterns(
        revision_history=revision_history,
        as_of_date=initial_date,
        reference_date=reference_date,
    )
    features.update(seasonal_features)

    # 4. Cross-indicator signals
    cross_features = compute_cross_indicator_signals(
        target_series_id=series_id,
        target_initial_date=initial_date,
        all_revisions=all_revisions,
        lag_periods=1,
    )
    features.update(cross_features)

    # 5. Revision correlations
    for window in [8, 12]:
        corr_features = compute_revision_correlation_features(
            target_series_id=series_id,
            as_of_date=initial_date,
            all_revisions=all_revisions,
            window_periods=window,
        )
        features.update(corr_features)

    # 6. Real-time regime features
    regime_features = compute_realtime_regime_features(
        as_of_date=initial_date,
        macro_data=macro_data,
    )
    features.update(regime_features)

    # 7. Initial release features
    if historical_values is not None:
        release_features = compute_initial_release_features(
            initial_value=initial_value,
            initial_date=initial_date,
            historical_values=historical_values,
        )
        features.update(release_features)

    # 8. Naive forecast surprise (proxy for consensus surprise)
    surprise_features = compute_naive_forecast_surprise(
        initial_value=initial_value,
        revision_history=revision_history,
        as_of_date=initial_date,
        window_periods=4,
    )
    features.update(surprise_features)

    # 9. Indicator-specific features
    indicator_features = compute_indicator_specific_features(
        series_id=series_id,
        reference_date=reference_date,
        initial_date=initial_date,
        revision_history=revision_history,
    )
    features.update(indicator_features)

    # Add audit timestamp
    features["_feature_computed_at"] = datetime.now().isoformat()
    features["_prediction_point"] = initial_date.isoformat()

    return features


def build_feature_matrix(
    all_revisions: Dict[str, pd.DataFrame],
    historical_values: Optional[Dict[str, pd.DataFrame]] = None,
    macro_data: Optional[pd.DataFrame] = None,
    regime_indicator: Optional[pd.DataFrame] = None,
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Build complete feature matrix for all indicators and time periods.

    This processes all revision observations and computes features for each,
    creating the training/evaluation dataset.

    Args:
        all_revisions: Dict mapping series_id -> revision DataFrame
        historical_values: Dict mapping series_id -> time series DataFrame
        macro_data: Real-time macro indicators
        regime_indicator: Real-time regime indicator
        min_date: Earliest initial_date to include
        max_date: Latest initial_date to include

    Returns:
        DataFrame with one row per prediction point, all features, and target
    """
    historical_values = historical_values or {}
    all_rows = []

    for series_id, rev_df in all_revisions.items():
        if rev_df.empty:
            continue

        logger.info(f"Computing features for {series_id} ({len(rev_df)} observations)")

        for _, obs in rev_df.iterrows():
            initial_date = obs["initial_date"]

            # Date filtering
            if min_date and initial_date < min_date:
                continue
            if max_date and initial_date > max_date:
                continue

            # Compute features
            features = compute_all_features(
                series_id=series_id,
                reference_date=obs["reference_date"],
                initial_value=obs["initial_value"],
                initial_date=initial_date,
                all_revisions=all_revisions,
                historical_values=historical_values.get(series_id),
                macro_data=macro_data,
                regime_indicator=regime_indicator,
            )

            # Add target variable
            features["target_revised_up"] = obs["revised_up"]
            features["target_revision_pct"] = obs["revision_pct"]

            all_rows.append(features)

    df = pd.DataFrame(all_rows)

    logger.info(f"Built feature matrix: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    # Simple test
    print("Feature engineering module loaded successfully.")
    print(f"Rolling windows configured: {ROLLING_WINDOWS}")
    print(f"Leakage buffer days: {LEAKAGE_BUFFER_DAYS}")
    print(f"Strict leakage checks: {STRICT_LEAKAGE_CHECKS}")

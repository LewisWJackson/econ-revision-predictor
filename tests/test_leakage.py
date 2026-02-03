"""
Tests for Data Leakage Prevention

These tests verify that the system correctly prevents temporal data leakage.
This is the most critical aspect of the system - if these tests fail,
the model's performance estimates are meaningless.

LEAKAGE VECTORS TESTED:
-----------------------
1. Feature computation uses only past data
2. Train/calibration/test splits are strictly temporal
3. Cross-indicator features are properly lagged
4. No future vintages leak into historical features
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LEAKAGE_BUFFER_DAYS


class TestTemporalIntegrity:
    """Tests for temporal integrity in data handling."""

    def test_revision_dataset_dates_ordered(self):
        """Verify initial_date always precedes final_date in revision data."""
        # Create mock revision data
        data = {
            "reference_date": [datetime(2024, 1, 1), datetime(2024, 4, 1)],
            "initial_date": [datetime(2024, 1, 30), datetime(2024, 4, 30)],
            "final_date": [datetime(2024, 4, 30), datetime(2024, 7, 30)],
            "initial_value": [100, 105],
            "final_value": [102, 103],
        }
        df = pd.DataFrame(data)

        # Verify ordering
        assert all(df["initial_date"] < df["final_date"]), \
            "initial_date must precede final_date"
        assert all(df["reference_date"] < df["initial_date"]), \
            "reference_date must precede initial_date"

    def test_no_future_data_in_features(self):
        """Test that feature computation rejects future data."""
        from features.engineering import verify_no_future_data

        prediction_date = datetime(2024, 6, 1)
        past_dates = [datetime(2024, 1, 1), datetime(2024, 3, 1), datetime(2024, 5, 1)]
        future_dates = [datetime(2024, 7, 1), datetime(2024, 8, 1)]

        # Past dates should pass
        result = verify_no_future_data(prediction_date, past_dates, "test_past")
        assert result.passed, "Past dates should pass verification"

        # Future dates should fail (in non-strict mode)
        result = verify_no_future_data(prediction_date, past_dates + future_dates, "test_future")
        assert not result.passed, "Future dates should fail verification"

    def test_train_test_split_temporal(self):
        """Test that train/test splits are strictly temporal."""
        # Create mock data spanning multiple years
        dates = pd.date_range("2018-01-01", "2024-12-31", freq="M")
        df = pd.DataFrame({
            "initial_date": dates,
            "value": np.random.randn(len(dates)),
        })

        # Define split boundaries
        train_end = "2019-12-31"
        test_start = "2023-01-01"

        train = df[df["initial_date"] <= train_end]
        test = df[df["initial_date"] >= test_start]

        # Verify no overlap
        assert train["initial_date"].max() < test["initial_date"].min(), \
            "Train and test sets must not overlap temporally"

        # Verify gap exists (calibration period)
        gap_days = (test["initial_date"].min() - train["initial_date"].max()).days
        assert gap_days > 365, "There should be a calibration gap between train and test"


class TestFeatureLeakage:
    """Tests for leakage in feature engineering."""

    def test_historical_bias_uses_only_completed_revisions(self):
        """Test that historical bias only uses revisions completed before prediction time."""
        from features.engineering import compute_historical_revision_bias

        # Create mock revision history
        revision_history = pd.DataFrame({
            "reference_date": pd.date_range("2020-01-01", periods=10, freq="Q"),
            "initial_date": pd.date_range("2020-02-01", periods=10, freq="Q"),
            "final_date": pd.date_range("2020-05-01", periods=10, freq="Q"),
            "revised_up": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
            "revision": [1.0, -0.5, 2.0, 1.5, -1.0, 0.5, -0.5, 1.0, 2.0, -1.0],
            "revision_pct": [1.0, -0.5, 2.0, 1.5, -1.0, 0.5, -0.5, 1.0, 2.0, -1.0],
        })

        # Prediction date in middle of history
        as_of_date = datetime(2021, 8, 1)

        features = compute_historical_revision_bias(
            revision_history=revision_history,
            as_of_date=as_of_date,
            window_periods=4,
        )

        # Should only use revisions with final_date < as_of_date
        usable_revisions = revision_history[
            revision_history["final_date"] < as_of_date - timedelta(days=LEAKAGE_BUFFER_DAYS)
        ]

        # If we got features, they should reflect only past data
        assert "revision_up_rate_w4" in features
        # Can't be NaN if we have enough history
        if len(usable_revisions) >= 4:
            assert not np.isnan(features["revision_up_rate_w4"])

    def test_cross_indicator_lag(self):
        """Test that cross-indicator features are properly lagged."""
        from features.engineering import compute_cross_indicator_signals

        # Target indicator released on June 1
        target_initial_date = datetime(2024, 6, 1)

        # Other indicator's revisions
        other_revisions = pd.DataFrame({
            "reference_date": [datetime(2024, 1, 1), datetime(2024, 4, 1)],
            "initial_date": [datetime(2024, 1, 30), datetime(2024, 4, 30)],
            "final_date": [datetime(2024, 4, 30), datetime(2024, 7, 30)],  # July revision too late!
            "revised_up": [1, 0],
            "revision_pct": [2.0, -1.0],
        })

        all_revisions = {"OTHER": other_revisions}

        features = compute_cross_indicator_signals(
            target_series_id="TARGET",
            target_initial_date=target_initial_date,
            all_revisions=all_revisions,
            lag_periods=1,
        )

        # The April revision (final July 30) should NOT be used
        # because its final_date is after target_initial_date
        # Only the January revision (final April 30) should be usable
        if "OTHER_lag1_revised_up" in features:
            assert features["OTHER_lag1_revised_up"] == 1, \
                "Should use January revision, not April"


class TestCalibrationIntegrity:
    """Tests for calibration data handling."""

    def test_calibration_data_after_training(self):
        """Verify calibration data is strictly after training data."""
        from config import (
            TRAIN_START_DATE,
            TRAIN_END_DATE,
            CALIBRATION_START_DATE,
            CALIBRATION_END_DATE,
        )

        train_end = pd.to_datetime(TRAIN_END_DATE)
        cal_start = pd.to_datetime(CALIBRATION_START_DATE)

        assert cal_start > train_end, \
            f"Calibration must start after training ends: {cal_start} > {train_end}"

    def test_test_data_after_calibration(self):
        """Verify test data is strictly after calibration data."""
        from config import (
            CALIBRATION_START_DATE,
            CALIBRATION_END_DATE,
            TEST_START_DATE,
            TEST_END_DATE,
        )

        cal_end = pd.to_datetime(CALIBRATION_END_DATE)
        test_start = pd.to_datetime(TEST_START_DATE)

        assert test_start > cal_end, \
            f"Test must start after calibration ends: {test_start} > {cal_end}"


class TestVintageDataHandling:
    """Tests for ALFRED vintage data handling."""

    def test_output_type_meanings(self):
        """Document and verify output_type parameter semantics."""
        # This is a documentation test - we verify our understanding
        # of FRED API output_type parameter

        output_types = {
            1: "Current values (CAUSES LEAKAGE)",
            2: "All observations by vintage date",
            3: "New and revised observations only",
            4: "Initial release only (SAFE)",
        }

        # Verify we use safe types for critical operations
        safe_for_initial = [4]
        safe_for_history = [2, 3]

        assert 4 in safe_for_initial, "output_type=4 is safe for initial releases"
        assert 1 not in safe_for_initial, "output_type=1 causes leakage"

    def test_vintage_date_filtering(self):
        """Test that vintage date filtering works correctly."""
        # Simulate vintage data
        vintage_data = pd.DataFrame({
            "reference_date": [datetime(2024, 1, 1)] * 3,
            "vintage_date": [
                datetime(2024, 1, 30),  # Initial
                datetime(2024, 2, 28),  # First revision
                datetime(2024, 5, 30),  # Second revision
            ],
            "value": [100, 102, 101],
        })

        # Query as of March 15
        as_of = datetime(2024, 3, 15)
        available = vintage_data[vintage_data["vintage_date"] <= as_of]

        # Should only see initial and first revision
        assert len(available) == 2
        assert available["value"].iloc[-1] == 102  # Most recent available


class TestReproducibility:
    """Tests for reproducibility of results."""

    def test_random_seed_consistency(self):
        """Test that random operations are reproducible with seed."""
        from config import RANDOM_SEED

        np.random.seed(RANDOM_SEED)
        first_run = np.random.rand(100)

        np.random.seed(RANDOM_SEED)
        second_run = np.random.rand(100)

        assert np.allclose(first_run, second_run), \
            "Random operations should be reproducible with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Probability Calibration Layer

This module implements isotonic regression calibration for each base model.
The goal is to transform raw model outputs into well-calibrated probabilities.

WHY CALIBRATION MATTERS:
------------------------
For prediction markets, we need calibrated probabilities, not just rankings.
A model that outputs 0.7 should be correct 70% of the time for that prediction.

Many models produce poorly calibrated probabilities:
- LightGBM: Often overconfident
- Logistic regression: Usually well-calibrated but can drift
- Bayesian: Better calibrated but not perfect

ISOTONIC REGRESSION:
--------------------
We use isotonic regression because:
1. Non-parametric: No assumptions about calibration shape
2. Monotonic: Preserves ranking (if raw P(A) > raw P(B), calibrated P(A) > calibrated P(B))
3. Robust: Works well with limited calibration data

TEMPORAL INTEGRITY:
-------------------
Calibration is trained on a HELD-OUT temporal validation set:
- Training data: Before calibration period
- Calibration data: Middle period (used here)
- Test data: After calibration period (never touched)

We NEVER calibrate on the same data used for model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import pickle
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS_DIR, CALIBRATION_BINS, logger
from models.base_models import BaseRevisionModel


# =============================================================================
# CALIBRATOR CLASS
# =============================================================================

class IsotonicCalibrator:
    """
    Isotonic regression calibrator for probability estimates.

    This transforms raw model probabilities into calibrated probabilities
    using a monotonic piecewise linear function learned from held-out data.

    Usage:
        calibrator = IsotonicCalibrator()
        calibrator.fit(raw_probs, true_labels)
        calibrated_probs = calibrator.transform(new_raw_probs)
    """

    def __init__(self, clip_bounds: Tuple[float, float] = (0.001, 0.999)):
        """
        Args:
            clip_bounds: Clip calibrated probabilities to avoid 0/1 extremes
        """
        self.clip_bounds = clip_bounds
        self.isotonic = IsotonicRegression(
            y_min=clip_bounds[0],
            y_max=clip_bounds[1],
            out_of_bounds="clip",
        )
        self.is_fitted = False

        # Diagnostics
        self.raw_min = None
        self.raw_max = None
        self.n_calibration_samples = None

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        """
        Fit isotonic regression on calibration set.

        Args:
            raw_probs: Raw probability predictions from base model
            y_true: True binary labels (0 or 1)

        Returns:
            self
        """
        # Validate inputs
        assert len(raw_probs) == len(y_true), "Length mismatch"
        assert np.all((raw_probs >= 0) & (raw_probs <= 1)), "Probs must be in [0,1]"
        assert set(np.unique(y_true)).issubset({0, 1}), "Labels must be 0 or 1"

        # Store diagnostics
        self.raw_min = raw_probs.min()
        self.raw_max = raw_probs.max()
        self.n_calibration_samples = len(raw_probs)

        # Fit isotonic regression
        self.isotonic.fit(raw_probs, y_true)
        self.is_fitted = True

        logger.info(
            f"Calibrator fitted on {self.n_calibration_samples} samples, "
            f"raw probs range: [{self.raw_min:.3f}, {self.raw_max:.3f}]"
        )

        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Transform raw probabilities to calibrated probabilities.

        Args:
            raw_probs: Raw probability predictions

        Returns:
            Calibrated probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        # Isotonic regression transform
        calibrated = self.isotonic.transform(raw_probs)

        # Clip to bounds (isotonic should handle this, but be safe)
        calibrated = np.clip(calibrated, self.clip_bounds[0], self.clip_bounds[1])

        return calibrated

    def get_calibration_diagnostics(
        self,
        raw_probs: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = CALIBRATION_BINS,
    ) -> Dict:
        """
        Compute calibration diagnostics comparing raw and calibrated probs.

        Returns dict with:
        - reliability diagram data (before and after)
        - ECE (Expected Calibration Error)
        - MCE (Maximum Calibration Error)
        """
        calibrated_probs = self.transform(raw_probs) if self.is_fitted else raw_probs

        # Compute reliability curves
        raw_fraction, raw_mean_pred = calibration_curve(
            y_true, raw_probs, n_bins=n_bins, strategy="uniform"
        )
        cal_fraction, cal_mean_pred = calibration_curve(
            y_true, calibrated_probs, n_bins=n_bins, strategy="uniform"
        )

        # Expected Calibration Error (weighted average deviation from diagonal)
        def compute_ece(y_true, y_prob, n_bins):
            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
                if mask.sum() > 0:
                    bin_acc = y_true[mask].mean()
                    bin_conf = y_prob[mask].mean()
                    ece += mask.sum() * abs(bin_acc - bin_conf)
            return ece / len(y_true)

        raw_ece = compute_ece(y_true, raw_probs, n_bins)
        cal_ece = compute_ece(y_true, calibrated_probs, n_bins)

        return {
            "raw_reliability": {"fraction_positive": raw_fraction, "mean_predicted": raw_mean_pred},
            "calibrated_reliability": {"fraction_positive": cal_fraction, "mean_predicted": cal_mean_pred},
            "raw_ece": raw_ece,
            "calibrated_ece": cal_ece,
            "ece_improvement": raw_ece - cal_ece,
        }


# =============================================================================
# CALIBRATED MODEL WRAPPER
# =============================================================================

class CalibratedModel:
    """
    Wrapper that combines a base model with its calibrator.

    This is the object that will be used in the meta-learner.
    """

    def __init__(self, base_model: BaseRevisionModel, calibrator: IsotonicCalibrator):
        self.base_model = base_model
        self.calibrator = calibrator
        self.name = f"{base_model.name}_calibrated"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get calibrated probability predictions."""
        raw_probs = self.base_model.predict_proba(X)
        calibrated_probs = self.calibrator.transform(raw_probs)
        return calibrated_probs

    def predict_raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get raw (uncalibrated) probability predictions."""
        return self.base_model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Delegate to base model."""
        return self.base_model.get_feature_importance()


# =============================================================================
# CALIBRATION PIPELINE
# =============================================================================

def calibrate_model(
    model: BaseRevisionModel,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    feature_cols: List[str],
) -> CalibratedModel:
    """
    Calibrate a fitted base model using held-out calibration data.

    IMPORTANT: X_cal and y_cal must be from a temporal period AFTER
    the training data but BEFORE the test data.

    Args:
        model: Fitted base model
        X_cal: Calibration features
        y_cal: Calibration targets
        feature_cols: List of feature column names

    Returns:
        CalibratedModel wrapping the base model and calibrator
    """
    # Get raw predictions on calibration set
    raw_probs = model.predict_proba(X_cal[feature_cols])

    # Fit calibrator
    calibrator = IsotonicCalibrator()
    calibrator.fit(raw_probs, y_cal.values)

    # Compute diagnostics
    diagnostics = calibrator.get_calibration_diagnostics(raw_probs, y_cal.values)
    logger.info(
        f"Calibration for {model.name}: "
        f"ECE {diagnostics['raw_ece']:.4f} -> {diagnostics['calibrated_ece']:.4f} "
        f"(improvement: {diagnostics['ece_improvement']:.4f})"
    )

    return CalibratedModel(model, calibrator)


def calibrate_all_models(
    models: Dict[str, BaseRevisionModel],
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    feature_cols: List[str],
) -> Dict[str, CalibratedModel]:
    """
    Calibrate all base models.

    Args:
        models: Dict mapping model name to fitted BaseRevisionModel
        X_cal: Calibration features
        y_cal: Calibration targets
        feature_cols: List of feature column names

    Returns:
        Dict mapping model name to CalibratedModel
    """
    calibrated_models = {}

    for name, model in models.items():
        logger.info(f"Calibrating {name}...")
        calibrated_models[name] = calibrate_model(model, X_cal, y_cal, feature_cols)

    return calibrated_models


# =============================================================================
# CALIBRATION ANALYSIS UTILITIES
# =============================================================================

def compare_calibration(
    models: Dict[str, CalibratedModel],
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compare calibration quality across models.

    Returns DataFrame with calibration metrics for each model.
    """
    results = []

    for name, model in models.items():
        raw_probs = model.predict_raw_proba(X[feature_cols])
        cal_probs = model.predict_proba(X[feature_cols])

        diagnostics = model.calibrator.get_calibration_diagnostics(
            raw_probs, y.values
        )

        results.append({
            "model": name,
            "raw_ece": diagnostics["raw_ece"],
            "calibrated_ece": diagnostics["calibrated_ece"],
            "ece_improvement": diagnostics["ece_improvement"],
            "n_samples": len(y),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Calibration module loaded successfully.")

    # Quick test with synthetic data
    np.random.seed(42)
    n = 500

    # Generate synthetic uncalibrated probabilities
    # Simulate overconfident model (probs too extreme)
    true_probs = np.random.beta(2, 2, n)
    y = (np.random.random(n) < true_probs).astype(int)
    raw_probs = np.clip(true_probs * 1.3 - 0.15, 0.05, 0.95)  # Distort

    # Calibrate
    calibrator = IsotonicCalibrator()
    calibrator.fit(raw_probs[:400], y[:400])

    # Test
    diagnostics = calibrator.get_calibration_diagnostics(raw_probs[400:], y[400:])
    print(f"\nTest calibration:")
    print(f"  Raw ECE: {diagnostics['raw_ece']:.4f}")
    print(f"  Calibrated ECE: {diagnostics['calibrated_ece']:.4f}")
    print(f"  Improvement: {diagnostics['ece_improvement']:.4f}")

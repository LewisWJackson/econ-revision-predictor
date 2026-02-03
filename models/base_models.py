"""
Base Models for Economic Revision Prediction

This module implements three diverse base models:
1. LightGBM - Gradient boosting for non-linear patterns
2. Logistic Regression - Interpretable linear model with hand-crafted interactions
3. Bayesian Logistic Regression - Explicit prior on base revision rate

MODEL DIVERSITY PRINCIPLE:
--------------------------
The ensemble benefits from diverse error patterns. These models differ in:
- LightGBM: Captures non-linear interactions automatically, may overfit
- Logistic: Linear relationships only, relies on manual feature engineering
- Bayesian: Regularizes toward historical base rate, uncertainty-aware

TEMPORAL INTEGRITY IN TRAINING:
-------------------------------
All models are trained on temporally separated data:
- Training set: Older data (before calibration period)
- Calibration set: Middle period (for isotonic regression)
- Test set: Most recent data (holdout, never seen during development)

Cross-validation (where used) is also temporal:
- TimeSeriesSplit, never random shuffle
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import pickle
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RANDOM_SEED,
    LIGHTGBM_PARAMS,
    LIGHTGBM_NUM_BOOST_ROUNDS,
    LIGHTGBM_EARLY_STOPPING_ROUNDS,
    LOGISTIC_REGRESSION_PARAMS,
    BAYESIAN_PARAMS,
    MODELS_DIR,
    logger,
)

# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class BaseRevisionModel(ABC):
    """
    Abstract base class for revision prediction models.

    All models must implement:
    - fit(): Train on features and targets
    - predict_proba(): Return calibrated probability estimates
    - get_feature_importance(): For interpretability
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.feature_names: List[str] = []

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseRevisionModel":
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of upward revision.

        Returns:
            1D array of probabilities in [0, 1]
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with columns [feature, importance]
        """
        pass

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / f"{self.name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved {self.name} to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "BaseRevisionModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded {model.name} from {path}")
        return model


# =============================================================================
# LIGHTGBM MODEL
# =============================================================================

class LightGBMModel(BaseRevisionModel):
    """
    LightGBM gradient boosting model for capturing non-linear patterns.

    Strengths:
    - Automatic feature interactions
    - Handles missing values natively
    - Fast training

    Weaknesses:
    - Can overfit with limited data
    - Less interpretable than linear models
    """

    def __init__(self):
        super().__init__(name="lightgbm")
        self.model = None
        self.params = LIGHTGBM_PARAMS.copy()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMModel":
        """
        Fit LightGBM model.

        Uses early stopping on validation set to prevent overfitting.
        """
        import lightgbm as lgb

        self.feature_names = list(X.columns)

        # Create datasets
        train_data = lgb.Dataset(X, label=y)

        callbacks = [lgb.log_evaluation(period=50)]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            callbacks.append(
                lgb.early_stopping(stopping_rounds=LIGHTGBM_EARLY_STOPPING_ROUNDS)
            )

            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=LIGHTGBM_NUM_BOOST_ROUNDS,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )
        else:
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=LIGHTGBM_NUM_BOOST_ROUNDS,
                callbacks=callbacks,
            )

        self.is_fitted = True
        logger.info(
            f"LightGBM fitted with {self.model.num_trees()} trees, "
            f"{len(self.feature_names)} features"
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of upward revision."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # LightGBM returns probabilities directly for binary classification
        probs = self.model.predict(X)
        return probs

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (gain-based)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importance = self.model.feature_importance(importance_type="gain")
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })
        return df.sort_values("importance", ascending=False)


# =============================================================================
# LOGISTIC REGRESSION WITH INTERACTIONS
# =============================================================================

class LogisticRegressionModel(BaseRevisionModel):
    """
    Logistic regression with hand-crafted interaction features.

    Strengths:
    - Fully interpretable coefficients
    - Stable probability estimates
    - Less prone to overfitting

    Weaknesses:
    - Requires manual feature engineering for non-linear patterns
    - May miss complex interactions

    INTERACTION FEATURES:
    - bias × regime: Different revision bias in expansion vs contraction
    - bias × quarter: Seasonal interaction with historical bias
    - surprise × regime: Surprises matter more in certain regimes
    """

    def __init__(self):
        super().__init__(name="logistic_regression")
        self.model = None
        self.scaler = None
        self.params = LOGISTIC_REGRESSION_PARAMS.copy()

    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create hand-crafted interaction features.

        These encode domain knowledge about revision patterns:
        - Revision bias interacts with economic regime
        - Initial surprise interacts with calendar
        """
        X = X.copy()

        # Bias × Regime interactions
        # Hypothesis: revision patterns differ in expansions vs contractions
        if "revision_up_rate_w8" in X.columns and "yield_curve_inverted" in X.columns:
            X["bias_x_inverted"] = X["revision_up_rate_w8"] * X["yield_curve_inverted"]

        if "revision_up_rate_w8" in X.columns and "sahm_recession_signal" in X.columns:
            X["bias_x_recession"] = X["revision_up_rate_w8"] * X["sahm_recession_signal"].fillna(0)

        # Bias × Quarter interactions
        # Hypothesis: seasonal patterns in revision bias
        if "revision_up_rate_w8" in X.columns:
            for q in [1, 2, 3, 4]:
                if f"is_q{q}" in X.columns or "quarter" in X.columns:
                    q_indicator = X.get(f"is_q{q}", (X.get("quarter", 0) == q).astype(int))
                    X[f"bias_x_q{q}"] = X["revision_up_rate_w8"] * q_indicator

        # Surprise × Regime interactions
        # Hypothesis: surprises matter more during uncertainty
        if "initial_surprise_magnitude" in X.columns and "vix_elevated" in X.columns:
            X["surprise_x_vix"] = (
                X["initial_surprise_magnitude"].fillna(0) *
                X["vix_elevated"].fillna(0)
            )

        # Magnitude × Direction interactions
        if "revision_magnitude_mean_w8" in X.columns and "revision_up_rate_w8" in X.columns:
            X["magnitude_x_bias"] = (
                X["revision_magnitude_mean_w8"].fillna(0) *
                X["revision_up_rate_w8"].fillna(0.5)
            )

        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LogisticRegressionModel":
        """
        Fit logistic regression with standardization.

        Standardization is important for:
        - Numerical stability
        - Comparable coefficient magnitudes
        - L2 regularization working correctly
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Create interaction features
        X_interact = self._create_interaction_features(X)
        self.feature_names = list(X_interact.columns)

        # Handle missing values (logistic regression doesn't handle them)
        X_clean = X_interact.fillna(0)

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)

        # Fit model
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_scaled, y)

        self.is_fitted = True
        logger.info(
            f"Logistic regression fitted with {len(self.feature_names)} features "
            f"(including interactions)"
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of upward revision."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_interact = self._create_interaction_features(X)
        X_clean = X_interact.fillna(0)
        X_scaled = self.scaler.transform(X_clean)

        # predict_proba returns [P(class=0), P(class=1)]
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return probs

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (absolute coefficient values)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Use absolute value of coefficients as importance
        importance = np.abs(self.model.coef_[0])
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
            "coefficient": self.model.coef_[0],
        })
        return df.sort_values("importance", ascending=False)


# =============================================================================
# BAYESIAN LOGISTIC REGRESSION
# =============================================================================

class BayesianLogisticModel(BaseRevisionModel):
    """
    Bayesian logistic regression with informative prior on base rate.

    Strengths:
    - Incorporates prior belief about revision patterns
    - Provides uncertainty estimates
    - Regularizes naturally toward base rate

    Weaknesses:
    - Slower to train (MCMC sampling)
    - Requires PyMC installation

    PRIOR SPECIFICATION:
    - Intercept prior centered on historical base rate
    - Coefficient priors are weakly informative (regularizing)

    The key insight: if we have historical evidence that this indicator
    revises up 55% of the time, we should incorporate that as a prior,
    not start from 50/50.
    """

    def __init__(self):
        super().__init__(name="bayesian_logistic")
        self.trace = None
        self.model = None
        self.base_rate = None
        self.params = BAYESIAN_PARAMS.copy()
        self.scaler = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BayesianLogisticModel":
        """
        Fit Bayesian logistic regression using PyMC.

        The prior on the intercept encodes the historical base rate
        of upward revisions for this indicator.
        """
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            logger.warning(
                "PyMC not installed. Falling back to sklearn BayesianRidge approximation."
            )
            return self._fit_fallback(X, y)

        from sklearn.preprocessing import StandardScaler

        self.feature_names = list(X.columns)

        # Calculate historical base rate for prior
        self.base_rate = y.mean()
        base_rate_logit = np.log(self.base_rate / (1 - self.base_rate + 1e-6))

        logger.info(f"Base rate of upward revision: {self.base_rate:.3f}")

        # Standardize features
        X_clean = X.fillna(0)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)

        # Build PyMC model
        with pm.Model() as self.model:
            # Data
            X_data = pm.Data("X", X_scaled)
            y_data = pm.Data("y", y.values)

            # Priors
            # Intercept: centered on historical base rate (informative)
            intercept = pm.Normal("intercept", mu=base_rate_logit, sigma=0.5)

            # Coefficients: weakly informative prior (regularizing)
            # sigma=1 on standardized features is moderately regularizing
            beta = pm.Normal("beta", mu=0, sigma=1, shape=X_scaled.shape[1])

            # Linear combination
            logit_p = intercept + pm.math.dot(X_data, beta)

            # Likelihood
            pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_data)

            # Sample
            self.trace = pm.sample(
                draws=self.params["draws"],
                tune=self.params["tune"],
                chains=self.params["chains"],
                target_accept=self.params["target_accept"],
                random_seed=self.params["random_seed"],
                progressbar=True,
                return_inferencedata=True,
            )

        self.is_fitted = True
        logger.info("Bayesian logistic regression fitted")

        return self

    def _fit_fallback(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "BayesianLogisticModel":
        """
        Fallback: use sklearn's regularized logistic regression
        with prior-informed regularization.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.feature_names = list(X.columns)
        self.base_rate = y.mean()

        X_clean = X.fillna(0)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)

        # Use stronger regularization as a simple Bayesian approximation
        self._fallback_model = LogisticRegression(
            C=0.5,  # Stronger regularization than default
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
        self._fallback_model.fit(X_scaled, y)

        self.is_fitted = True
        self._using_fallback = True
        logger.info("Bayesian model using fallback (sklearn LogisticRegression)")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of upward revision.

        For Bayesian model, returns posterior predictive mean.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_clean = X.fillna(0)
        X_scaled = self.scaler.transform(X_clean)

        if hasattr(self, "_using_fallback") and self._using_fallback:
            return self._fallback_model.predict_proba(X_scaled)[:, 1]

        # Use posterior mean of parameters
        intercept_mean = self.trace.posterior["intercept"].mean().values
        beta_mean = self.trace.posterior["beta"].mean(dim=["chain", "draw"]).values

        logit_p = intercept_mean + X_scaled @ beta_mean
        probs = 1 / (1 + np.exp(-logit_p))

        return probs

    def predict_proba_with_uncertainty(
        self,
        X: pd.DataFrame,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates (Bayesian specific).

        Returns:
            mean: Point estimate (posterior mean)
            lower: 5th percentile
            upper: 95th percentile
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if hasattr(self, "_using_fallback") and self._using_fallback:
            probs = self._fallback_model.predict_proba(
                self.scaler.transform(X.fillna(0))
            )[:, 1]
            # No uncertainty in fallback
            return probs, probs, probs

        X_clean = X.fillna(0)
        X_scaled = self.scaler.transform(X_clean)

        # Sample from posterior
        intercepts = self.trace.posterior["intercept"].values.flatten()
        betas = self.trace.posterior["beta"].values.reshape(-1, X_scaled.shape[1])

        # Subsample for efficiency
        idx = np.random.choice(len(intercepts), size=min(n_samples, len(intercepts)), replace=False)

        # Compute predictions for each posterior sample
        all_probs = []
        for i in idx:
            logit_p = intercepts[i] + X_scaled @ betas[i]
            probs = 1 / (1 + np.exp(-logit_p))
            all_probs.append(probs)

        all_probs = np.array(all_probs)  # Shape: (n_samples, n_observations)

        mean_probs = all_probs.mean(axis=0)
        lower_probs = np.percentile(all_probs, 5, axis=0)
        upper_probs = np.percentile(all_probs, 95, axis=0)

        return mean_probs, lower_probs, upper_probs

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (posterior mean of absolute coefficients)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if hasattr(self, "_using_fallback") and self._using_fallback:
            importance = np.abs(self._fallback_model.coef_[0])
            coef_mean = self._fallback_model.coef_[0]
            coef_std = np.zeros_like(coef_mean)
        else:
            beta_samples = self.trace.posterior["beta"].values.reshape(
                -1, len(self.feature_names)
            )
            coef_mean = beta_samples.mean(axis=0)
            coef_std = beta_samples.std(axis=0)
            importance = np.abs(coef_mean)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
            "coefficient_mean": coef_mean,
            "coefficient_std": coef_std,
        })
        return df.sort_values("importance", ascending=False)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_base_models() -> Dict[str, BaseRevisionModel]:
    """
    Create instances of all base models.

    Returns:
        Dictionary mapping model name to model instance
    """
    return {
        "lightgbm": LightGBMModel(),
        "logistic_regression": LogisticRegressionModel(),
        "bayesian_logistic": BayesianLogisticModel(),
    }


def get_feature_columns(X: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns, excluding metadata and target columns.

    This is used to ensure consistent feature sets across models.
    """
    exclude_prefixes = ("_", "target_", "series_id", "reference_date", "initial_date")
    return [
        col for col in X.columns
        if not any(col.startswith(p) or col == p for p in exclude_prefixes)
        and X[col].dtype in [np.float64, np.int64, np.float32, np.int32, float, int]
    ]


if __name__ == "__main__":
    print("Base models module loaded successfully.")
    print("Available models:")
    for name, model in create_base_models().items():
        print(f"  - {name}: {model.__class__.__name__}")

"""
Configuration for Economic Revision Predictor

IMPORTANT: Set your FRED API key as an environment variable:
    export FRED_API_KEY="your_key_here"

Or create a .env file in the project root with:
    FRED_API_KEY=your_key_here
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv(Path(__file__).parent / ".env")
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import date

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
EVALUATION_DIR = PROJECT_ROOT / "evaluation" / "outputs"

# Create directories if they don't exist (skip on read-only filesystems like Streamlit Cloud)
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, EVALUATION_DIR]:
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # Read-only filesystem (e.g. Streamlit Cloud)

# =============================================================================
# API CONFIGURATION
# =============================================================================

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Rate limiting: FRED allows 120 requests/minute
FRED_RATE_LIMIT_REQUESTS = 120
FRED_RATE_LIMIT_WINDOW_SECONDS = 60

# =============================================================================
# INDICATOR CONFIGURATION
# =============================================================================

@dataclass
class IndicatorConfig:
    """Configuration for a single economic indicator."""
    series_id: str
    name: str
    category: str
    frequency: str  # 'Q' = quarterly, 'M' = monthly, 'W' = weekly
    revision_window_days: int  # Days until value is considered "final"
    typical_revisions: int  # Expected number of revision stages
    start_date: str  # Earliest date to pull data from

    # For computing "final" value - how many days after initial release
    # to consider the value settled for training purposes
    final_lag_days: int = field(default=None)

    def __post_init__(self):
        if self.final_lag_days is None:
            self.final_lag_days = self.revision_window_days


# Tier 1: Richest revision histories
INDICATORS: Dict[str, IndicatorConfig] = {
    # GDP: Advance (T+30d) -> Second (T+60d) -> Third (T+90d)
    # We consider "final" as the third estimate for prediction purposes
    "GDPC1": IndicatorConfig(
        series_id="GDPC1",
        name="Real Gross Domestic Product",
        category="gdp",
        frequency="Q",
        revision_window_days=90,  # ~3 months to third estimate
        typical_revisions=3,
        start_date="1990-01-01",
        final_lag_days=95,  # Slightly past third estimate
    ),

    # Nonfarm Payroll: 3 monthly revisions
    # Initial -> 1st revision (next month) -> 2nd revision (month after)
    "PAYEMS": IndicatorConfig(
        series_id="PAYEMS",
        name="Total Nonfarm Payroll",
        category="employment",
        frequency="M",
        revision_window_days=60,  # ~2 months for sample-based revisions
        typical_revisions=3,
        start_date="1990-01-01",
        final_lag_days=65,
    ),

    # Initial Claims: Weekly revisions
    "ICSA": IndicatorConfig(
        series_id="ICSA",
        name="Initial Unemployment Claims",
        category="employment",
        frequency="W",
        revision_window_days=7,  # Revised weekly
        typical_revisions=1,
        start_date="1990-01-01",
        final_lag_days=14,
    ),

    # Industrial Production: 5-month revision window
    "INDPRO": IndicatorConfig(
        series_id="INDPRO",
        name="Industrial Production Index",
        category="manufacturing",
        frequency="M",
        revision_window_days=150,  # 5 months
        typical_revisions=5,
        start_date="1990-01-01",
        final_lag_days=155,
    ),

    # Durable Goods: Advance -> Full (2-month window)
    "DGORDER": IndicatorConfig(
        series_id="DGORDER",
        name="Durable Goods Orders",
        category="manufacturing",
        frequency="M",
        revision_window_days=60,
        typical_revisions=2,
        start_date="1992-01-01",  # Shorter history
        final_lag_days=65,
    ),

    # Retail Sales: Advance -> Preliminary -> Final
    "RSXFS": IndicatorConfig(
        series_id="RSXFS",
        name="Retail Sales (ex Food Services)",
        category="consumption",
        frequency="M",
        revision_window_days=45,  # ~6 weeks
        typical_revisions=3,
        start_date="1992-01-01",
        final_lag_days=50,
    ),

    # Housing Starts
    "HOUST": IndicatorConfig(
        series_id="HOUST",
        name="Housing Starts",
        category="housing",
        frequency="M",
        revision_window_days=30,
        typical_revisions=2,
        start_date="1990-01-01",
        final_lag_days=35,
    ),
}

# List of series IDs for convenience
INDICATOR_SERIES_IDS: List[str] = list(INDICATORS.keys())

# =============================================================================
# TEMPORAL CONFIGURATION
# =============================================================================

# Training data range
# We need enough history for the revision window to close on all training samples
TRAIN_START_DATE = "1995-01-01"  # After indicators have stable histories
TRAIN_END_DATE = "2019-12-31"   # Before COVID disruption

# Validation set for calibration (temporal, not random)
CALIBRATION_START_DATE = "2020-01-01"
CALIBRATION_END_DATE = "2022-12-31"

# Test set (holdout, never touched during development)
TEST_START_DATE = "2023-01-01"
TEST_END_DATE = "2024-12-31"

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Rolling windows for historical features (in number of periods, not days)
ROLLING_WINDOWS = [4, 8, 12, 20]  # e.g., 4 quarters = 1 year for quarterly data

# Minimum observations required to compute rolling statistics
MIN_OBSERVATIONS_FOR_ROLLING = 4

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Random seed for all models (reproducibility)
RANDOM_SEED = 42

# LightGBM parameters
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "deterministic": True,
    "force_row_wise": True,  # For reproducibility
}

LIGHTGBM_NUM_BOOST_ROUNDS = 200
LIGHTGBM_EARLY_STOPPING_ROUNDS = 20

# Logistic Regression parameters
LOGISTIC_REGRESSION_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}

# Bayesian model parameters (PyMC)
BAYESIAN_PARAMS = {
    "draws": 2000,
    "tune": 1000,
    "chains": 4,
    "target_accept": 0.9,
    "random_seed": RANDOM_SEED,
}

# Meta-learner (stacking) parameters
META_LEARNER_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}

# =============================================================================
# LEAKAGE PREVENTION CONFIGURATION
# =============================================================================

# Buffer days to ensure no forward-looking information
# When computing features for a prediction made on date D,
# we only use data with vintage_date <= D - LEAKAGE_BUFFER_DAYS
LEAKAGE_BUFFER_DAYS = 1

# Enable strict leakage checks (slower but safer during development)
STRICT_LEAKAGE_CHECKS = True

# Log all leakage check results
LOG_LEAKAGE_CHECKS = True

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Number of bins for calibration plots
CALIBRATION_BINS = 10

# Bootstrap iterations for confidence intervals
BOOTSTRAP_ITERATIONS = 1000

# =============================================================================
# LOGGING
# =============================================================================

import logging

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("econ_revision_predictor")

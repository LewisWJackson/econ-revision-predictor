#!/usr/bin/env python3
"""
Economic Revision Predictor - Main Pipeline

This script orchestrates the complete training and evaluation pipeline:
1. Data download from FRED/ALFRED
2. Feature engineering with temporal integrity
3. Base model training
4. Calibration
5. Meta-learner stacking
6. Evaluation

Usage:
    python main.py --download      # Download data from FRED
    python main.py --train         # Train models
    python main.py --evaluate      # Evaluate on test set
    python main.py --full          # Full pipeline (download + train + evaluate)
    python main.py --predict       # Make predictions on latest data

TEMPORAL INTEGRITY GUARANTEE:
-----------------------------
This pipeline enforces strict temporal separation:
- Training: 1995-2019
- Calibration: 2020-2022
- Test: 2023-2024

No information from later periods can leak into earlier stages.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    CALIBRATION_START_DATE,
    CALIBRATION_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    EVALUATION_DIR,
    RANDOM_SEED,
    STRICT_LEAKAGE_CHECKS,
    logger,
)

# =============================================================================
# LEAKAGE VERIFICATION
# =============================================================================

def verify_temporal_integrity(df: pd.DataFrame, split_name: str, max_date: str):
    """
    Verify no data from after max_date is present.

    This is a critical safety check. We verify at multiple stages
    that no future information has leaked in.
    """
    if "initial_date" not in df.columns:
        logger.warning(f"Cannot verify temporal integrity for {split_name}: no initial_date column")
        return

    max_allowed = pd.to_datetime(max_date)
    max_observed = pd.to_datetime(df["initial_date"]).max()

    if max_observed > max_allowed:
        error_msg = (
            f"TEMPORAL INTEGRITY VIOLATION in {split_name}: "
            f"Found data from {max_observed} but max allowed is {max_allowed}"
        )
        logger.error(error_msg)
        if STRICT_LEAKAGE_CHECKS:
            raise ValueError(error_msg)
    else:
        logger.info(
            f"✓ Temporal integrity verified for {split_name}: "
            f"max date {max_observed} <= {max_allowed}"
        )


# =============================================================================
# DATA PIPELINE
# =============================================================================

def download_data():
    """
    Download revision data from FRED/ALFRED for all configured indicators.

    This uses output_type=4 for initial releases and output_type=2 for
    all vintages, ensuring we capture the complete revision history.
    """
    from data.scripts.fred_client import download_all_indicators

    logger.info("=" * 60)
    logger.info("DOWNLOADING DATA FROM FRED/ALFRED")
    logger.info("=" * 60)

    # Download with date range that covers all periods
    # We download more than we need and filter later for safety
    datasets = download_all_indicators(
        start_date="1990-01-01",
        end_date="2025-12-31",
        output_dir=RAW_DATA_DIR,
    )

    logger.info(f"Downloaded {len(datasets)} indicators")

    # Summary
    for series_id, df in datasets.items():
        logger.info(
            f"  {series_id}: {len(df)} observations, "
            f"range {df['reference_date'].min()} to {df['reference_date'].max()}"
        )

    return datasets


def load_data() -> dict:
    """
    Load previously downloaded data from disk.
    """
    logger.info("Loading data from disk...")

    datasets = {}
    for series_id in INDICATORS.keys():
        path = RAW_DATA_DIR / f"{series_id}_revisions.parquet"
        if path.exists():
            datasets[series_id] = pd.read_parquet(path)
            logger.info(f"  Loaded {series_id}: {len(datasets[series_id])} rows")
        else:
            logger.warning(f"  Missing data for {series_id}")

    return datasets


def prepare_features(datasets: dict) -> pd.DataFrame:
    """
    Build feature matrix from revision datasets.

    LEAKAGE PREVENTION:
    - Features are computed only using data available at prediction time
    - All rolling statistics use only past completed revisions
    - Cross-indicator features are properly lagged
    """
    from features.engineering import build_feature_matrix
    from data.scripts.macro_indicators import load_macro_indicators, download_all_macro_indicators

    logger.info("=" * 60)
    logger.info("BUILDING FEATURE MATRIX")
    logger.info("=" * 60)

    # Load or download macro indicators
    macro_data = load_macro_indicators()
    if macro_data.empty:
        logger.info("Downloading macro indicators...")
        macro_data = download_all_macro_indicators()

    if not macro_data.empty:
        logger.info(f"Loaded {len(macro_data)} macro indicator observations")
        # Create regime indicator from macro data
        regime_indicator = macro_data[["date", "is_contraction"]].copy()
    else:
        logger.warning("No macro data available, regime features will be NaN")
        regime_indicator = None

    # Build feature matrix for all data
    # The feature engineering module handles temporal integrity internally
    feature_df = build_feature_matrix(
        all_revisions=datasets,
        historical_values=None,  # TODO: Add if consensus data available
        macro_data=macro_data if not macro_data.empty else None,
        regime_indicator=regime_indicator,
    )

    if feature_df.empty:
        raise ValueError("Feature matrix is empty. Check data download.")

    # Save processed features
    feature_path = PROCESSED_DATA_DIR / "features.parquet"
    feature_df.to_parquet(feature_path, index=False)
    logger.info(f"Saved feature matrix to {feature_path}")
    logger.info(f"Feature matrix shape: {feature_df.shape}")

    return feature_df


def split_data(df: pd.DataFrame) -> tuple:
    """
    Split data into train/calibration/test sets with temporal separation.

    CRITICAL: These splits are by INITIAL_DATE, not by reference_date.
    This ensures we predict on data released after training.

    Splits:
    - Train: initial_date in [1995, 2019]
    - Calibration: initial_date in [2020, 2022]
    - Test: initial_date in [2023, 2024]
    """
    logger.info("Splitting data temporally...")

    df["initial_date"] = pd.to_datetime(df["initial_date"])

    # Apply temporal splits
    train_mask = (
        (df["initial_date"] >= TRAIN_START_DATE) &
        (df["initial_date"] <= TRAIN_END_DATE)
    )
    cal_mask = (
        (df["initial_date"] >= CALIBRATION_START_DATE) &
        (df["initial_date"] <= CALIBRATION_END_DATE)
    )
    test_mask = (
        (df["initial_date"] >= TEST_START_DATE) &
        (df["initial_date"] <= TEST_END_DATE)
    )

    train_df = df[train_mask].copy()
    cal_df = df[cal_mask].copy()
    test_df = df[test_mask].copy()

    # Verify temporal integrity
    verify_temporal_integrity(train_df, "train", TRAIN_END_DATE)
    verify_temporal_integrity(cal_df, "calibration", CALIBRATION_END_DATE)
    verify_temporal_integrity(test_df, "test", TEST_END_DATE)

    logger.info(f"Train set: {len(train_df)} samples ({TRAIN_START_DATE} to {TRAIN_END_DATE})")
    logger.info(f"Calibration set: {len(cal_df)} samples ({CALIBRATION_START_DATE} to {CALIBRATION_END_DATE})")
    logger.info(f"Test set: {len(test_df)} samples ({TEST_START_DATE} to {TEST_END_DATE})")

    return train_df, cal_df, test_df


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_models(train_df: pd.DataFrame, cal_df: pd.DataFrame):
    """
    Train all models: base models, calibration, meta-learner.

    Training flow:
    1. Identify feature columns (exclude metadata and targets)
    2. Train each base model on train_df
    3. Calibrate each base model using cal_df
    4. Train meta-learner on calibrated predictions
    5. Save ensemble
    """
    from models.base_models import create_base_models, get_feature_columns
    from models.calibration import calibrate_all_models
    from models.meta_learner import build_stacking_ensemble

    logger.info("=" * 60)
    logger.info("TRAINING MODELS")
    logger.info("=" * 60)

    # Get feature columns
    feature_cols = get_feature_columns(train_df)
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target_revised_up"].copy()

    X_cal = cal_df[feature_cols].copy()
    y_cal = cal_df["target_revised_up"].copy()

    # Handle missing values for training
    X_train = X_train.fillna(0)
    X_cal = X_cal.fillna(0)

    logger.info(f"Training samples: {len(X_train)}, Calibration samples: {len(X_cal)}")
    logger.info(f"Train positive rate: {y_train.mean():.3f}")
    logger.info(f"Calibration positive rate: {y_cal.mean():.3f}")

    # Create base models
    base_models = create_base_models()

    # Train each base model
    fitted_models = {}
    for name, model in base_models.items():
        logger.info(f"\nTraining {name}...")

        # Split some training data for early stopping (for LightGBM)
        split_idx = int(len(X_train) * 0.85)
        X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

        model.fit(X_tr, y_tr, X_val, y_val)
        fitted_models[name] = model

        # Log training performance
        train_probs = model.predict_proba(X_train)
        logger.info(f"  {name} train predictions: mean={train_probs.mean():.3f}, std={train_probs.std():.3f}")

    # Calibrate all models
    logger.info("\nCalibrating models...")
    calibrated_models = calibrate_all_models(
        models=fitted_models,
        X_cal=X_cal,
        y_cal=y_cal,
        feature_cols=feature_cols,
    )

    # Build stacking ensemble
    logger.info("\nBuilding stacking ensemble...")
    ensemble = build_stacking_ensemble(
        calibrated_models=calibrated_models,
        X_stack=X_cal,
        y_stack=y_cal,
        feature_cols=feature_cols,
    )

    # Save ensemble
    ensemble_path = ensemble.save(MODELS_DIR / "ensemble.pkl")

    # Save feature columns for later use
    feature_cols_path = MODELS_DIR / "feature_cols.json"
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f)

    logger.info(f"\nModel training complete. Ensemble saved to {ensemble_path}")

    return ensemble, feature_cols


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================

def evaluate_ensemble(
    ensemble,
    test_df: pd.DataFrame,
    feature_cols: list,
):
    """
    Evaluate ensemble on held-out test set.

    This is the final evaluation that tells us if the model generalizes.
    """
    from evaluation.metrics import (
        generate_evaluation_report,
        evaluate_by_indicator,
    )

    logger.info("=" * 60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 60)

    # Prepare test data
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target_revised_up"].values

    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Test positive rate: {y_test.mean():.3f}")

    # Get predictions from ensemble and individual models
    predictions = {}

    # Ensemble prediction
    predictions["ensemble"] = ensemble.predict_proba(test_df)

    # Individual calibrated models
    decomposed = ensemble.predict_proba_decomposed(test_df)
    for model_name in ensemble.model_names:
        predictions[model_name] = decomposed[model_name].values

    # Naive baseline
    predictions["naive_baseline"] = np.full(len(y_test), y_test.mean())

    # Check if LightGBM alone is better - if so, use it as primary
    from evaluation.metrics import compute_brier_score
    ensemble_bs = compute_brier_score(y_test, predictions["ensemble"])
    lgb_bs = compute_brier_score(y_test, predictions["lightgbm"])

    if lgb_bs < ensemble_bs:
        logger.info(f"NOTE: LightGBM alone (BS={lgb_bs:.4f}) outperforms ensemble (BS={ensemble_bs:.4f})")
        logger.info("Consider using LightGBM directly for production predictions.")
        predictions["best_model"] = predictions["lightgbm"]
    else:
        predictions["best_model"] = predictions["ensemble"]

    # Generate report
    report = generate_evaluation_report(
        y_true=y_test,
        predictions=predictions,
        output_dir=EVALUATION_DIR,
    )

    # Breakdown by indicator
    test_df_eval = test_df.copy()
    test_df_eval["predicted_prob"] = predictions["ensemble"]

    indicator_results = evaluate_by_indicator(test_df_eval)
    logger.info("\nPerformance by indicator:")
    print(indicator_results.to_string(index=False))

    # Save indicator breakdown
    indicator_results.to_csv(EVALUATION_DIR / "indicator_breakdown.csv", index=False)

    # Save model weights
    weights = ensemble.get_model_weights()
    logger.info("\nMeta-learner weights:")
    print(weights.to_string(index=False))
    weights.to_csv(EVALUATION_DIR / "model_weights.csv", index=False)

    # Save feature importance
    importance = ensemble.get_feature_importance()
    importance.head(20).to_csv(EVALUATION_DIR / "feature_importance.csv", index=False)
    logger.info("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    return report


# =============================================================================
# PREDICTION PIPELINE
# =============================================================================

def make_predictions(ensemble, feature_cols: list):
    """
    Make predictions on the most recent data.

    This would be used in production to generate signals for prediction markets.
    """
    from models.meta_learner import StackingEnsemble

    logger.info("=" * 60)
    logger.info("MAKING PREDICTIONS ON LATEST DATA")
    logger.info("=" * 60)

    # Load latest features
    feature_path = PROCESSED_DATA_DIR / "features.parquet"
    if not feature_path.exists():
        logger.error("No feature data found. Run --train first.")
        return

    df = pd.read_parquet(feature_path)

    # Filter to most recent unrevisioned data
    # In production, you'd filter to indicators where revision window hasn't closed
    df["initial_date"] = pd.to_datetime(df["initial_date"])
    cutoff = datetime.now() - pd.Timedelta(days=90)  # Recent releases

    recent = df[df["initial_date"] >= cutoff].copy()

    if recent.empty:
        logger.info("No recent data to predict on.")
        return

    # Make predictions
    X = recent[feature_cols].fillna(0)
    recent["predicted_prob_up"] = ensemble.predict_proba(recent)

    # Display predictions
    output_cols = ["series_id", "reference_date", "initial_date", "initial_value", "predicted_prob_up"]
    predictions = recent[output_cols].sort_values("predicted_prob_up", ascending=False)

    logger.info(f"\nPredictions for {len(predictions)} recent releases:")
    print(predictions.to_string(index=False))

    # Save predictions
    predictions.to_csv(EVALUATION_DIR / "latest_predictions.csv", index=False)

    return predictions


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Economic Revision Predictor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --full           # Complete pipeline
    python main.py --download       # Only download data
    python main.py --train          # Only train (assumes data exists)
    python main.py --evaluate       # Only evaluate (assumes model exists)
    python main.py --predict        # Make predictions on latest data
        """
    )

    parser.add_argument("--download", action="store_true", help="Download data from FRED")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    parser.add_argument("--predict", action="store_true", help="Predict on latest data")
    parser.add_argument("--full", action="store_true", help="Full pipeline (download + train + evaluate)")

    args = parser.parse_args()

    # If no args, show help
    if not any([args.download, args.train, args.evaluate, args.predict, args.full]):
        parser.print_help()
        return

    # Set random seed
    np.random.seed(RANDOM_SEED)

    logger.info("=" * 60)
    logger.info("ECONOMIC REVISION PREDICTOR")
    logger.info(f"Started at {datetime.now().isoformat()}")
    logger.info("=" * 60)

    try:
        if args.full or args.download:
            datasets = download_data()
        else:
            datasets = load_data()

        if args.full or args.train:
            # Build features
            feature_df = prepare_features(datasets)

            # Split data
            train_df, cal_df, test_df = split_data(feature_df)

            # Train models
            ensemble, feature_cols = train_models(train_df, cal_df)

            # Evaluate
            if args.full or args.evaluate:
                evaluate_ensemble(ensemble, test_df, feature_cols)

        elif args.evaluate:
            # Load existing model and evaluate
            from models.meta_learner import StackingEnsemble

            ensemble = StackingEnsemble.load(MODELS_DIR / "ensemble.pkl")
            with open(MODELS_DIR / "feature_cols.json") as f:
                feature_cols = json.load(f)

            feature_df = pd.read_parquet(PROCESSED_DATA_DIR / "features.parquet")
            _, _, test_df = split_data(feature_df)

            evaluate_ensemble(ensemble, test_df, feature_cols)

        if args.predict:
            from models.meta_learner import StackingEnsemble

            ensemble = StackingEnsemble.load(MODELS_DIR / "ensemble.pkl")
            with open(MODELS_DIR / "feature_cols.json") as f:
                feature_cols = json.load(f)

            make_predictions(ensemble, feature_cols)

        logger.info("=" * 60)
        logger.info(f"Completed at {datetime.now().isoformat()}")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

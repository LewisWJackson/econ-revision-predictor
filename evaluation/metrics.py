"""
Evaluation Framework for Economic Revision Predictions

This module provides comprehensive evaluation of probabilistic forecasts:
- Brier Score (primary metric)
- Calibration analysis (reliability diagrams)
- Performance breakdown by indicator type
- Comparison against naive base rate prediction

WHY BRIER SCORE:
----------------
Brier score measures the mean squared error of probability forecasts:
    BS = (1/n) * Σ(p_i - o_i)²

where p_i is predicted probability and o_i is actual outcome (0 or 1).

It rewards:
- Calibration: Predictions close to true probabilities
- Resolution: Ability to discriminate between events and non-events
- (Implicitly) correct base rate

Lower is better. Range: [0, 1].

Decomposition:
    BS = Reliability - Resolution + Uncertainty

- Reliability: How far predictions are from observed frequencies (calibration)
- Resolution: How much predictions differ from the base rate (discrimination)
- Uncertainty: Base rate variance (inherent unpredictability)

NAIVE BASELINE:
---------------
The simplest prediction is the historical base rate:
    p_naive = observed_rate_of_upward_revisions

Any useful model must beat this baseline's Brier score.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CALIBRATION_BINS, BOOTSTRAP_ITERATIONS, EVALUATION_DIR, logger


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    brier_score: float
    brier_skill_score: float  # Relative to naive baseline
    log_loss: float
    auc_roc: float
    calibration_error: float  # ECE
    n_samples: int
    base_rate: float

    def to_dict(self) -> Dict:
        return {
            "brier_score": self.brier_score,
            "brier_skill_score": self.brier_skill_score,
            "log_loss": self.log_loss,
            "auc_roc": self.auc_roc,
            "calibration_error": self.calibration_error,
            "n_samples": self.n_samples,
            "base_rate": self.base_rate,
        }


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (lower is better)."""
    return brier_score_loss(y_true, y_prob)


def compute_brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    reference_prob: Optional[float] = None,
) -> float:
    """
    Compute Brier Skill Score relative to a reference forecast.

    BSS = 1 - (BS_model / BS_reference)

    If reference_prob is None, uses the climatological base rate.

    BSS > 0: Model beats reference
    BSS = 0: Model equals reference
    BSS < 0: Model worse than reference
    """
    bs_model = compute_brier_score(y_true, y_prob)

    if reference_prob is None:
        reference_prob = y_true.mean()

    bs_reference = compute_brier_score(y_true, np.full_like(y_prob, reference_prob))

    if bs_reference == 0:
        return 0.0  # Perfect reference, can't beat it

    return 1 - (bs_model / bs_reference)


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ(n_b / n) * |accuracy_b - confidence_b|

    where b indexes bins, n_b is bin count, accuracy_b is actual
    positive rate in bin, confidence_b is mean predicted probability.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            ece += mask.sum() * abs(bin_accuracy - bin_confidence)

    return ece / len(y_true)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics for a set of predictions.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        EvaluationMetrics dataclass
    """
    base_rate = y_true.mean()

    # Handle edge case of single class
    if len(np.unique(y_true)) > 1:
        ll = log_loss(y_true, np.clip(y_prob, 1e-10, 1 - 1e-10))
        auc = roc_auc_score(y_true, y_prob)
    else:
        ll = np.nan
        auc = 0.5

    metrics = EvaluationMetrics(
        brier_score=compute_brier_score(y_true, y_prob),
        brier_skill_score=compute_brier_skill_score(y_true, y_prob),
        log_loss=ll,
        auc_roc=auc,
        calibration_error=compute_calibration_error(y_true, y_prob),
        n_samples=len(y_true),
        base_rate=base_rate,
    )

    logger.info(
        f"{model_name} evaluation: "
        f"Brier={metrics.brier_score:.4f}, "
        f"BSS={metrics.brier_skill_score:.4f}, "
        f"ECE={metrics.calibration_error:.4f}, "
        f"AUC={metrics.auc_roc:.4f}"
    )

    return metrics


# =============================================================================
# BRIER SCORE DECOMPOSITION
# =============================================================================

def decompose_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> Dict[str, float]:
    """
    Decompose Brier score into reliability, resolution, and uncertainty.

    BS = Reliability - Resolution + Uncertainty

    - Reliability (REL): Calibration term, want LOW
    - Resolution (RES): Discrimination term, want HIGH
    - Uncertainty (UNC): Base rate variance, fixed for given data

    This decomposition helps diagnose model issues:
    - High REL: Poor calibration, consider recalibrating
    - Low RES: Poor discrimination, need better features/model
    """
    base_rate = y_true.mean()
    n = len(y_true)

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        n_k = mask.sum()

        if n_k > 0:
            o_k = y_true[mask].mean()  # Observed frequency in bin
            f_k = y_prob[mask].mean()  # Mean forecast in bin

            reliability += n_k * (f_k - o_k) ** 2
            resolution += n_k * (o_k - base_rate) ** 2

    reliability /= n
    resolution /= n
    uncertainty = base_rate * (1 - base_rate)

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_score": reliability - resolution + uncertainty,
        "base_rate": base_rate,
    }


# =============================================================================
# CALIBRATION PLOTS
# =============================================================================

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = CALIBRATION_BINS,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create reliability diagram (calibration plot).

    A well-calibrated model follows the diagonal: when it predicts 0.7,
    the actual positive rate should be ~70%.

    Also shows histogram of predictions to identify where most
    predictions fall.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    fraction_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.plot(mean_predicted, fraction_positives, "o-", label=model_name)
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Reliability Diagram")
    ax1.legend(loc="lower right")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Add ECE annotation
    ece = compute_calibration_error(y_true, y_prob, n_bins)
    ax1.annotate(
        f"ECE = {ece:.4f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
    )

    # Histogram of predictions
    ax2.hist(y_prob, bins=n_bins, edgecolor="black", alpha=0.7)
    ax2.axvline(y_true.mean(), color="red", linestyle="--", label=f"Base rate: {y_true.mean():.3f}")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predictions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved reliability diagram to {save_path}")

    return fig


def plot_calibration_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = CALIBRATION_BINS,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare calibration across multiple models.

    Args:
        results: Dict mapping model_name -> (y_true, y_prob)
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, (y_true, y_prob)), color in zip(results.items(), colors):
        fraction_positives, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        ece = compute_calibration_error(y_true, y_prob, n_bins)
        ax.plot(
            mean_predicted, fraction_positives,
            "o-", color=color, label=f"{name} (ECE={ece:.4f})"
        )

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title("Calibration Comparison", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved calibration comparison to {save_path}")

    return fig


# =============================================================================
# BREAKDOWN BY INDICATOR TYPE
# =============================================================================

def evaluate_by_indicator(
    df: pd.DataFrame,
    prob_col: str = "predicted_prob",
    target_col: str = "target_revised_up",
    indicator_col: str = "series_id",
) -> pd.DataFrame:
    """
    Break down evaluation metrics by indicator type.

    This reveals if the model performs differently across indicators.
    """
    results = []

    for indicator in df[indicator_col].unique():
        mask = df[indicator_col] == indicator
        y_true = df.loc[mask, target_col].values
        y_prob = df.loc[mask, prob_col].values

        if len(y_true) < 10:
            continue

        metrics = evaluate_predictions(y_true, y_prob, model_name=indicator)

        results.append({
            "indicator": indicator,
            **metrics.to_dict(),
        })

    return pd.DataFrame(results).sort_values("brier_score")


def evaluate_by_category(
    df: pd.DataFrame,
    category_mapping: Dict[str, str],
    prob_col: str = "predicted_prob",
    target_col: str = "target_revised_up",
    indicator_col: str = "series_id",
) -> pd.DataFrame:
    """
    Break down evaluation by indicator category (e.g., GDP, employment).

    Args:
        category_mapping: Dict mapping series_id -> category
    """
    df = df.copy()
    df["category"] = df[indicator_col].map(category_mapping)

    results = []

    for category in df["category"].unique():
        mask = df["category"] == category
        y_true = df.loc[mask, target_col].values
        y_prob = df.loc[mask, prob_col].values

        if len(y_true) < 10:
            continue

        metrics = evaluate_predictions(y_true, y_prob, model_name=category)

        results.append({
            "category": category,
            **metrics.to_dict(),
        })

    return pd.DataFrame(results).sort_values("brier_score")


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for Brier score.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    n = len(y_true)
    bootstrap_scores = []

    np.random.seed(42)  # Reproducibility

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        bs = compute_brier_score(y_true[idx], y_prob[idx])
        bootstrap_scores.append(bs)

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
    point = compute_brier_score(y_true, y_prob)

    return point, lower, upper


# =============================================================================
# COMPARISON REPORT
# =============================================================================

def generate_evaluation_report(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    indicator_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Generate comprehensive evaluation report comparing models.

    Args:
        y_true: True labels
        predictions: Dict mapping model_name -> predicted probabilities
        indicator_df: Optional DataFrame with indicator metadata
        output_dir: Directory to save plots and reports

    Returns:
        Dictionary with all evaluation results
    """
    output_dir = output_dir or EVALUATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "n_samples": len(y_true),
        "base_rate": y_true.mean(),
        "models": {},
    }

    # Evaluate each model
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Samples: {len(y_true)}")
    print(f"Base rate (upward revisions): {y_true.mean():.3f}")
    print()

    # Naive baseline
    naive_bs = compute_brier_score(y_true, np.full(len(y_true), y_true.mean()))
    print(f"Naive baseline Brier score: {naive_bs:.4f}")
    print("-" * 60)

    for name, y_prob in predictions.items():
        metrics = evaluate_predictions(y_true, y_prob, model_name=name)
        decomp = decompose_brier_score(y_true, y_prob)
        bs_point, bs_lower, bs_upper = bootstrap_brier_score(y_true, y_prob)

        report["models"][name] = {
            "metrics": metrics.to_dict(),
            "decomposition": decomp,
            "brier_ci": {"point": bs_point, "lower": bs_lower, "upper": bs_upper},
        }

        print(f"\n{name}:")
        print(f"  Brier Score: {metrics.brier_score:.4f} [{bs_lower:.4f}, {bs_upper:.4f}]")
        print(f"  Brier Skill Score: {metrics.brier_skill_score:.4f}")
        print(f"  Log Loss: {metrics.log_loss:.4f}")
        print(f"  AUC-ROC: {metrics.auc_roc:.4f}")
        print(f"  Calibration Error (ECE): {metrics.calibration_error:.4f}")
        print(f"  Decomposition:")
        print(f"    Reliability: {decomp['reliability']:.4f} (want LOW)")
        print(f"    Resolution:  {decomp['resolution']:.4f} (want HIGH)")
        print(f"    Uncertainty: {decomp['uncertainty']:.4f} (fixed)")

        # Save reliability diagram
        fig = plot_reliability_diagram(
            y_true, y_prob,
            model_name=name,
            save_path=output_dir / f"reliability_{name}.png",
        )
        plt.close(fig)

    # Comparison plot
    comparison_data = {name: (y_true, probs) for name, probs in predictions.items()}
    fig = plot_calibration_comparison(
        comparison_data,
        save_path=output_dir / "calibration_comparison.png",
    )
    plt.close(fig)

    print("\n" + "=" * 60)
    print(f"Plots saved to {output_dir}")

    return report


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")

    # Quick test with synthetic data
    np.random.seed(42)
    n = 500

    # True labels with 55% positive rate
    y_true = np.random.binomial(1, 0.55, n)

    # Simulated model predictions
    # Good model: correlated with truth
    good_probs = np.clip(y_true * 0.5 + 0.25 + np.random.normal(0, 0.15, n), 0.1, 0.9)

    # Poor model: random
    poor_probs = np.random.uniform(0.3, 0.7, n)

    # Evaluate
    print("\nGood model:")
    good_metrics = evaluate_predictions(y_true, good_probs, "good_model")
    print(f"  BSS: {good_metrics.brier_skill_score:.4f} (>0 beats naive)")

    print("\nPoor model:")
    poor_metrics = evaluate_predictions(y_true, poor_probs, "poor_model")
    print(f"  BSS: {poor_metrics.brier_skill_score:.4f}")

"""
Economic Revision Predictor - Streamlit Dashboard

Predicts whether US economic indicator revisions will exceed initial estimates,
and generates trading signals based on historical revision biases.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Paths (relative to this file, works on Streamlit Cloud)
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = APP_DIR / "models" / "saved"
EVAL_DIR = APP_DIR / "evaluation" / "outputs"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Economic Revision Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached so files are read once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_predictions():
    path = EVAL_DIR / "latest_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_indicator_breakdown():
    path = EVAL_DIR / "indicator_breakdown.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_feature_importance():
    path = EVAL_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_model_weights():
    path = EVAL_DIR / "model_weights.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_revision_data():
    """Load raw revision parquet files and compute summary stats."""
    all_data = {}
    for parquet_file in sorted(RAW_DATA_DIR.glob("*_revisions.parquet")):
        indicator = parquet_file.stem.replace("_revisions", "")
        try:
            df = pd.read_parquet(parquet_file)
            all_data[indicator] = df
        except Exception:
            pass
    return all_data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDICATOR_NAMES = {
    "GDPC1": "Real GDP",
    "PAYEMS": "Nonfarm Payrolls",
    "ICSA": "Initial Claims",
    "INDPRO": "Industrial Production",
    "RSXFS": "Retail Sales",
    "DGORDER": "Durable Goods",
    "HOUST": "Housing Starts",
}

REVISION_BIASES = {
    "GDPC1": {"direction": "up", "probability": 0.993, "colour": "#22c55e"},
    "PAYEMS": {"direction": "up", "probability": 0.618, "colour": "#22c55e"},
    "ICSA": {"direction": "neutral", "probability": 0.50, "colour": "#f59e0b"},
    "INDPRO": {"direction": "down", "probability": 0.921, "colour": "#ef4444"},
    "RSXFS": {"direction": "down", "probability": 0.983, "colour": "#ef4444"},
    "DGORDER": {"direction": "down", "probability": 0.686, "colour": "#ef4444"},
    "HOUST": {"direction": "neutral", "probability": 0.507, "colour": "#f59e0b"},
}

INDICATOR_TRADES = {
    "GDPC1": {
        "miss": [{"symbol": "SPY", "dir": "BUY", "size": "5%"}, {"symbol": "QQQ", "dir": "BUY", "size": "3%"}],
        "beat": [],
    },
    "PAYEMS": {
        "miss": [{"symbol": "SPY", "dir": "BUY", "size": "4%"}, {"symbol": "XLF", "dir": "BUY", "size": "3%"}],
        "beat": [{"symbol": "TLT", "dir": "BUY", "size": "2%"}],
    },
    "INDPRO": {
        "miss": [],
        "beat": [{"symbol": "XLI", "dir": "SELL", "size": "3%"}, {"symbol": "SPY", "dir": "SELL", "size": "2%"}],
    },
    "RSXFS": {
        "miss": [],
        "beat": [{"symbol": "XLY", "dir": "SELL", "size": "3%"}, {"symbol": "SPY", "dir": "SELL", "size": "2%"}],
    },
    "DGORDER": {
        "miss": [],
        "beat": [{"symbol": "XLI", "dir": "SELL", "size": "2%"}],
    },
    "ICSA": {"miss": [], "beat": []},
    "HOUST": {"miss": [], "beat": []},
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Live Predictions",
        "Trade Simulator",
        "Model Performance",
        "Revision Explorer",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Economic Revision Predictor v1.0")
st.sidebar.caption("Model: LightGBM + Logistic + Bayesian Ensemble")
st.sidebar.caption(
    "Data: FRED/ALFRED vintage series (1995-2024)"
)


# ===================================================================
# PAGE: Dashboard
# ===================================================================

def page_dashboard():
    st.title("Economic Revision Predictor")
    st.markdown(
        "Predicts whether the **final** reading of US economic indicators "
        "will exceed the **initial** release, exploiting systematic revision biases."
    )

    # --- Revision bias overview ---
    st.header("Revision Bias Overview")
    st.markdown(
        "Each indicator has a historical tendency to be revised in a specific direction. "
        "Our model quantifies these biases and generates trading signals."
    )

    cols = st.columns(len(REVISION_BIASES))
    for i, (code, info) in enumerate(REVISION_BIASES.items()):
        with cols[i]:
            direction = info["direction"].upper()
            prob = info["probability"]
            arrow = "↑" if direction == "UP" else "↓" if direction == "DOWN" else "→"
            st.metric(
                label=INDICATOR_NAMES.get(code, code),
                value=f"{prob:.0%} {arrow}",
                delta=f"Revises {direction}",
                delta_color="normal" if direction == "UP" else "inverse" if direction == "DOWN" else "off",
            )

    # --- Strategy overview ---
    st.header("Trading Strategy")
    st.markdown("Based on revision biases, we trade the **reaction** to economic releases:")

    strategy_data = []
    for code in ["GDPC1", "PAYEMS", "RSXFS", "INDPRO", "DGORDER"]:
        bias = REVISION_BIASES[code]
        trades_miss = INDICATOR_TRADES.get(code, {}).get("miss", [])
        trades_beat = INDICATOR_TRADES.get(code, {}).get("beat", [])

        miss_action = ", ".join(f"{t['dir']} {t['symbol']}" for t in trades_miss) if trades_miss else "HOLD"
        beat_action = ", ".join(f"{t['dir']} {t['symbol']}" for t in trades_beat) if trades_beat else "HOLD"

        strategy_data.append({
            "Indicator": INDICATOR_NAMES[code],
            "Revision Bias": f"{bias['direction'].upper()} ({bias['probability']:.0%})",
            "On MISS": miss_action,
            "On BEAT": beat_action,
        })

    st.dataframe(
        pd.DataFrame(strategy_data),
        use_container_width=True,
        hide_index=True,
    )

    # --- How it works ---
    st.header("How It Works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Data")
        st.markdown(
            "- Historical vintages from FRED/ALFRED API\n"
            "- 7 indicators from 1995-2024\n"
            "- Point-in-time data to prevent leakage"
        )

    with col2:
        st.subheader("2. Model")
        st.markdown(
            "- 3-model ensemble (LightGBM, Logistic, Bayesian)\n"
            "- Isotonic calibration layer\n"
            "- Meta-learner stacking"
        )

    with col3:
        st.subheader("3. Trade")
        st.markdown(
            "- Monitor economic releases\n"
            "- Compare actual vs consensus\n"
            "- Execute when revision bias gives edge"
        )


# ===================================================================
# PAGE: Live Predictions
# ===================================================================

def page_live_predictions():
    st.title("Live Predictions")
    st.markdown("Current model predictions for recent economic releases.")

    predictions = load_predictions()
    if predictions.empty:
        st.warning("No prediction data found. Run the pipeline first: `python3 main.py --full`")
        return

    # Convert columns
    predictions["predicted_prob_up"] = predictions["predicted_prob_up"].astype(float)
    predictions["reference_date"] = pd.to_datetime(predictions["reference_date"])
    predictions["initial_date"] = pd.to_datetime(predictions["initial_date"])

    # Indicator filter
    indicators = sorted(predictions["series_id"].unique())
    selected = st.multiselect(
        "Filter by indicator",
        indicators,
        default=indicators,
        format_func=lambda x: f"{INDICATOR_NAMES.get(x, x)} ({x})",
    )
    filtered = predictions[predictions["series_id"].isin(selected)]

    # Confidence filter
    min_conf = st.slider("Minimum confidence (P(up))", 0.0, 1.0, 0.0, 0.05)
    filtered = filtered[
        (filtered["predicted_prob_up"] >= min_conf) |
        (filtered["predicted_prob_up"] <= 1 - min_conf)
    ]

    # Sort by strength of conviction
    filtered["conviction"] = (filtered["predicted_prob_up"] - 0.5).abs()
    filtered = filtered.sort_values("conviction", ascending=False)

    # Display
    for _, row in filtered.iterrows():
        prob = row["predicted_prob_up"]
        series = row["series_id"]
        name = INDICATOR_NAMES.get(series, series)
        ref_date = row["reference_date"].strftime("%Y-%m-%d")

        if prob > 0.6:
            signal = "UP"
            colour = "green"
            icon = "🟢"
        elif prob < 0.4:
            signal = "DOWN"
            colour = "red"
            icon = "🔴"
        else:
            signal = "NEUTRAL"
            colour = "orange"
            icon = "🟡"

        with st.container():
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                st.markdown(f"**{name}** ({series})")
                st.caption(f"Period: {ref_date}")
            with c2:
                st.metric("P(Revises Up)", f"{prob:.1%}")
            with c3:
                st.markdown(f"### {icon} {signal}")
            with c4:
                st.metric("Initial Value", f"{row['initial_value']:,.1f}")
            st.divider()

    st.caption(f"Showing {len(filtered)} of {len(predictions)} predictions")


# ===================================================================
# PAGE: Trade Simulator
# ===================================================================

def page_trade_simulator():
    st.title("Trade Simulator")
    st.markdown(
        "Simulate what the bot would do when an economic release comes out. "
        "Enter the release details below."
    )

    col1, col2 = st.columns(2)

    with col1:
        indicator = st.selectbox(
            "Economic Indicator",
            list(INDICATOR_NAMES.keys()),
            format_func=lambda x: f"{INDICATOR_NAMES[x]} ({x})",
        )

        actual = st.number_input(
            "Actual Release Value",
            value=150000.0 if indicator == "PAYEMS" else 2.0,
            format="%.2f",
        )

    with col2:
        consensus = st.number_input(
            "Consensus / Forecast",
            value=200000.0 if indicator == "PAYEMS" else 2.5,
            format="%.2f",
        )

        portfolio_size = st.number_input(
            "Portfolio Size ($)",
            value=100000,
            min_value=1000,
            step=10000,
        )

    if st.button("Simulate Release", type="primary"):
        # Calculate surprise
        surprise = actual - consensus
        surprise_pct = (surprise / abs(consensus)) * 100 if consensus != 0 else 0
        is_miss = surprise < 0
        is_big = abs(surprise_pct) > 5

        bias = REVISION_BIASES.get(indicator, {})
        direction = bias.get("direction", "neutral")
        prob = bias.get("probability", 0.5)

        st.markdown("---")

        # Release summary
        st.subheader("Release Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual", f"{actual:,.2f}")
        c2.metric("Consensus", f"{consensus:,.2f}")
        c3.metric("Surprise", f"{surprise:+,.2f}", delta=f"{surprise_pct:+.1f}%")
        c4.metric("Type", "MISS" if is_miss else "BEAT")

        # Revision analysis
        st.subheader("Revision Analysis")
        st.info(
            f"**{INDICATOR_NAMES[indicator]}** has a **{direction.upper()}** revision bias "
            f"({prob:.0%} probability). "
            f"{'This miss is likely to be revised up.' if is_miss and direction == 'up' else ''}"
            f"{'This beat is likely to be revised down.' if not is_miss and direction == 'down' else ''}"
            f"{'No strong revision edge here.' if direction == 'neutral' else ''}"
        )

        # Trading signal
        st.subheader("Trading Signal")
        trades_config = INDICATOR_TRADES.get(indicator, {})
        scenario = "miss" if is_miss else "beat"
        trades = trades_config.get(scenario, [])

        if not trades:
            st.warning("**HOLD** - No trading edge for this scenario.")
        else:
            # Determine signal
            if is_big:
                signal_type = "STRONG_BUY" if trades[0]["dir"] == "BUY" else "STRONG_SELL"
            else:
                signal_type = trades[0]["dir"]

            if "BUY" in signal_type:
                st.success(f"**Signal: {signal_type}**")
            else:
                st.error(f"**Signal: {signal_type}**")

            st.markdown("**Trades to Execute:**")
            trade_data = []
            for t in trades:
                size_pct = float(t["size"].replace("%", ""))
                if is_big:
                    size_pct = min(size_pct * 1.5, 7.0)
                dollar_amount = portfolio_size * (size_pct / 100)
                trade_data.append({
                    "Symbol": t["symbol"],
                    "Direction": t["dir"],
                    "Size (%)": f"{size_pct:.1f}%",
                    "Dollar Amount": f"${dollar_amount:,.0f}",
                    "Stop Loss": "2.0%",
                    "Take Profit": "4.0%",
                })

            st.dataframe(
                pd.DataFrame(trade_data),
                use_container_width=True,
                hide_index=True,
            )

            # Reasoning
            st.subheader("Reasoning")
            if is_miss and direction == "up":
                st.markdown(
                    f"- Release **missed** consensus by {abs(surprise_pct):.1f}%\n"
                    f"- Historical data shows **{prob:.0%}** chance of upward revision\n"
                    f"- Market is overreacting to the downside\n"
                    f"- **Buy the dip** - the number will likely be revised better"
                )
            elif not is_miss and direction == "down":
                st.markdown(
                    f"- Release **beat** consensus by {surprise_pct:.1f}%\n"
                    f"- Historical data shows **{prob:.0%}** chance of downward revision\n"
                    f"- Market is overreacting to the upside\n"
                    f"- **Fade the rally** - the number will likely be revised lower"
                )


# ===================================================================
# PAGE: Model Performance
# ===================================================================

def page_model_performance():
    st.title("Model Performance")

    # --- Model weights ---
    weights = load_model_weights()
    if not weights.empty:
        st.header("Ensemble Model Weights")
        st.markdown(
            "The meta-learner assigns weights to each base model's calibrated output:"
        )

        for _, row in weights.iterrows():
            st.progress(
                float(row["relative_weight"]),
                text=f"**{row['model'].replace('_', ' ').title()}**: {row['relative_weight']:.1%}",
            )

    # --- Indicator breakdown ---
    breakdown = load_indicator_breakdown()
    if not breakdown.empty:
        st.header("Performance by Indicator")

        display_df = breakdown.copy()
        display_df["indicator"] = display_df["indicator"].map(
            lambda x: f"{INDICATOR_NAMES.get(x, x)} ({x})"
        )
        display_df = display_df.rename(columns={
            "indicator": "Indicator",
            "brier_score": "Brier Score",
            "brier_skill_score": "Brier Skill",
            "auc_roc": "AUC-ROC",
            "calibration_error": "Calibration Error",
            "n_samples": "Samples",
            "base_rate": "Base Rate (% Up)",
        })

        # Format base rate as percentage
        display_df["Base Rate (% Up)"] = display_df["Base Rate (% Up)"].apply(
            lambda x: f"{x:.0%}"
        )

        st.dataframe(
            display_df[["Indicator", "Brier Score", "Brier Skill", "Calibration Error", "Samples", "Base Rate (% Up)"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "**Brier Skill Score** > 0 means the model outperforms the naive baseline "
            "(always predicting the historical base rate)."
        )

    # --- Feature importance ---
    importance = load_feature_importance()
    if not importance.empty:
        st.header("Top Feature Importances")
        chart_df = importance.head(15).copy()
        chart_df = chart_df.sort_values("importance", ascending=True)
        st.bar_chart(
            chart_df.set_index("feature")["importance"],
            horizontal=True,
        )

    # --- Calibration plots ---
    st.header("Calibration Plots")
    cal_img = EVAL_DIR / "calibration_comparison.png"
    if cal_img.exists():
        st.image(str(cal_img), caption="Calibration Comparison Across Models")
    else:
        st.info("Run the pipeline to generate calibration plots: `python3 main.py --full`")

    # Reliability diagrams
    st.subheader("Reliability Diagrams")
    rel_cols = st.columns(3)
    model_names = ["lightgbm", "logistic_regression", "bayesian_logistic"]
    for i, model in enumerate(model_names):
        img_path = EVAL_DIR / f"reliability_{model}.png"
        if img_path.exists():
            with rel_cols[i % 3]:
                st.image(str(img_path), caption=model.replace("_", " ").title())


# ===================================================================
# PAGE: Revision Explorer
# ===================================================================

def page_revision_explorer():
    st.title("Revision History Explorer")
    st.markdown("Explore the raw revision data from FRED/ALFRED vintage archives.")

    revision_data = load_revision_data()
    if not revision_data:
        st.warning("No revision data found. Run: `python3 main.py --download`")
        return

    indicator = st.selectbox(
        "Select Indicator",
        list(revision_data.keys()),
        format_func=lambda x: f"{INDICATOR_NAMES.get(x, x)} ({x})",
    )

    df = revision_data[indicator]
    st.markdown(f"**{len(df)} revision records** for {INDICATOR_NAMES.get(indicator, indicator)}")

    # Show column info
    st.markdown(f"**Columns:** {', '.join(df.columns.tolist())}")

    # If we have initial_value and final_value, compute revision stats
    if "initial_value" in df.columns and "final_value" in df.columns:
        df_clean = df.dropna(subset=["initial_value", "final_value"])
        df_clean["revision"] = df_clean["final_value"] - df_clean["initial_value"]
        df_clean["revised_up"] = df_clean["revision"] > 0

        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revisions", len(df_clean))
        c2.metric("Revised Up", f"{df_clean['revised_up'].mean():.1%}")
        c3.metric("Avg Revision", f"{df_clean['revision'].mean():+.4f}")
        c4.metric("Median Revision", f"{df_clean['revision'].median():+.4f}")

        # Revision distribution
        st.subheader("Revision Distribution")
        st.bar_chart(df_clean["revision"].value_counts().sort_index().head(50))

        # Time series of revisions
        if "reference_date" in df_clean.columns:
            df_clean["reference_date"] = pd.to_datetime(df_clean["reference_date"])
            ts = df_clean.set_index("reference_date")["revision"].sort_index()
            st.subheader("Revisions Over Time")
            st.line_chart(ts)

    # Raw data table
    with st.expander("View Raw Data"):
        st.dataframe(df.head(200), use_container_width=True)


# ===================================================================
# Router
# ===================================================================

if page == "Dashboard":
    page_dashboard()
elif page == "Live Predictions":
    page_live_predictions()
elif page == "Trade Simulator":
    page_trade_simulator()
elif page == "Model Performance":
    page_model_performance()
elif page == "Revision Explorer":
    page_revision_explorer()

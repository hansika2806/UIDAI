"""
Robust Statistical Indicators for UIDAI Governance Framework.

Replaces fragile min-max scaling with MAD normalization and
simple lag-correlation TPS with Shannon entropy-based predictability.

Mathematical foundations:
    MAD normalization:  x_tilde = (x - median(X)) / (1.4826 * MAD(X))
    Shannon entropy:    H(X) = -sum(p(x_t) * log2(p(x_t)))
    Normalized TPS:     TPS_entropy = 1 - H(X) / H_max
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-10


def mad_normalize(series: pd.Series) -> pd.Series:
    """Median Absolute Deviation normalization.

    Robust alternative to min-max scaling. Uses the consistency constant
    1.4826 so that MAD estimates the standard deviation for Gaussian data.

    Formula:
        x_tilde = (x - median(X)) / (1.4826 * MAD(X))

    Returns values centred at 0, where |x_tilde| > 3 suggests outliers.
    """
    median_val = series.median()
    mad_val = (series - median_val).abs().median()
    if mad_val < EPS:
        logger.warning("MAD is near-zero; returning zero-centred series.")
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - median_val) / (1.4826 * mad_val)


def mad_normalize_to_unit(series: pd.Series) -> pd.Series:
    """MAD-normalize then squash to [0, 1] via sigmoid.

    This gives an outlier-robust score on a bounded scale:
        score = 1 / (1 + exp(-x_tilde))
    """
    z = mad_normalize(series)
    return 1.0 / (1.0 + np.exp(-z))


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a discrete distribution.

    H(X) = -sum(p_i * log2(p_i))

    Parameters
    ----------
    counts : array-like
        Non-negative counts or values that will be normalized to probabilities.

    Returns
    -------
    float
        Entropy in bits. Zero if the distribution is degenerate.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total < EPS:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def compute_entropy_tps(state_month: pd.DataFrame) -> pd.DataFrame:
    """Compute entropy-based Temporal Predictability Score per state.

    For each state, discretises the monthly Total_Activity into bins
    and measures Shannon entropy. Low entropy => predictable pattern.

    TPS_entropy = 1 - H(X) / H_max

    H_max = log2(n_bins) is the maximum possible entropy (uniform dist).
    """
    n_bins = 12  # monthly seasonality granularity
    results = []

    for state in sorted(state_month["state"].unique()):
        df_s = state_month[state_month["state"] == state]

        if len(df_s) < 3:
            results.append({
                "state": state,
                "entropy_raw": np.nan,
                "entropy_normalized": np.nan,
                "TPS_entropy": np.nan,
            })
            continue

        activity = df_s["Total_Activity"].values
        hist_counts, _ = np.histogram(activity, bins=min(n_bins, len(activity)))
        h = shannon_entropy(hist_counts)
        h_max = np.log2(len(hist_counts)) if len(hist_counts) > 1 else 1.0
        tps = 1.0 - (h / h_max) if h_max > 0 else 0.0

        results.append({
            "state": state,
            "entropy_raw": h,
            "entropy_normalized": h / h_max if h_max > 0 else 0.0,
            "TPS_entropy": tps,
        })

    return pd.DataFrame(results)


def compute_robust_indicators(state_master: pd.DataFrame, state_month: pd.DataFrame) -> pd.DataFrame:
    """Recompute AMI, UPI, VSI, TPS using robust statistical methods.

    Replaces min-max with MAD-sigmoid normalization and lag-corr TPS
    with entropy-based TPS.

    Returns a copy of state_master with new robust columns.
    """
    logger.info("Computing robust statistical indicators (MAD + Entropy)...")
    result = state_master.copy()

    # --- Robust AMI: MAD-sigmoid of raw rank sum ---
    if "AMI_raw" in result.columns:
        result["AMI_robust"] = 1.0 - mad_normalize_to_unit(result["AMI_raw"])
    else:
        rank_cols = ["E_total", "D_total", "B_total"]
        for col in rank_cols:
            result[f"{col}_rank"] = result[col].rank(ascending=False, method="average")
        raw_rank_sum = sum(result[f"{col}_rank"] for col in rank_cols)
        result["AMI_robust"] = 1.0 - mad_normalize_to_unit(raw_rank_sum)

    # --- Robust UPI: MAD-sigmoid of log update pressure ---
    upi_raw = np.log1p(result["D_total"] + result["B_total"]) - np.log1p(result["E_total"])
    result["UPI_robust"] = mad_normalize_to_unit(upi_raw)

    # --- Robust VSI: MAD-sigmoid of ratio CV ---
    if "ratio_cv" in result.columns:
        result["VSI_robust"] = mad_normalize_to_unit(result["ratio_cv"])
    elif "VSI_raw" in result.columns:
        result["VSI_robust"] = mad_normalize_to_unit(result["VSI_raw"])
    else:
        result["VSI_robust"] = mad_normalize_to_unit(result.get("VSI", pd.Series(0, index=result.index)))

    # --- Entropy-based TPS ---
    entropy_df = compute_entropy_tps(state_month)
    result = result.merge(entropy_df[["state", "TPS_entropy", "entropy_raw"]], on="state", how="left")
    result["TPS_robust"] = result["TPS_entropy"].fillna(0.0)

    logger.info("Robust indicators computed successfully.")
    return result


def compute_all_robust_outputs(
    state_master: pd.DataFrame, state_month: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Entry point: compute all robust indicator outputs.

    Returns
    -------
    dict with keys:
        - 'state_master_robust': state_master augmented with robust columns
        - 'entropy_details': per-state entropy breakdown
    """
    entropy_df = compute_entropy_tps(state_month)
    robust_master = compute_robust_indicators(state_master, state_month)

    return {
        "state_master_robust": robust_master,
        "entropy_details": entropy_df,
    }

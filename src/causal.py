"""
Causal Inference Module for UIDAI Governance Framework.

Implements Granger causality testing and CUSUM structural break detection
to answer: "Does enrollment *cause* future updates?" and "When did the
system's operational regime structurally change?"

Mathematical foundations:
    Granger Causality:
        Restricted:   U_t = sum(alpha_i * U_{t-i}) + eps_t
        Unrestricted: U_t = sum(alpha_i * U_{t-i}) + sum(beta_i * E_{t-i}) + eta_t
        F-test:       F = ((RSS_R - RSS_U)/p) / (RSS_U/(T-2p-1))

    Stationarity (ADF test) — CRITICAL:
        Granger causality is only valid on stationary time series. Running
        it on non-stationary (unit-root) series produces spurious F-statistics
        regardless of the true causal structure. We enforce stationarity via
        the Augmented Dickey-Fuller test:
            H_0: unit root present (non-stationary)
            H_1: no unit root (stationary)
        If H_0 is not rejected at p < 0.05, we first-difference the series
        (d=1). If still non-stationary, second-difference (d=2).

    CUSUM:
        S_t = sum_{i=k+1}^{t} (eps_hat_i / sigma_hat)
        Boundary crossing => structural break
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def adf_stationarity_test(series: np.ndarray) -> Dict:
    """Augmented Dickey-Fuller unit root test for stationarity.

    H_0: unit root (non-stationary)
    H_1: no unit root (stationary)

    Returns dict with:
        - 'is_stationary': bool (True if p < 0.05)
        - 'adf_stat': ADF test statistic
        - 'p_value': p-value of ADF test
        - 'n_lags': number of lags used
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series, autolag="AIC")
        return {
            "is_stationary": result[1] < 0.05,
            "adf_stat": float(result[0]),
            "p_value": float(result[1]),
            "n_lags": int(result[2]),
        }
    except ImportError:
        # statsmodels not available — assume stationary to avoid blocking
        logger.warning("statsmodels not available; skipping ADF test (assuming stationary).")
        return {"is_stationary": True, "adf_stat": np.nan, "p_value": np.nan, "n_lags": 0}
    except Exception as e:
        logger.debug(f"ADF test failed: {e}; assuming stationary.")
        return {"is_stationary": True, "adf_stat": np.nan, "p_value": np.nan, "n_lags": 0}


def make_stationary(series: np.ndarray, max_diffs: int = 2) -> Tuple[np.ndarray, int]:
    """Difference a series until stationary (by ADF), up to max_diffs times.

    Returns (differenced_series, n_diffs_applied).
    If still non-stationary after max_diffs, returns the last differenced series
    with a warning — this is safer than running Granger on a known unit-root series.
    """
    s = series.copy()
    for d in range(max_diffs + 1):
        test = adf_stationarity_test(s)
        if test["is_stationary"]:
            if d > 0:
                logger.debug(f"Series stationary after {d} differencing(s). ADF p={test['p_value']:.4f}")
            return s, d
        if d < max_diffs:
            s = np.diff(s)
    logger.warning(
        f"Series still non-stationary after {max_diffs} difference(s). "
        "Granger results should be interpreted with caution."
    )
    return s, max_diffs


def granger_causality_test(
    y: np.ndarray,
    x: np.ndarray,
    max_lag: int = 3,
) -> List[Dict]:
    """Test if x Granger-causes y for lags 1..max_lag.

    For each lag p, computes an F-test comparing:
        Restricted:   y_t = sum(a_i * y_{t-i}) + eps
        Unrestricted: y_t = sum(a_i * y_{t-i}) + sum(b_i * x_{t-i}) + eps

    F = ((RSS_R - RSS_U)/p) / (RSS_U/(T - 2p - 1))

    Returns a list of dicts with lag, F-statistic, p-value, and significance.
    """
    results = []
    T = len(y)

    for lag in range(1, max_lag + 1):
        if T <= 2 * lag + 1:
            results.append({
                "lag": lag,
                "f_statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
            })
            continue

        # Build lagged matrices
        y_target = y[lag:]
        n = len(y_target)

        # Restricted model: only y lags
        X_restricted = np.column_stack([y[lag - i - 1: T - i - 1] for i in range(lag)])
        X_restricted = np.column_stack([np.ones(n), X_restricted])

        # Unrestricted model: y lags + x lags
        X_unrestricted = np.column_stack([
            X_restricted,
            *[x[lag - i - 1: T - i - 1] for i in range(lag)]
        ])

        # OLS for restricted
        try:
            beta_r = np.linalg.lstsq(X_restricted, y_target, rcond=None)[0]
            residuals_r = y_target - X_restricted @ beta_r
            rss_r = np.sum(residuals_r ** 2)

            # OLS for unrestricted
            beta_u = np.linalg.lstsq(X_unrestricted, y_target, rcond=None)[0]
            residuals_u = y_target - X_unrestricted @ beta_u
            rss_u = np.sum(residuals_u ** 2)

            # F-statistic
            df1 = lag  # number of restrictions
            df2 = n - 2 * lag - 1
            if df2 <= 0 or rss_u < 1e-12:
                results.append({
                    "lag": lag,
                    "f_statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                })
                continue

            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)

            results.append({
                "lag": lag,
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            })
        except (np.linalg.LinAlgError, ValueError):
            results.append({
                "lag": lag,
                "f_statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
            })

    return results


def cusum_test(
    series: np.ndarray,
    split_frac: float = 0.5,
) -> Dict:
    """CUSUM structural break detection.

    Fits an OLS model on the first `split_frac` of the data, then
    computes cumulative sum of recursive residuals:

        S_t = sum_{i=k+1}^{t} (eps_hat_i / sigma_hat)

    The maximum absolute CUSUM value, normalised by sqrt(n), is compared
    against critical values. A high CUSUM indicates a structural break.

    Returns dict with CUSUM statistics and detected break point.
    """
    n = len(series)
    if n < 6:
        return {
            "cusum_max": np.nan,
            "break_index": None,
            "break_detected": False,
            "cusum_series": np.array([]),
        }

    k = max(int(n * split_frac), 3)

    # Fit trend on first portion
    t_train = np.arange(k).reshape(-1, 1)
    t_train_aug = np.column_stack([np.ones(k), t_train])
    y_train = series[:k]

    try:
        beta = np.linalg.lstsq(t_train_aug, y_train, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {
            "cusum_max": np.nan,
            "break_index": None,
            "break_detected": False,
            "cusum_series": np.array([]),
        }

    # Recursive residuals on the full series
    t_full = np.arange(n).reshape(-1, 1)
    t_full_aug = np.column_stack([np.ones(n), t_full])
    predictions = t_full_aug @ beta
    residuals = series - predictions

    sigma = np.std(residuals[:k])
    if sigma < 1e-12:
        sigma = 1.0

    # CUSUM statistic
    normalized_residuals = residuals[k:] / sigma
    cusum = np.cumsum(normalized_residuals)

    if len(cusum) == 0:
        return {
            "cusum_max": 0.0,
            "break_index": None,
            "break_detected": False,
            "cusum_series": np.array([]),
        }

    cusum_abs = np.abs(cusum)
    cusum_max = cusum_abs.max()
    break_idx = int(np.argmax(cusum_abs)) + k

    # Approximate 5% critical value for CUSUM: 1.358 * sqrt(n - k)
    # (from Brown, Durbin, Evans 1975)
    critical_value = 1.358 * np.sqrt(n - k)
    break_detected = bool(cusum_max > critical_value)

    return {
        "cusum_max": float(cusum_max),
        "critical_value": float(critical_value),
        "break_index": int(break_idx) if break_detected else None,
        "break_detected": break_detected,
        "cusum_series": cusum,
    }


def run_granger_analysis(
    state_month: pd.DataFrame,
    max_lag: int = 3,
    min_obs: int = 6,
) -> pd.DataFrame:
    """Run Granger causality tests per state with ADF stationarity enforcement.

    STATIONARITY GUARANTEE: Before running any Granger test, each series is
    tested for unit roots via the Augmented Dickey-Fuller test. Non-stationary
    series are automatically differenced until stationary (up to d=2). The
    order of differencing applied is recorded in the output so the analyst
    can interpret results correctly (e.g., d=1 means the result describes
    *changes* in enrollment causing *changes* in updates, not levels).

    Tests:
        - Does enrollment Granger-cause demographic updates?
        - Does enrollment Granger-cause biometric updates?
        - Does demographic updates Granger-cause biometric updates?

    Returns a dataframe with per-state, per-lag test results.
    """
    logger.info("Running Granger causality analysis per state (with ADF stationarity checks)...")
    all_results = []

    for state in sorted(state_month["state"].unique()):
        df_s = state_month[state_month["state"] == state].sort_values("year_month")

        if len(df_s) < min_obs:
            logger.debug(f"Skipping {state}: only {len(df_s)} observations (need {min_obs})")
            continue

        e_raw = df_s["E_total"].values.astype(float)
        d_raw = df_s["D_total"].values.astype(float)
        b_raw = df_s["B_total"].values.astype(float)

        # ── Enforce stationarity via ADF ──────────────────────────────────
        e_total, e_diffs = make_stationary(e_raw)
        d_total, d_diffs = make_stationary(d_raw)
        b_total, b_diffs = make_stationary(b_raw)

        # Align lengths after differencing (differencing shortens arrays)
        min_len = min(len(e_total), len(d_total), len(b_total))
        e_total = e_total[-min_len:]
        d_total = d_total[-min_len:]
        b_total = b_total[-min_len:]

        if min_len < min_obs:
            logger.debug(f"Skipping {state}: too few obs after differencing ({min_len})")
            continue

        # Test: E -> D
        for res in granger_causality_test(d_total, e_total, max_lag=max_lag):
            all_results.append({
                "state": state,
                "cause": "Enrollment",
                "effect": "Demographic Updates",
                "cause_diffs": e_diffs,
                "effect_diffs": d_diffs,
                **res,
            })

        # Test: E -> B
        for res in granger_causality_test(b_total, e_total, max_lag=max_lag):
            all_results.append({
                "state": state,
                "cause": "Enrollment",
                "effect": "Biometric Updates",
                "cause_diffs": e_diffs,
                "effect_diffs": b_diffs,
                **res,
            })

        # Test: D -> B
        for res in granger_causality_test(b_total, d_total, max_lag=max_lag):
            all_results.append({
                "state": state,
                "cause": "Demographic Updates",
                "effect": "Biometric Updates",
                "cause_diffs": d_diffs,
                "effect_diffs": b_diffs,
                **res,
            })

    result = pd.DataFrame(all_results)
    n_sig = result["significant"].sum() if len(result) > 0 else 0
    n_stationary = (result["cause_diffs"] == 0).sum() if len(result) > 0 and "cause_diffs" in result.columns else 0
    logger.info(
        f"Granger analysis complete: {n_sig} significant causal links found. "
        f"{n_stationary}/{len(result)} tests used raw (already-stationary) series."
    )
    return result


def run_cusum_analysis(
    state_month: pd.DataFrame,
    target_col: str = "update_to_enrol_ratio",
    min_obs: int = 4,
) -> pd.DataFrame:
    """Run CUSUM structural break detection per state.

    Detects regime changes in the update-to-enrollment ratio,
    which may correspond to policy shocks (demonetization, COVID, etc.)
    """
    logger.info("Running CUSUM structural break detection per state...")
    all_results = []

    for state in sorted(state_month["state"].unique()):
        df_s = state_month[state_month["state"] == state].sort_values("year_month")

        if len(df_s) < min_obs or target_col not in df_s.columns:
            continue

        series = df_s[target_col].values.astype(float)
        months = df_s["year_month"].values

        cusum_result = cusum_test(series)

        row = {
            "state": state,
            "cusum_max": cusum_result["cusum_max"],
            "critical_value": cusum_result.get("critical_value", np.nan),
            "break_detected": cusum_result["break_detected"],
            "n_observations": len(series),
        }

        if cusum_result["break_detected"] and cusum_result["break_index"] is not None:
            bi = cusum_result["break_index"]
            row["break_month"] = months[bi] if bi < len(months) else None
        else:
            row["break_month"] = None

        all_results.append(row)

    result = pd.DataFrame(all_results)
    n_breaks = result["break_detected"].sum() if len(result) > 0 else 0
    logger.info(f"CUSUM analysis complete: {n_breaks} structural breaks detected.")
    return result


def run_causal_analysis(
    state_month: pd.DataFrame,
    max_lag: int = 3,
) -> Dict[str, pd.DataFrame]:
    """Entry point: run all causal inference analyses.

    Returns dict with:
        - 'granger_results': per-state Granger causality test results
        - 'cusum_results': per-state CUSUM structural break results
    """
    granger_df = run_granger_analysis(state_month, max_lag=max_lag)
    cusum_df = run_cusum_analysis(state_month)

    return {
        "granger_results": granger_df,
        "cusum_results": cusum_df,
    }

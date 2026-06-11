"""
Data & Model Drift Monitoring for UIDAI Governance Framework.

Implements two statistical drift detection strategies:

1. Covariate / Feature Drift (KS Test)
   Run two-sample Kolmogorov-Smirnov tests on feature distributions.
   H₀: Baseline and current feature distributions are identical.
   If p < α (default 0.05), drift is detected on that feature.

2. Concept / Model Drift (Population Stability Index — PSI)
   Measure how much the GMM soft cluster assignment distribution has
   shifted between a baseline run and the current run.

   PSI = Σ (P_current - P_baseline) * ln(P_current / P_baseline)

   Interpretation:
       PSI < 0.10  → Stable (no meaningful shift)
       PSI < 0.25  → Moderate shift (monitor closely)
       PSI ≥ 0.25  → Significant shift (trigger retraining)
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

INDICATOR_COLS = ["AMI", "UPI", "VSI", "TPS"]
PSI_STABLE = 0.10
PSI_WARNING = 0.25

DRIFT_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "analysis",
    "drift_report.csv",
)


def detect_covariate_drift(
    baseline: pd.Series,
    current: pd.Series,
    alpha: float = 0.05,
) -> Dict:
    """Two-sample Kolmogorov-Smirnov test for feature distribution drift.

    KS statistic: D = sup_x |F₁(x) - F₂(x)|

    Parameters
    ----------
    baseline : pd.Series
        Distribution of the feature during baseline / training period.
    current : pd.Series
        Distribution of the feature in the current pipeline run.
    alpha : float
        Significance level for the hypothesis test.

    Returns
    -------
    dict with keys:
        - feature         : column name
        - ks_stat         : KS test statistic
        - p_value         : p-value
        - drift_detected  : bool (True = drift detected)
        - severity        : 'stable' | 'warning' | 'critical'
    """
    baseline_clean = baseline.dropna().values
    current_clean = current.dropna().values

    if len(baseline_clean) < 2 or len(current_clean) < 2:
        return {
            "ks_stat": float("nan"),
            "p_value": float("nan"),
            "drift_detected": False,
            "severity": "insufficient_data",
        }

    ks_stat, p_value = stats.ks_2samp(baseline_clean, current_clean)
    drift_detected = bool(p_value < alpha)

    if not drift_detected:
        severity = "stable"
    elif p_value < alpha / 10:
        severity = "critical"
    else:
        severity = "warning"

    return {
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "drift_detected": drift_detected,
        "severity": severity,
    }


def calculate_psi(
    baseline_probs: np.ndarray,
    current_probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Population Stability Index between two distributions.

    PSI = Σ (P_current_bin - P_baseline_bin) * ln(P_current_bin / P_baseline_bin)

    Parameters
    ----------
    baseline_probs : np.ndarray
        Predicted probabilities / scores from the training/baseline run.
    current_probs : np.ndarray
        Predicted probabilities / scores from the current run.
    n_bins : int
        Number of equal-width bins for the empirical distribution.

    Returns
    -------
    float : PSI score (0 = no shift, >0.25 = significant shift)
    """
    # Create bins from the combined range
    combined = np.concatenate([baseline_probs, current_probs])
    bin_edges = np.linspace(combined.min(), combined.max() + 1e-10, n_bins + 1)

    # Compute bin probabilities for each distribution
    baseline_counts = np.histogram(baseline_probs, bins=bin_edges)[0]
    current_counts = np.histogram(current_probs, bins=bin_edges)[0]

    # Normalise to proportions, smoothing zeros to avoid log(0)
    eps = 1e-4
    baseline_pct = (baseline_counts + eps) / (len(baseline_probs) + eps * n_bins)
    current_pct = (current_counts + eps) / (len(current_probs) + eps * n_bins)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def run_drift_check(
    current_state_master: pd.DataFrame,
    indicator_cols: Optional[List[str]] = None,
    baseline_path: Optional[str] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run full covariate + model drift check.

    1. Load baseline indicator distributions from the saved baseline CSV
       (or skip PSI if not available yet — first run).
    2. Run KS-test on each indicator column.
    3. Compute PSI on the Ensemble_Anomaly_Score (if present).
    4. Save the drift report to data/analysis/drift_report.csv.

    Parameters
    ----------
    current_state_master : pd.DataFrame
        The fully enriched state master from the current pipeline run.
    indicator_cols : list, optional
        Indicator features to check. Defaults to ['AMI', 'UPI', 'VSI', 'TPS'].
    baseline_path : str, optional
        Path to a saved baseline state_master CSV from a previous run.
        Defaults to data/analysis/state_master_full.csv (previous run's snapshot).
    alpha : float
        Significance level for KS tests.

    Returns
    -------
    pd.DataFrame
        Drift report with one row per feature.
    """
    if indicator_cols is None:
        indicator_cols = INDICATOR_COLS

    logger.info("=" * 60)
    logger.info("DRIFT MONITORING: Covariate Shift (KS Test) + Concept Drift (PSI)")
    logger.info("=" * 60)

    # Locate baseline file
    if baseline_path is None:
        baseline_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "analysis",
            "state_master_full.csv",
        )

    rows = []

    if not os.path.exists(baseline_path):
        logger.warning(
            f"Drift check skipped — no baseline found at {baseline_path}. "
            "This is expected on the first pipeline run. "
            "A baseline will be established after this run completes."
        )
        return pd.DataFrame()

    try:
        baseline_df = pd.read_csv(baseline_path)
    except Exception as e:
        logger.warning(f"Could not load baseline for drift check: {e}")
        return pd.DataFrame()

    # --- Covariate Drift (KS Test per indicator) ---
    for col in indicator_cols:
        if col not in baseline_df.columns or col not in current_state_master.columns:
            continue

        result = detect_covariate_drift(
            baseline=baseline_df[col],
            current=current_state_master[col],
            alpha=alpha,
        )

        rows.append(
            {
                "feature": col,
                "check_type": "KS_covariate",
                "ks_stat": result.get("ks_stat"),
                "p_value": result.get("p_value"),
                "drift_detected": result.get("drift_detected"),
                "severity": result.get("severity"),
                "psi": float("nan"),
            }
        )

        if result["drift_detected"]:
            logger.warning(
                f"⚠️  Covariate drift detected on '{col}': "
                f"KS={result['ks_stat']:.4f}, p={result['p_value']:.4f} "
                f"[{result['severity'].upper()}]"
            )
        else:
            logger.info(
                f"✓  '{col}' stable: KS={result['ks_stat']:.4f}, "
                f"p={result['p_value']:.4f}"
            )

    # --- Concept Drift (PSI on ensemble anomaly score, if available) ---
    if (
        "Ensemble_Anomaly_Score" in baseline_df.columns
        and "Ensemble_Anomaly_Score" in current_state_master.columns
    ):

        psi_score = calculate_psi(
            baseline_probs=baseline_df["Ensemble_Anomaly_Score"].dropna().values,
            current_probs=current_state_master["Ensemble_Anomaly_Score"]
            .dropna()
            .values,
        )

        if psi_score < PSI_STABLE:
            severity = "stable"
        elif psi_score < PSI_WARNING:
            severity = "warning"
        else:
            severity = "critical"

        rows.append(
            {
                "feature": "Ensemble_Anomaly_Score",
                "check_type": "PSI_concept",
                "ks_stat": float("nan"),
                "p_value": float("nan"),
                "drift_detected": psi_score >= PSI_STABLE,
                "severity": severity,
                "psi": psi_score,
            }
        )

        if psi_score >= PSI_WARNING:
            logger.warning(
                f"⚠️  Concept drift detected: PSI={psi_score:.4f} on "
                f"Ensemble_Anomaly_Score [{severity.upper()}]. "
                "Consider retraining anomaly detection models."
            )
        elif psi_score >= PSI_STABLE:
            logger.info(
                f"⚡ Moderate concept drift: PSI={psi_score:.4f} (threshold={PSI_STABLE})"
            )
        else:
            logger.info(f"✓  Anomaly score distribution stable: PSI={psi_score:.4f}")

    if not rows:
        return pd.DataFrame()

    drift_df = pd.DataFrame(rows)

    # Persist the report
    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    drift_df.to_csv(DRIFT_REPORT_PATH, index=False)
    logger.info(f"Drift report saved → {DRIFT_REPORT_PATH}")

    return drift_df

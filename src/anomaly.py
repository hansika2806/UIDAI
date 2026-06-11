"""
Advanced Anomaly Detection for UIDAI Governance Framework.

Replaces simple IQR-based anomaly detection with an ensemble of
Isolation Forest and ECOD (Empirical Cumulative Distribution-based
Outlier Detection).

Mathematical foundations:
    Isolation Forest:
        Anomaly score: s(x, n) = 2^(-E[h(x)] / c(n))
        where h(x) is isolation depth, c(n) = 2H(n-1) - 2(n-1)/n
        Score near 1 => anomaly, near 0.5 => normal

    ECOD:
        score(x) = -log(F_hat(x))
        where F_hat is the joint empirical CDF estimated per dimension.
        No hyperparameters, fully interpretable.

    Ensemble:
        Final score = weighted average of normalized IF and ECOD scores.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

INDICATOR_COLS = ["AMI", "UPI", "VSI", "TPS"]


def run_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.1,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Run Isolation Forest anomaly detection.

    The anomaly score is based on the average path length to isolate
    a point in random binary trees:

        s(x, n) = 2^(-E[h(x)] / c(n))

    where c(n) = 2*H(n-1) - 2*(n-1)/n is the average path length
    of unsuccessful search in a BST.

    Returns raw scores (more negative = more anomalous) and labels.
    """
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    labels = iso_forest.fit_predict(X)
    raw_scores = iso_forest.decision_function(X)

    # Convert to anomaly scores in [0, 1] where 1 = most anomalous
    # decision_function: negative = anomaly, positive = normal
    anomaly_scores = 1.0 - (raw_scores - raw_scores.min()) / (
        raw_scores.max() - raw_scores.min() + 1e-10
    )

    return {
        "labels": labels,  # -1 = anomaly, 1 = normal
        "raw_scores": raw_scores,
        "anomaly_scores": anomaly_scores,
    }


def run_ecod(
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run ECOD (Empirical Cumulative Distribution-based Outlier Detection).

    For each dimension, computes the empirical CDF and measures how
    extreme each observation is in the tails:

        score_j(x) = max(-log(F_j(x_j)), -log(1 - F_j(x_j)))
        score(x) = sum_j score_j(x)

    No hyperparameters. Fully interpretable: tells you exactly which
    dimension is driving the anomaly.
    """
    try:
        from pyod.models.ecod import ECOD

        ecod = ECOD(contamination=0.1)
        ecod.fit(X)
        raw_scores = ecod.decision_scores_
        labels = ecod.labels_  # 0 = normal, 1 = anomaly

        # Normalize to [0, 1]
        score_range = raw_scores.max() - raw_scores.min()
        if score_range < 1e-10:
            anomaly_scores = np.zeros(len(raw_scores))
        else:
            anomaly_scores = (raw_scores - raw_scores.min()) / score_range

        return {
            "labels": labels,
            "raw_scores": raw_scores,
            "anomaly_scores": anomaly_scores,
        }
    except ImportError:
        logger.warning(
            "pyod not installed; falling back to manual ECOD implementation."
        )
        return _manual_ecod(X)


def _manual_ecod(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Manual ECOD implementation without pyod dependency.

    For each feature j and observation i:
        left_tail_j  = -log(rank_j(x_ij) / (n+1))
        right_tail_j = -log(1 - rank_j(x_ij) / (n+1))
        score_j(x_i) = max(left_tail_j, right_tail_j)
        score(x_i)   = sum_j score_j(x_i)
    """
    n, d = X.shape
    scores = np.zeros(n)

    for j in range(d):
        col = X[:, j]
        ranks = pd.Series(col).rank(method="average").values

        # Empirical CDF: F(x) = rank / (n + 1)
        ecdf = ranks / (n + 1)

        # Tail scores: take the more extreme tail
        left_score = -np.log(ecdf + 1e-10)
        right_score = -np.log(1 - ecdf + 1e-10)
        dimension_score = np.maximum(left_score, right_score)

        scores += dimension_score

    # Normalize to [0, 1]
    score_range = scores.max() - scores.min()
    if score_range < 1e-10:
        anomaly_scores = np.zeros(n)
    else:
        anomaly_scores = (scores - scores.min()) / score_range

    # Label top 10% as anomalies
    threshold = np.percentile(scores, 90)
    labels = (scores >= threshold).astype(int)

    return {
        "labels": labels,
        "raw_scores": scores,
        "anomaly_scores": anomaly_scores,
    }


def ensemble_anomaly_scores(
    if_scores: np.ndarray,
    ecod_scores: np.ndarray,
    weight_if: float = 0.5,
    weight_ecod: float = 0.5,
) -> np.ndarray:
    """Combine Isolation Forest and ECOD scores via weighted average.

    ensemble_score = w_IF * IF_score + w_ECOD * ECOD_score

    Both scores should be in [0, 1] where 1 = most anomalous.
    """
    return weight_if * if_scores + weight_ecod * ecod_scores


def run_advanced_anomaly_detection(
    state_master: pd.DataFrame,
    indicator_cols: Optional[list] = None,
    contamination: float = 0.1,
) -> Dict[str, object]:
    """Full anomaly detection pipeline.

    1. Standardize indicator features
    2. Run Isolation Forest
    3. Run ECOD
    4. Compute ensemble anomaly scores
    5. Rank states by composite anomaly severity

    Returns dict with augmented state_master and component results,
    including the fitted scaler and Isolation Forest for serialization.
    """
    if indicator_cols is None:
        indicator_cols = INDICATOR_COLS

    logger.info("Running advanced anomaly detection (IF + ECOD ensemble)...")

    X_raw = state_master[indicator_cols].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    if_result = run_isolation_forest(X, contamination=contamination)
    iso_forest.fit(X)  # also keep the fitted model for serialization

    # ECOD
    ecod_result = run_ecod(X)

    # Ensemble
    ensemble_scores = ensemble_anomaly_scores(
        if_result["anomaly_scores"],
        ecod_result["anomaly_scores"],
    )

    # Build result
    result = state_master.copy()
    result["IF_Anomaly_Score"] = if_result["anomaly_scores"]
    result["IF_Anomaly_Label"] = (
        (if_result["labels"] == -1).astype(int)
        if if_result["labels"].min() < 0
        else if_result["labels"]
    )
    result["ECOD_Anomaly_Score"] = ecod_result["anomaly_scores"]
    result["ECOD_Anomaly_Label"] = ecod_result["labels"]
    result["Ensemble_Anomaly_Score"] = ensemble_scores

    # Classify: top 10% of ensemble scores as anomalous
    threshold = np.percentile(ensemble_scores, 90)
    result["Ensemble_Anomaly"] = (ensemble_scores >= threshold).astype(int)

    # Severity ranking
    result["Anomaly_Severity_Rank"] = (
        result["Ensemble_Anomaly_Score"].rank(ascending=False, method="min").astype(int)
    )

    n_anomalies = result["Ensemble_Anomaly"].sum()
    logger.info(
        f"Advanced anomaly detection complete: {n_anomalies} anomalies flagged."
    )

    return {
        "state_master_anomalies": result,
        "isolation_forest_result": if_result,
        "ecod_result": ecod_result,
        "ensemble_scores": ensemble_scores,
        # Fitted estimators for the model registry
        "scaler": scaler,
        "iso_forest": iso_forest,
    }

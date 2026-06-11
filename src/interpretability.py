"""
SHAP Interpretability Module for UIDAI Governance Framework.

Trains a Gradient Boosted Tree model to predict Priority_Score from
raw features, then uses TreeSHAP to explain each state's score.

Mathematical foundation (Shapley values from cooperative game theory):
    phi_j = sum_{S subset F\\{j}} (|S|! * (|F|-|S|-1)!) / |F|!
            * [f(S union {j}) - f(S)]

    This decomposes the model prediction into additive contributions
    from each feature, satisfying efficiency, symmetry, and linearity.

    TreeSHAP computes exact Shapley values in O(TLD^2) time for tree
    models, where T=trees, L=leaves, D=depth.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features used to predict Priority_Score
FEATURE_COLS = [
    "E_total",
    "D_total",
    "B_total",
    "Total_Activity",
    "log_activity",
    "Avg_Update_Ratio",
    "Avg_Demo_Ratio",
    "Avg_Bio_Ratio",
    "E_child_share",
    "E_minor_share",
    "D_adult_share",
    "B_adult_share",
    "Active_Months",
]

TARGET_COL = "Priority_Score"


def train_priority_model(
    state_master: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
) -> Optional[object]:
    """Train a GradientBoostingRegressor to predict Priority_Score.

    We train on the full dataset (not for generalisation, but for
    explanation). The goal is to build a model that SHAP can decompose.

    Uses sklearn GradientBoostingRegressor for TreeSHAP compatibility.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Only use features that exist in the dataframe
    available_features = [c for c in feature_cols if c in state_master.columns]
    if len(available_features) < 3:
        logger.warning(
            f"Only {len(available_features)} features available; skipping SHAP."
        )
        return None

    if target_col not in state_master.columns:
        logger.warning(f"Target column '{target_col}' not found; skipping SHAP.")
        return None

    from sklearn.ensemble import GradientBoostingRegressor

    X = state_master[available_features].fillna(0).values
    y = state_master[target_col].values

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    r2 = model.score(X, y)
    logger.info(
        f"Priority model trained: R² = {r2:.4f} on {len(available_features)} features."
    )

    model._feature_names = available_features
    return model


def compute_shap_values(
    model: object,
    state_master: pd.DataFrame,
) -> Optional[Dict]:
    """Compute SHAP values using TreeExplainer.

    TreeSHAP computes exact Shapley values for tree-based models:
        phi_j(x) = contribution of feature j to prediction f(x)

    Properties:
        sum_j phi_j(x) = f(x) - E[f(X)]  (efficiency/local accuracy)
    """
    feature_cols = model._feature_names
    X = state_master[feature_cols].fillna(0).values

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value

        logger.info(
            f"SHAP values computed for {X.shape[0]} states, {X.shape[1]} features."
        )

        return {
            "shap_values": shap_values,
            "expected_value": (
                float(expected_value.item())
                if hasattr(expected_value, "item")
                else float(expected_value)
            ),
            "feature_names": feature_cols,
            "X": X,
        }
    except ImportError:
        logger.warning(
            "shap library not installed; falling back to permutation importance."
        )
        return _fallback_importance(model, X, feature_cols, state_master)
    except Exception as e:
        logger.warning(
            f"SHAP computation failed ({e}); falling back to permutation importance."
        )
        return _fallback_importance(model, X, feature_cols, state_master)


def _fallback_importance(
    model: object,
    X: np.ndarray,
    feature_cols: List[str],
    state_master: pd.DataFrame,
) -> Dict:
    """Fallback: use sklearn feature importances if SHAP is unavailable."""
    importances = model.feature_importances_

    # Create pseudo-SHAP values proportional to feature importance * feature value
    pseudo_shap = np.zeros_like(X)
    for j, imp in enumerate(importances):
        col_std = X[:, j].std()
        if col_std > 0:
            pseudo_shap[:, j] = imp * (X[:, j] - X[:, j].mean()) / col_std
        else:
            pseudo_shap[:, j] = 0.0

    predictions = model.predict(X)
    expected_value = float(predictions.mean())

    return {
        "shap_values": pseudo_shap,
        "expected_value": expected_value,
        "feature_names": feature_cols,
        "X": X,
        "is_fallback": True,
    }


def build_shap_summary(
    shap_result: Dict,
    state_master: pd.DataFrame,
) -> pd.DataFrame:
    """Build a summary DataFrame of SHAP contributions per state.

    For each state, shows the top 3 most influential features
    and their SHAP contribution values.
    """
    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    states = state_master["state"].values

    rows = []
    for i, state in enumerate(states):
        sv = shap_values[i]
        # Sort features by absolute SHAP contribution
        sorted_indices = np.argsort(np.abs(sv))[::-1]

        top_features = []
        top_values = []
        for rank in range(min(3, len(sorted_indices))):
            idx = sorted_indices[rank]
            top_features.append(feature_names[idx])
            top_values.append(float(sv[idx]))

        row = {
            "state": state,
            "predicted_priority": float(shap_result["expected_value"] + sv.sum()),
            "shap_sum": float(sv.sum()),
            "top_1_feature": top_features[0] if len(top_features) > 0 else None,
            "top_1_shap": top_values[0] if len(top_values) > 0 else None,
            "top_2_feature": top_features[1] if len(top_features) > 1 else None,
            "top_2_shap": top_values[1] if len(top_values) > 1 else None,
            "top_3_feature": top_features[2] if len(top_features) > 2 else None,
            "top_3_shap": top_values[2] if len(top_values) > 2 else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_importance_df(shap_result: Dict) -> pd.DataFrame:
    """Compute global feature importance from mean |SHAP| values."""
    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    df["importance_pct"] = (
        100.0 * df["mean_abs_shap"] / (df["mean_abs_shap"].sum() + 1e-10)
    )

    return df


def run_interpretability(
    state_master: pd.DataFrame,
) -> Dict[str, object]:
    """Entry point: train model, compute SHAP, build summaries.

    Returns dict with:
        - 'shap_result': raw SHAP values and metadata
        - 'shap_summary': per-state top-3 feature contributions
        - 'feature_importance': global feature importance ranking
        - 'model': trained GBM model
    """
    logger.info("Running SHAP interpretability pipeline...")

    model = train_priority_model(state_master)
    if model is None:
        logger.warning("Model training failed; returning empty results.")
        return {
            "shap_result": None,
            "shap_summary": pd.DataFrame(),
            "feature_importance": pd.DataFrame(),
            "model": None,
        }

    shap_result = compute_shap_values(model, state_master)
    if shap_result is None:
        return {
            "shap_result": None,
            "shap_summary": pd.DataFrame(),
            "feature_importance": pd.DataFrame(),
            "model": model,
        }

    shap_summary = build_shap_summary(shap_result, state_master)
    feature_importance = build_feature_importance_df(shap_result)

    logger.info("SHAP interpretability complete.")

    return {
        "shap_result": shap_result,
        "shap_summary": shap_summary,
        "feature_importance": feature_importance,
        "model": model,
        "priority_model": model,  # explicit alias for model registry
    }

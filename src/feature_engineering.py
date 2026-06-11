import os
import logging
from typing import Dict

import pandas as pd

from analytics import build_analysis_outputs
from config import ANALYSIS_DATA_DIR, CLEANED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pandera schema validation (non-fatal — warn on schema errors, don't crash)  #
# --------------------------------------------------------------------------- #
try:
    from schemas import EnrolmentSchema, DemographicUpdateSchema, BiometricUpdateSchema
    import pandera.pandas as pa
    _SCHEMAS_AVAILABLE = True
except ImportError:
    _SCHEMAS_AVAILABLE = False
    logger.warning("Pandera schemas not available; skipping data contract validation.")

INPUT_FILES = {
    "enrolment": "aadhaar_enrolment_clean_final.csv",
    "demographic": "aadhaar_demographic_update_clean_final.csv",
    "biometric": "aadhaar_biometric_update_clean_final.csv",
}


def _validate_frame(df: pd.DataFrame, schema_class, name: str) -> pd.DataFrame:
    """Validate a DataFrame against a Pandera schema, logging errors non-fatally."""
    if not _SCHEMAS_AVAILABLE:
        return df
    try:
        import pandera as pa
        validated = schema_class.validate(df, lazy=True)
        logger.info(f"[Schema] ✓ '{name}' passed data contract validation.")
        return validated
    except pa.errors.SchemaErrors as err:
        logger.warning(
            f"[Schema] ⚠️  Data contract violations detected in '{name}' "
            f"({len(err.failure_cases)} failures). Pipeline continues with raw data.\n"
            f"{err.failure_cases.to_string(index=False)}"
        )
        return df
    except Exception as e:
        logger.warning(f"[Schema] Validation skipped for '{name}': {e}")
        return df


def load_cleaned_inputs() -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    missing = []

    for name, filename in INPUT_FILES.items():
        path = os.path.join(CLEANED_DATA_DIR, filename)
        if not os.path.exists(path):
            missing.append(path)
            continue
        frames[name] = pd.read_csv(path)

    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required cleaned datasets are missing. Run src/data_cleaning.py first.\n"
            f"{missing_text}"
        )

    # --- Pandera validation ---
    logger.info("Running Pandera data contract validation on cleaned inputs...")
    if _SCHEMAS_AVAILABLE:
        frames["enrolment"]  = _validate_frame(frames["enrolment"],  EnrolmentSchema,        "enrolment")
        frames["demographic"] = _validate_frame(frames["demographic"], DemographicUpdateSchema, "demographic")
        frames["biometric"]   = _validate_frame(frames["biometric"],   BiometricUpdateSchema,   "biometric")

    return frames


def save_outputs(outputs: Dict[str, pd.DataFrame]) -> None:
    filename_map = {
        "enrolment_analysis_ready": "enrolment_analysis_ready.csv",
        "demographic_analysis_ready": "demographic_analysis_ready.csv",
        "biometric_analysis_ready": "biometric_analysis_ready.csv",
        "state_month_master": "state_month_master.csv",
        "state_stability": "state_stability.csv",
        "lag_features": "lag_features.csv",
        "rank_persistence": "rank_persistence.csv",
        "rank_correlations": "rank_correlations.csv",
        "state_master_full": "state_master_full.csv",
        "state_anomalies": "state_anomalies.csv",
        "state_focus_summary": "state_focus_summary.csv",
        "national_monthly_summary": "national_monthly_summary.csv",
        "pareto_activity": "pareto_activity.csv",
        "activity_correlation_matrix": "activity_correlation_matrix.csv",
        "indicator_correlation_matrix": "indicator_correlation_matrix.csv",
    }

    for key, filename in filename_map.items():
        if key in outputs and isinstance(outputs[key], pd.DataFrame):
            outputs[key].to_csv(os.path.join(ANALYSIS_DATA_DIR, filename), index=False)

    state_master = outputs["state_master_full"].copy()
    state_master[
        ["state", "AMI", "UPI", "VSI", "TPS", "Policy_Category"]
    ].to_csv(
        os.path.join(ANALYSIS_DATA_DIR, "state_policy_indicators_policy_view.csv"),
        index=False,
    )
    state_master.to_csv(
        os.path.join(ANALYSIS_DATA_DIR, "state_policy_indicators_full.csv"),
        index=False,
    )
    state_master[
        [
            "state",
            "Governance_Status",
            "Policy_Action",
            "Policy_Category",
            "AMI",
            "UPI",
            "VSI",
            "TPS",
        ]
    ].to_csv(os.path.join(ANALYSIS_DATA_DIR, "state_policy_actions.csv"), index=False)
    outputs["state_month_master"].to_csv(
        os.path.join(ANALYSIS_DATA_DIR, "state_month_combined.csv"),
        index=False,
    )


def save_advanced_outputs(advanced: Dict[str, object]) -> None:
    """Save outputs from advanced ML modules to data/analysis/."""
    for key, value in advanced.items():
        if isinstance(value, pd.DataFrame) and not value.empty:
            path = os.path.join(ANALYSIS_DATA_DIR, f"{key}.csv")
            value.to_csv(path, index=False)
            logger.info(f"  Saved {key}.csv ({value.shape})")


def run_advanced_analytics(outputs: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Run all 6 advanced ML modules and collect results."""
    state_master = outputs["state_master_full"]
    state_month = outputs["state_month_master"]
    advanced = {}

    # --- Module 1: Robust Indicators ---
    logger.info("=" * 60)
    logger.info("MODULE 1: Robust Statistical Indicators")
    logger.info("=" * 60)
    try:
        from robust_indicators import compute_all_robust_outputs
        robust_out = compute_all_robust_outputs(state_master, state_month)
        advanced["robust_indicators"] = robust_out["state_master_robust"]
        advanced["entropy_details"] = robust_out["entropy_details"]
        # Use robust master for downstream modules
        state_master = robust_out["state_master_robust"]
    except Exception as e:
        logger.error(f"Robust indicators failed: {e}")

    # --- Module 2: Clustering ---
    logger.info("=" * 60)
    logger.info("MODULE 2: Unsupervised Governance Clustering")
    logger.info("=" * 60)
    try:
        from clustering import run_governance_clustering
        from registry import save_model_artifact
        cluster_out = run_governance_clustering(state_master)
        advanced["clustered_states"] = cluster_out["state_master_clustered"]
        advanced["bic_selection"] = cluster_out["bic_selection"]
        advanced["cluster_centroids"] = cluster_out["centroids"]
        state_master = cluster_out["state_master_clustered"]
        # Persist fitted estimators
        save_model_artifact(cluster_out["gmm_model"], "gmm_model")
        save_model_artifact(cluster_out["scaler"], "clustering_scaler")
        logger.info("[Registry] GMM model and clustering scaler saved.")
    except Exception as e:
        logger.error(f"Clustering failed: {e}")

    # --- Module 3: Causal Inference ---
    logger.info("=" * 60)
    logger.info("MODULE 3: Causal Inference (Granger + CUSUM)")
    logger.info("=" * 60)
    try:
        from causal import run_causal_analysis
        causal_out = run_causal_analysis(state_month)
        advanced["granger_results"] = causal_out["granger_results"]
        advanced["cusum_results"] = causal_out["cusum_results"]
    except Exception as e:
        logger.error(f"Causal analysis failed: {e}")

    # --- Module 4: Forecasting ---
    logger.info("=" * 60)
    logger.info("MODULE 4: Probabilistic Demand Forecasting (SARIMA)")
    logger.info("=" * 60)
    try:
        from forecasting import run_demand_forecasting
        forecast_out = run_demand_forecasting(state_month, target_col="D_total", horizon=12)
        advanced["forecast_results"] = forecast_out["forecast_df"]
        advanced["forecast_model_summary"] = forecast_out["model_summary"]
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")

    # --- Module 5: Advanced Anomaly Detection ---
    logger.info("=" * 60)
    logger.info("MODULE 5: Advanced Anomaly Detection (IF + ECOD)")
    logger.info("=" * 60)
    try:
        from anomaly import run_advanced_anomaly_detection
        from registry import save_model_artifact
        anomaly_out = run_advanced_anomaly_detection(state_master)
        advanced["advanced_anomalies"] = anomaly_out["state_master_anomalies"]
        # Persist fitted estimators
        save_model_artifact(anomaly_out["scaler"], "anomaly_scaler")
        save_model_artifact(anomaly_out["iso_forest"], "isolation_forest")
        logger.info("[Registry] Anomaly scaler and Isolation Forest saved.")
    except Exception as e:
        logger.error(f"Advanced anomaly detection failed: {e}")

    # --- Module 6: SHAP Interpretability ---
    logger.info("=" * 60)
    logger.info("MODULE 6: SHAP Interpretability")
    logger.info("=" * 60)
    try:
        from interpretability import run_interpretability
        from registry import save_model_artifact
        shap_out = run_interpretability(state_master)
        advanced["shap_summary"] = shap_out["shap_summary"]
        advanced["feature_importance"] = shap_out["feature_importance"]
        # Persist fitted priority model
        if shap_out.get("priority_model") is not None:
            save_model_artifact(shap_out["priority_model"], "priority_model")
            logger.info("[Registry] SHAP priority model saved.")
    except Exception as e:
        logger.error(f"SHAP interpretability failed: {e}")

    # --- Module 7: Drift Monitoring ---
    logger.info("=" * 60)
    logger.info("MODULE 7: Data & Model Drift Monitoring (KS + PSI)")
    logger.info("=" * 60)
    try:
        from drift import run_drift_check
        drift_df = run_drift_check(state_master)
        if drift_df is not None and not drift_df.empty:
            advanced["drift_report"] = drift_df
    except Exception as e:
        logger.error(f"Drift monitoring failed: {e}")

    return advanced


def main() -> bool:
    logger.info("Starting feature engineering and analysis pipeline...")
    frames = load_cleaned_inputs()

    # Core analytics
    outputs = build_analysis_outputs(
        enrolment=frames["enrolment"],
        demographic=frames["demographic"],
        biometric=frames["biometric"],
    )
    save_outputs(outputs)

    logger.info("Core analysis outputs generated.")
    logger.info(f"Saved core datasets to: {ANALYSIS_DATA_DIR}")
    for name in sorted(outputs):
        logger.info(f"  - {name}.csv")

    # Advanced ML analytics
    logger.info("")
    logger.info("=" * 60)
    logger.info("RUNNING ADVANCED ML ANALYTICS PIPELINE")
    logger.info("=" * 60)

    advanced = run_advanced_analytics(outputs)
    save_advanced_outputs(advanced)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL PIPELINES COMPLETE")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        logger.error(str(exc))
    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")

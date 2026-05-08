import os
from typing import Dict

import pandas as pd

from analytics import build_analysis_outputs
from config import ANALYSIS_DATA_DIR, CLEANED_DATA_DIR


INPUT_FILES = {
    "enrolment": "aadhaar_enrolment_clean_final.csv",
    "demographic": "aadhaar_demographic_update_clean_final.csv",
    "biometric": "aadhaar_biometric_update_clean_final.csv",
}


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


def main() -> bool:
    print("Starting feature engineering and analysis pipeline...")
    frames = load_cleaned_inputs()

    outputs = build_analysis_outputs(
        enrolment=frames["enrolment"],
        demographic=frames["demographic"],
        biometric=frames["biometric"],
    )
    save_outputs(outputs)

    print("Analysis outputs generated successfully.")
    print(f"Saved datasets to: {ANALYSIS_DATA_DIR}")
    for name in sorted(outputs):
        print(f"- {name}.csv")
    return True


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc)

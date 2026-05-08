from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-6


def minmax_scale(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if pd.isna(span) or span <= EPS:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - series.min()) / (span + EPS)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year_month"] = out["date"].dt.to_period("M").astype(str)
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.to_period("Q").astype(str)
    return out


def prepare_analysis_frames(
    enrolment: pd.DataFrame,
    demographic: pd.DataFrame,
    biometric: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enrolment = add_time_columns(enrolment)
    demographic = add_time_columns(demographic)
    biometric = add_time_columns(biometric)

    enrolment["total_enrolments"] = (
        enrolment["age_0_5"] + enrolment["age_5_17"] + enrolment["age_18_greater"]
    )
    demographic["total_demographic_updates"] = (
        demographic["demo_age_5_17"] + demographic["demo_age_17_"]
    )
    biometric["total_biometric_updates"] = (
        biometric["bio_age_5_17"] + biometric["bio_age_17_"]
    )
    return enrolment, demographic, biometric


def aggregate_state_month(
    enrolment: pd.DataFrame,
    demographic: pd.DataFrame,
    biometric: pd.DataFrame,
) -> pd.DataFrame:
    state_month_enrol = enrolment.groupby(["state", "year_month"], as_index=False).agg(
        E_total=("total_enrolments", "sum"),
        E_0_5=("age_0_5", "sum"),
        E_5_17=("age_5_17", "sum"),
        E_18_plus=("age_18_greater", "sum"),
    )

    state_month_demo = demographic.groupby(["state", "year_month"], as_index=False).agg(
        D_total=("total_demographic_updates", "sum"),
        D_5_17=("demo_age_5_17", "sum"),
        D_17_plus=("demo_age_17_", "sum"),
    )

    state_month_bio = biometric.groupby(["state", "year_month"], as_index=False).agg(
        B_total=("total_biometric_updates", "sum"),
        B_5_17=("bio_age_5_17", "sum"),
        B_17_plus=("bio_age_17_", "sum"),
    )

    state_month = (
        state_month_enrol.merge(state_month_demo, on=["state", "year_month"], how="inner")
        .merge(state_month_bio, on=["state", "year_month"], how="inner")
        .sort_values(["state", "year_month"])
        .reset_index(drop=True)
    )

    state_month["update_to_enrol_ratio"] = (
        (state_month["D_total"] + state_month["B_total"]) / (state_month["E_total"] + EPS)
    )
    state_month["demo_update_ratio"] = state_month["D_total"] / (state_month["E_total"] + EPS)
    state_month["biometric_update_ratio"] = state_month["B_total"] / (state_month["E_total"] + EPS)
    state_month["E_child_share"] = state_month["E_0_5"] / (state_month["E_total"] + EPS)
    state_month["E_minor_share"] = state_month["E_5_17"] / (state_month["E_total"] + EPS)
    state_month["D_adult_share"] = state_month["D_17_plus"] / (state_month["D_total"] + EPS)
    state_month["B_adult_share"] = state_month["B_17_plus"] / (state_month["B_total"] + EPS)
    state_month["Total_Activity"] = (
        state_month["E_total"] + state_month["D_total"] + state_month["B_total"]
    )
    return state_month


def compute_activity_correlation_matrix(state_month: pd.DataFrame) -> pd.DataFrame:
    corr_cols = [
        "E_total",
        "D_total",
        "B_total",
        "update_to_enrol_ratio",
        "E_child_share",
        "D_adult_share",
        "B_adult_share",
    ]
    return state_month[corr_cols].corr().reset_index().rename(columns={"index": "metric"})


def compute_state_stability(state_month: pd.DataFrame) -> pd.DataFrame:
    return (
        state_month.groupby("state")
        .agg(
            E_cv=("E_total", lambda x: x.std() / (x.mean() + EPS)),
            D_cv=("D_total", lambda x: x.std() / (x.mean() + EPS)),
            B_cv=("B_total", lambda x: x.std() / (x.mean() + EPS)),
            ratio_cv=("update_to_enrol_ratio", lambda x: x.std() / (x.mean() + EPS)),
        )
        .reset_index()
    )


def compute_rank_persistence(state_month: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    state_totals = state_month.groupby("state", as_index=False).agg(
        E_total=("E_total", "sum"),
        D_total=("D_total", "sum"),
        B_total=("B_total", "sum"),
    )
    state_totals["E_rank"] = state_totals["E_total"].rank(ascending=False, method="average")
    state_totals["D_rank"] = state_totals["D_total"].rank(ascending=False, method="average")
    state_totals["B_rank"] = state_totals["B_total"].rank(ascending=False, method="average")
    rank_corr = {
        "rho_enrolment_demographic": state_totals["E_rank"].corr(
            state_totals["D_rank"], method="spearman"
        ),
        "rho_enrolment_biometric": state_totals["E_rank"].corr(
            state_totals["B_rank"], method="spearman"
        ),
    }
    return state_totals, rank_corr


def compute_lag_features(state_month: pd.DataFrame) -> pd.DataFrame:
    lag_results: List[Dict[str, float]] = []
    for state in sorted(state_month["state"].unique()):
        df_state = state_month[state_month["state"] == state].sort_values("year_month")
        if len(df_state) < 3:
            lag_results.append({"state": state, "lag1_corr": np.nan, "lag2_corr": np.nan})
            continue

        lag1_corr = df_state["E_5_17"].corr(df_state["B_17_plus"].shift(-1))
        lag2_corr = df_state["E_5_17"].corr(df_state["B_17_plus"].shift(-2))
        lag_results.append(
            {"state": state, "lag1_corr": lag1_corr, "lag2_corr": lag2_corr}
        )

    return pd.DataFrame(lag_results)


def classify_state(row: pd.Series) -> str:
    if row["AMI"] > 0.7 and row["VSI"] < 0.3:
        return "Stable Mature System"
    if row["UPI"] > 0.7 and row["VSI"] > 0.7:
        return "High Maintenance Stress"
    if row["AMI"] < 0.4 and row["UPI"] < 0.4:
        return "Expansion Phase"
    if row["TPS"] < 0.3:
        return "Unpredictable - Needs Monitoring"
    return "Balanced / Transitional"


def map_policy_action(row: pd.Series) -> Tuple[str, str]:
    if row["AMI"] < 0.3 and row["UPI"] > 0.6 and row["VSI"] > 0.5:
        return (
            "Mature-Overloaded",
            "Expand update capacity, decentralize service points, and introduce load-balancing.",
        )
    if row["AMI"] < 0.3 and row["VSI"] < 0.3:
        return (
            "Stable Mature",
            "Optimize operations and invest in fraud detection plus service quality.",
        )
    if row["AMI"] > 0.7 and row["UPI"] < 0.4:
        return (
            "Expansion Phase",
            "Increase enrollment infrastructure and run targeted outreach programs.",
        )
    if row["TPS"] < 0.3:
        return (
            "Unpredictable System",
            "Increase monitoring and investigate administrative or reporting irregularities.",
        )
    return (
        "Transitional",
        "Balance investment across enrollment and update services.",
    )


def compute_state_indicators(
    state_month: pd.DataFrame,
    state_stability: pd.DataFrame,
    lag_df: pd.DataFrame,
) -> pd.DataFrame:
    state_master = state_month.groupby("state", as_index=False).agg(
        E_total=("E_total", "sum"),
        D_total=("D_total", "sum"),
        B_total=("B_total", "sum"),
        E_child_share=("E_child_share", "mean"),
        E_minor_share=("E_minor_share", "mean"),
        D_adult_share=("D_adult_share", "mean"),
        B_adult_share=("B_adult_share", "mean"),
        Avg_Update_Ratio=("update_to_enrol_ratio", "mean"),
        Avg_Demo_Ratio=("demo_update_ratio", "mean"),
        Avg_Bio_Ratio=("biometric_update_ratio", "mean"),
        Active_Months=("year_month", "nunique"),
    )

    state_master["Total_Activity"] = (
        state_master["E_total"] + state_master["D_total"] + state_master["B_total"]
    )
    state_master["log_activity"] = np.log1p(state_master["Total_Activity"])

    state_master["E_rank"] = state_master["E_total"].rank(ascending=False, method="average")
    state_master["D_rank"] = state_master["D_total"].rank(ascending=False, method="average")
    state_master["B_rank"] = state_master["B_total"].rank(ascending=False, method="average")
    state_master["AMI_raw"] = (
        state_master["E_rank"] + state_master["D_rank"] + state_master["B_rank"]
    )
    state_master["UPI_raw"] = np.log1p(
        state_master["D_total"] + state_master["B_total"]
    ) - np.log1p(state_master["E_total"])

    state_master = state_master.merge(
        state_stability.rename(columns={"ratio_cv": "VSI_raw"}), on="state", how="left"
    )
    state_master = state_master.merge(lag_df, on="state", how="left")
    state_master["TPS_raw"] = state_master["lag1_corr"].abs().fillna(0)

    for col in ["AMI_raw", "UPI_raw", "VSI_raw", "TPS_raw"]:
        state_master[col] = minmax_scale(state_master[col].fillna(0))

    state_master["AMI"] = 1 - state_master["AMI_raw"]
    state_master["UPI"] = state_master["UPI_raw"]
    state_master["VSI"] = state_master["VSI_raw"]
    state_master["TPS"] = state_master["TPS_raw"]
    state_master["Policy_Category"] = state_master.apply(classify_state, axis=1)
    state_master[["Governance_Status", "Policy_Action"]] = state_master.apply(
        map_policy_action, axis=1, result_type="expand"
    )
    return state_master.sort_values("state").reset_index(drop=True)


def compute_indicator_correlation_matrix(state_master: pd.DataFrame) -> pd.DataFrame:
    corr = state_master[["AMI", "UPI", "VSI", "TPS"]].corr(method="pearson")
    return corr.reset_index().rename(columns={"index": "metric"})


def compute_national_monthly_summary(state_month: pd.DataFrame) -> pd.DataFrame:
    national = (
        state_month.groupby("year_month", as_index=False)
        .agg(
            E_total=("E_total", "sum"),
            D_total=("D_total", "sum"),
            B_total=("B_total", "sum"),
            demo_update_ratio=("demo_update_ratio", "mean"),
            biometric_update_ratio=("biometric_update_ratio", "mean"),
            update_to_enrol_ratio=("update_to_enrol_ratio", "mean"),
            E_child_share=("E_child_share", "mean"),
            D_adult_share=("D_adult_share", "mean"),
            B_adult_share=("B_adult_share", "mean"),
        )
        .sort_values("year_month")
        .reset_index(drop=True)
    )
    national["Total_Activity"] = national["E_total"] + national["D_total"] + national["B_total"]
    return national


def compute_pareto_activity(state_master: pd.DataFrame) -> pd.DataFrame:
    pareto = state_master.sort_values("Total_Activity", ascending=False).reset_index(drop=True)
    pareto["activity_rank"] = np.arange(1, len(pareto) + 1)
    pareto["cum_share"] = pareto["Total_Activity"].cumsum() / (
        pareto["Total_Activity"].sum() + EPS
    )
    return pareto


def compute_anomalies(state_master: pd.DataFrame) -> pd.DataFrame:
    df = state_master.copy()

    q1 = df["VSI"].quantile(0.25)
    q3 = df["VSI"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    df["Anomaly_High_Stress"] = df["VSI"] > upper_bound

    df["Anomaly_Pressure_Mismatch"] = (
        (df["UPI"] > df["UPI"].quantile(0.75)) & (df["AMI"] < df["AMI"].quantile(0.25))
    )

    df["Expected_VSI"] = df["log_activity"].rank(pct=True)
    df["Volatility_Excess"] = df["VSI"] - df["Expected_VSI"]
    df["Anomaly_Scale_Volatility"] = df["Volatility_Excess"] > df["Volatility_Excess"].quantile(0.9)

    ami_med = df["AMI"].median()
    upi_med = df["UPI"].median()
    df["Governance_Distance"] = np.sqrt((df["AMI"] - ami_med) ** 2 + (df["UPI"] - upi_med) ** 2)
    df["Anomaly_Governance_Outlier"] = (
        df["Governance_Distance"] > df["Governance_Distance"].quantile(0.9)
    )

    anomaly_cols = [
        "Anomaly_High_Stress",
        "Anomaly_Pressure_Mismatch",
        "Anomaly_Scale_Volatility",
        "Anomaly_Governance_Outlier",
    ]
    df["Anomaly_Flag_Count"] = df[anomaly_cols].sum(axis=1)
    return df.sort_values(
        ["Anomaly_Flag_Count", "VSI", "UPI"], ascending=[False, False, False]
    ).reset_index(drop=True)


def compute_state_focus_summary(state_master: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    summary = state_master.merge(
        anomalies[
            [
                "state",
                "Anomaly_Flag_Count",
                "Anomaly_High_Stress",
                "Anomaly_Pressure_Mismatch",
                "Anomaly_Scale_Volatility",
                "Anomaly_Governance_Outlier",
                "Volatility_Excess",
                "Governance_Distance",
            ]
        ],
        on="state",
        how="left",
    )
    summary["Priority_Score"] = (
        0.30 * summary["UPI"]
        + 0.25 * summary["VSI"]
        + 0.20 * (1 - summary["TPS"])
        + 0.15 * summary["Anomaly_Flag_Count"].fillna(0) / 4.0
        + 0.10 * (1 - summary["AMI"])
    )
    return summary.sort_values("Priority_Score", ascending=False).reset_index(drop=True)


def build_analysis_outputs(
    enrolment: pd.DataFrame,
    demographic: pd.DataFrame,
    biometric: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    enrolment_ready, demographic_ready, biometric_ready = prepare_analysis_frames(
        enrolment, demographic, biometric
    )
    state_month = aggregate_state_month(enrolment_ready, demographic_ready, biometric_ready)
    state_stability = compute_state_stability(state_month)
    lag_df = compute_lag_features(state_month)
    rank_persistence, rank_corr = compute_rank_persistence(state_month)
    state_master = compute_state_indicators(state_month, state_stability, lag_df)
    anomalies = compute_anomalies(state_master)
    state_focus = compute_state_focus_summary(state_master, anomalies)
    national_monthly = compute_national_monthly_summary(state_month)
    pareto_activity = compute_pareto_activity(state_master)
    activity_corr = compute_activity_correlation_matrix(state_month)
    indicator_corr = compute_indicator_correlation_matrix(state_master)

    rank_corr_df = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in rank_corr.items()
        ]
    )

    return {
        "enrolment_analysis_ready": enrolment_ready,
        "demographic_analysis_ready": demographic_ready,
        "biometric_analysis_ready": biometric_ready,
        "state_month_master": state_month,
        "state_stability": state_stability,
        "lag_features": lag_df,
        "rank_persistence": rank_persistence,
        "rank_correlations": rank_corr_df,
        "state_master_full": state_focus,
        "state_anomalies": anomalies,
        "state_focus_summary": state_focus,
        "national_monthly_summary": national_monthly,
        "pareto_activity": pareto_activity,
        "activity_correlation_matrix": activity_corr,
        "indicator_correlation_matrix": indicator_corr,
    }

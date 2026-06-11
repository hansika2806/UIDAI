import pandas as pd
import streamlit as st


def format_indian_number(value: float) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 1e7:
        return f"{value / 1e7:.2f} Cr"
    if abs(value) >= 1e5:
        return f"{value / 1e5:.2f} L"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f} K"
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(
    state_master: pd.DataFrame, state_month: pd.DataFrame, anomalies: pd.DataFrame
) -> None:
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        metric_card(
            "States in Scope",
            str(state_master["state"].nunique()),
            "Filtered analytical universe",
        )
    with kpi_cols[1]:
        metric_card(
            "Months Covered",
            str(state_month["year_month"].nunique()),
            "Active state-month observations",
        )
    with kpi_cols[2]:
        metric_card(
            "Total Activity",
            format_indian_number(state_master["Total_Activity"].sum()),
            "Enrollments plus demographic and biometric updates",
        )
    with kpi_cols[3]:
        metric_card(
            "Flagged States",
            str((anomalies["Anomaly_Flag_Count"] > 0).sum()),
            "At least one anomaly rule triggered",
        )
    with kpi_cols[4]:
        if not state_master.empty:
            top_category = state_master["Policy_Category"].mode().iloc[0]
        else:
            top_category = "-"
        metric_card("Dominant Category", top_category, "Most common governance profile")

"""Metric cards and KPI strip with risk-aware styling for UIDAI Intelligence Studio."""

import pandas as pd
import streamlit as st


def format_indian_number(value: float) -> str:
    """Format a number in the Indian numbering system (Crore, Lakh, Thousand)."""
    if pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1e7:
        return f"{value / 1e7:.2f} Cr"
    if abs(value) >= 1e5:
        return f"{value / 1e5:.2f} L"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f} K"
    if value == int(value):
        return f"{int(value)}"
    return f"{value:.2f}"


def metric_card(
    label: str,
    value: str,
    note: str,
    risk_class: str = "",
    color: str = "",
) -> None:
    """Render a single metric card with optional risk styling and dynamic text scaling."""
    cls = f"metric-card {risk_class}".strip()

    # Calculate responsive font size to prevent overlapping or overflow
    font_size = "2.1rem"
    if len(value) > 20:
        font_size = "1.05rem"
    elif len(value) > 14:
        font_size = "1.25rem"
    elif len(value) > 9:
        font_size = "1.45rem"

    value_style = (
        f"color:{color}; font-size:{font_size}; word-break:break-word; line-height:1.25;"
        if color
        else f"font-size:{font_size}; word-break:break-word; line-height:1.25;"
    )
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="{value_style}">{value}</div>
            <div class="metric-delta">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(
    state_master: pd.DataFrame,
    state_month: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:
    """Render the top-level KPI strip with risk-aware colouring."""
    n_states = state_master["state"].nunique()
    n_months = state_month["year_month"].nunique()
    total_activity = state_master["Total_Activity"].sum()
    flagged = int((anomalies["Anomaly_Flag_Count"] > 0).sum())
    high_risk = int((anomalies["Anomaly_Flag_Count"] >= 2).sum())

    # Determine dominant category
    if not state_master.empty:
        top_category = state_master["Policy_Category"].mode().iloc[0]
    else:
        top_category = "—"

    # Risk class for flagged card
    flagged_risk = ""
    flagged_color = "#16a34a"
    if high_risk > 0:
        flagged_risk = "risk-critical"
        flagged_color = "#dc2626"
    elif flagged > 0:
        flagged_risk = "risk-warning"
        flagged_color = "#d97706"

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        metric_card(
            "States in Scope",
            str(n_states),
            "Filtered analytical universe",
            color="#0e7490",
        )
    with kpi_cols[1]:
        metric_card(
            "Months Covered",
            str(n_months),
            "Active state-month observations",
            color="#334155",
        )
    with kpi_cols[2]:
        metric_card(
            "Total Activity",
            format_indian_number(total_activity),
            "Enrollments + demographic + biometric updates",
            color="#0e7490",
        )
    with kpi_cols[3]:
        metric_card(
            "Flagged States",
            str(flagged),
            (
                f"{high_risk} with 2+ simultaneous anomaly flags"
                if high_risk > 0
                else "At least one anomaly rule triggered"
            ),
            risk_class=flagged_risk,
            color=flagged_color,
        )
    with kpi_cols[4]:
        metric_card(
            "Dominant Category",
            top_category,
            "Most common governance profile",
        )

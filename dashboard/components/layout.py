import pandas as pd
import streamlit as st
from components.charts import (
    build_national_trend_chart,
    build_category_mix_chart,
    build_pareto_chart,
    build_quadrant_chart,
    build_stress_scale_chart,
    build_indicator_heatmap,
    build_rank_persistence_chart,
    build_ratio_trend_chart,
    build_volatility_bar_chart,
    build_lifecycle_chart,
    build_anomaly_chart,
    build_mismatch_queue_chart,
    build_state_month_chart,
    build_peer_comparison,
    build_state_ratio_trend_chart,
)


# ── Index interpretation helper ───────────────────────────────────────────────
def _index_interpretation(label: str, value: float) -> str:
    """Return a plain-English summary for the given index value."""
    if label == "AMI":
        if value >= 0.7:
            return "Highly mature — most eligible residents are enrolled."
        elif value < 0.4:
            return "Active expansion phase — large portion not yet registered."
        return "Moderate maturity — enrollment is ongoing."
    elif label == "UPI":
        if value >= 0.7:
            return "High update pressure — system handling many record changes."
        elif value < 0.4:
            return "Low update pressure — record amendment activity minimal."
        return "Moderate update pressure — steady but not overloading."
    elif label == "VSI":
        if value >= 0.7:
            return "High volume stress — significant monthly fluctuation detected."
        elif value < 0.4:
            return "Stable volumes — activity is predictably consistent."
        return "Moderate volatility — some fluctuation observed."
    elif label == "TPS":
        if value < 0.3:
            return "Highly unpredictable — erratic activity, difficult to forecast."
        elif value >= 0.7:
            return "Strong temporal predictability — consistent seasonal rhythm."
        return "Moderate predictability — some seasonal structure present."
    return ""


def _percentile_badge(value: float, all_values: pd.Series) -> str:
    """Return an HTML badge showing the national percentile for this value."""
    pct = int(100 * (all_values < value).sum() / max(len(all_values), 1))
    if pct >= 90:
        color = "#16a34a"
        label = f"Top {100 - pct}%"
    elif pct >= 70:
        color = "#0e7490"
        label = f"{pct}th pct"
    elif pct >= 40:
        color = "#d97706"
        label = f"{pct}th pct"
    else:
        color = "#dc2626"
        label = f"Bottom {pct + 1}%"
    return (
        f'<span style="font-size:0.72rem;font-weight:500;background:rgba(0,0,0,0.04);'
        f"color:{color};padding:0.15rem 0.45rem;border-radius:9999px;"
        f'border:1px solid {color};margin-left:0.4rem;vertical-align:middle;">{label}</span>'
    )


def render_executive_overview(
    national: pd.DataFrame,
    state_master: pd.DataFrame,
    pareto: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:
    st.markdown(
        '<div class="interpret-box" style="margin-top:0;">'
        "National movement, state concentration, category spread, and the policy queue that deserves attention first. "
        "These are the <strong>summary-level signals</strong> — click into a state for the full ML diagnostic."
        "</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.6, 1])
    with left:
        st.plotly_chart(
            build_national_trend_chart(national),
            width="stretch",
            config={"displayModeBar": False},
        )
    with right:
        st.plotly_chart(
            build_category_mix_chart(state_master),
            width="stretch",
            config={"displayModeBar": False},
        )

    left, right = st.columns([1.4, 1.1])
    with left:
        st.plotly_chart(
            build_pareto_chart(pareto),
            width="stretch",
            config={"displayModeBar": False},
        )
    with right:
        st.markdown("#### Priority States")
        preview_cols = [
            "state",
            "Priority_Score",
            "Governance_Status",
            "Policy_Action",
            "Anomaly_Flag_Count",
        ]
        st.dataframe(
            anomalies[preview_cols].head(12),
            width="stretch",
            hide_index=True,
        )


def render_governance_diagnostics(
    state_master: pd.DataFrame, selected_state: str, outputs: dict
) -> None:
    st.markdown(
        '<div class="headline">📐 Governance Diagnostics</div>', unsafe_allow_html=True
    )
    st.plotly_chart(build_quadrant_chart(state_master, selected_state), width="stretch")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_stress_scale_chart(state_master), width="stretch")
    with right:
        indicator_corr = outputs["indicator_correlation_matrix"]
        st.plotly_chart(
            build_indicator_heatmap(indicator_corr, "Indicator Correlation Heatmap"),
            width="stretch",
        )

    compare_mode = st.radio(
        "Rank Persistence View",
        options=["Demographic Rank", "Biometric Rank"],
        horizontal=True,
    )
    rank_df = outputs["rank_persistence"].copy()
    st.plotly_chart(
        build_rank_persistence_chart(rank_df, compare_mode), width="stretch"
    )


def render_lifecycle_operations(
    national: pd.DataFrame, state_master: pd.DataFrame, outputs: dict
) -> None:
    st.markdown(
        '<div class="headline">⚙️ Lifecycle &amp; Operations</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(build_ratio_trend_chart(national), width="stretch")
    with right:
        st.plotly_chart(build_volatility_bar_chart(state_master), width="stretch")

    left, right = st.columns([1.3, 1.05])
    with left:
        st.plotly_chart(build_lifecycle_chart(state_master), width="stretch")
    with right:
        activity_corr = outputs["activity_correlation_matrix"]
        st.plotly_chart(
            build_indicator_heatmap(
                activity_corr, "Activity and Lifecycle Correlations"
            ),
            width="stretch",
        )


def render_anomalies_risk(anomalies: pd.DataFrame) -> None:
    st.markdown(
        '<div class="headline">🚨 Anomalies &amp; Risk Detection</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="interpret-box" style="margin-top:0; line-height:1.65;">'
        "<strong>Consensus Ensemble Anomaly Detection</strong>: Combining ECOD and Isolation Forest models.<br>"
        "• <strong>ECOD (Empirical Cumulative Distribution)</strong>: Targets extreme univariate outliers at the distribution tails (e.g., massive spikes in specific update queues).<br>"
        "• <strong>Isolation Forest</strong>: Isolate multivariate anomalies (states with subtle but unusual correlation profiles across multiple indexes).<br>"
        "By summing ECOD and Isolation Forest flags, the system builds a robust, false-positive resilient consensus score to prioritize states where multiple warning rules trigger simultaneously."
        "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_anomaly_chart(anomalies),
        width="stretch",
        config={"displayModeBar": False},
    )
    left, right = st.columns([1.2, 1])
    with left:
        flag_cols = [
            "state",
            "Anomaly_Flag_Count",
            "Anomaly_High_Stress",
            "Anomaly_Pressure_Mismatch",
            "Anomaly_Scale_Volatility",
            "Anomaly_Governance_Outlier",
            "Volatility_Excess",
            "Governance_Distance",
        ]
        st.dataframe(
            anomalies[flag_cols].sort_values("Anomaly_Flag_Count", ascending=False),
            width="stretch",
            hide_index=True,
        )
    with right:
        pressure_mismatch = anomalies[anomalies["Anomaly_Pressure_Mismatch"]]
        if pressure_mismatch.empty:
            st.info(
                "No pressure-maturity mismatch states under the active filter window."
            )
        else:
            st.plotly_chart(
                build_mismatch_queue_chart(pressure_mismatch),
                width="stretch",
                config={"displayModeBar": False},
            )


def render_state_drilldown(
    state_master: pd.DataFrame, state_month: pd.DataFrame, selected_state: str
) -> None:
    state_row = state_master[state_master["state"] == selected_state].iloc[0]
    state_series = state_month[state_month["state"] == selected_state].copy()

    # ── State header ──────────────────────────────────────────────────────────
    risk_color = "#dc2626"
    cat = str(state_row.get("Policy_Category", ""))
    if "Maintenance Stress" in cat or "Overloaded" in cat:
        risk_color = "#dc2626"
    elif "Unpredictable" in cat:
        risk_color = "#d97706"
    else:
        risk_color = "#16a34a"

    st.markdown(
        f'<div class="headline" style="color:{risk_color};">{selected_state}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:#475569;margin-bottom:1rem;">'
        f'{state_row["Governance_Status"]} &nbsp;·&nbsp; '
        f'{state_row["Policy_Category"]} &nbsp;·&nbsp; '
        f'Priority score <strong>{state_row["Priority_Score"]:.3f}</strong>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Metric cards with percentile badges ───────────────────────────────────
    drill_cols = st.columns(4)
    index_meta = [
        ("AMI", "Aadhaar Maturity Index", "#0e7490"),
        ("UPI", "Update Propensity Index", "#7c3aed"),
        ("VSI", "Volume Stress Index", "#16a34a"),
        ("TPS", "Temporal Performance Score", "#d97706"),
    ]
    for col, (idx, label, color) in zip(drill_cols, index_meta):
        with col:
            val = float(state_row[idx])
            badge = _percentile_badge(val, state_master[idx])
            interp = _index_interpretation(idx, val)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div>
                        <span class="metric-value" style="color:{color};">{val:.3f}</span>
                        {badge}
                    </div>
                    <div style="font-size:0.8rem;color:#475569;margin-top:0.4rem;font-style:italic;">{interp}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Recommended action ────────────────────────────────────────────────────
    st.markdown(
        f'<div class="interpret-box" style="margin-top:1.5rem;">'
        f'<strong>Recommended Action:</strong> {state_row["Policy_Action"]}'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Time-series charts ────────────────────────────────────────────────────
    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(
            build_state_month_chart(state_series, selected_state),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            build_peer_comparison(state_master, selected_state),
            width="stretch",
        )

    st.plotly_chart(
        build_state_ratio_trend_chart(state_series, selected_state),
        width="stretch",
    )


def render_data_export(
    state_master: pd.DataFrame,
    state_month: pd.DataFrame,
    anomalies: pd.DataFrame,
    national: pd.DataFrame,
    pareto: pd.DataFrame,
    outputs: dict,
) -> None:
    st.markdown(
        '<div class="headline">📦 Download Analytical Outputs</div>',
        unsafe_allow_html=True,
    )
    export_map = {
        "State Master": state_master,
        "State Month Master": state_month,
        "State Focus Summary": anomalies,
        "National Monthly Summary": national,
        "Pareto Activity": pareto,
        "Rank Persistence": outputs["rank_persistence"],
        "Lag Features": outputs["lag_features"],
    }
    export_choice = st.selectbox("Choose dataset", list(export_map.keys()))
    export_df = export_map[export_choice]
    st.dataframe(export_df.head(50), width="stretch", hide_index=True)
    st.download_button(
        label=f"⬇ Download {export_choice} CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{export_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

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
from components.metrics import metric_card


def render_executive_overview(
    national: pd.DataFrame,
    state_master: pd.DataFrame,
    pareto: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:
    st.markdown(
        '<div class="panel-note">This view keeps the big picture intact: national movement, state concentration, category spread, and the policy queue that deserves attention first.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.6, 1])
    with left:
        st.plotly_chart(build_national_trend_chart(national), use_container_width=True)
    with right:
        st.plotly_chart(
            build_category_mix_chart(state_master), use_container_width=True
        )

    left, right = st.columns([1.4, 1.1])
    with left:
        st.plotly_chart(build_pareto_chart(pareto), use_container_width=True)
    with right:
        st.subheader("Priority States")
        preview_cols = [
            "state",
            "Priority_Score",
            "Governance_Status",
            "Policy_Action",
            "Anomaly_Flag_Count",
        ]
        st.dataframe(
            anomalies[preview_cols].head(12),
            use_container_width=True,
            hide_index=True,
        )


def render_governance_diagnostics(
    state_master: pd.DataFrame, selected_state: str, outputs: dict
) -> None:
    st.plotly_chart(
        build_quadrant_chart(state_master, selected_state), use_container_width=True
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            build_stress_scale_chart(state_master), use_container_width=True
        )
    with right:
        indicator_corr = outputs["indicator_correlation_matrix"]
        st.plotly_chart(
            build_indicator_heatmap(indicator_corr, "Indicator Correlation Heatmap"),
            use_container_width=True,
        )

    compare_mode = st.radio(
        "Rank Persistence View",
        options=["Demographic Rank", "Biometric Rank"],
        horizontal=True,
    )
    rank_df = outputs["rank_persistence"].copy()
    st.plotly_chart(
        build_rank_persistence_chart(rank_df, compare_mode), use_container_width=True
    )


def render_lifecycle_operations(
    national: pd.DataFrame, state_master: pd.DataFrame, outputs: dict
) -> None:
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(build_ratio_trend_chart(national), use_container_width=True)
    with right:
        st.plotly_chart(
            build_volatility_bar_chart(state_master), use_container_width=True
        )

    left, right = st.columns([1.3, 1.05])
    with left:
        st.plotly_chart(build_lifecycle_chart(state_master), use_container_width=True)
    with right:
        activity_corr = outputs["activity_correlation_matrix"]
        st.plotly_chart(
            build_indicator_heatmap(
                activity_corr, "Activity and Lifecycle Correlations"
            ),
            use_container_width=True,
        )


def render_anomalies_risk(anomalies: pd.DataFrame) -> None:
    st.plotly_chart(build_anomaly_chart(anomalies), use_container_width=True)
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
            use_container_width=True,
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
                build_mismatch_queue_chart(pressure_mismatch), use_container_width=True
            )


def render_state_drilldown(
    state_master: pd.DataFrame, state_month: pd.DataFrame, selected_state: str
) -> None:
    state_row = state_master[state_master["state"] == selected_state].iloc[0]
    state_series = state_month[state_month["state"] == selected_state].copy()

    st.subheader(selected_state)
    st.caption(
        f"{state_row['Governance_Status']} | {state_row['Policy_Category']} | Priority score {state_row['Priority_Score']:.2f}"
    )
    drill_cols = st.columns(4)
    with drill_cols[0]:
        metric_card("AMI", f"{state_row['AMI']:.2f}", "Maturity index")
    with drill_cols[1]:
        metric_card("UPI", f"{state_row['UPI']:.2f}", "Update pressure index")
    with drill_cols[2]:
        metric_card("VSI", f"{state_row['VSI']:.2f}", "Volatility stress index")
    with drill_cols[3]:
        metric_card("TPS", f"{state_row['TPS']:.2f}", "Temporal predictability")

    st.markdown(
        f'<div class="panel-note"><strong>Recommended action:</strong> {state_row["Policy_Action"]}</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(
            build_state_month_chart(state_series, selected_state),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            build_peer_comparison(state_master, selected_state),
            use_container_width=True,
        )

    st.plotly_chart(
        build_state_ratio_trend_chart(state_series, selected_state),
        use_container_width=True,
    )


def render_data_export(
    state_master: pd.DataFrame,
    state_month: pd.DataFrame,
    anomalies: pd.DataFrame,
    national: pd.DataFrame,
    pareto: pd.DataFrame,
    outputs: dict,
) -> None:
    st.subheader("Download Analytical Outputs")
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
    st.dataframe(export_df.head(50), use_container_width=True, hide_index=True)
    st.download_button(
        label=f"Download {export_choice} CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{export_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

"""Causal Inference panel: Granger causality heatmap and CUSUM structural breaks."""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "analysis")


def _load_granger_results():
    path = os.path.join(ANALYSIS_DIR, "granger_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_cusum_results():
    path = os.path.join(ANALYSIS_DIR, "cusum_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def render_causal_panel() -> None:
    """Render the Causal Inference panel."""
    st.subheader("🔗 Causal Inference & Structural Breaks")
    st.markdown(
        '<div class="panel-note"><strong>Granger Causality</strong> tests whether historical trends in enrollment '
        'or updates statistically help predict future updates. <strong>CUSUM (Cumulative Sum)</strong> detects structural '
        'breaks (regime shifts) in the update-to-enrollment ratio due to policy shocks, lockdowns, or administrative changes.</div>',
        unsafe_allow_html=True,
    )

    granger = _load_granger_results()
    cusum = _load_cusum_results()

    if granger is None or cusum is None:
        st.warning("Causal inference data not available. Run the advanced analytics pipeline first.")
        return

    # Stationarity context banner (Fix 1)
    if "cause_diffs" in granger.columns:
        n_differenced = int((granger["cause_diffs"] > 0).sum())
        n_total = len(granger)
        if n_differenced > 0:
            pct = round(100 * n_differenced / n_total)
            st.info(
                f"🧮 **Stationarity enforced (ADF):** {n_differenced}/{n_total} test pairs ({pct}%) used "
                f"differenced series. Results describe *changes* in the causal variable, not raw levels. "
                f"Non-differenced tests used raw (already-stationary) series."
            )
        else:
            st.success(
                "✅ **Stationarity check passed:** All input series were already stationary. "
                "Granger results reflect raw-level causal relationships."
            )

    # High-level metrics
    cols = st.columns(3)
    with cols[0]:
        sig_granger = granger[granger["significant"] == True]
        st.metric(
            label="Significant Causal Links",
            value=f"{len(sig_granger)} / {len(granger)}",
            help="Total tested state-lag pairs where the Granger causality p-value < 0.05",
        )
    with cols[1]:
        breaks = cusum[cusum["break_detected"] == True]
        st.metric(
            label="Regime Shifts Detected",
            value=f"{len(breaks)} / {len(cusum)}",
            help="States showing a statistically significant structural break in their update pattern",
        )
    with cols[2]:
        most_common_break = breaks["break_month"].mode().iloc[0] if not breaks.empty else "N/A"
        st.metric(
            label="Peak Shift Month",
            value=most_common_break,
            help="The month in which most structural breaks were clustered",
        )

    # 2 columns layout: Granger overview on left, CUSUM on right
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Granger Causality Overview")
        st.markdown(
            "Which causal directions are most active across states? (Showing count of significant links)"
        )
        
        summary_granger = granger[granger["significant"] == True].groupby(["cause", "effect"]).size().reset_index(name="count")
        
        fig_granger = px.bar(
            summary_granger,
            x="count",
            y="cause",
            color="effect",
            orientation="h",
            title="Significant Causal Links Across All States",
            color_discrete_sequence=px.colors.qualitative.Prism,
            labels={"count": "Number of States", "cause": "Causal Variable", "effect": "Effect Variable"},
            template="plotly_white",
        )
        fig_granger.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_granger, use_container_width=True)

    with right:
        st.markdown("### Structural Breaks (CUSUM) Timeline")
        st.markdown(
            "Distribution of structural breaks detected over time. Clusters indicate national policy shocks."
        )
        
        break_months = breaks.groupby("break_month").size().reset_index(name="count").sort_values("break_month")
        
        if not break_months.empty:
            fig_cusum = px.bar(
                break_months,
                x="break_month",
                y="count",
                title="Number of Regime Shifts by Month",
                color_discrete_sequence=["#ea580c"],
                labels={"break_month": "Month of Shift", "count": "States with Shift"},
                template="plotly_white",
            )
            fig_cusum.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_cusum, use_container_width=True)
        else:
            st.info("No structural breaks detected to display.")

    # State Detail Section
    st.markdown("---")
    st.markdown("### 🔍 State-Level Deep Dive")
    all_states = sorted(granger["state"].unique())
    selected_state = st.selectbox("Select State to Inspect Causal Dynamics:", all_states)

    state_granger = granger[granger["state"] == selected_state]
    state_cusum = cusum[cusum["state"] == selected_state].iloc[0]

    d_cols = st.columns(2)

    with d_cols[0]:
        st.markdown(f"#### Granger Causality for {selected_state}")
        display_granger = state_granger.copy()
        display_granger["Significant?"] = display_granger["significant"].map({True: "✅ Yes", False: "❌ No"})
        display_granger["p-value"] = display_granger["p_value"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        display_granger["F-Stat"] = display_granger["f_statistic"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        show_cols = ["cause", "effect", "lag", "F-Stat", "p-value", "Significant?"]
        # Add differencing context if available
        if "cause_diffs" in display_granger.columns:
            display_granger["Differenced (d)"] = display_granger["cause_diffs"].apply(
                lambda x: f"d={int(x)}" if pd.notna(x) else "raw"
            )
            show_cols.insert(3, "Differenced (d)")

        st.dataframe(
            display_granger[show_cols],
            use_container_width=True,
            hide_index=True,
        )

    with d_cols[1]:
        st.markdown(f"#### Structural Break Analysis for {selected_state}")
        cusum_max  = state_cusum.get("cusum_max",  float("nan")) if hasattr(state_cusum, "get") else state_cusum["cusum_max"]
        crit_val   = state_cusum.get("critical_value", float("nan")) if hasattr(state_cusum, "get") else state_cusum.get("critical_value", float("nan"))
        break_det  = bool(state_cusum.get("break_detected", False)) if hasattr(state_cusum, "get") else bool(state_cusum["break_detected"])

        cusum_str  = f"{cusum_max:.2f}" if pd.notna(cusum_max) else "N/A"
        crit_str   = f"{crit_val:.2f}" if pd.notna(crit_val) else "N/A"

        if break_det:
            break_month = state_cusum.get("break_month", "Unknown") if hasattr(state_cusum, "get") else state_cusum["break_month"]
            st.success(
                f"**Regime Shift Detected!**\n\n"
                f"- **Shift Month:** {break_month}\n"
                f"- **CUSUM Maximum:** {cusum_str} (Critical Threshold: {crit_str})"
            )
            st.info(
                "A structural break indicates a permanent shift in the relationship between Aadhaar enrollment "
                "and update volumes in this state. This is typically caused by major policy directives, system outages, "
                "or demographic saturation."
            )
        else:
            st.info(
                f"**No Structural Break Detected.**\n\n"
                f"The update-to-enrollment ratio remains statistically stable over the observed period.\n"
                f"- **CUSUM Maximum:** {cusum_str} (Critical Threshold: {crit_str})"
            )

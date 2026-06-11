"""SHAP panel: Model interpretability and waterfall charts for Priority Score explanations."""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ANALYSIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "analysis",
)

FEATURE_NAME_MAP = {
    "E_total": "Total Enrollments",
    "D_total": "Demographic Updates",
    "B_total": "Biometric Updates",
    "Total_Activity": "Total Activity",
    "log_activity": "Log(Total Activity)",
    "Avg_Update_Ratio": "Avg Update Ratio",
    "Avg_Demo_Ratio": "Avg Demographic Update Ratio",
    "Avg_Bio_Ratio": "Avg Biometric Update Ratio",
    "E_child_share": "Enrollment Child Share (0-5)",
    "E_minor_share": "Enrollment Minor Share (5-17)",
    "D_adult_share": "Demo Update Adult Share (17+)",
    "B_adult_share": "Bio Update Adult Share (17+)",
    "Active_Months": "Active Months",
}


def _load_shap_summary():
    path = os.path.join(ANALYSIS_DIR, "shap_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_feature_importance():
    path = os.path.join(ANALYSIS_DIR, "feature_importance.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def render_shap_panel() -> None:
    """Render the SHAP Interpretability panel."""
    st.subheader("🔮 ML Interpretability & Feature Attribution (SHAP)")
    st.markdown(
        '<div class="panel-note"><strong>SHAP (SHapley Additive exPlanations)</strong> uses game theory to decompose the '
        "contributions of various raw metrics to the composite <strong>Priority Score</strong>. This explains "
        "<em>why</em> a particular state is flagged for attention, rather than just showing the score.</div>",
        unsafe_allow_html=True,
    )

    shap_summary = _load_shap_summary()
    feat_imp = _load_feature_importance()

    if shap_summary is None or feat_imp is None:
        st.warning(
            "SHAP explanation data not available. Run the advanced analytics pipeline first."
        )
        return

    # Two columns: global feature importance on left, local waterfall on right
    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("### Global Feature Importance")
        st.markdown(
            "Average absolute SHAP value representing the overall impact of each feature on state Priority Scores."
        )

        display_feat = feat_imp.copy()
        display_feat["feature_readable"] = (
            display_feat["feature"]
            .map(FEATURE_NAME_MAP)
            .fillna(display_feat["feature"])
        )

        fig_imp = px.bar(
            display_feat.sort_values("importance_pct"),
            x="importance_pct",
            y="feature_readable",
            orientation="h",
            title="Global Feature Impact (% of total attribution)",
            color="importance_pct",
            color_continuous_scale="Viridis",
            labels={
                "importance_pct": "Relative Importance (%)",
                "feature_readable": "Feature Name",
            },
            template="plotly_white",
        )
        fig_imp.update_layout(
            height=450, margin=dict(l=20, r=20, t=50, b=20), coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with right:
        st.markdown("### State-Level Waterfall Explanations")
        st.markdown(
            "Select a state to inspect the additive contributions driving its composite Priority Score."
        )

        all_states = sorted(shap_summary["state"].unique())
        selected_state = st.selectbox("Select State to Explain:", all_states)

        row = shap_summary[shap_summary["state"] == selected_state].iloc[0]

        # Calculate base value (expected value) from row
        predicted_val = row["predicted_priority"]
        shap_sum = row["shap_sum"]
        base_val = predicted_val - shap_sum

        # Extract features and their contributions
        t1_feat = row["top_1_feature"]
        t1_val = row["top_1_shap"]
        t2_feat = row["top_2_feature"]
        t2_val = row["top_2_shap"]
        t3_feat = row["top_3_feature"]
        t3_val = row["top_3_shap"]

        # Calculate remainder
        top_3_sum = (
            (t1_val if pd.notna(t1_val) else 0)
            + (t2_val if pd.notna(t2_val) else 0)
            + (t3_val if pd.notna(t3_val) else 0)
        )
        other_val = shap_sum - top_3_sum

        # Human-readable labels
        def get_readable(f):
            return FEATURE_NAME_MAP.get(f, str(f)) if pd.notna(f) else "N/A"

        # Construct waterfall data
        x_labels = [
            "Base Value",
            get_readable(t1_feat),
            get_readable(t2_feat),
            get_readable(t3_feat),
            "Other Features",
            "Predicted Score",
        ]
        y_vals = [base_val, t1_val, t2_val, t3_val, other_val, predicted_val]
        measures = ["absolute", "relative", "relative", "relative", "relative", "total"]

        fig_waterfall = go.Figure(
            go.Waterfall(
                name="SHAP Decomp",
                orientation="v",
                measure=measures,
                x=x_labels,
                textposition="outside",
                text=[
                    f"{v:+.3f}" if m == "relative" else f"{v:.3f}"
                    for v, m in zip(y_vals, measures)
                ],
                y=y_vals,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#ef4444"}},
                increasing={"marker": {"color": "#10b981"}},
                totals={"marker": {"color": "#3b82f6"}},
            )
        )

        fig_waterfall.update_layout(
            title=f"SHAP Waterfall Chart for {selected_state}",
            waterfallgap=0.3,
            height=450,
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    # Detailed table view
    st.markdown("---")
    st.markdown("### 📊 Comprehensive SHAP Contributions Table")
    display_summary = shap_summary.copy()
    display_summary["expected_value"] = (
        display_summary["predicted_priority"] - display_summary["shap_sum"]
    )

    # Map top feature columns to readable names
    display_summary["top_1_feature"] = (
        display_summary["top_1_feature"]
        .map(FEATURE_NAME_MAP)
        .fillna(display_summary["top_1_feature"])
    )
    display_summary["top_2_feature"] = (
        display_summary["top_2_feature"]
        .map(FEATURE_NAME_MAP)
        .fillna(display_summary["top_2_feature"])
    )
    display_summary["top_3_feature"] = (
        display_summary["top_3_feature"]
        .map(FEATURE_NAME_MAP)
        .fillna(display_summary["top_3_feature"])
    )

    # Round numerical columns
    num_cols = [
        "predicted_priority",
        "expected_value",
        "shap_sum",
        "top_1_shap",
        "top_2_shap",
        "top_3_shap",
    ]
    for col in num_cols:
        if col in display_summary.columns:
            display_summary[col] = display_summary[col].round(4)

    st.dataframe(
        display_summary[
            [
                "state",
                "predicted_priority",
                "expected_value",
                "shap_sum",
                "top_1_feature",
                "top_1_shap",
                "top_2_feature",
                "top_2_shap",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

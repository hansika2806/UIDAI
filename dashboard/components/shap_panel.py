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
    "Avg_Update_Ratio": "Aadhaar Record Update Pressure",
    "Avg_Demo_Ratio": "Demographic Update Volume Share",
    "E_child_share": "Child Enrollment Share (0-5)",
    "B_adult_share": "Biometric Update Share (Adults 17+)",
    "E_total": "Total Enrollment Velocity",
    "Active_Months": "Operational Months Density",
    "E_minor_share": "Minor Enrollment Share (5-17)",
    "Avg_Bio_Ratio": "Biometric Update Volume Share",
    "D_adult_share": "Demographic Update Share (Adults 17+)",
    "D_total": "Demographic Updates",
    "B_total": "Biometric Updates",
    "Total_Activity": "Total Activity",
    "log_activity": "Log(Total Activity)",
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


def _readable(feature_code: str) -> str:
    """Convert a raw feature name to a human-readable label."""
    return FEATURE_NAME_MAP.get(str(feature_code), str(feature_code)) if pd.notna(feature_code) else "N/A"


def _shap_interpreter(row: pd.Series) -> str:
    """Generate dynamic interpreter text from the top SHAP contribution."""
    t1_feat = row.get("top_1_feature", None)
    t1_val = row.get("top_1_shap", None)

    if pd.isna(t1_feat) or pd.isna(t1_val):
        return "Insufficient SHAP data to generate an interpretation for this state."

    readable_feat = _readable(t1_feat)
    predicted = row.get("predicted_priority", 0.5)

    if t1_val > 0:
        direction_phrase = (
            f"This state's priority score is elevated primarily because its "
            f"<strong>{readable_feat}</strong> is significantly above the national baseline, "
            f"indicating increased operational stress."
        )
    else:
        direction_phrase = (
            f"This state's priority score sits below the national baseline, primarily because its "
            f"<strong>{readable_feat}</strong> remains low, reducing immediate operational pressure."
        )

    # Check second feature for supporting context
    t2_feat = row.get("top_2_feature", None)
    t2_val = row.get("top_2_shap", None)
    support = ""
    if pd.notna(t2_feat) and pd.notna(t2_val):
        r2 = _readable(t2_feat)
        if t2_val > 0:
            support = f" A secondary upward push comes from <strong>{r2}</strong>."
        else:
            support = f" <strong>{r2}</strong> partially offsets this, pulling the score downward."

    priority_context = ""
    if predicted >= 0.55:
        priority_context = " This state requires <strong>active policy attention</strong>."
    elif predicted <= 0.35:
        priority_context = " This state is currently in a <strong>lower-priority zone</strong> — but may need monitoring for emerging risks."

    return direction_phrase + support + priority_context


def render_shap_panel() -> None:
    """Render the SHAP Interpretability panel."""
    st.markdown(
        '<div class="headline">🔮 ML Interpretability &amp; Feature Attribution</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="interpret-box" style="margin-top:0;">'
        '<strong>SHAP (SHapley Additive exPlanations)</strong> uses game theory to decompose '
        "the contributions of various raw metrics to the composite <strong>Priority Score</strong>. "
        "This explains <em>why</em> a particular state is flagged for attention, not just the score itself."
        "</div>",
        unsafe_allow_html=True,
    )

    shap_summary = _load_shap_summary()
    feat_imp = _load_feature_importance()

    if shap_summary is None or feat_imp is None:
        st.warning("SHAP explanation data not available. Run the advanced analytics pipeline first.")
        return

    # ── Two columns: global importance (left) + waterfall (right) ────────────
    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("### Global Feature Importance")
        st.markdown(
            "Average absolute SHAP value — how much each feature influences Priority Scores nationwide."
        )

        display_feat = feat_imp.copy()
        display_feat["feature_readable"] = (
            display_feat["feature"].map(FEATURE_NAME_MAP).fillna(display_feat["feature"])
        )

        fig_imp = px.bar(
            display_feat.sort_values("importance_pct"),
            x="importance_pct",
            y="feature_readable",
            orientation="h",
            title="Global Feature Impact (% of total attribution)",
            color="importance_pct",
            color_continuous_scale="Teal",
            labels={
                "importance_pct": "Relative Importance (%)",
                "feature_readable": "",
            },
            template="plotly_white",
        )
        fig_imp.update_layout(
            height=420,
            margin=dict(l=10, r=20, t=50, b=20),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_imp, width="stretch", config={"displayModeBar": False})

    with right:
        st.markdown("### State-Level Waterfall Explanation")
        all_states = sorted(shap_summary["state"].unique())
        selected_state = st.selectbox("Select State to Explain:", all_states, key="shap_state_select")

        row = shap_summary[shap_summary["state"] == selected_state].iloc[0]
        predicted_val = row["predicted_priority"]
        shap_sum = row["shap_sum"]
        base_val = predicted_val - shap_sum

        t1_feat, t1_val = row["top_1_feature"], row["top_1_shap"]
        t2_feat, t2_val = row["top_2_feature"], row["top_2_shap"]
        t3_feat, t3_val = row["top_3_feature"], row["top_3_shap"]

        top_3_sum = sum(v if pd.notna(v) else 0 for v in [t1_val, t2_val, t3_val])
        other_val = shap_sum - top_3_sum

        x_labels = [
            "National Baseline",
            _readable(t1_feat),
            _readable(t2_feat),
            _readable(t3_feat),
            "Other Features",
            "Priority Score",
        ]
        y_vals = [base_val, t1_val, t2_val, t3_val, other_val, predicted_val]
        measures = ["absolute", "relative", "relative", "relative", "relative", "total"]

        fig_waterfall = go.Figure(
            go.Waterfall(
                name="SHAP Decomposition",
                orientation="v",
                measure=measures,
                x=x_labels,
                textposition="outside",
                text=[
                    f"{v:+.3f}" if m == "relative" else f"{v:.3f}"
                    for v, m in zip(y_vals, measures)
                ],
                y=y_vals,
                connector={"line": {"color": "rgba(148,163,184,0.5)", "width": 1}},
                decreasing={"marker": {"color": "#ef4444"}},
                increasing={"marker": {"color": "#10b981"}},
                totals={"marker": {"color": "#3b82f6"}},
            )
        )
        fig_waterfall.update_layout(
            title=f"Priority Score Build-Up — {selected_state}",
            waterfallgap=0.3,
            height=400,
            template="plotly_white",
            margin=dict(l=10, r=10, t=50, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
        )
        st.plotly_chart(fig_waterfall, width="stretch")

        # ── Dynamic interpreter below waterfall ───────────────────────────────
        interp_text = _shap_interpreter(row)
        st.markdown(
            f'<div class="interpret-box">{interp_text}</div>',
            unsafe_allow_html=True,
        )

    # ── Comprehensive table (behind expander) ─────────────────────────────────
    with st.expander("Show full SHAP contributions table for all states"):
        display_summary = shap_summary.copy()
        display_summary["expected_value"] = (
            display_summary["predicted_priority"] - display_summary["shap_sum"]
        )
        for col in ["top_1_feature", "top_2_feature", "top_3_feature"]:
            display_summary[col] = (
                display_summary[col].map(FEATURE_NAME_MAP).fillna(display_summary[col])
            )
        for col in ["predicted_priority", "expected_value", "shap_sum", "top_1_shap", "top_2_shap", "top_3_shap"]:
            if col in display_summary.columns:
                display_summary[col] = display_summary[col].round(4)

        st.dataframe(
            display_summary[[
                "state", "predicted_priority", "expected_value", "shap_sum",
                "top_1_feature", "top_1_shap", "top_2_feature", "top_2_shap",
            ]],
            width="stretch",
            hide_index=True,
        )

"""Causal Inference panel: Granger causality Plotly flowchart and CUSUM structural breaks."""

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

# Human-readable labels for causal variables
CAUSAL_LABEL_MAP = {
    "Enrollment": "New Enrollments",
    "Demographic Updates": "Demographic Record Changes",
    "Biometric Updates": "Biometric Capture Updates",
}

# Predefined policy actions derived from significant causal links
CAUSAL_ACTION_MAP = {
    ("Enrollment", "Demographic Updates"): (
        "Past enrollment surges reliably predict a spike in "
        "<strong>demographic update requests</strong> 1–2 months later. "
        "Pre-position update capacity before any enrollment drive."
    ),
    ("Enrollment", "Biometric Updates"): (
        "Past enrollment surges reliably predict a spike in "
        "<strong>biometric update requests</strong> 1–2 months later. "
        "Pre-position biometric capture capacity before any enrollment drive."
    ),
    ("Demographic Updates", "Biometric Updates"): (
        "Past demographic update volumes reliably predict a surge in "
        "<strong>biometric update requests</strong> 1–2 months later. "
        "Coordinate scheduling across demographic and biometric queues."
    ),
}


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


def _build_causal_flow_chart(state_granger: pd.DataFrame) -> go.Figure:
    """Build a Plotly flowchart showing Granger causal links with hover stats."""
    all_tests = state_granger.copy()

    # Nodes: 3 variables at fixed positions
    node_labels = list(CAUSAL_LABEL_MAP.values())
    node_x = [0.1, 0.5, 0.9]
    node_y = [0.5, 0.85, 0.5]

    fig = go.Figure()

    # Background node circles
    for i, (lbl, x, y) in enumerate(zip(node_labels, node_x, node_y)):
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=55, color="#102a43", opacity=0.9),
                text=[lbl],
                textposition="middle center",
                textfont=dict(color="white", size=10, family="Inter, sans-serif"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    var_index = {
        "Enrollment": 0,
        "Demographic Updates": 1,
        "Biometric Updates": 2,
    }

    # Draw edges for each tested cause→effect pair
    for _, row in all_tests.drop_duplicates(subset=["cause", "effect"]).iterrows():
        c = row["cause"]
        e = row["effect"]
        ci = var_index.get(c, 0)
        ei = var_index.get(e, 2)
        is_sig = bool(row.get("significant", False))

        x0, y0 = node_x[ci], node_y[ci]
        x1, y1 = node_x[ei], node_y[ei]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.08

        f_stat = row.get("f_statistic", float("nan"))
        p_val = row.get("p_value", float("nan"))
        lag = row.get("lag", "?")

        hover_text = (
            f"<b>{c} → {e}</b><br>"
            f"Lag: {lag} month(s)<br>"
            f"F-stat: {f_stat:.2f}<br>"
            f"p-value: {p_val:.4f}<br>"
            f"{'✅ Significant' if is_sig else '❌ Not significant'}"
        )

        line_color = "#16a34a" if is_sig else "rgba(148,163,184,0.4)"
        line_width = 3.5 if is_sig else 1.5
        line_dash = "solid" if is_sig else "dot"

        fig.add_trace(
            go.Scatter(
                x=[x0, mx, x1],
                y=[y0, my, y1],
                mode="lines",
                line=dict(color=line_color, width=line_width, dash=line_dash),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
            )
        )

        # Arrowhead indicator (midpoint marker)
        label_txt = "✓" if is_sig else "✗"
        fig.add_trace(
            go.Scatter(
                x=[mx],
                y=[my],
                mode="markers+text",
                marker=dict(size=18, color=line_color, symbol="circle"),
                text=[label_txt],
                textfont=dict(color="white", size=11),
                textposition="middle center",
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Granger Causality Network (hover for F-statistic & p-value)",
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[0.2, 1.1]),
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(255,255,255,0.92)",
        plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
    )
    return fig


def _causal_interpreter(state_granger: pd.DataFrame) -> str:
    """Generate dynamic Granger causality interpretation based on significant links."""
    sig = state_granger[state_granger["significant"]]
    disclaimer = (
        "<div style='font-size:0.78rem;color:#64748b;margin-top:0.6rem;"
        "border-top:1px dashed rgba(14,116,144,0.15);padding-top:0.5rem;line-height:1.4;'>"
        "⚠️ <strong>Statistical Note:</strong> Granger causality measures <em>predictive precedence</em> "
        "(whether historical fluctuations in variable X help forecast variable Y) rather than direct "
        "physical cause-and-effect. Use these signals to coordinate administrative queue schedules, "
        "but verify insights via local field audits before implementing hard policy changes."
        "</div>"
    )

    if sig.empty:
        return (
            "No reliable predictive pattern between enrollment and update volumes was detected. "
            "This state's update demand appears <strong>independent of enrollment timing</strong>. "
            "Policy interventions for update capacity can be decoupled from enrollment campaigns."
            + disclaimer
        )

    lines = []
    seen_pairs = set()
    for _, row in sig.iterrows():
        pair = (row["cause"], row["effect"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        action_text = CAUSAL_ACTION_MAP.get(pair)
        if action_text:
            lag = row.get("lag", "?")
            f_stat = row.get("f_statistic", float("nan"))
            lines.append(
                f"{action_text} "
                f"(F={f_stat:.2f}, lag={lag} month{'s' if int(lag) != 1 else ''})"
            )

    if not lines:
        return (
            "Significant links detected but no standard interpretation available for this pair."
            + disclaimer
        )

    intro = (
        f"<strong>{len(seen_pairs)} significant causal link(s) detected:</strong><br>"
    )
    body = "<br><br>".join(f"• {line}" for line in lines)
    return intro + body + disclaimer


def render_causal_panel() -> None:
    """Render the Causal Inference panel."""
    st.markdown(
        '<div class="headline">🔗 Causal Inference &amp; Structural Breaks</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="interpret-box" style="margin-top:0;">'
        "<strong>Granger Causality</strong> tests whether historical trends in enrollment "
        "statistically help predict future update volumes. "
        "<strong>CUSUM (Cumulative Sum)</strong> detects structural breaks — permanent shifts "
        "in the update-to-enrollment ratio caused by policy shocks, lockdowns, or saturation."
        "</div>",
        unsafe_allow_html=True,
    )

    granger = _load_granger_results()
    cusum = _load_cusum_results()

    if granger is None or cusum is None:
        st.warning(
            "Causal inference data not available. Run the advanced analytics pipeline first."
        )
        return

    # ── Stationarity banner ───────────────────────────────────────────────────
    if "cause_diffs" in granger.columns:
        n_differenced = int((granger["cause_diffs"] > 0).sum())
        n_total = len(granger)
        if n_differenced > 0:
            pct = round(100 * n_differenced / n_total)
            st.info(
                f"🧮 **Stationarity enforced (ADF):** {n_differenced}/{n_total} test pairs ({pct}%) "
                f"used differenced series. Results describe *changes* in the causal variable, not raw levels."
            )

    # ── High-level metrics ────────────────────────────────────────────────────
    sig_granger = granger[granger["significant"]]
    breaks = cusum[cusum["break_detected"]]
    most_common_break = (
        breaks["break_month"].mode().iloc[0] if not breaks.empty else "N/A"
    )

    cols = st.columns(3)
    with cols[0]:
        st.metric(
            "Significant Causal Links",
            f"{len(sig_granger)} / {len(granger.drop_duplicates(['state','cause','effect','lag']))}",
            help="Granger causality p-value < 0.05",
        )
    with cols[1]:
        st.metric(
            "Regime Shifts Detected",
            f"{len(breaks)} / {len(cusum)}",
            help="States with CUSUM break detected",
        )
    with cols[2]:
        st.metric(
            "Peak Shift Month",
            most_common_break,
            help="Month with most structural breaks clustered",
        )

    # ── Summary overview charts ───────────────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Granger Causality Overview")
        summary_granger = (
            granger[granger["significant"]]
            .groupby(["cause", "effect"])
            .size()
            .reset_index(name="count")
        )
        if not summary_granger.empty:
            fig_granger = px.bar(
                summary_granger,
                x="count",
                y="cause",
                color="effect",
                orientation="h",
                title="Significant Causal Links Across All States",
                color_discrete_sequence=["#0e7490", "#d97706", "#dc2626"],
                labels={"count": "Number of States", "cause": ""},
                template="plotly_white",
            )
            fig_granger.update_layout(
                height=360,
                margin=dict(l=10, r=20, t=50, b=20),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_granger, width="stretch")
        else:
            st.info("No significant causal links across the entire dataset.")

    with right:
        st.markdown("### Structural Breaks (CUSUM) Timeline")
        break_months = (
            breaks.groupby("break_month")
            .size()
            .reset_index(name="count")
            .sort_values("break_month")
        )
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
            fig_cusum.update_layout(
                height=360,
                margin=dict(l=10, r=20, t=50, b=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
            )
            st.plotly_chart(fig_cusum, width="stretch")
        else:
            st.info("No structural breaks detected to display.")

    # ── State-level deep dive ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 State-Level Causal Deep Dive")
    all_states = sorted(granger["state"].unique())
    selected_state = st.selectbox(
        "Select State to Inspect Causal Dynamics:",
        all_states,
        key="causal_state_select",
    )

    state_granger = granger[granger["state"] == selected_state]

    # Plotly causal flow chart
    st.plotly_chart(_build_causal_flow_chart(state_granger), width="stretch")

    # Dynamic interpreter
    interp_text = _causal_interpreter(state_granger)
    st.markdown(
        f'<div class="interpret-box">{interp_text}</div>',
        unsafe_allow_html=True,
    )

    # CUSUM detail for selected state
    state_cusum_df = cusum[cusum["state"] == selected_state]
    if not state_cusum_df.empty:
        state_cusum = state_cusum_df.iloc[0]
        cusum_max = float(state_cusum.get("cusum_max", float("nan")))
        crit_val = float(state_cusum.get("critical_value", float("nan")))
        break_det = bool(state_cusum.get("break_detected", False))

        cusum_str = f"{cusum_max:.2f}" if pd.notna(cusum_max) else "N/A"
        crit_str = f"{crit_val:.2f}" if pd.notna(crit_val) else "N/A"

        st.markdown("#### Structural Break Analysis")
        if break_det:
            break_month = state_cusum.get("break_month", "Unknown")
            st.success(
                f"**Regime Shift Detected in {selected_state}!** "
                f"Break month: **{break_month}** — "
                f"CUSUM max: {cusum_str} (threshold: {crit_str}). "
                "This indicates a permanent shift in the update-to-enrollment ratio, typically caused by "
                "major policy directives, system outages, or demographic saturation."
            )
        else:
            st.info(
                f"**No structural break detected** in {selected_state}. "
                f"The update-to-enrollment ratio remains statistically stable. "
                f"CUSUM max: {cusum_str} (threshold: {crit_str})."
            )

    # Technical table behind expander
    with st.expander("Show full Granger causality table for this state"):
        display_granger = state_granger.copy()
        display_granger["Significant?"] = display_granger["significant"].map(
            {True: "✅ Yes", False: "❌ No"}
        )
        display_granger["p-value"] = display_granger["p_value"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )
        display_granger["F-Stat"] = display_granger["f_statistic"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
        show_cols = ["cause", "effect", "lag", "F-Stat", "p-value", "Significant?"]
        if "cause_diffs" in display_granger.columns:
            display_granger["Differenced (d)"] = display_granger["cause_diffs"].apply(
                lambda x: f"d={int(x)}" if pd.notna(x) else "raw"
            )
            show_cols.insert(3, "Differenced (d)")
        st.dataframe(display_granger[show_cols], width="stretch", hide_index=True)

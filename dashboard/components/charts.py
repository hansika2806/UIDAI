"""
Plotly chart builders for the UIDAI Intelligence Studio.

Two-register chart chrome approach:
  • Summary charts (executive overview): minimal grid, no axis labels, clean
  • Detail charts (state drilldown, diagnostics): full grid, axis labels, tooltips
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Shared design tokens ─────────────────────────────────────────────────────
PALETTE = {
    "navy": "#102a43",
    "teal": "#0e7490",
    "teal_lt": "rgba(14,116,144,0.08)",
    "amber": "#d97706",
    "coral": "#ea580c",
    "green": "#16a34a",
    "red": "#dc2626",
    "slate": "#334155",
    "muted": "#64748b",
    "mist": "#f8fafc",
    "border": "rgba(148,163,184,0.20)",
    "surface": "rgba(255,255,255,0.92)",
}

FONT = dict(family="Inter, Segoe UI, sans-serif", size=12, color=PALETTE["slate"])
GRID = dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
NO_GRID = dict(showgrid=False)


def _summary_layout(
    fig: go.Figure, title: str, height: int = 400, x_title: str = "", y_title: str = ""
) -> go.Figure:
    """Apply detailed summary-register chrome: full grid, axis titles."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PALETTE["navy"])),
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor="rgba(248,250,252,0.6)",
        font=FONT,
        xaxis=dict(**GRID, title=x_title),
        yaxis=dict(**GRID, title=y_title),
    )
    return fig


def _detail_layout(
    fig: go.Figure, title: str, height: int = 460, x_title: str = "", y_title: str = ""
) -> go.Figure:
    """Apply full detail-register chrome: axis titles, grid, tooltips."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PALETTE["navy"])),
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor="rgba(248,250,252,0.6)",
        font=FONT,
        xaxis=dict(**GRID, title=x_title),
        yaxis=dict(**GRID, title=y_title),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Summary-register charts (executive overview)
# ══════════════════════════════════════════════════════════════════════════════

def build_national_trend_chart(national: pd.DataFrame) -> go.Figure:
    """National UIDAI activity trend — summary register."""
    fig = go.Figure()
    traces = [
        ("E_total", "Enrollments", PALETTE["navy"], 2.5),
        ("D_total", "Demographic Updates", PALETTE["teal"], 2.5),
        ("B_total", "Biometric Updates", PALETTE["amber"], 2.5),
    ]
    for col, name, color, width in traces:
        fig.add_trace(go.Scatter(
            x=national["year_month"],
            y=national[col],
            mode="lines",
            name=name,
            line=dict(color=color, width=width),
        ))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _summary_layout(fig, "National UIDAI Activity Trend", 420, "Month", "Activity Volume")


def build_category_mix_chart(state_master: pd.DataFrame) -> go.Figure:
    """Policy category distribution — summary register."""
    counts = (
        state_master["Policy_Category"]
        .value_counts()
        .rename_axis("Policy_Category")
        .reset_index(name="Count")
    )
    fig = px.bar(
        counts,
        x="Count",
        y="Policy_Category",
        orientation="h",
        color="Policy_Category",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(showlegend=False)
    return _summary_layout(fig, "Policy Category Mix", 420, "Count", "Policy Category")


def build_pareto_chart(pareto_df: pd.DataFrame) -> go.Figure:
    """Activity concentration Pareto — summary register."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=pareto_df["state"],
            y=pareto_df["Total_Activity"],
            name="Total Activity",
            marker_color=PALETTE["navy"],
            marker_line_width=0,
            opacity=0.85,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=pareto_df["state"],
            y=pareto_df["cum_share"],
            mode="lines+markers",
            name="Cumulative Share",
            line=dict(color=PALETTE["coral"], width=3),
            marker=dict(size=4, color=PALETTE["coral"]),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="", secondary_y=False, **GRID)
    fig.update_yaxes(title_text="", tickformat=".0%", secondary_y=True, **NO_GRID)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _summary_layout(fig, "Activity Concentration by State", 420, "State", "Activity Volume / Share")


# ══════════════════════════════════════════════════════════════════════════════
# Detail-register charts (diagnostics, drilldowns)
# ══════════════════════════════════════════════════════════════════════════════

def build_indicator_heatmap(corr_df: pd.DataFrame, title: str) -> go.Figure:
    """Correlation heatmap — detail register."""
    matrix = corr_df.set_index("metric")
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )
    return _detail_layout(fig, title, 420)


def build_quadrant_chart(
    state_master: pd.DataFrame, highlight_state: str = ""
) -> go.Figure:
    """Governance quadrant (AMI vs UPI) — detail register."""
    fig = px.scatter(
        state_master,
        x="AMI",
        y="UPI",
        color="Policy_Category",
        size="VSI",
        size_max=44,
        hover_name="state",
        hover_data={"TPS": ":.3f", "Total_Activity": ":,.0f", "Governance_Status": True},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.add_vline(x=state_master["AMI"].median(), line_dash="dash", line_color="#94a3b8")
    fig.add_hline(y=state_master["UPI"].median(), line_dash="dash", line_color="#94a3b8")

    if highlight_state and highlight_state in set(state_master["state"]):
        chosen = state_master[state_master["state"] == highlight_state]
        fig.add_trace(go.Scatter(
            x=chosen["AMI"], y=chosen["UPI"],
            mode="markers+text",
            text=chosen["state"],
            textposition="top center",
            marker=dict(size=18, color=PALETTE["coral"], line=dict(width=2, color="white")),
            name="Selected State",
        ))

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _detail_layout(fig, "Governance Quadrant: Maturity vs Maintenance Pressure", 520,
                          x_title="AMI (Aadhaar Maturity Index)",
                          y_title="UPI (Update Propensity Index)")


def build_state_month_chart(state_series: pd.DataFrame, state_name: str) -> go.Figure:
    """State monthly activity mix — detail register."""
    chart_df = state_series.sort_values("year_month")
    fig = go.Figure()
    traces = [
        ("E_total", "Enrollments", PALETTE["navy"], 2.5),
        ("D_total", "Demographic Updates", PALETTE["teal"], 2.5),
        ("B_total", "Biometric Updates", PALETTE["amber"], 2.5),
    ]
    for col, name, color, width in traces:
        fig.add_trace(go.Scatter(
            x=chart_df["year_month"], y=chart_df[col],
            mode="lines", name=name,
            line=dict(color=color, width=width),
        ))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _detail_layout(fig, f"{state_name}: Monthly Activity Mix", 420,
                          x_title="Month", y_title="Volume")


def build_ratio_trend_chart(national: pd.DataFrame) -> go.Figure:
    """Maintenance intensity ratios — detail register."""
    fig = go.Figure()
    traces = [
        ("demo_update_ratio", "Demographic / Enrollment", PALETTE["teal"], 3),
        ("biometric_update_ratio", "Biometric / Enrollment", PALETTE["amber"], 3),
        ("update_to_enrol_ratio", "Combined Update / Enrollment", PALETTE["coral"], 4),
    ]
    for col, name, color, width in traces:
        fig.add_trace(go.Scatter(
            x=national["year_month"], y=national[col],
            mode="lines", name=name,
            line=dict(color=color, width=width),
        ))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _detail_layout(fig, "Maintenance Intensity Through Time", 420,
                          x_title="Month", y_title="Ratio")


def build_stress_scale_chart(state_master: pd.DataFrame) -> go.Figure:
    """Stress vs scale diagnostic — detail register."""
    fig = px.scatter(
        state_master,
        x="Total_Activity", y="VSI",
        color="AMI", size="UPI", size_max=40,
        hover_name="state",
        hover_data=["TPS", "Governance_Status"],
        color_continuous_scale="Viridis",
        log_x=True,
    )
    fig.add_hline(y=state_master["VSI"].median(), line_dash="dash", line_color="#94a3b8")
    fig.add_vline(x=max(state_master["Total_Activity"].median(), 1), line_dash="dash", line_color="#94a3b8")
    return _detail_layout(fig, "Stress vs Scale Diagnostic", 500,
                          x_title="Total Activity (log scale)",
                          y_title="VSI (Volume Stress Index)")


def build_lifecycle_chart(state_master: pd.DataFrame) -> go.Figure:
    """Lifecycle transition pattern — detail register."""
    fig = px.scatter(
        state_master,
        x="E_child_share", y="D_adult_share",
        size="Total_Activity", color="TPS",
        hover_name="state", size_max=50,
        color_continuous_scale="Tealgrn",
    )
    return _detail_layout(fig, "Lifecycle Transition Pattern", 500,
                          x_title="Child Enrollment Share (0-5)",
                          y_title="Adult Demographic Update Share (17+)")


def build_anomaly_chart(anomalies: pd.DataFrame) -> go.Figure:
    """Anomaly pressure map — summary register (overview context)."""
    plot_df = anomalies.copy()
    plot_df["Anomaly_Flag_Count"] = plot_df["Anomaly_Flag_Count"].astype(float).tolist()
    fig = px.scatter(
        plot_df,
        x="AMI", y="UPI",
        size="Anomaly_Flag_Count",
        color="Anomaly_Flag_Count",
        hover_name="state",
        hover_data=["VSI", "TPS", "Governance_Status"],
        size_max=48,
        color_continuous_scale="OrRd",
    )
    return _detail_layout(fig, "Anomaly Pressure Map", 500,
                          x_title="AMI (Aadhaar Maturity Index)",
                          y_title="UPI (Update Propensity Index)")


def build_peer_comparison(state_master: pd.DataFrame, selected_state: str) -> go.Figure:
    """State vs median peer radar — detail register."""
    row = state_master[state_master["state"] == selected_state].iloc[0]
    metrics = ["AMI", "UPI", "VSI", "TPS"]
    peer = state_master[metrics].median()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[row[m] for m in metrics] + [row[metrics[0]]],
        theta=metrics + [metrics[0]],
        fill="toself", name=selected_state,
        line=dict(color=PALETTE["teal"], width=2.5),
        fillcolor="rgba(14,116,144,0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[peer[m] for m in metrics] + [peer[metrics[0]]],
        theta=metrics + [metrics[0]],
        fill="toself", name="National Median",
        line=dict(color=PALETTE["muted"], width=2, dash="dash"),
        fillcolor="rgba(100,116,139,0.08)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=13, color=PALETTE["slate"])),
        ),
        showlegend=True,
        legend=dict(x=0.85, y=1.0),
        margin=dict(l=30, r=30, t=50, b=30),
        height=420,
        paper_bgcolor=PALETTE["surface"],
        font=FONT,
        title=dict(
            text=f"{selected_state}: Indicator Profile vs Median",
            font=dict(size=14, color=PALETTE["navy"]),
        ),
    )
    return fig


def build_rank_persistence_chart(rank_df: pd.DataFrame, compare_mode: str) -> go.Figure:
    """Enrollment rank vs update rank — detail register."""
    y_metric = "D_rank" if compare_mode == "Demographic Rank" else "B_rank"
    fig = px.scatter(
        rank_df,
        x="E_rank", y=y_metric,
        hover_name="state",
        color_discrete_sequence=[PALETTE["teal"]],
    )
    # Add 45-degree reference line
    max_rank = max(rank_df["E_rank"].max(), rank_df[y_metric].max()) if not rank_df.empty else 35
    fig.add_trace(go.Scatter(
        x=[1, max_rank], y=[1, max_rank],
        mode="lines",
        line=dict(dash="dot", color="#94a3b8", width=1.5),
        name="Perfect Persistence",
        showlegend=True,
    ))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _detail_layout(fig, f"Enrollment Rank vs {compare_mode}", 430,
                          x_title="Enrollment Rank",
                          y_title=compare_mode)


def build_volatility_bar_chart(state_master: pd.DataFrame) -> go.Figure:
    """Highest volatility states — detail register."""
    top_vol = state_master.nlargest(12, "VSI")[["state", "VSI", "UPI", "TPS"]]
    fig = px.bar(
        top_vol.sort_values("VSI"),
        x="VSI", y="state",
        orientation="h",
        color="UPI",
        color_continuous_scale="YlOrRd",
    )
    return _detail_layout(fig, "Highest Volatility States", 430,
                          x_title="VSI (Volume Stress Index)")


def build_mismatch_queue_chart(pressure_mismatch: pd.DataFrame) -> go.Figure:
    """Pressure-maturity mismatch queue — detail register."""
    fig = px.bar(
        pressure_mismatch.sort_values("Priority_Score"),
        x="Priority_Score", y="state",
        color="UPI",
        orientation="h",
        color_continuous_scale="Oranges",
    )
    return _detail_layout(fig, "Pressure-Maturity Mismatch Queue", 460,
                          x_title="Priority Score")


def build_state_ratio_trend_chart(
    state_series: pd.DataFrame, selected_state: str
) -> go.Figure:
    """State-level update intensity trend — detail register."""
    ratio_cols = ["year_month", "demo_update_ratio", "biometric_update_ratio", "update_to_enrol_ratio"]
    available = [c for c in ratio_cols if c in state_series.columns]
    ratio_series = state_series[available]
    fig = px.line(
        ratio_series.melt(id_vars="year_month", var_name="Metric", value_name="Value"),
        x="year_month", y="Value", color="Metric",
        color_discrete_sequence=[PALETTE["teal"], PALETTE["amber"], PALETTE["coral"]],
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _detail_layout(fig, f"{selected_state}: Update Intensity Through Time", 420,
                          x_title="Month", y_title="Ratio")

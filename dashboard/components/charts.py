import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = {
    "navy": "#102a43",
    "teal": "#0e7490",
    "gold": "#f59e0b",
    "coral": "#ea580c",
    "green": "#15803d",
    "slate": "#475569",
    "mist": "#f8fafc",
}


def build_pareto_chart(pareto_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=pareto_df["state"],
            y=pareto_df["Total_Activity"],
            name="Total Activity",
            marker_color=PALETTE["navy"],
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
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Activity Concentration by State",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Total Activity", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Share", tickformat=".0%", secondary_y=True)
    return fig


def build_indicator_heatmap(corr_df: pd.DataFrame, title: str) -> go.Figure:
    matrix = corr_df.set_index("metric")
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_quadrant_chart(
    state_master: pd.DataFrame, highlight_state: str = ""
) -> go.Figure:
    fig = px.scatter(
        state_master,
        x="AMI",
        y="UPI",
        color="Policy_Category",
        size="VSI",
        size_max=44,
        hover_name="state",
        hover_data={
            "TPS": ":.3f",
            "Total_Activity": ":,.0f",
            "Governance_Status": True,
        },
        color_discrete_sequence=px.colors.qualitative.Safe,
        template="plotly_white",
        title="Governance Quadrant: Maturity vs Maintenance Pressure",
    )
    fig.add_vline(
        x=state_master["AMI"].median(), line_dash="dash", line_color="#64748b"
    )
    fig.add_hline(
        y=state_master["UPI"].median(), line_dash="dash", line_color="#64748b"
    )
    if highlight_state and highlight_state in set(state_master["state"]):
        chosen = state_master[state_master["state"] == highlight_state]
        fig.add_trace(
            go.Scatter(
                x=chosen["AMI"],
                y=chosen["UPI"],
                mode="markers+text",
                text=chosen["state"],
                textposition="top center",
                marker=dict(
                    size=18, color=PALETTE["coral"], line=dict(width=2, color="white")
                ),
                name="Selected State",
            )
        )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=70, b=20))
    return fig


def build_state_month_chart(state_series: pd.DataFrame, state_name: str) -> go.Figure:
    chart_df = state_series.sort_values("year_month")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["year_month"],
            y=chart_df["E_total"],
            mode="lines",
            name="Enrollments",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["year_month"],
            y=chart_df["D_total"],
            mode="lines",
            name="Demographic Updates",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["year_month"],
            y=chart_df["B_total"],
            mode="lines",
            name="Biometric Updates",
        )
    )
    fig.update_layout(
        title=f"{state_name}: Monthly Activity Mix",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_ratio_trend_chart(national: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=national["year_month"],
            y=national["demo_update_ratio"],
            mode="lines",
            name="Demographic / Enrollment",
            line=dict(color=PALETTE["teal"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=national["year_month"],
            y=national["biometric_update_ratio"],
            mode="lines",
            name="Biometric / Enrollment",
            line=dict(color=PALETTE["gold"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=national["year_month"],
            y=national["update_to_enrol_ratio"],
            mode="lines",
            name="Combined Update / Enrollment",
            line=dict(color=PALETTE["coral"], width=4),
        )
    )
    fig.update_layout(
        title="Maintenance Intensity Through Time",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_stress_scale_chart(state_master: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        state_master,
        x="Total_Activity",
        y="VSI",
        color="AMI",
        size="UPI",
        size_max=40,
        hover_name="state",
        hover_data=["TPS", "Governance_Status"],
        color_continuous_scale="Viridis",
        log_x=True,
        template="plotly_white",
        title="Stress vs Scale Diagnostic",
    )
    fig.add_hline(
        y=state_master["VSI"].median(), line_dash="dash", line_color="#94a3b8"
    )
    fig.add_vline(
        x=max(state_master["Total_Activity"].median(), 1),
        line_dash="dash",
        line_color="#94a3b8",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_lifecycle_chart(state_master: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        state_master,
        x="E_child_share",
        y="D_adult_share",
        size="Total_Activity",
        color="TPS",
        hover_name="state",
        size_max=50,
        color_continuous_scale="Tealgrn",
        template="plotly_white",
        title="Lifecycle Transition Pattern",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_xaxes(title="Child Enrollment Share (0-5)")
    fig.update_yaxes(title="Adult Demographic Update Share (17+)")
    return fig


def build_anomaly_chart(anomalies: pd.DataFrame) -> go.Figure:
    # Explicitly coerce to plain lists to avoid narwhals.Series incompatibility
    # with plotly's 'size' validator in newer pandas/plotly versions.
    plot_df = anomalies.copy()
    plot_df["Anomaly_Flag_Count"] = plot_df["Anomaly_Flag_Count"].astype(float).tolist()
    fig = px.scatter(
        plot_df,
        x="AMI",
        y="UPI",
        size="Anomaly_Flag_Count",
        color="Anomaly_Flag_Count",
        hover_name="state",
        hover_data=["VSI", "TPS", "Governance_Status"],
        size_max=48,
        color_continuous_scale="OrRd",
        template="plotly_white",
        title="Anomaly Pressure Map",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_peer_comparison(state_master: pd.DataFrame, selected_state: str) -> go.Figure:
    row = state_master[state_master["state"] == selected_state].iloc[0]
    metrics = ["AMI", "UPI", "VSI", "TPS"]
    peer = state_master[metrics].median()
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=metrics,
            fill="toself",
            name=selected_state,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[peer[m] for m in metrics],
            theta=metrics,
            fill="toself",
            name="Median Peer",
        )
    )
    fig.update_layout(
        title=f"{selected_state}: Indicator Profile vs Median",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_national_trend_chart(national: pd.DataFrame) -> go.Figure:
    trend_df = national.melt(
        id_vars="year_month",
        value_vars=["E_total", "D_total", "B_total"],
        var_name="Series",
        value_name="Value",
    )
    fig = px.line(
        trend_df,
        x="year_month",
        y="Value",
        color="Series",
        template="plotly_white",
        title="National UIDAI Activity Trend",
        color_discrete_sequence=[PALETTE["navy"], PALETTE["teal"], PALETTE["gold"]],
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_category_mix_chart(state_master: pd.DataFrame) -> go.Figure:
    category_counts = (
        state_master["Policy_Category"]
        .value_counts()
        .rename_axis("Policy_Category")
        .reset_index(name="Count")
    )
    fig = px.bar(
        category_counts,
        x="Count",
        y="Policy_Category",
        orientation="h",
        color="Policy_Category",
        template="plotly_white",
        title="Policy Category Mix",
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    return fig


def build_rank_persistence_chart(rank_df: pd.DataFrame, compare_mode: str) -> go.Figure:
    y_metric = "D_rank" if compare_mode == "Demographic Rank" else "B_rank"
    fig = px.scatter(
        rank_df,
        x="E_rank",
        y=y_metric,
        hover_name="state",
        template="plotly_white",
        title=f"Enrollment Rank vs {compare_mode}",
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_volatility_bar_chart(state_master: pd.DataFrame) -> go.Figure:
    top_vol = state_master.nlargest(12, "VSI")[["state", "VSI", "UPI", "TPS"]]
    fig = px.bar(
        top_vol.sort_values("VSI"),
        x="VSI",
        y="state",
        orientation="h",
        color="UPI",
        template="plotly_white",
        title="Highest Volatility States",
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_mismatch_queue_chart(pressure_mismatch: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        pressure_mismatch.sort_values("Priority_Score"),
        x="Priority_Score",
        y="state",
        color="UPI",
        orientation="h",
        template="plotly_white",
        title="Pressure-Maturity Mismatch Queue",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_state_ratio_trend_chart(
    state_series: pd.DataFrame, selected_state: str
) -> go.Figure:
    ratio_series = state_series[
        [
            "year_month",
            "demo_update_ratio",
            "biometric_update_ratio",
            "update_to_enrol_ratio",
        ]
    ]
    fig = px.line(
        ratio_series.melt(id_vars="year_month", var_name="Metric", value_name="Value"),
        x="year_month",
        y="Value",
        color="Metric",
        template="plotly_white",
        title=f"{selected_state}: Update Intensity Through Time",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig

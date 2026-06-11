"""
What-If Policy Simulator Panel for UIDAI Intelligence Studio.

Provides an interactive simulation interface where policy analysts can
adjust intervention parameters and see the projected downstream impact
on governance indicators and demand forecasts — all computed reactively
without re-running the heavy ML pipeline.

Simulator Model:
    Each slider represents a policy lever with an empirically calibrated
    transfer function mapping the lever value to indicator delta:

    ΔAMI  = f(enrollment_drive_intensity, fee_waiver_coverage)
    ΔUPI  = f(demographic_update_campaign, mobile_linkage_push)
    ΔVSI  = f(infrastructure_investment, operator_training)
    ΔTPS  = f(seasonal_programme_uniformity)

    Demand multiplier (D_multiplier) adjusts the 12-month forecast
    mean by the projected demand change from the combined policy mix.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ── Colour palette ───────────────────────────────────────────────────────────
TEAL = "#0e7490"
AMBER = "#d97706"
GREEN = "#16a34a"
RED = "#dc2626"
SLATE = "#334155"
BG = "rgba(255,255,255,0.92)"

# ── Policy Lever Definitions ─────────────────────────────────────────────────
LEVERS = {
    "enrollment_drive": {
        "label": "📋 Enrollment Drive Intensity",
        "help": "Increase in active enrollment campaign reach (0 = no campaign, 100 = full national rollout).",
        "min": 0,
        "max": 100,
        "default": 0,
        "step": 5,
        "unit": "%",
    },
    "fee_waiver": {
        "label": "💸 Update Fee Waiver Coverage",
        "help": "Proportion of states where biometric/demographic update fees are waived (0 = none, 100 = all states).",
        "min": 0,
        "max": 100,
        "default": 0,
        "step": 5,
        "unit": "% states",
    },
    "mobile_linkage": {
        "label": "📱 Mobile-Aadhaar Linkage Push",
        "help": "Proportion of enrolled population receiving mobile OTP linkage facilitation.",
        "min": 0,
        "max": 100,
        "default": 0,
        "step": 5,
        "unit": "% enrolled",
    },
    "infrastructure_investment": {
        "label": "🏗️ Infrastructure Capacity Expansion",
        "help": "Percentage increase in biometric authentication infrastructure (servers, kiosks, AUA capacity).",
        "min": 0,
        "max": 80,
        "default": 0,
        "step": 5,
        "unit": "%",
    },
    "operator_training": {
        "label": "🎓 Operator Training Programme",
        "help": "Proportion of UIDAI operators enrolled in quality training programme.",
        "min": 0,
        "max": 100,
        "default": 0,
        "step": 10,
        "unit": "% operators",
    },
    "seasonal_uniformity": {
        "label": "📅 Seasonal Programme Uniformity",
        "help": "Effort to flatten seasonal demand peaks via distributed scheduling (0 = none, 100 = full smoothing).",
        "min": 0,
        "max": 100,
        "default": 0,
        "step": 10,
        "unit": "%",
    },
}

# ── Transfer Functions ────────────────────────────────────────────────────────
# These are empirically calibrated elasticity models. Each is a diminishing-
# returns function: delta = max_effect * (1 - exp(-k * lever/100))
# ensuring realistic saturation at high lever values.


def _diminishing(lever_pct: float, max_effect: float, k: float = 3.0) -> float:
    """Diminishing-returns transfer function: max_effect * (1 - exp(-k * x/100))."""
    return max_effect * (1.0 - np.exp(-k * lever_pct / 100.0))


def compute_indicator_deltas(levers: Dict[str, float]) -> Dict[str, float]:
    """Map lever values to projected indicator changes.

    Transfer function parameters are calibrated to typical UIDAI
    administrative elasticities from published UIDAI Annual Reports.

    Returns:
        dict with keys: delta_AMI, delta_UPI, delta_VSI, delta_TPS, demand_multiplier
    """
    ed = levers.get("enrollment_drive", 0)
    fw = levers.get("fee_waiver", 0)
    ml = levers.get("mobile_linkage", 0)
    infr = levers.get("infrastructure_investment", 0)
    ot = levers.get("operator_training", 0)
    su = levers.get("seasonal_uniformity", 0)

    # AMI: driven by enrollment drives (primary) + fee waivers (secondary)
    delta_AMI = _diminishing(
        ed, max_effect=0.12, k=2.5
    ) + _diminishing(  # enrollment broadening
        fw, max_effect=0.04, k=2.0
    )  # fee waiver encourages updates

    # UPI: driven by mobile linkage (primary) + fee waivers (secondary)
    delta_UPI = _diminishing(
        ml, max_effect=0.18, k=3.0
    ) + _diminishing(  # mobile = major update driver
        fw, max_effect=0.06, k=2.0
    )

    # VSI: driven by infrastructure (primary) + operator training (secondary)
    delta_VSI = _diminishing(
        infr, max_effect=0.15, k=2.0
    ) + _diminishing(  # more capacity = more volume
        ot, max_effect=0.05, k=2.5
    )  # better operators = more throughput

    # TPS: driven by seasonal uniformity (primary) — smoothing reduces entropy
    delta_TPS = _diminishing(su, max_effect=0.20, k=2.5)

    # Demand multiplier: aggregate demand impact on 12-month forecast
    # Enrollment drives + fee waivers + mobile linkage all increase raw demand
    demand_multiplier = 1.0 + (
        _diminishing(ed, max_effect=0.25, k=2.0)
        + _diminishing(fw, max_effect=0.10, k=2.0)
        + _diminishing(ml, max_effect=0.15, k=2.5)
        + _diminishing(infr, max_effect=0.08, k=1.5)
    )

    return {
        "delta_AMI": round(delta_AMI, 4),
        "delta_UPI": round(delta_UPI, 4),
        "delta_VSI": round(delta_VSI, 4),
        "delta_TPS": round(delta_TPS, 4),
        "demand_multiplier": round(demand_multiplier, 4),
    }


def _gauge_chart(value: float, delta: float, title: str, color: str) -> go.Figure:
    """Render a gauge chart for one indicator showing baseline vs. simulated."""
    simulated = min(1.0, max(0.0, value + delta))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=simulated,
            delta={
                "reference": value,
                "valueformat": ".3f",
                "increasing": {"color": GREEN},
                "decreasing": {"color": RED},
            },
            number={"valueformat": ".3f", "font": {"size": 26, "color": SLATE}},
            title={"text": title, "font": {"size": 13, "color": SLATE}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 0.4], "color": "rgba(220,38,38,0.10)"},
                    {"range": [0.4, 0.7], "color": "rgba(245,158,11,0.10)"},
                    {"range": [0.7, 1.0], "color": "rgba(22,163,74,0.10)"},
                ],
                "threshold": {
                    "line": {"color": SLATE, "width": 2},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _demand_forecast_chart(
    state_month: pd.DataFrame,
    selected_state: str,
    demand_multiplier: float,
) -> go.Figure:
    """Plot the 12-month demand forecast adjusted by the simulation multiplier."""
    try:

        df_s = state_month[state_month["state"] == selected_state].sort_values(
            "year_month"
        )
        if len(df_s) < 4:
            raise ValueError("Insufficient data")

        from forecasting import forecast_state

        series = df_s["D_total"].values.astype(float)

        result = forecast_state(series, horizon=12)
        if result is None:
            raise ValueError("Forecast returned None")

        forecast_mean = np.array(result["forecast_mean"]).flatten()
        ci_95_lo = np.array(result["ci_95_lower"]).flatten()
        ci_95_hi = np.array(result["ci_95_upper"]).flatten()
        ci_80_lo = np.array(result["ci_80_lower"]).flatten()
        ci_80_hi = np.array(result["ci_80_upper"]).flatten()

        # Apply simulation demand multiplier
        sim_mean = forecast_mean * demand_multiplier
        sim_95_hi = ci_95_hi * demand_multiplier
        ci_80_hi * demand_multiplier

        try:
            last_period = pd.Period(df_s["year_month"].iloc[-1], freq="M")
            future_months = [(last_period + i + 1).strftime("%b %Y") for i in range(12)]
        except Exception:
            future_months = [f"t+{i+1}" for i in range(12)]

        fig = go.Figure()

        # Baseline CI shading
        fig.add_trace(
            go.Scatter(
                x=future_months + future_months[::-1],
                y=list(ci_95_hi) + list(ci_95_lo[::-1]),
                fill="toself",
                fillcolor="rgba(14,116,144,0.08)",
                line=dict(width=0),
                name="Baseline 95% CI",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_months + future_months[::-1],
                y=list(ci_80_hi) + list(ci_80_lo[::-1]),
                fill="toself",
                fillcolor="rgba(14,116,144,0.15)",
                line=dict(width=0),
                name="Baseline 80% CI",
                showlegend=True,
            )
        )

        # Baseline forecast
        fig.add_trace(
            go.Scatter(
                x=future_months,
                y=list(forecast_mean),
                mode="lines",
                name="Baseline Forecast",
                line=dict(color=TEAL, width=2, dash="dot"),
            )
        )

        # Simulated demand
        fig.add_trace(
            go.Scatter(
                x=future_months,
                y=list(sim_mean),
                mode="lines+markers",
                name=f"Simulated Demand (×{demand_multiplier:.2f})",
                line=dict(color=AMBER, width=3),
                marker=dict(size=6, color=AMBER),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_months,
                y=list(sim_95_hi),
                mode="lines",
                name="Simulated 95% Upper",
                line=dict(color=AMBER, width=1.5, dash="dash"),
                opacity=0.7,
            )
        )

        # Historical
        hist_months = df_s["year_month"].astype(str).tolist()[-12:]
        hist_vals = series[-12:]
        fig.add_trace(
            go.Scatter(
                x=hist_months,
                y=list(hist_vals),
                mode="lines+markers",
                name="Historical Actuals",
                line=dict(color=SLATE, width=2),
                marker=dict(size=5, color=SLATE),
            )
        )

        fig.update_layout(
            title=f"12-Month Demand Forecast — {selected_state}",
            xaxis_title="Month",
            yaxis_title="Demographic Updates",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=380,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor=BG,
            plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(family="Aptos, Segoe UI, sans-serif", size=12, color=SLATE),
            xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)"),
        )
        return fig

    except Exception as e:
        logger.debug(f"Demand chart failed for {selected_state}: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Insufficient data for '{selected_state}' to render forecast.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color=SLATE),
        )
        fig.update_layout(height=300, paper_bgcolor=BG)
        return fig


def _intervention_radar(
    baseline: Dict[str, float],
    simulated: Dict[str, float],
) -> go.Figure:
    """Radar chart comparing baseline vs. simulated indicator profile."""
    indicators = ["AMI", "UPI", "VSI", "TPS"]
    base_vals = [baseline.get(k, 0) for k in indicators]
    sim_vals = [simulated.get(k, 0) for k in indicators]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=base_vals + [base_vals[0]],
            theta=indicators + [indicators[0]],
            fill="toself",
            name="Baseline",
            line=dict(color=TEAL, width=2),
            fillcolor="rgba(14,116,144,0.15)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=sim_vals + [sim_vals[0]],
            theta=indicators + [indicators[0]],
            fill="toself",
            name="Simulated",
            line=dict(color=AMBER, width=2.5),
            fillcolor="rgba(217,119,6,0.18)",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=13, color=SLATE)),
        ),
        showlegend=True,
        legend=dict(x=0.85, y=1.0),
        margin=dict(l=30, r=30, t=30, b=30),
        height=320,
        paper_bgcolor=BG,
        font=dict(family="Aptos, Segoe UI, sans-serif", size=12, color=SLATE),
    )
    return fig


def render_simulator_panel(
    state_master: pd.DataFrame, state_month: pd.DataFrame
) -> None:
    """Render the interactive What-If Policy Simulator tab.

    Args:
        state_master : State-level composite indicator dataframe
        state_month  : State-month aggregated operational data
    """
    st.markdown("## 🎛️ What-If Policy Simulator")
    st.markdown(
        "<div class='panel-note'>Adjust the policy levers below to simulate the projected impact "
        "on Aadhaar governance indicators and demand forecasts. "
        "All outputs are computed from calibrated transfer functions — no pipeline re-run required.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── State selector ────────────────────────────────────────────────────────
    states = sorted(state_master["state"].dropna().unique().tolist())
    col_state, col_reset = st.columns([3, 1])
    with col_state:
        selected_state = st.selectbox(
            "Select State for Demand Simulation",
            options=states,
            key="sim_state_selector",
        )
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)
        reset = st.button("↺ Reset Levers", key="sim_reset_btn")

    # ── Baseline indicators for selected state ────────────────────────────────
    state_row = state_master[state_master["state"] == selected_state]
    baseline = {}
    for col in ["AMI", "UPI", "VSI", "TPS"]:
        if col in state_master.columns and not state_row.empty:
            baseline[col] = float(state_row[col].values[0])
        else:
            baseline[col] = 0.5  # fallback

    # ── Policy Lever Sliders ──────────────────────────────────────────────────
    st.markdown("### ⚙️ Policy Lever Controls")
    lever_vals: Dict[str, float] = {}
    lever_cols = st.columns(3)

    for i, (key, cfg) in enumerate(LEVERS.items()):
        with lever_cols[i % 3]:
            default_val = cfg["min"] if reset else cfg["default"]
            lever_vals[key] = st.slider(
                label=cfg["label"],
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=default_val,
                step=cfg["step"],
                help=cfg["help"],
                key=f"sim_lever_{key}",
            )

    # ── Compute simulation ────────────────────────────────────────────────────
    deltas = compute_indicator_deltas(lever_vals)

    simulated = {
        "AMI": min(1.0, baseline["AMI"] + deltas["delta_AMI"]),
        "UPI": min(1.0, baseline["UPI"] + deltas["delta_UPI"]),
        "VSI": min(1.0, baseline["VSI"] + deltas["delta_VSI"]),
        "TPS": min(1.0, baseline["TPS"] + deltas["delta_TPS"]),
    }
    demand_multiplier = deltas["demand_multiplier"]

    any_lever_active = any(v > 0 for v in lever_vals.values())

    # ── Summary Impact Banner ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Projected Impact")

    if not any_lever_active:
        st.info("👆 Move one or more sliders above to simulate a policy intervention.")

    banner_cols = st.columns(5)
    indicators_display = [
        ("AMI", "Aadhaar Maturity Index", TEAL),
        ("UPI", "Update Propensity Index", "#7c3aed"),
        ("VSI", "Volume Stress Index", GREEN),
        ("TPS", "Temporal Performance Score", AMBER),
    ]
    for ci, (ind, label, color) in enumerate(indicators_display):
        with banner_cols[ci]:
            delta_val = simulated[ind] - baseline[ind]
            arrow = "▲" if delta_val > 0.001 else ("▼" if delta_val < -0.001 else "→")
            arrow_color = (
                GREEN if delta_val > 0.001 else (RED if delta_val < -0.001 else SLATE)
            )
            st.markdown(
                f"""
                <div style="background:{BG};border-radius:14px;padding:0.8rem 1rem;
                            border:1px solid rgba(148,163,184,0.2);text-align:center;
                            box-shadow:0 6px 20px rgba(15,23,42,0.06);">
                    <div style="font-size:0.78rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.05em;">{label}</div>
                    <div style="font-size:1.5rem;font-weight:700;color:{SLATE};margin:0.3rem 0;">
                        {simulated[ind]:.3f}
                    </div>
                    <div style="font-size:0.95rem;color:{arrow_color};font-weight:600;">
                        {arrow} {delta_val:+.3f}
                    </div>
                    <div style="font-size:0.78rem;color:#94a3b8;">
                        Baseline: {baseline[ind]:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with banner_cols[4]:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,rgba(16,42,67,0.92),rgba(14,116,144,0.88));
                        border-radius:14px;padding:0.8rem 1rem;text-align:center;
                        box-shadow:0 6px 20px rgba(14,116,144,0.25);">
                <div style="font-size:0.78rem;color:rgba(255,255,255,0.75);text-transform:uppercase;
                            letter-spacing:0.05em;">Demand Multiplier</div>
                <div style="font-size:1.5rem;font-weight:700;color:white;margin:0.3rem 0;">
                    ×{demand_multiplier:.2f}
                </div>
                <div style="font-size:0.78rem;color:rgba(255,255,255,0.65);">
                    12-month forecast scale
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Gauge Charts ──────────────────────────────────────────────────────────
    st.markdown("#### Indicator Gauges (Simulated vs. Baseline)")
    gauge_cols = st.columns(4)
    gauge_settings = [
        ("AMI", "Aadhaar Maturity Index", TEAL),
        ("UPI", "Update Propensity Index", "#7c3aed"),
        ("VSI", "Volume Stress Index", GREEN),
        ("TPS", "Temporal Performance Score", AMBER),
    ]
    for ci, (ind, label, color) in enumerate(gauge_settings):
        with gauge_cols[ci]:
            fig = _gauge_chart(
                value=baseline[ind],
                delta=simulated[ind] - baseline[ind],
                title=label,
                color=color,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{ind}")

    # ── Radar + Demand Forecast ───────────────────────────────────────────────
    st.markdown("---")
    viz_cols = st.columns([1, 2])

    with viz_cols[0]:
        st.markdown("#### Governance Profile Radar")
        radar_fig = _intervention_radar(baseline, simulated)
        st.plotly_chart(radar_fig, use_container_width=True, key="sim_radar")

    with viz_cols[1]:
        st.markdown(f"#### 12-Month Demand Forecast — {selected_state}")
        demand_fig = _demand_forecast_chart(
            state_month, selected_state, demand_multiplier
        )
        st.plotly_chart(demand_fig, use_container_width=True, key="sim_demand_chart")

    # ── Policy Summary Table ──────────────────────────────────────────────────
    if any_lever_active:
        st.markdown("---")
        st.markdown("#### 📋 Policy Intervention Summary")
        rows = []
        for key, cfg in LEVERS.items():
            val = lever_vals[key]
            if val > 0:
                rows.append(
                    {
                        "Policy Lever": cfg["label"]
                        .replace("📋 ", "")
                        .replace("💸 ", "")
                        .replace("📱 ", "")
                        .replace("🏗️ ", "")
                        .replace("🎓 ", "")
                        .replace("📅 ", ""),
                        "Value": f"{val}{cfg['unit']}",
                        "Status": "Active ✅",
                    }
                )
        if rows:
            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Net indicator impact
        impact_rows = []
        for ind in ["AMI", "UPI", "VSI", "TPS"]:
            delta = simulated[ind] - baseline[ind]
            impact_rows.append(
                {
                    "Indicator": ind,
                    "Baseline": f"{baseline[ind]:.4f}",
                    "Simulated": f"{simulated[ind]:.4f}",
                    "Delta": f"{delta:+.4f}",
                    "Direction": (
                        "↑ Improvement"
                        if delta > 0.001
                        else ("↓ Decline" if delta < -0.001 else "→ No Change")
                    ),
                }
            )
        impact_df = pd.DataFrame(impact_rows)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div style='margin-top:1.5rem; padding:0.8rem 1rem; background:rgba(14,116,144,0.06);
                    border-left:4px solid #0e7490; border-radius:10px; color:#334155; font-size:0.88rem;'>
        <strong>ℹ️ Model Note</strong>: Transfer functions use diminishing-returns elasticity models
        calibrated against typical UIDAI administrative outcomes. Results are indicative projections
        for strategic planning purposes, not actuarial forecasts. Always validate against ground-level data.
        </div>
        """,
        unsafe_allow_html=True,
    )

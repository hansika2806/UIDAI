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

Baseline logic:
    The chart opens showing the actual SARIMA forecast (no-intervention
    trajectory) read from forecast_results.csv as the red "do-nothing" line.
    Levers adjust a multiplier on top of that baseline.
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

ANALYSIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "analysis",
)

# ── Colour palette ────────────────────────────────────────────────────────────
TEAL = "#0e7490"
AMBER = "#d97706"
GREEN = "#16a34a"
RED = "#dc2626"
SLATE = "#334155"
BG = "rgba(255,255,255,0.92)"

# ── Policy Lever Definitions ──────────────────────────────────────────────────
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

# ── Policy Action → Preset Lever Values ───────────────────────────────────────
PRESET_LEVERS: Dict[str, Dict[str, float]] = {
    "Balance investment across enrollment and update services.": {
        "enrollment_drive": 30.0,
        "fee_waiver": 20.0,
        "mobile_linkage": 40.0,
        "infrastructure_investment": 20.0,
        "operator_training": 50.0,
        "seasonal_uniformity": 40.0,
    },
    "Optimize operations and invest in fraud detection plus service quality.": {
        "enrollment_drive": 10.0,
        "fee_waiver": 10.0,
        "mobile_linkage": 30.0,
        "infrastructure_investment": 40.0,
        "operator_training": 80.0,
        "seasonal_uniformity": 60.0,
    },
    "Increase monitoring and investigate administrative or reporting irregularities.": {
        "enrollment_drive": 0.0,
        "fee_waiver": 0.0,
        "mobile_linkage": 20.0,
        "infrastructure_investment": 10.0,
        "operator_training": 90.0,
        "seasonal_uniformity": 30.0,
    },
    "Increase enrollment infrastructure and run targeted outreach programs.": {
        "enrollment_drive": 70.0,
        "fee_waiver": 40.0,
        "mobile_linkage": 30.0,
        "infrastructure_investment": 50.0,
        "operator_training": 40.0,
        "seasonal_uniformity": 20.0,
    },
    "Expand update capacity, decentralize service points, and introduce load-balancing.": {
        "enrollment_drive": 10.0,
        "fee_waiver": 30.0,
        "mobile_linkage": 50.0,
        "infrastructure_investment": 70.0,
        "operator_training": 60.0,
        "seasonal_uniformity": 70.0,
    },
}


# ── Transfer Functions ────────────────────────────────────────────────────────
def _diminishing(lever_pct: float, max_effect: float, k: float = 3.0) -> float:
    """Diminishing-returns transfer function: max_effect * (1 - exp(-k * x/100))."""
    return max_effect * (1.0 - np.exp(-k * lever_pct / 100.0))


def compute_indicator_deltas(levers: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    """Map lever values to projected indicator changes, dynamically scaled by baseline state sensitivities."""
    ed = levers.get("enrollment_drive", 0)
    fw = levers.get("fee_waiver", 0)
    ml = levers.get("mobile_linkage", 0)
    infr = levers.get("infrastructure_investment", 0)
    ot = levers.get("operator_training", 0)
    su = levers.get("seasonal_uniformity", 0)

    # State-aware sensitivity scaling factors:
    # 1. Enrollment drives have high impact in low-maturity states, but diminishing returns in highly mature states.
    ami_base = baseline.get("AMI", 0.5)
    enrollment_sensitivity = max(0.1, 1.2 - ami_base)  # Higher impact if AMI is low
    
    # 2. Fee waivers and mobile linkages have larger impacts in states with low/moderate update propensity (UPI).
    upi_base = baseline.get("UPI", 0.5)
    update_sensitivity = max(0.2, 1.3 - upi_base)
    
    # 3. Infrastructure expansion has more impact on stabilizing volume stress (VSI) in high-stress states.
    vsi_base = baseline.get("VSI", 0.5)
    infra_sensitivity = max(0.3, 0.5 + vsi_base)
    
    # 4. Seasonal uniformity effort (TPS) has more stabilization potential in erratic (low TPS) states.
    tps_base = baseline.get("TPS", 0.5)
    temporal_sensitivity = max(0.2, 1.2 - tps_base)

    delta_AMI = (
        _diminishing(ed, max_effect=0.12, k=2.5) * enrollment_sensitivity
        + _diminishing(fw, max_effect=0.04, k=2.0) * update_sensitivity
    )
    delta_UPI = (
        _diminishing(ml, max_effect=0.18, k=3.0) * update_sensitivity
        + _diminishing(fw, max_effect=0.06, k=2.0) * update_sensitivity
    )
    delta_VSI = (
        _diminishing(infr, max_effect=0.15, k=2.0) * infra_sensitivity
        + _diminishing(ot, max_effect=0.05, k=2.5) * update_sensitivity
    )
    delta_TPS = _diminishing(su, max_effect=0.20, k=2.5) * temporal_sensitivity
    
    demand_multiplier = 1.0 + (
        _diminishing(ed, max_effect=0.25, k=2.0) * enrollment_sensitivity
        + _diminishing(fw, max_effect=0.10, k=2.0) * update_sensitivity
        + _diminishing(ml, max_effect=0.15, k=2.5) * update_sensitivity
        + _diminishing(infr, max_effect=0.08, k=1.5) * infra_sensitivity
    )

    return {
        "delta_AMI": round(delta_AMI, 4),
        "delta_UPI": round(delta_UPI, 4),
        "delta_VSI": round(-delta_VSI, 4),  # Note: Stabilizing infrastructure / training should REDUCE volume stress (VSI)
        "delta_TPS": round(delta_TPS, 4),   # Smoothing seasonality INCREASES temporal predictability
        "demand_multiplier": round(demand_multiplier, 4),
    }


def _load_forecast_results() -> Optional[pd.DataFrame]:
    path = os.path.join(ANALYSIS_DIR, "forecast_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_model_summary() -> Optional[pd.DataFrame]:
    path = os.path.join(ANALYSIS_DIR, "forecast_model_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Gauge Chart ───────────────────────────────────────────────────────────────
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


# ── Demand Forecast Chart (with SARIMA baseline from CSV) ─────────────────────
def _demand_forecast_chart(
    state_month: pd.DataFrame,
    selected_state: str,
    demand_multiplier: float,
    forecast_df: Optional[pd.DataFrame],
    model_summary: Optional[pd.DataFrame],
) -> go.Figure:
    """Plot the baseline SARIMA forecast in red + simulated trajectory overlay."""
    fig = go.Figure()
    error_msg = None

    # --- Historical actuals ---
    df_s = state_month[state_month["state"] == selected_state].sort_values("year_month")
    if not df_s.empty:
        hist_months = df_s["year_month"].astype(str).tolist()[-18:]
        hist_vals = df_s["D_total"].values.astype(float)[-18:]
        fig.add_trace(
            go.Scatter(
                x=hist_months,
                y=list(hist_vals),
                mode="lines+markers",
                name="Historical Actuals",
                line=dict(color=SLATE, width=2),
                marker=dict(size=4, color=SLATE),
            )
        )

    # --- Precomputed SARIMA baseline (the "do-nothing" trajectory) ---
    if forecast_df is not None:
        state_fc = forecast_df[forecast_df["state"] == selected_state].sort_values("forecast_step")
        if not state_fc.empty:
            months = state_fc["forecast_month"].astype(str).tolist()
            fc_mean = state_fc["forecast_mean"].values
            ci95_lo = state_fc["ci_95_lower"].values
            ci95_hi = state_fc["ci_95_upper"].values

            # 95% CI shading for baseline
            fig.add_trace(
                go.Scatter(
                    x=months + months[::-1],
                    y=list(ci95_hi) + list(ci95_lo[::-1]),
                    fill="toself",
                    fillcolor="rgba(220,38,38,0.07)",
                    line=dict(width=0),
                    name="No-Intervention 95% CI",
                    showlegend=True,
                )
            )

            # Bold red baseline = do-nothing trajectory
            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=list(fc_mean),
                    mode="lines",
                    name="Current Trajectory (No Intervention)",
                    line=dict(color=RED, width=3),
                )
            )

            # Capacity threshold line + dynamic operational variance envelope
            if model_summary is not None and not model_summary.empty:
                state_ms = model_summary[model_summary["state"] == selected_state]
                if not state_ms.empty and "capacity_threshold" in state_ms.columns:
                    thresh = state_ms["capacity_threshold"].iloc[0]
                    if pd.notna(thresh):
                        # Shaded region representing operational capacity variance (±10% due to server/network latency)
                        fig.add_trace(
                            go.Scatter(
                                x=[months[0], months[-1]],
                                y=[thresh * 1.1, thresh * 1.1],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=[months[0], months[-1]],
                                y=[thresh * 0.9, thresh * 0.9],
                                mode="lines",
                                fill="tonexty",
                                fillcolor="rgba(220,38,38,0.05)",
                                line=dict(width=0),
                                name="Operational Variance (±10%)",
                                showlegend=True,
                            )
                        )
                        fig.add_hline(
                            y=thresh,
                            line_dash="dot",
                            line_color="#991b1b",
                            line_width=2,
                            annotation_text=f"⚠ Capacity Limit ({thresh:,.0f})",
                            annotation_position="bottom right",
                            annotation_font_color="#991b1b",
                        )

            # Simulated overlay
            if demand_multiplier != 1.0:
                sim_mean = fc_mean * demand_multiplier
                sim_hi = ci95_hi * demand_multiplier
                sim_lo = ci95_lo * demand_multiplier

                fig.add_trace(
                    go.Scatter(
                        x=months + months[::-1],
                        y=list(sim_hi) + list(sim_lo[::-1]),
                        fill="toself",
                        fillcolor="rgba(22,163,74,0.08)",
                        line=dict(width=0),
                        name="Intervention 95% CI",
                        showlegend=True,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=months,
                        y=list(sim_mean),
                        mode="lines+markers",
                        name=f"Simulated Demand (×{demand_multiplier:.2f})",
                        line=dict(color=GREEN, width=3),
                        marker=dict(size=5, color=GREEN),
                    )
                )
        else:
            error_msg = f"No precomputed forecast available for {selected_state}."
    else:
        error_msg = "Forecast results file not found. Run the analytics pipeline first."

    if error_msg:
        fig.add_annotation(
            text=error_msg,
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color=SLATE),
        )

    fig.update_layout(
        title=dict(text=f"12-Month Demand Forecast — {selected_state}", font=dict(size=14)),
        xaxis_title="Month",
        yaxis_title="Demographic Updates",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor=BG,
        plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=SLATE),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
    )
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
            line=dict(color=RED, width=2),
            fillcolor="rgba(220,38,38,0.10)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=sim_vals + [sim_vals[0]],
            theta=indicators + [indicators[0]],
            fill="toself",
            name="Simulated",
            line=dict(color=GREEN, width=2.5),
            fillcolor="rgba(22,163,74,0.15)",
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
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=SLATE),
    )
    return fig


# ── Index interpretation helper ───────────────────────────────────────────────
def _index_interpretation(label: str, value: float) -> str:
    """Return a plain-English sentence for the given index value."""
    if label == "AMI":
        if value >= 0.7:
            return "Aadhaar system is highly mature — most eligible residents are enrolled."
        elif value < 0.4:
            return "Still in active enrollment expansion — a large portion of residents are not yet registered."
        else:
            return "Aadhaar maturity is moderate — enrollment is ongoing but not yet complete."
    elif label == "UPI":
        if value >= 0.7:
            return "Update pressure is high — the system is handling many record changes relative to its enrollment base."
        elif value < 0.4:
            return "Low update pressure — record amendment activity is minimal relative to the enrolled population."
        else:
            return "Update pressure is moderate — record changes are steady but not overloading the system."
    elif label == "VSI":
        if value >= 0.7:
            return "High volume stress — monthly activity fluctuates significantly, signalling potential operational instability."
        elif value < 0.4:
            return "Volume is stable — monthly activity is predictably consistent."
        else:
            return "Moderate volume stress — some fluctuation in monthly activity observed."
    elif label == "TPS":
        if value < 0.3:
            return "Highly unpredictable — activity patterns are erratic and difficult to forecast reliably."
        elif value >= 0.7:
            return "Strong temporal predictability — this state's activity follows a consistent seasonal rhythm."
        else:
            return "Moderate temporal predictability — some seasonal structure present but not strongly consistent."
    return ""


# ── Main render function ──────────────────────────────────────────────────────
def render_simulator_panel(
    state_master: pd.DataFrame, state_month: pd.DataFrame
) -> None:
    """Render the interactive What-If Policy Simulator tab."""

    # Load forecast data once
    forecast_df = _load_forecast_results()
    model_summary = _load_model_summary()

    st.markdown(
        '<div class="headline">🎛️ What-If Policy Simulator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="interpret-box" style="margin-top:0;">'
        "The chart below shows the <strong>current do-nothing trajectory</strong> — "
        "what will happen if no new policies are enacted. Adjust the levers to overlay "
        "a simulated intervention and see the projected improvement."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── State selector ────────────────────────────────────────────────────────
    states = sorted(state_master["state"].dropna().unique().tolist())
    col_state, col_reset, col_apply = st.columns([2.5, 0.8, 1.7])
    with col_state:
        selected_state = st.selectbox(
            "Select State for Demand Simulation",
            options=states,
            key="sim_state_selector",
        )
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)
        reset = st.button("↺ Reset", key="sim_reset_btn")

    # ── Policy_Action context ─────────────────────────────────────────────────
    state_row = state_master[state_master["state"] == selected_state]
    baseline = {}
    policy_action = "Balance investment across enrollment and update services."
    for col_name in ["AMI", "UPI", "VSI", "TPS"]:
        if col_name in state_master.columns and not state_row.empty:
            baseline[col_name] = float(state_row[col_name].values[0])
        else:
            baseline[col_name] = 0.5

    if not state_row.empty and "Policy_Action" in state_row.columns:
        policy_action = str(state_row["Policy_Action"].values[0])

    st.markdown(
        f'<div class="interpret-box" style="margin-bottom:1rem;">'
        f'<strong>Recommended Action for {selected_state}:</strong><br>'
        f'<em style="color:#0e7490;">{policy_action}</em>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with col_apply:
        st.markdown("<br>", unsafe_allow_html=True)
        apply_preset = st.button("⚡ Apply Suggested Intervention", key="sim_apply_preset_btn")

    if apply_preset:
        preset = PRESET_LEVERS.get(policy_action, {})
        for key_name, val in preset.items():
            st.session_state[f"sim_lever_{key_name}"] = int(val)
        st.rerun()

    # ── Policy Lever Sliders ──────────────────────────────────────────────────
    st.markdown("### ⚙️ Policy Lever Controls")
    lever_vals: Dict[str, float] = {}
    lever_cols = st.columns(3)

    for i, (key_name, cfg) in enumerate(LEVERS.items()):
        with lever_cols[i % 3]:
            default_val = cfg["min"] if reset else int(
                st.session_state.get(f"sim_lever_{key_name}", cfg["default"])
            )
            lever_vals[key_name] = st.slider(
                label=cfg["label"],
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=default_val,
                step=cfg["step"],
                help=cfg["help"],
                key=f"sim_lever_{key_name}",
            )

    # ── Compute simulation (with state-aware baseline sensitivities) ──────────
    deltas = compute_indicator_deltas(lever_vals, baseline)
    simulated = {
        "AMI": max(0.0, min(1.0, baseline["AMI"] + deltas["delta_AMI"])),
        "UPI": max(0.0, min(1.0, baseline["UPI"] + deltas["delta_UPI"])),
        "VSI": max(0.0, min(1.0, baseline["VSI"] + deltas["delta_VSI"])),
        "TPS": max(0.0, min(1.0, baseline["TPS"] + deltas["delta_TPS"])),
    }
    demand_multiplier = deltas["demand_multiplier"]
    any_lever_active = any(v > 0 for v in lever_vals.values())

    # ── Summary Impact Banner ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Projected Impact on Governance Indices")

    if not any_lever_active:
        st.info("👆 Move the sliders above — or click **Apply Suggested Intervention** — to simulate a policy change.")

    banner_cols = st.columns(4)
    indicators_display = [
        ("AMI", "Aadhaar Maturity Index", TEAL),
        ("UPI", "Update Propensity Index", "#7c3aed"),
        ("VSI", "Volume Stress Index", GREEN),
        ("TPS", "Temporal Performance Score", AMBER),
    ]
    for ci, (ind, label, color) in enumerate(indicators_display):
        with banner_cols[ci]:
            delta_val = simulated[ind] - baseline[ind]
            if ind == "VSI":
                # Volume Stress decrease (negative delta) is an improvement (green)
                arrow = "▼" if delta_val < -0.001 else ("▲" if delta_val > 0.001 else "→")
                arrow_color = GREEN if delta_val < -0.001 else (RED if delta_val > 0.001 else SLATE)
            else:
                arrow = "▲" if delta_val > 0.001 else ("▼" if delta_val < -0.001 else "→")
                arrow_color = GREEN if delta_val > 0.001 else (RED if delta_val < -0.001 else SLATE)
            interp = _index_interpretation(ind, simulated[ind])
            st.markdown(
                f"""
                <div style="background:{BG};border-radius:16px;padding:1rem;
                            border:1px solid rgba(148,163,184,0.2);
                            box-shadow:0 6px 20px rgba(15,23,42,0.06);">
                    <div style="font-size:0.78rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.05em;">{label}</div>
                    <div class="metric-value" style="color:{color};">{simulated[ind]:.3f}</div>
                    <div style="font-size:0.95rem;color:{arrow_color};">{arrow} {delta_val:+.3f}</div>
                    <div style="font-size:0.75rem;color:#94a3b8;">Baseline: {baseline[ind]:.3f}</div>
                    <div style="font-size:0.8rem;color:#475569;margin-top:0.4rem;font-style:italic;">{interp}</div>
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
            st.plotly_chart(fig, width="stretch", key=f"gauge_{ind}")

    # ── Radar + Demand Forecast ───────────────────────────────────────────────
    st.markdown("---")
    viz_cols = st.columns([1, 2])

    with viz_cols[0]:
        st.markdown("#### Governance Profile Radar")
        radar_fig = _intervention_radar(baseline, simulated)
        st.plotly_chart(radar_fig, width="stretch", key="sim_radar")

    with viz_cols[1]:
        st.markdown(f"#### 12-Month Demand Forecast — {selected_state}")
        demand_fig = _demand_forecast_chart(
            state_month, selected_state, demand_multiplier, forecast_df, model_summary
        )
        st.plotly_chart(demand_fig, width="stretch", key="sim_demand_chart")

    # ── Simulation Interpreter ────────────────────────────────────────────────
    if any_lever_active:
        net_delta = sum(simulated[k] - baseline[k] for k in ["AMI", "UPI", "VSI", "TPS"])
        direction = "improvement" if net_delta > 0 else "decline"
        demand_change_pct = (demand_multiplier - 1.0) * 100

        interpreter_html = (
            f"<strong>Simulation Summary for {selected_state}:</strong><br>"
            f"The selected interventions project an overall governance <em>{direction}</em> "
            f"of {abs(net_delta):.3f} points across all four indices. "
            f"Demand is expected to change by <strong>{demand_change_pct:+.1f}%</strong> "
            f"relative to the do-nothing baseline. "
        )
        if demand_multiplier > 1.1:
            interpreter_html += (
                "This increased demand requires pre-positioning infrastructure capacity "
                "before the intervention is deployed, or the system risks a capacity breach."
            )
        elif demand_multiplier <= 1.0:
            interpreter_html += (
                "Demand is not expected to increase significantly — the intervention focuses "
                "on quality and efficiency rather than volume expansion."
            )

        st.markdown(
            f'<div class="interpret-box">{interpreter_html}</div>',
            unsafe_allow_html=True,
        )

        # ── Policy Summary Table ──────────────────────────────────────────────
        st.markdown("---")
        with st.expander("Show full intervention impact table"):
            rows = []
            for key_name, cfg in LEVERS.items():
                val = lever_vals[key_name]
                if val > 0:
                    rows.append({
                        "Policy Lever": cfg["label"].replace("📋 ", "").replace("💸 ", "")
                            .replace("📱 ", "").replace("🏗️ ", "").replace("🎓 ", "").replace("📅 ", ""),
                        "Value": f"{val}{cfg['unit']}",
                        "Status": "Active ✅",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            impact_rows = []
            for ind in ["AMI", "UPI", "VSI", "TPS"]:
                delta = simulated[ind] - baseline[ind]
                impact_rows.append({
                    "Indicator": ind,
                    "Baseline": f"{baseline[ind]:.4f}",
                    "Simulated": f"{simulated[ind]:.4f}",
                    "Delta": f"{delta:+.4f}",
                    "Direction": "↑ Improvement" if delta > 0.001 else ("↓ Decline" if delta < -0.001 else "→ No Change"),
                })
            st.dataframe(pd.DataFrame(impact_rows), width="stretch", hide_index=True)

    st.markdown(
        """
        <div style='margin-top:1.5rem;padding:0.8rem 1rem;background:rgba(14,116,144,0.06);
                    border-left:4px solid #0e7490;border-radius:10px;color:#334155;font-size:0.88rem;'>
        <strong>ℹ️ Model Note</strong>: Transfer functions use diminishing-returns elasticity models
        calibrated against typical UIDAI administrative outcomes. Results are indicative projections
        for strategic planning — always validate against ground-level data before policy deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )

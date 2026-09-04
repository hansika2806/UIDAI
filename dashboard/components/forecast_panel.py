"""Forecast panel: SARIMA demand forecast charts with dynamic confidence-band opacity."""

import os
from typing import Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ANALYSIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "analysis",
)


def _load_forecast_data():
    path = os.path.join(ANALYSIS_DIR, "forecast_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_model_summary():
    path = os.path.join(ANALYSIS_DIR, "forecast_model_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _mape_to_opacity(mape: float, min_mape: float, max_mape: float) -> float:
    """
    Normalize MAPE to a confidence-band opacity.
    High accuracy (low MAPE) → high opacity (bold bands).
    Low accuracy (high MAPE) → low opacity (faint bands).
    Formula: opacity = max(0.15, 0.85 - (mape - min_mape) / (max_mape - min_mape) * 0.70)
    """
    if max_mape <= min_mape:
        return 0.50
    normalized = (mape - min_mape) / (max_mape - min_mape)
    return max(0.15, 0.85 - normalized * 0.70)


def build_forecast_chart(
    forecast_df: pd.DataFrame,
    state: str,
    state_month: pd.DataFrame,
    model_summary: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Build a forecast chart with historical data, dynamic confidence bands, and capacity threshold."""
    state_fc = forecast_df[forecast_df["state"] == state].sort_values("forecast_step")
    state_hist = state_month[state_month["state"] == state].sort_values("year_month")

    # Compute opacity for this state
    ci_opacity_95 = 0.12
    ci_opacity_80 = 0.22
    if (
        model_summary is not None
        and not model_summary.empty
        and "backtest_mape" in model_summary.columns
    ):
        valid_mapes = model_summary["backtest_mape"].dropna()
        if len(valid_mapes) >= 2:
            min_mape = float(valid_mapes.min())
            max_mape = float(valid_mapes.max())
            state_row = model_summary[model_summary["state"] == state]
            if not state_row.empty and pd.notna(state_row["backtest_mape"].iloc[0]):
                state_mape = float(state_row["backtest_mape"].iloc[0])
                base_opacity = _mape_to_opacity(state_mape, min_mape, max_mape)
                ci_opacity_95 = round(base_opacity * 0.50, 3)
                ci_opacity_80 = round(base_opacity * 0.90, 3)

    fig = go.Figure()

    # Historical data
    if not state_hist.empty:
        fig.add_trace(
            go.Scatter(
                x=state_hist["year_month"],
                y=state_hist["D_total"],
                mode="lines",
                name="Historical",
                line=dict(color="#102a43", width=2.5),
            )
        )

    if state_fc.empty:
        fig.update_layout(
            title=f"{state}: No forecast available", template="plotly_white"
        )
        return fig

    months = state_fc["forecast_month"].values

    # 95% CI band (opacity dynamic)
    fig.add_trace(
        go.Scatter(
            x=list(months) + list(months[::-1]),
            y=list(state_fc["ci_95_upper"]) + list(state_fc["ci_95_lower"][::-1]),
            fill="toself",
            fillcolor=f"rgba(14,116,144,{ci_opacity_95})",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Band",
            showlegend=True,
        )
    )

    # 80% CI band (opacity dynamic)
    fig.add_trace(
        go.Scatter(
            x=list(months) + list(months[::-1]),
            y=list(state_fc["ci_80_upper"]) + list(state_fc["ci_80_lower"][::-1]),
            fill="toself",
            fillcolor=f"rgba(14,116,144,{ci_opacity_80})",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence Band",
            showlegend=True,
        )
    )

    # Forecast mean
    fig.add_trace(
        go.Scatter(
            x=months,
            y=state_fc["forecast_mean"],
            mode="lines+markers",
            name="SARIMA Forecast",
            line=dict(color="#ea580c", width=3, dash="dash"),
            marker=dict(size=5),
        )
    )

    # Capacity threshold
    if model_summary is not None and not model_summary.empty:
        state_ms = model_summary[model_summary["state"] == state]
        if not state_ms.empty and "capacity_threshold" in state_ms.columns:
            thresh = state_ms["capacity_threshold"].iloc[0]
            if pd.notna(thresh):
                fig.add_hline(
                    y=thresh,
                    line_dash="dot",
                    line_color="#ef4444",
                    line_width=2,
                    annotation_text=f"⚠ Capacity Limit ({thresh:,.0f})",
                    annotation_position="bottom right",
                    annotation_font_color="#ef4444",
                )

    fig.update_layout(
        title=dict(
            text=f"{state}: 12-Month Demographic Update Demand Forecast",
            font=dict(size=14),
        ),
        xaxis_title="Month",
        yaxis_title="Demographic Updates",
        template="plotly_white",
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#334155"),
        paper_bgcolor="rgba(255,255,255,0.92)",
    )
    return fig


def _forecast_interpreter(
    state: str, model_summary: Optional[pd.DataFrame], months_to_breach
) -> str:
    """Generate a plain-English forecast summary for this state."""
    if model_summary is None:
        return "Model summary not available for interpretation."

    state_ms = model_summary[model_summary["state"] == state]
    if state_ms.empty:
        return f"No model summary available for {state}."

    mape = (
        state_ms["backtest_mape"].iloc[0]
        if "backtest_mape" in state_ms.columns
        else None
    )
    order = state_ms["order"].iloc[0] if "order" in state_ms.columns else "Unknown"

    accuracy_phrase = ""
    if pd.notna(mape):
        if mape < 20:
            accuracy_phrase = f"The forecast is <strong>highly accurate</strong> (walk-forward MAPE: {mape:.1f}%) — confidence bands are narrow and actionable."
        elif mape < 50:
            accuracy_phrase = f"The forecast has <strong>moderate accuracy</strong> (MAPE: {mape:.1f}%) — use the outer confidence bounds for planning headroom."
        else:
            accuracy_phrase = f"The forecast has <strong>high uncertainty</strong> (MAPE: {mape:.1f}%) — treat the confidence band as a wide planning range, not a precise prediction."

    breach_phrase = ""
    if pd.notna(months_to_breach) and months_to_breach > -1:
        breach_phrase = (
            f" ⚠️ <strong>Infrastructure capacity is projected to be breached within {int(months_to_breach)} month(s)</strong>. "
            "Immediate capacity pre-positioning is recommended."
        )
    else:
        breach_phrase = " ✅ Demand is projected to stay within infrastructure capacity over the 12-month horizon."

    return f"Model: <strong>{order}</strong>. {accuracy_phrase}{breach_phrase}"


def render_forecast_panel(state_month: pd.DataFrame) -> None:
    """Render the full forecast panel in the dashboard."""
    st.markdown(
        '<div class="headline">📈 Probabilistic Demand Forecasting (SARIMA)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="interpret-box" style="margin-top:0;">'
        "<strong>SARIMA(p,d,q)(P,D,Q)[12]</strong> models are fitted per state with AIC-based order selection. "
        "Confidence intervals show 80% and 95% prediction bands. "
        "<strong>Band opacity is tied to model accuracy</strong> — states with low walk-forward error "
        "show bold, narrow bands; high-uncertainty states show faint, wide bands."
        "</div>",
        unsafe_allow_html=True,
    )

    forecast_df = _load_forecast_data()
    model_summary = _load_model_summary()

    if forecast_df is None or forecast_df.empty:
        st.warning(
            "No forecast data available. Run the advanced analytics pipeline first."
        )
        return

    # ── Capacity alerts overview ──────────────────────────────────────────────
    if (
        model_summary is not None
        and not model_summary.empty
        and "months_to_breach" in model_summary.columns
    ):
        breached_states = model_summary[
            model_summary["months_to_breach"].notna()
            & (model_summary["months_to_breach"] > -1)
        ].sort_values("months_to_breach")
        if not breached_states.empty:
            st.error(
                f"⚠️ **Infrastructure Capacity Alert**: {len(breached_states)} states are projected "
                f"to breach their capacity threshold within 12 months!"
            )
            st.dataframe(
                breached_states[
                    ["state", "months_to_breach", "capacity_threshold"]
                ].rename(
                    columns={
                        "months_to_breach": "Months to Breach",
                        "capacity_threshold": "Capacity Limit",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.success(
                "✅ **Capacity Status**: All states are projected to stay within operational infrastructure thresholds."
            )

    # ── State selector ────────────────────────────────────────────────────────
    forecasted_states = sorted(forecast_df["state"].unique())
    selected = st.selectbox(
        "Select state for forecast", forecasted_states, key="forecast_state"
    )

    # Months to breach for selected state
    months_to_breach = None
    if (
        model_summary is not None
        and not model_summary.empty
        and "months_to_breach" in model_summary.columns
    ):
        state_ms = model_summary[model_summary["state"] == selected]
        if not state_ms.empty:
            months_to_breach = state_ms["months_to_breach"].iloc[0]

    st.plotly_chart(
        build_forecast_chart(forecast_df, selected, state_month, model_summary),
        width="stretch",
    )

    # ── Forecast interpreter ──────────────────────────────────────────────────
    interp = _forecast_interpreter(selected, model_summary, months_to_breach)
    st.markdown(f'<div class="interpret-box">{interp}</div>', unsafe_allow_html=True)

    # ── Model summary behind expander ─────────────────────────────────────────
    if model_summary is not None and not model_summary.empty:
        with st.expander("Show model accuracy & validation details"):
            has_backtest = "backtest_mape" in model_summary.columns
            fitted_mask = model_summary["model_fitted"]

            metric_cols = st.columns(4)
            with metric_cols[0]:
                fitted_count = int(model_summary["model_fitted"].sum())
                total_count = len(model_summary)
                st.metric("Models Fitted", f"{fitted_count}/{total_count}")
            with metric_cols[1]:
                if "aic" in model_summary.columns:
                    median_aic = model_summary.loc[fitted_mask, "aic"].median()
                    st.metric(
                        "Median AIC",
                        f"{median_aic:.1f}" if pd.notna(median_aic) else "—",
                    )
            with metric_cols[2]:
                if has_backtest:
                    med_mape = (
                        model_summary.loc[fitted_mask, "backtest_mape"]
                        .dropna()
                        .median()
                    )
                    st.metric(
                        "Median Walk-Forward MAPE",
                        f"{med_mape:.1f}%" if pd.notna(med_mape) else "—",
                    )
            with metric_cols[3]:
                if has_backtest:
                    med_rmse = (
                        model_summary.loc[fitted_mask, "backtest_rmse"]
                        .dropna()
                        .median()
                    )
                    st.metric(
                        "Median Walk-Forward RMSE",
                        f"{med_rmse:,.0f}" if pd.notna(med_rmse) else "—",
                    )

            display_cols = [
                "state",
                "n_obs",
                "model_fitted",
                "aic",
                "order",
                "months_to_breach",
            ]
            if has_backtest:
                display_cols += ["backtest_mape", "backtest_rmse", "backtest_mae"]
            available = [c for c in display_cols if c in model_summary.columns]
            rename_map = {
                "n_obs": "Obs",
                "model_fitted": "Fitted",
                "aic": "AIC",
                "order": "ARIMA Order",
                "months_to_breach": "Months to Breach",
                "backtest_mape": "MAPE (%)",
                "backtest_rmse": "RMSE",
                "backtest_mae": "MAE",
            }
            display_df = model_summary[available].copy()
            for col in ["aic", "backtest_mape", "backtest_rmse", "backtest_mae"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(2)
            st.dataframe(
                display_df.rename(columns=rename_map).sort_values(
                    (
                        "AIC"
                        if "AIC" in display_df.rename(columns=rename_map).columns
                        else display_df.columns[0]
                    ),
                    na_position="last",
                ),
                width="stretch",
                hide_index=True,
            )

            if has_backtest:
                valid_bt = model_summary.loc[fitted_mask].dropna(
                    subset=["backtest_mape"]
                )
                if len(valid_bt) >= 2:
                    sp_left, sp_right = st.columns(2)
                    best = valid_bt.nsmallest(3, "backtest_mape")[
                        ["state", "backtest_mape", "backtest_rmse"]
                    ]
                    worst = valid_bt.nlargest(3, "backtest_mape")[
                        ["state", "backtest_mape", "backtest_rmse"]
                    ]
                    with sp_left:
                        st.markdown("🟢 **Most Accurate Forecasts** (lowest MAPE)")
                        best.columns = ["State", "MAPE (%)", "RMSE"]
                        best["MAPE (%)"] = best["MAPE (%)"].round(1)
                        st.dataframe(best, width="stretch", hide_index=True)
                    with sp_right:
                        st.markdown("🔴 **Least Accurate Forecasts** (highest MAPE)")
                        worst.columns = ["State", "MAPE (%)", "RMSE"]
                        worst["MAPE (%)"] = worst["MAPE (%)"].round(1)
                        st.dataframe(worst, width="stretch", hide_index=True)

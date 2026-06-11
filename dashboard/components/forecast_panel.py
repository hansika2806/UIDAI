"""Forecast panel: SARIMA demand forecast charts with confidence bands."""

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


def build_forecast_chart(
    forecast_df: pd.DataFrame,
    state: str,
    state_month: pd.DataFrame,
    model_summary: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Build a forecast chart with historical data, confidence bands, and capacity threshold."""
    state_fc = forecast_df[forecast_df["state"] == state].sort_values("forecast_step")
    state_hist = state_month[state_month["state"] == state].sort_values("year_month")

    fig = go.Figure()

    # Historical data
    if not state_hist.empty:
        fig.add_trace(
            go.Scatter(
                x=state_hist["year_month"],
                y=state_hist["D_total"],
                mode="lines",
                name="Historical",
                line=dict(color="#102a43", width=2),
            )
        )

    if state_fc.empty:
        fig.update_layout(
            title=f"{state}: No forecast available", template="plotly_white"
        )
        return fig

    months = state_fc["forecast_month"].values

    # 95% CI band
    fig.add_trace(
        go.Scatter(
            x=list(months) + list(months[::-1]),
            y=list(state_fc["ci_95_upper"]) + list(state_fc["ci_95_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(14,116,144,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
            showlegend=True,
        )
    )

    # 80% CI band
    fig.add_trace(
        go.Scatter(
            x=list(months) + list(months[::-1]),
            y=list(state_fc["ci_80_upper"]) + list(state_fc["ci_80_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(14,116,144,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% CI",
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
        )
    )

    # Add capacity threshold line
    if model_summary is not None and not model_summary.empty:
        state_row = model_summary[model_summary["state"] == state]
        if not state_row.empty and "capacity_threshold" in state_row.columns:
            thresh = state_row["capacity_threshold"].iloc[0]
            if pd.notna(thresh):
                fig.add_hline(
                    y=thresh,
                    line_dash="dot",
                    line_color="#ef4444",
                    line_width=2,
                    annotation_text=f"Infrastructure Capacity Threshold ({thresh:,.1f})",
                    annotation_position="bottom right",
                )

    fig.update_layout(
        title=f"{state}: 12-Month Demographic Update Demand Forecast",
        xaxis_title="Month",
        yaxis_title="Demographic Updates",
        template="plotly_white",
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def render_forecast_panel(state_month: pd.DataFrame) -> None:
    """Render the full forecast panel in the dashboard."""
    st.subheader("📈 Probabilistic Demand Forecasting (SARIMA)")
    st.markdown(
        '<div class="panel-note"><strong>SARIMA(p,d,q)(P,D,Q)[12]</strong> models are fitted per state '
        "with AIC-based order selection. Confidence intervals show 80% and 95% prediction bands. "
        "Capacity checks compare forecasts against the state infrastructure limit (1.25x max historical demand).</div>",
        unsafe_allow_html=True,
    )

    forecast_df = _load_forecast_data()
    model_summary = _load_model_summary()

    if forecast_df is None or forecast_df.empty:
        st.warning(
            "No forecast data available. Run the advanced analytics pipeline first."
        )
        return

    # Capacity alerts overview
    if (
        model_summary is not None
        and not model_summary.empty
        and "months_to_breach" in model_summary.columns
    ):
        breached_states = model_summary[
            model_summary["months_to_breach"] > -1
        ].sort_values("months_to_breach")
        if not breached_states.empty:
            st.error(
                f"⚠️ **Infrastructure Capacity Alert**: {len(breached_states)} states are projected to breach their capacity threshold within 12 months!"
            )
            alert_cols = ["state", "months_to_breach", "capacity_threshold"]
            st.dataframe(
                breached_states[alert_cols].rename(
                    columns={
                        "months_to_breach": "Months to Breach",
                        "capacity_threshold": "Capacity Limit",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success(
                "✅ **Capacity Status**: All states are projected to stay within operational infrastructure thresholds."
            )

    # State selector
    forecasted_states = sorted(forecast_df["state"].unique())
    selected = st.selectbox(
        "Select state for forecast", forecasted_states, key="forecast_state"
    )

    st.plotly_chart(
        build_forecast_chart(forecast_df, selected, state_month, model_summary),
        use_container_width=True,
    )

    # Model summary
    if model_summary is not None and not model_summary.empty:
        st.markdown("---")
        st.markdown("#### 📐 Model Selection & Backtest Accuracy")

        # ── Walk-forward accuracy metrics ────────────────────────────────────
        has_backtest = "backtest_mape" in model_summary.columns
        fitted_mask = model_summary["model_fitted"]

        metric_cols = st.columns(4)
        fitted_count = int(model_summary["model_fitted"].sum())
        total_count = len(model_summary)

        with metric_cols[0]:
            st.metric(
                "Models Fitted",
                f"{fitted_count}/{total_count}",
                help="Number of states where SARIMA converged successfully.",
            )
        with metric_cols[1]:
            if "aic" in model_summary.columns:
                median_aic = model_summary.loc[fitted_mask, "aic"].median()
                st.metric(
                    "Median AIC",
                    f"{median_aic:.1f}" if pd.notna(median_aic) else "—",
                    help="Lower AIC = better-fitting model.",
                )
        with metric_cols[2]:
            if has_backtest:
                med_mape = (
                    model_summary.loc[fitted_mask, "backtest_mape"].dropna().median()
                )
                st.metric(
                    "Median Walk-Forward MAPE",
                    f"{med_mape:.1f}%" if pd.notna(med_mape) else "—",
                    help="Mean Absolute Percentage Error on held-out 3-month test window.",
                )
        with metric_cols[3]:
            if has_backtest:
                med_rmse = (
                    model_summary.loc[fitted_mask, "backtest_rmse"].dropna().median()
                )
                st.metric(
                    "Median Walk-Forward RMSE",
                    f"{med_rmse:,.0f}" if pd.notna(med_rmse) else "—",
                    help="Root Mean Squared Error on held-out 3-month test window.",
                )

        # ── Full summary table ───────────────────────────────────────────────
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

        # Round float columns for cleaner display
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
            use_container_width=True,
            hide_index=True,
        )

        # ── Accuracy spotlight ───────────────────────────────────────────────
        if has_backtest:
            valid_bt = model_summary.loc[fitted_mask].dropna(subset=["backtest_mape"])
            if len(valid_bt) >= 2:
                best = valid_bt.nsmallest(3, "backtest_mape")[
                    ["state", "backtest_mape", "backtest_rmse"]
                ]
                worst = valid_bt.nlargest(3, "backtest_mape")[
                    ["state", "backtest_mape", "backtest_rmse"]
                ]
                sp_left, sp_right = st.columns(2)
                with sp_left:
                    st.markdown("🟢 **Most Accurate Forecasts** (lowest MAPE)")
                    best.columns = ["State", "MAPE (%)", "RMSE"]
                    best["MAPE (%)"] = best["MAPE (%)"].round(1)
                    st.dataframe(best, use_container_width=True, hide_index=True)
                with sp_right:
                    st.markdown("🔴 **Least Accurate Forecasts** (highest MAPE)")
                    worst.columns = ["State", "MAPE (%)", "RMSE"]
                    worst["MAPE (%)"] = worst["MAPE (%)"].round(1)
                    st.dataframe(worst, use_container_width=True, hide_index=True)

                st.caption(
                    "Walk-forward validation: model trained on all-but-last-3 months, "
                    "tested on held-out 3-month window. High MAPE states may benefit from "
                    "a global pooled model or additional data collection."
                )

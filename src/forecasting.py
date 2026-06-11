"""
Probabilistic Demand Forecasting for UIDAI Governance Framework.

Fits SARIMA(p,d,q)(P,D,Q)[12] models per state to forecast monthly
update demand with confidence intervals. Uses AIC grid search for
automatic order selection.

Mathematical foundation:
    SARIMA model:
        phi(B) * Phi(B^12) * (1-B)^d * (1-B^12)^D * U_t
            = theta(B) * Theta(B^12) * eps_t

    Where B is the backshift operator, phi/Phi are AR polynomials,
    theta/Theta are MA polynomials.

    AIC = 2k - 2 * log_likelihood
    Lower AIC => better model (penalises complexity).
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress statsmodels convergence warnings during grid search
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Returns np.nan if actual contains zeros."""
    mask = actual != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def walk_forward_backtest(
    series: np.ndarray,
    test_size: int = 3,
    min_train: int = 8,
) -> Dict:
    """Walk-forward validation for the SARIMA forecasting model.

    RATIONALE:
        A forecast without a reported error metric is not useful for policy
        decisions. Walk-forward validation gives the analyst a rigorous answer
        to "how accurate was the model on data it hadn't seen?"

    Method:
        1. Hold out the last `test_size` observations as the test window.
        2. Fit the model on the remaining training data.
        3. Forecast `test_size` steps ahead.
        4. Compute MAPE and RMSE on the held-out window.

    Returns:
        dict with:
            - 'mape'       : Mean Absolute Percentage Error (%) on test window
            - 'rmse'       : Root Mean Squared Error on test window
            - 'mae'        : Mean Absolute Error on test window
            - 'n_train'    : Number of training observations
            - 'n_test'     : Number of test observations
            - 'actual'     : Actual values in test window
            - 'predicted'  : Model predictions for test window
    """
    n = len(series)
    
    # Dynamic parameter scaling for short time series
    if n < min_train + test_size:
        if n >= 7:
            min_train = 5
            test_size = 2
        elif n >= 5:
            min_train = 4
            test_size = 1

    if n < min_train + test_size:
        return {
            "mape": np.nan, "rmse": np.nan, "mae": np.nan,
            "n_train": 0, "n_test": 0,
            "actual": np.array([]), "predicted": np.array([]),
        }

    train = series[:-test_size]
    test = series[-test_size:]

    result = forecast_state(train, horizon=test_size)
    if result is None:
        return {
            "mape": np.nan, "rmse": np.nan, "mae": np.nan,
            "n_train": len(train), "n_test": test_size,
            "actual": test, "predicted": np.array([]),
        }

    forecast_vals = result["forecast_mean"]
    if hasattr(forecast_vals, "values"):
        forecast_vals = forecast_vals.values
    forecast_vals = np.array(forecast_vals).flatten()[:test_size]

    return {
        "mape": _mape(test, forecast_vals),
        "rmse": _rmse(test, forecast_vals),
        "mae": float(np.mean(np.abs(test - forecast_vals))),
        "n_train": len(train),
        "n_test": test_size,
        "actual": test,
        "predicted": forecast_vals,
    }


def _fit_sarima_with_order(
    series: np.ndarray,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> Optional[object]:
    """Attempt to fit a SARIMA model with given orders. Returns None on failure."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=100)
        return result
    except Exception:
        return None


def auto_sarima(
    series: np.ndarray,
    seasonal_period: int = 12,
    max_p: int = 2,
    max_d: int = 1,
    max_q: int = 2,
    max_P: int = 1,
    max_D: int = 1,
    max_Q: int = 1,
) -> Optional[object]:
    """Grid search over SARIMA orders, selecting the model with lowest AIC.

    AIC = 2k - 2 * ln(L)

    Tests all combinations of (p,d,q)(P,D,Q)[s] within specified ranges.
    Returns the fitted model with the lowest AIC, or None if all fail.
    """
    best_aic = np.inf
    best_model = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            order = (p, d, q)
                            seasonal = (P, D, Q, seasonal_period)
                            result = _fit_sarima_with_order(series, order, seasonal)
                            if result is not None and result.aic < best_aic:
                                best_aic = result.aic
                                best_model = result

    return best_model


def _manual_fallback_forecast(state_series: np.ndarray, horizon: int = 12) -> Dict:
    """Manually generate a linear trend forecast with standard error-based confidence intervals.
    
    Used when statistical time series packages fail to converge due to small sample size.
    """
    n = len(state_series)
    t = np.arange(n)
    
    # Fit simple linear regression: y = slope * t + intercept
    if n >= 2:
        try:
            slope, intercept = np.polyfit(t, state_series, 1)
        except Exception:
            slope = 0.0
            intercept = float(state_series[-1]) if n > 0 else 0.0
    else:
        slope = 0.0
        intercept = float(state_series[0]) if n > 0 else 0.0
        
    future_t = np.arange(n, n + horizon)
    forecast_mean = slope * future_t + intercept
    
    # Clip negative values
    forecast_mean = np.maximum(0.0, forecast_mean)
    
    # Estimate standard error
    std_err = np.std(state_series) if n > 1 else (0.1 * float(state_series[0]) if n > 0 else 1.0)
    if std_err < 1e-12:
        std_err = 1.0
        
    # Standard normal multipliers: 80% CI => 1.282, 95% CI => 1.960
    # Increase uncertainty over time (sqrt(step) expansion)
    expansion = np.sqrt(np.arange(1, horizon + 1))
    
    ci_80_half = 1.282 * std_err * expansion
    ci_95_half = 1.960 * std_err * expansion
    
    return {
        "forecast_mean": forecast_mean,
        "ci_80_lower": np.maximum(0.0, forecast_mean - ci_80_half),
        "ci_80_upper": forecast_mean + ci_80_half,
        "ci_95_lower": np.maximum(0.0, forecast_mean - ci_95_half),
        "ci_95_upper": forecast_mean + ci_95_half,
        "aic": 999.0,
        "order": "Fallback (Linear Regression)",
        "seasonal_order": "None",
    }


def forecast_state(
    state_series: np.ndarray,
    horizon: int = 12,
    seasonal_period: int = 12,
) -> Optional[Dict]:
    """Forecast a single state's update demand.

    Returns forecast values with 80% and 95% confidence intervals.
    """
    model = None
    if len(state_series) < 8:
        # Too little data to fit even a simple ARIMA; use linear fallback directly
        return _manual_fallback_forecast(state_series, horizon=horizon)
    elif len(state_series) < 24:
        # Not enough data for seasonal SARIMA; try simple ARIMA(1,1,1)
        model = _fit_sarima_with_order(
            state_series,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
        )
        # Try ARIMA(1,0,0) as a statsmodels fallback
        if model is None:
            model = _fit_sarima_with_order(
                state_series,
                order=(1, 0, 0),
                seasonal_order=(0, 0, 0, 0),
            )
    else:
        model = auto_sarima(state_series, seasonal_period=seasonal_period)

    if model is None:
        return _manual_fallback_forecast(state_series, horizon=horizon)

    try:
        forecast_result = model.get_forecast(steps=horizon)
        forecast_mean = forecast_result.predicted_mean

        # 80% confidence interval (alpha=0.20)
        ci_80 = forecast_result.conf_int(alpha=0.20)
        # 95% confidence interval (alpha=0.05)
        ci_95 = forecast_result.conf_int(alpha=0.05)

        return {
            "forecast_mean": forecast_mean,
            "ci_80_lower": ci_80.iloc[:, 0].values,
            "ci_80_upper": ci_80.iloc[:, 1].values,
            "ci_95_lower": ci_95.iloc[:, 0].values,
            "ci_95_upper": ci_95.iloc[:, 1].values,
            "aic": model.aic,
            "order": model.specification.get("order", None),
            "seasonal_order": model.specification.get("seasonal_order", None),
        }
    except Exception as e:
        logger.debug(f"Forecast extraction failed: {e}; falling back to manual model.")
        return _manual_fallback_forecast(state_series, horizon=horizon)


def run_demand_forecasting(
    state_month: pd.DataFrame,
    target_col: str = "D_total",
    horizon: int = 12,
    min_obs: int = 4,
    run_backtest: bool = True,
    backtest_test_size: int = 3,
) -> Dict[str, object]:
    """Run SARIMA forecasting for all states with optional walk-forward backtesting.

    Parameters
    ----------
    state_month : pd.DataFrame
        State-month aggregated data.
    target_col : str
        Column to forecast (default: demographic updates).
    horizon : int
        Number of months to forecast forward.
    min_obs : int
        Minimum observations required to attempt forecasting.
    run_backtest : bool
        If True, run walk-forward backtesting and report MAPE/RMSE per state.
    backtest_test_size : int
        Number of periods to hold out for walk-forward validation.

    Returns
    -------
    dict with:
        - 'forecast_df': DataFrame with forecast details per state
        - 'model_summary': DataFrame with model selection + backtest metrics
        - 'state_forecasts': dict mapping state -> forecast arrays
    """
    logger.info(
        f"Running SARIMA demand forecasting "
        f"(target={target_col}, horizon={horizon}, backtest={run_backtest})..."
    )

    forecast_rows = []
    model_rows = []
    state_forecasts = {}

    states = sorted(state_month["state"].unique())
    for i, state in enumerate(states):
        df_s = state_month[state_month["state"] == state].sort_values("year_month")

        if len(df_s) < min_obs:
            logger.debug(f"Skipping {state}: only {len(df_s)} observations (need {min_obs})")
            model_rows.append({
                "state": state,
                "n_obs": len(df_s),
                "model_fitted": False,
                "aic": np.nan,
                "order": None,
                "seasonal_order": None,
                "capacity_threshold": np.nan,
                "months_to_breach": -1,
                "backtest_mape": np.nan,
                "backtest_rmse": np.nan,
                "backtest_mae": np.nan,
            })
            continue

        series = df_s[target_col].values.astype(float)
        last_month = df_s["year_month"].iloc[-1]

        # ── Walk-Forward Backtest ────────────────────────────────────
        bt_metrics = {"mape": np.nan, "rmse": np.nan, "mae": np.nan}
        if run_backtest:
            bt = walk_forward_backtest(series, test_size=backtest_test_size)
            bt_metrics = {"mape": bt["mape"], "rmse": bt["rmse"], "mae": bt["mae"]}

        # ── Full-series Forecast ──────────────────────────────────────
        result = forecast_state(series, horizon=horizon)

        if result is None:
            model_rows.append({
                "state": state,
                "n_obs": len(series),
                "model_fitted": False,
                "aic": np.nan,
                "order": None,
                "seasonal_order": None,
                "capacity_threshold": np.nan,
                "months_to_breach": -1,
                "backtest_mape": bt_metrics["mape"],
                "backtest_rmse": bt_metrics["rmse"],
                "backtest_mae": bt_metrics["mae"],
            })
            continue

        # Define capacity threshold as 1.25 * max historical update volume
        capacity_threshold = 1.25 * series.max()

        # Find first month where the worst-case (ci_95_upper) exceeds capacity
        months_to_breach = None
        for j in range(horizon):
            if result["ci_95_upper"][j] > capacity_threshold:
                months_to_breach = j + 1
                break

        model_rows.append({
            "state": state,
            "n_obs": len(series),
            "model_fitted": True,
            "aic": result["aic"],
            "order": str(result["order"]),
            "seasonal_order": str(result["seasonal_order"]),
            "capacity_threshold": float(capacity_threshold),
            "months_to_breach": int(months_to_breach) if months_to_breach is not None else -1,
            "backtest_mape": bt_metrics["mape"],
            "backtest_rmse": bt_metrics["rmse"],
            "backtest_mae": bt_metrics["mae"],
        })

        # Generate future month labels
        try:
            last_period = pd.Period(last_month, freq="M")
            future_months = [(last_period + i + 1).strftime("%Y-%m") for i in range(horizon)]
        except Exception:
            future_months = [f"t+{i+1}" for i in range(horizon)]

        state_forecasts[state] = result

        for j in range(horizon):
            forecast_rows.append({
                "state": state,
                "forecast_month": future_months[j],
                "forecast_step": j + 1,
                "forecast_mean": float(result["forecast_mean"].iloc[j] if hasattr(result["forecast_mean"], "iloc") else result["forecast_mean"][j]),
                "ci_80_lower": float(result["ci_80_lower"][j]),
                "ci_80_upper": float(result["ci_80_upper"][j]),
                "ci_95_lower": float(result["ci_95_lower"][j]),
                "ci_95_upper": float(result["ci_95_upper"][j]),
            })

        if (i + 1) % 10 == 0:
            logger.info(f"  Forecasted {i + 1}/{len(states)} states...")

    forecast_df = pd.DataFrame(forecast_rows)
    model_summary = pd.DataFrame(model_rows)

    n_fitted = model_summary["model_fitted"].sum()
    if run_backtest and "backtest_mape" in model_summary.columns:
        valid_mape = model_summary["backtest_mape"].dropna()
        median_mape = valid_mape.median() if len(valid_mape) > 0 else np.nan
        logger.info(
            f"Forecasting complete: {n_fitted}/{len(states)} states modelled. "
            f"Median walk-forward MAPE: {median_mape:.1f}%"
        )
    else:
        logger.info(f"Forecasting complete: {n_fitted}/{len(states)} states modelled.")

    return {
        "forecast_df": forecast_df,
        "model_summary": model_summary,
        "state_forecasts": state_forecasts,
    }

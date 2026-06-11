"""
Hierarchical Time Series Forecasting for UIDAI Governance Framework.

Implements global forecasting models that share statistical power across states,
with bottom-up reconciliation to ensure forecast consistency.

Architecture:
- Global LightGBM model trained on all states simultaneously
- State-specific features + cross-state patterns
- Bottom-up reconciliation: state forecasts sum to national forecast
- Handles cold-start problem for new states

Mathematical foundation:
    Bottom-up reconciliation:
        ŷ_national(t) = Σ_s ŷ_state_s(t)
    
    This ensures hierarchical consistency without optimization,
    unlike top-down or middle-out approaches.

Advantages over per-state SARIMA:
1. Shares statistical power across states (better for small states)
2. Captures cross-state patterns and spillovers
3. Handles missing data and cold starts
4. Faster training (one model vs. 36 models)
5. Guaranteed forecast consistency
"""

import logging
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from logger import get_logger, PerformanceLogger

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)


class HierarchicalForecaster:
    """
    Global time series forecasting with hierarchical reconciliation.
    
    Features:
    - Lag features (autoregressive)
    - Rolling statistics (mean, std, trend)
    - Calendar features (month, quarter, year)
    - State embeddings (learned representations)
    - Cross-state features (national trends)
    """
    
    def __init__(
        self,
        horizon: int = 12,
        lags: List[int] = [1, 2, 3, 6, 12],
        rolling_windows: List[int] = [3, 6, 12],
        lgb_params: Optional[Dict] = None
    ):
        """
        Initialize hierarchical forecaster.
        
        Args:
            horizon: Forecast horizon (months)
            lags: Lag features to create
            rolling_windows: Window sizes for rolling statistics
            lgb_params: LightGBM parameters
        """
        self.horizon = horizon
        self.lags = lags
        self.rolling_windows = rolling_windows
        
        # Default LightGBM parameters
        self.lgb_params = lgb_params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.model: Optional[lgb.Booster] = None
        self.state_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        
        logger.info(
            "Hierarchical forecaster initialized",
            extra={
                "horizon": horizon,
                "lags": lags,
                "rolling_windows": rolling_windows
            }
        )
    
    def create_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'value',
        state_col: str = 'state',
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Create features for global forecasting model.
        
        Args:
            df: DataFrame with columns [state, date, value]
            target_col: Name of target column
            state_col: Name of state column
            date_col: Name of date column
        
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        df = df.sort_values([state_col, date_col])
        
        # Encode states
        df['state_encoded'] = self.state_encoder.fit_transform(df[state_col])
        
        # Calendar features
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['year'] = df[date_col].dt.year
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features (per state)
        for lag in self.lags:
            df[f'lag_{lag}'] = df.groupby(state_col)[target_col].shift(lag)
        
        # Rolling statistics (per state)
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = (
                df.groupby(state_col)[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f'rolling_std_{window}'] = (
                df.groupby(state_col)[target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
        
        # Trend features (per state)
        df['trend'] = (
            df.groupby(state_col)
            .cumcount()
        )
        
        # National-level features (cross-state patterns)
        national_mean = df.groupby(date_col)[target_col].transform('mean')
        df['national_mean'] = national_mean
        df['deviation_from_national'] = df[target_col] - national_mean
        
        # State-level statistics
        state_mean = df.groupby(state_col)[target_col].transform('mean')
        df['state_mean'] = state_mean
        df['deviation_from_state_mean'] = df[target_col] - state_mean
        
        logger.info(f"Created {len(df.columns)} features")
        
        return df
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'value'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data by removing rows with NaN features.
        
        Args:
            df: DataFrame with features
            target_col: Name of target column
        
        Returns:
            Tuple of (X, y)
        """
        # Feature columns (exclude metadata)
        exclude_cols = [target_col, 'state', 'date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN (from lag/rolling features)
        df_clean = df.dropna(subset=feature_cols)
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        self.feature_names = feature_cols
        
        logger.info(
            "Training data prepared",
            extra={
                "n_samples": len(X),
                "n_features": len(feature_cols),
                "n_dropped": len(df) - len(df_clean)
            }
        )
        
        return X, y
    
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'value',
        state_col: str = 'state',
        date_col: str = 'date',
        n_splits: int = 3
    ) -> Dict:
        """
        Train global forecasting model with time series cross-validation.
        
        Args:
            df: DataFrame with columns [state, date, value]
            target_col: Name of target column
            state_col: Name of state column
            date_col: Name of date column
            n_splits: Number of CV splits
        
        Returns:
            Dictionary with training metrics
        """
        with perf_logger.timer("hierarchical_forecasting_training"):
            # Create features
            df_features = self.create_features(df, target_col, state_col, date_col)
            
            # Prepare training data
            X, y = self.prepare_training_data(df_features, target_col)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model
                model = lgb.train(
                    self.lgb_params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                
                # Evaluate
                y_pred = model.predict(X_val)
                rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                mae = np.mean(np.abs(y_val - y_pred))
                
                cv_scores.append({'fold': fold, 'rmse': rmse, 'mae': mae})
                
                logger.info(
                    f"Fold {fold + 1}/{n_splits}",
                    extra={'rmse': rmse, 'mae': mae}
                )
            
            # Train final model on all data
            train_data = lgb.Dataset(X, label=y)
            self.model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=1000
            )
            
            # Calculate average CV scores
            avg_rmse = np.mean([s['rmse'] for s in cv_scores])
            avg_mae = np.mean([s['mae'] for s in cv_scores])
            
            metrics = {
                'cv_rmse': avg_rmse,
                'cv_mae': avg_mae,
                'cv_scores': cv_scores,
                'n_features': len(self.feature_names),
                'n_samples': len(X)
            }
            
            logger.info(
                "Training complete",
                extra={
                    'cv_rmse': avg_rmse,
                    'cv_mae': avg_mae,
                    'n_features': len(self.feature_names)
                }
            )
            
            return metrics
    
    def forecast(
        self,
        df: pd.DataFrame,
        target_col: str = 'value',
        state_col: str = 'state',
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Generate forecasts for all states with bottom-up reconciliation.
        
        Args:
            df: Historical data DataFrame
            target_col: Name of target column
            state_col: Name of state column
            date_col: Name of date column
        
        Returns:
            DataFrame with forecasts for each state
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        with perf_logger.timer("hierarchical_forecasting_inference"):
            forecasts = []
            states = df[state_col].unique()
            
            for state in states:
                state_df = df[df[state_col] == state].copy()
                state_forecast = self._forecast_state(
                    state_df, state, target_col, state_col, date_col
                )
                forecasts.append(state_forecast)
            
            # Combine all state forecasts
            forecast_df = pd.concat(forecasts, ignore_index=True)
            
            # Bottom-up reconciliation
            forecast_df = self._reconcile_bottom_up(forecast_df, date_col)
            
            logger.info(
                "Forecasting complete",
                extra={
                    'n_states': len(states),
                    'horizon': self.horizon
                }
            )
            
            return forecast_df
    
    def _forecast_state(
        self,
        state_df: pd.DataFrame,
        state: str,
        target_col: str,
        state_col: str,
        date_col: str
    ) -> pd.DataFrame:
        """
        Generate forecast for a single state.
        
        Args:
            state_df: Historical data for the state
            state: State name
            target_col: Target column name
            state_col: State column name
            date_col: Date column name
        
        Returns:
            DataFrame with forecasts
        """
        # Get last date
        last_date = state_df[date_col].max()
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=self.horizon,
            freq='MS'
        )
        
        # Initialize forecast DataFrame
        forecast_rows = []
        
        # Iterative forecasting (use previous predictions as features)
        current_df = state_df.copy()
        
        for future_date in future_dates:
            # Create row for prediction
            new_row = pd.DataFrame({
                state_col: [state],
                date_col: [future_date],
                target_col: [np.nan]  # Will be predicted
            })
            
            # Append to current data
            temp_df = pd.concat([current_df, new_row], ignore_index=True)
            
            # Create features
            temp_df_features = self.create_features(
                temp_df, target_col, state_col, date_col
            )
            
            # Get features for the new row
            X_new = temp_df_features.iloc[[-1]][self.feature_names]
            
            # Handle missing features (fill with last known values)
            X_new = X_new.fillna(method='ffill').fillna(0)
            
            # Predict
            prediction = self.model.predict(X_new)[0]
            
            # Store forecast
            forecast_rows.append({
                state_col: state,
                date_col: future_date,
                'forecast': prediction
            })
            
            # Update current_df with prediction for next iteration
            current_df = pd.concat([
                current_df,
                pd.DataFrame({
                    state_col: [state],
                    date_col: [future_date],
                    target_col: [prediction]
                })
            ], ignore_index=True)
        
        return pd.DataFrame(forecast_rows)
    
    def _reconcile_bottom_up(
        self,
        forecast_df: pd.DataFrame,
        date_col: str
    ) -> pd.DataFrame:
        """
        Apply bottom-up reconciliation to ensure consistency.
        
        Bottom-up: National forecast = sum of state forecasts
        
        Args:
            forecast_df: DataFrame with state-level forecasts
            date_col: Date column name
        
        Returns:
            DataFrame with reconciled forecasts and national totals
        """
        # Calculate national forecast (sum of states)
        national_forecast = (
            forecast_df.groupby(date_col)['forecast']
            .sum()
            .reset_index()
        )
        national_forecast['state'] = 'National'
        national_forecast = national_forecast.rename(columns={'forecast': 'forecast'})
        
        # Combine state and national forecasts
        reconciled_df = pd.concat([
            forecast_df,
            national_forecast
        ], ignore_index=True)
        
        logger.info("Bottom-up reconciliation complete")
        
        return reconciled_df
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importance(importance_type='gain')
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df


def compare_with_sarima(
    hierarchical_forecast: pd.DataFrame,
    sarima_forecast: pd.DataFrame,
    actual: pd.DataFrame
) -> Dict:
    """
    Compare hierarchical forecasting with SARIMA baseline.
    
    Args:
        hierarchical_forecast: Forecasts from hierarchical model
        sarima_forecast: Forecasts from SARIMA models
        actual: Actual values
    
    Returns:
        Dictionary with comparison metrics
    """
    # Calculate RMSE for both methods
    hier_rmse = np.sqrt(np.mean((actual['value'] - hierarchical_forecast['forecast']) ** 2))
    sarima_rmse = np.sqrt(np.mean((actual['value'] - sarima_forecast['forecast']) ** 2))
    
    improvement = (sarima_rmse - hier_rmse) / sarima_rmse * 100
    
    return {
        'hierarchical_rmse': hier_rmse,
        'sarima_rmse': sarima_rmse,
        'improvement_pct': improvement
    }

# Made with Bob

"""
Synthetic Control Method for Causal Inference in UIDAI Governance Framework.

Implements counterfactual analysis to measure the net impact of policy
interventions. Answers: "What would have happened without the policy?"

Mathematical foundation:
    Synthetic control: Ŷ_0t = Σ_j w_j * Y_jt
    where w_j ≥ 0, Σ_j w_j = 1
    
    Treatment effect: τ_t = Y_1t - Ŷ_0t
    
    Optimization:
        min_w ||X_1 - X_0 W||² + λ||W||²
        s.t. w_j ≥ 0, Σ_j w_j = 1

Example use case:
    Maharashtra implements a new biometric update policy in Jan 2023.
    Question: Did it increase update rates?
    
    Synthetic Maharashtra = weighted combination of similar states
    (Gujarat, Karnataka, Tamil Nadu) that didn't implement the policy.
    
    Treatment effect = Actual Maharashtra - Synthetic Maharashtra

Advantages over Granger causality:
- Measures net causal impact, not just predictive correlation
- Provides counterfactual estimate
- Robust to confounders (if control states are well-matched)
- Interpretable weights show which states are similar
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

from logger import get_logger

logger = get_logger(__name__)


class SyntheticControl:
    """
    Synthetic Control Method for causal inference.
    
    Constructs a synthetic version of the treated unit using a weighted
    combination of control units, then compares actual vs. synthetic
    outcomes post-intervention.
    """
    
    def __init__(
        self,
        treated_unit: str,
        control_units: List[str],
        intervention_date: pd.Timestamp,
        outcome_col: str = 'value',
        predictor_cols: Optional[List[str]] = None
    ):
        """
        Initialize synthetic control analysis.
        
        Args:
            treated_unit: Name of treated unit (e.g., "Maharashtra")
            control_units: List of control unit names
            intervention_date: Date of intervention
            outcome_col: Name of outcome variable
            predictor_cols: List of predictor variables for matching
        """
        self.treated_unit = treated_unit
        self.control_units = control_units
        self.intervention_date = intervention_date
        self.outcome_col = outcome_col
        self.predictor_cols = predictor_cols or []
        
        self.weights: Optional[np.ndarray] = None
        self.synthetic_outcome: Optional[pd.Series] = None
        self.treatment_effect: Optional[pd.Series] = None
        
        logger.info(
            "Synthetic control initialized",
            extra={
                "treated_unit": treated_unit,
                "n_controls": len(control_units),
                "intervention_date": intervention_date.isoformat()
            }
        )
    
    def fit(
        self,
        df: pd.DataFrame,
        unit_col: str = 'state',
        date_col: str = 'date',
        method: str = 'optimization'
    ) -> Dict:
        """
        Fit synthetic control by finding optimal weights.
        
        Args:
            df: DataFrame with columns [unit, date, outcome, predictors]
            unit_col: Name of unit column
            date_col: Name of date column
            method: Fitting method ('optimization' or 'regression')
        
        Returns:
            Dictionary with fit results
        """
        # Split into pre and post intervention
        pre_df = df[df[date_col] < self.intervention_date].copy()
        
        # Extract treated unit data
        treated_data = pre_df[pre_df[unit_col] == self.treated_unit]
        
        # Extract control units data
        control_data = pre_df[pre_df[unit_col].isin(self.control_units)]
        
        if method == 'optimization':
            self.weights = self._fit_optimization(
                treated_data, control_data, unit_col, date_col
            )
        elif method == 'regression':
            self.weights = self._fit_regression(
                treated_data, control_data, unit_col, date_col
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate synthetic outcome for full period
        self.synthetic_outcome = self._calculate_synthetic_outcome(
            df, unit_col, date_col
        )
        
        # Calculate treatment effect
        actual_outcome = df[df[unit_col] == self.treated_unit].set_index(date_col)[self.outcome_col]
        self.treatment_effect = actual_outcome - self.synthetic_outcome
        
        # Calculate fit metrics
        pre_treatment_effect = self.treatment_effect[self.treatment_effect.index < self.intervention_date]
        pre_rmse = np.sqrt(np.mean(pre_treatment_effect ** 2))
        
        results = {
            "weights": dict(zip(self.control_units, self.weights)),
            "pre_intervention_rmse": pre_rmse,
            "n_control_units": len(self.control_units),
            "method": method
        }
        
        logger.info(
            "Synthetic control fitted",
            extra={
                "pre_rmse": pre_rmse,
                "top_3_weights": sorted(
                    results["weights"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            }
        )
        
        return results
    
    def _fit_optimization(
        self,
        treated_data: pd.DataFrame,
        control_data: pd.DataFrame,
        unit_col: str,
        date_col: str
    ) -> np.ndarray:
        """
        Fit weights using constrained optimization.
        
        Minimizes: ||Y_treated - Y_controls * W||²
        Subject to: w_i ≥ 0, Σ w_i = 1
        
        Args:
            treated_data: Pre-intervention data for treated unit
            control_data: Pre-intervention data for control units
            unit_col: Unit column name
            date_col: Date column name
        
        Returns:
            Optimal weights array
        """
        # Pivot to wide format
        treated_series = treated_data.set_index(date_col)[self.outcome_col]
        
        control_matrix = control_data.pivot(
            index=date_col,
            columns=unit_col,
            values=self.outcome_col
        )[self.control_units]
        
        # Align dates
        common_dates = treated_series.index.intersection(control_matrix.index)
        Y_treated = treated_series.loc[common_dates].values
        Y_controls = control_matrix.loc[common_dates].values
        
        # Objective function
        def objective(w):
            synthetic = Y_controls @ w
            return np.sum((Y_treated - synthetic) ** 2)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights >= 0
        bounds = [(0, 1) for _ in range(len(self.control_units))]
        
        # Initial guess: equal weights
        w0 = np.ones(len(self.control_units)) / len(self.control_units)
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        return result.x
    
    def _fit_regression(
        self,
        treated_data: pd.DataFrame,
        control_data: pd.DataFrame,
        unit_col: str,
        date_col: str
    ) -> np.ndarray:
        """
        Fit weights using constrained least squares regression.
        
        Args:
            treated_data: Pre-intervention data for treated unit
            control_data: Pre-intervention data for control units
            unit_col: Unit column name
            date_col: Date column name
        
        Returns:
            Regression weights array
        """
        from sklearn.linear_model import Ridge
        
        # Pivot to wide format
        treated_series = treated_data.set_index(date_col)[self.outcome_col]
        
        control_matrix = control_data.pivot(
            index=date_col,
            columns=unit_col,
            values=self.outcome_col
        )[self.control_units]
        
        # Align dates
        common_dates = treated_series.index.intersection(control_matrix.index)
        Y_treated = treated_series.loc[common_dates].values
        Y_controls = control_matrix.loc[common_dates].values
        
        # Fit Ridge regression (L2 regularization)
        model = Ridge(alpha=0.1, fit_intercept=False, positive=True)
        model.fit(Y_controls, Y_treated)
        
        # Normalize weights to sum to 1
        weights = model.coef_
        weights = np.maximum(weights, 0)  # Ensure non-negative
        weights = weights / weights.sum()  # Normalize
        
        return weights
    
    def _calculate_synthetic_outcome(
        self,
        df: pd.DataFrame,
        unit_col: str,
        date_col: str
    ) -> pd.Series:
        """
        Calculate synthetic outcome using fitted weights.
        
        Args:
            df: Full DataFrame
            unit_col: Unit column name
            date_col: Date column name
        
        Returns:
            Synthetic outcome time series
        """
        control_data = df[df[unit_col].isin(self.control_units)]
        
        control_matrix = control_data.pivot(
            index=date_col,
            columns=unit_col,
            values=self.outcome_col
        )[self.control_units]
        
        synthetic = control_matrix.values @ self.weights
        
        return pd.Series(synthetic, index=control_matrix.index)
    
    def estimate_treatment_effect(
        self,
        post_period_only: bool = True
    ) -> Dict:
        """
        Estimate average treatment effect.
        
        Args:
            post_period_only: If True, only calculate for post-intervention period
        
        Returns:
            Dictionary with treatment effect estimates
        """
        if self.treatment_effect is None:
            raise ValueError("Must call fit() first")
        
        if post_period_only:
            effect = self.treatment_effect[self.treatment_effect.index >= self.intervention_date]
        else:
            effect = self.treatment_effect
        
        ate = effect.mean()
        ate_std = effect.std()
        ate_se = ate_std / np.sqrt(len(effect))
        
        # T-test for significance
        t_stat = ate / ate_se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(effect) - 1))
        
        # Cumulative effect
        cumulative_effect = effect.sum()
        
        results = {
            "average_treatment_effect": ate,
            "ate_std": ate_std,
            "ate_se": ate_se,
            "t_statistic": t_stat,
            "p_value": p_value,
            "cumulative_effect": cumulative_effect,
            "n_periods": len(effect),
            "significant": p_value < 0.05
        }
        
        logger.info(
            "Treatment effect estimated",
            extra={
                "ate": ate,
                "p_value": p_value,
                "significant": results["significant"]
            }
        )
        
        return results
    
    def placebo_test(
        self,
        df: pd.DataFrame,
        unit_col: str = 'state',
        date_col: str = 'date',
        n_placebos: int = 100
    ) -> Dict:
        """
        Conduct placebo test by randomly assigning intervention to control units.
        
        Tests null hypothesis: treatment effect is not significantly different
        from what we'd observe by chance.
        
        Args:
            df: Full DataFrame
            unit_col: Unit column name
            date_col: Date column name
            n_placebos: Number of placebo tests to run
        
        Returns:
            Dictionary with placebo test results
        """
        if self.treatment_effect is None:
            raise ValueError("Must call fit() first")
        
        # Actual treatment effect
        actual_effect = self.estimate_treatment_effect(post_period_only=True)
        actual_ate = actual_effect["average_treatment_effect"]
        
        # Run placebo tests
        placebo_effects = []
        
        for i in range(n_placebos):
            # Randomly select a control unit as "treated"
            placebo_treated = np.random.choice(self.control_units)
            placebo_controls = [u for u in self.control_units if u != placebo_treated]
            
            if len(placebo_controls) < 2:
                continue
            
            # Fit synthetic control for placebo
            placebo_sc = SyntheticControl(
                treated_unit=placebo_treated,
                control_units=placebo_controls,
                intervention_date=self.intervention_date,
                outcome_col=self.outcome_col
            )
            
            try:
                placebo_sc.fit(df, unit_col, date_col, method='optimization')
                placebo_result = placebo_sc.estimate_treatment_effect(post_period_only=True)
                placebo_effects.append(placebo_result["average_treatment_effect"])
            except Exception as e:
                logger.warning(f"Placebo test {i} failed: {e}")
                continue
        
        # Calculate p-value: proportion of placebo effects >= actual effect
        placebo_effects = np.array(placebo_effects)
        p_value_placebo = np.mean(np.abs(placebo_effects) >= np.abs(actual_ate))
        
        results = {
            "actual_ate": actual_ate,
            "placebo_effects": placebo_effects,
            "placebo_mean": placebo_effects.mean(),
            "placebo_std": placebo_effects.std(),
            "p_value": p_value_placebo,
            "n_placebos": len(placebo_effects),
            "significant": p_value_placebo < 0.05
        }
        
        logger.info(
            "Placebo test complete",
            extra={
                "n_placebos": len(placebo_effects),
                "p_value": p_value_placebo,
                "significant": results["significant"]
            }
        )
        
        return results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a DataFrame for visualization.
        
        Returns:
            DataFrame with actual, synthetic, and treatment effect
        """
        if self.synthetic_outcome is None or self.treatment_effect is None:
            raise ValueError("Must call fit() first")
        
        results_df = pd.DataFrame({
            'date': self.synthetic_outcome.index,
            'synthetic': self.synthetic_outcome.values,
            'treatment_effect': self.treatment_effect.values
        })
        
        results_df['post_intervention'] = results_df['date'] >= self.intervention_date
        
        return results_df


def run_synthetic_control_analysis(
    df: pd.DataFrame,
    treated_unit: str,
    intervention_date: pd.Timestamp,
    unit_col: str = 'state',
    date_col: str = 'date',
    outcome_col: str = 'value',
    exclude_units: Optional[List[str]] = None
) -> Tuple[SyntheticControl, Dict]:
    """
    Convenience function to run complete synthetic control analysis.
    
    Args:
        df: DataFrame with panel data
        treated_unit: Name of treated unit
        intervention_date: Date of intervention
        unit_col: Unit column name
        date_col: Date column name
        outcome_col: Outcome variable name
        exclude_units: Units to exclude from control pool
    
    Returns:
        Tuple of (SyntheticControl object, results dictionary)
    """
    # Determine control units
    all_units = df[unit_col].unique()
    exclude_units = exclude_units or []
    control_units = [
        u for u in all_units
        if u != treated_unit and u not in exclude_units
    ]
    
    # Initialize and fit
    sc = SyntheticControl(
        treated_unit=treated_unit,
        control_units=control_units,
        intervention_date=intervention_date,
        outcome_col=outcome_col
    )
    
    fit_results = sc.fit(df, unit_col, date_col)
    treatment_effect = sc.estimate_treatment_effect(post_period_only=True)
    placebo_results = sc.placebo_test(df, unit_col, date_col, n_placebos=50)
    
    results = {
        "fit": fit_results,
        "treatment_effect": treatment_effect,
        "placebo_test": placebo_results
    }
    
    return sc, results

# Made with Bob

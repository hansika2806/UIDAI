"""
Data & Model Drift Monitoring for UIDAI Governance Framework.

Implements statistical tests to detect:
1. Feature Drift - Changes in input data distribution
2. Prediction Drift - Changes in model output distribution
3. Concept Drift - Changes in the relationship between features and target

Uses:
- Kolmogorov-Smirnov (KS) test for continuous features
- Chi-squared test for categorical features
- Population Stability Index (PSI) for binned distributions
- Jensen-Shannon Divergence for probability distributions

Mathematical foundations:
    KS statistic: D = sup_x |F_ref(x) - F_prod(x)|
    PSI = sum((p_prod - p_ref) * log(p_prod / p_ref))
    JS divergence: JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from logger import get_logger

logger = get_logger(__name__)


class DriftMonitor:
    """
    Monitor for data and model drift detection.

    Tracks:
    - Feature distributions over time
    - Prediction distributions
    - Statistical drift metrics
    - Automated alerting thresholds
    """

    def __init__(
        self,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.1,
        js_threshold: float = 0.1,
    ):
        """
        Initialize drift monitor.

        Args:
            ks_threshold: P-value threshold for KS test (default: 0.05)
            psi_threshold: PSI threshold for drift alert (default: 0.1)
            js_threshold: JS divergence threshold (default: 0.1)

        Thresholds:
            PSI < 0.1: No significant drift
            0.1 <= PSI < 0.2: Moderate drift (monitor)
            PSI >= 0.2: Significant drift (retrain)
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.js_threshold = js_threshold

        self.drift_history: List[Dict] = []

        logger.info(
            "Drift monitor initialized",
            extra={
                "ks_threshold": ks_threshold,
                "psi_threshold": psi_threshold,
                "js_threshold": js_threshold,
            },
        )

    def ks_test(
        self, reference: np.ndarray, production: np.ndarray, feature_name: str
    ) -> Dict:
        """
        Kolmogorov-Smirnov test for continuous feature drift.

        Tests null hypothesis: reference and production come from same distribution

        Args:
            reference: Reference (training) data
            production: Production (current) data
            feature_name: Name of the feature

        Returns:
            Dictionary with test results
        """
        # Remove NaNs
        reference = reference[~np.isnan(reference)]
        production = production[~np.isnan(production)]

        if len(reference) == 0 or len(production) == 0:
            return {
                "feature": feature_name,
                "test": "KS",
                "statistic": np.nan,
                "p_value": np.nan,
                "drift_detected": False,
                "error": "Empty arrays after NaN removal",
            }

        # Perform KS test
        statistic, p_value = ks_2samp(reference, production)

        drift_detected = p_value < self.ks_threshold

        result = {
            "feature": feature_name,
            "test": "KS",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "severity": self._classify_drift_severity(p_value, "ks"),
        }

        if drift_detected:
            logger.warning(f"Feature drift detected: {feature_name}", extra=result)

        return result

    def psi(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        feature_name: str,
        n_bins: int = 10,
    ) -> Dict:
        """
        Population Stability Index for feature drift.

        PSI = sum((p_prod - p_ref) * log(p_prod / p_ref))

        Args:
            reference: Reference data
            production: Production data
            feature_name: Name of the feature
            n_bins: Number of bins for discretization

        Returns:
            Dictionary with PSI results
        """
        # Remove NaNs
        reference = reference[~np.isnan(reference)]
        production = production[~np.isnan(production)]

        if len(reference) == 0 or len(production) == 0:
            return {
                "feature": feature_name,
                "test": "PSI",
                "psi": np.nan,
                "drift_detected": False,
                "error": "Empty arrays",
            }

        # Create bins based on reference data
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicates

        if len(bins) < 2:
            return {
                "feature": feature_name,
                "test": "PSI",
                "psi": np.nan,
                "drift_detected": False,
                "error": "Insufficient unique values for binning",
            }

        # Bin the data
        ref_binned = np.digitize(reference, bins[1:-1])
        prod_binned = np.digitize(production, bins[1:-1])

        # Calculate proportions
        ref_counts = np.bincount(ref_binned, minlength=len(bins))
        prod_counts = np.bincount(prod_binned, minlength=len(bins))

        ref_props = ref_counts / len(reference)
        prod_props = prod_counts / len(production)

        # Add small constant to avoid log(0)
        epsilon = 1e-10
        ref_props = np.maximum(ref_props, epsilon)
        prod_props = np.maximum(prod_props, epsilon)

        # Calculate PSI
        psi_value = np.sum((prod_props - ref_props) * np.log(prod_props / ref_props))

        drift_detected = psi_value > self.psi_threshold

        result = {
            "feature": feature_name,
            "test": "PSI",
            "psi": float(psi_value),
            "drift_detected": drift_detected,
            "severity": self._classify_drift_severity(psi_value, "psi"),
        }

        if drift_detected:
            logger.warning(
                f"Feature drift detected (PSI): {feature_name}", extra=result
            )

        return result

    def js_divergence(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        feature_name: str,
        n_bins: int = 50,
    ) -> Dict:
        """
        Jensen-Shannon divergence for distribution comparison.

        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        where M = 0.5 * (P + Q)

        Args:
            reference: Reference data
            production: Production data
            feature_name: Name of the feature
            n_bins: Number of bins for histogram

        Returns:
            Dictionary with JS divergence results
        """
        # Remove NaNs
        reference = reference[~np.isnan(reference)]
        production = production[~np.isnan(production)]

        if len(reference) == 0 or len(production) == 0:
            return {
                "feature": feature_name,
                "test": "JS",
                "js_divergence": np.nan,
                "drift_detected": False,
                "error": "Empty arrays",
            }

        # Create common bins
        all_data = np.concatenate([reference, production])
        bins = np.linspace(all_data.min(), all_data.max(), n_bins + 1)

        # Create histograms
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        prod_hist, _ = np.histogram(production, bins=bins, density=True)

        # Normalize to probability distributions
        ref_hist = ref_hist / ref_hist.sum()
        prod_hist = prod_hist / prod_hist.sum()

        # Calculate JS divergence
        js_div = jensenshannon(ref_hist, prod_hist)

        drift_detected = js_div > self.js_threshold

        result = {
            "feature": feature_name,
            "test": "JS",
            "js_divergence": float(js_div),
            "drift_detected": drift_detected,
            "severity": self._classify_drift_severity(js_div, "js"),
        }

        if drift_detected:
            logger.warning(f"Feature drift detected (JS): {feature_name}", extra=result)

        return result

    def chi2_test(
        self, reference: np.ndarray, production: np.ndarray, feature_name: str
    ) -> Dict:
        """
        Chi-squared test for categorical feature drift.

        Args:
            reference: Reference categorical data
            production: Production categorical data
            feature_name: Name of the feature

        Returns:
            Dictionary with chi-squared test results
        """
        # Get unique categories
        np.unique(np.concatenate([reference, production]))

        # Create contingency table
        ref_counts = pd.Series(reference).value_counts()
        prod_counts = pd.Series(production).value_counts()

        contingency = pd.DataFrame(
            {"reference": ref_counts, "production": prod_counts}
        ).fillna(0)

        # Perform chi-squared test
        chi2, p_value, dof, expected = chi2_contingency(contingency.values)

        drift_detected = p_value < self.ks_threshold

        result = {
            "feature": feature_name,
            "test": "Chi2",
            "statistic": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof),
            "drift_detected": drift_detected,
            "severity": self._classify_drift_severity(p_value, "chi2"),
        }

        if drift_detected:
            logger.warning(f"Categorical drift detected: {feature_name}", extra=result)

        return result

    def monitor_dataframe(
        self,
        reference_df: pd.DataFrame,
        production_df: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> Dict:
        """
        Monitor drift across all features in a DataFrame.

        Args:
            reference_df: Reference (training) DataFrame
            production_df: Production (current) DataFrame
            categorical_features: List of categorical feature names

        Returns:
            Dictionary with drift results for all features
        """
        categorical_features = categorical_features or []
        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_features": len(reference_df.columns),
            "features": {},
        }

        for col in reference_df.columns:
            if col in categorical_features:
                # Categorical feature
                result = self.chi2_test(
                    reference_df[col].values, production_df[col].values, col
                )
            else:
                # Continuous feature - run multiple tests
                ks_result = self.ks_test(
                    reference_df[col].values, production_df[col].values, col
                )
                psi_result = self.psi(
                    reference_df[col].values, production_df[col].values, col
                )

                result = {
                    "ks": ks_result,
                    "psi": psi_result,
                    "drift_detected": ks_result["drift_detected"]
                    or psi_result["drift_detected"],
                }

            results["features"][col] = result

        # Count drifted features
        drifted_features = [
            col
            for col, res in results["features"].items()
            if res.get("drift_detected", False)
            or (
                isinstance(res, dict) and res.get("ks", {}).get("drift_detected", False)
            )
        ]

        results["n_drifted_features"] = len(drifted_features)
        results["drifted_features"] = drifted_features
        results["drift_percentage"] = (
            len(drifted_features) / len(reference_df.columns) * 100
        )

        # Store in history
        self.drift_history.append(results)

        logger.info(
            "Drift monitoring complete",
            extra={
                "n_features": results["n_features"],
                "n_drifted": results["n_drifted_features"],
                "drift_pct": results["drift_percentage"],
            },
        )

        return results

    def _classify_drift_severity(self, value: float, test_type: str) -> str:
        """
        Classify drift severity based on test statistic.

        Args:
            value: Test statistic value
            test_type: Type of test ('ks', 'psi', 'js', 'chi2')

        Returns:
            Severity classification string
        """
        if test_type == "psi":
            if value < 0.1:
                return "none"
            elif value < 0.2:
                return "moderate"
            else:
                return "severe"
        elif test_type in ["ks", "chi2"]:
            # P-value based
            if value > 0.1:
                return "none"
            elif value > 0.01:
                return "moderate"
            else:
                return "severe"
        elif test_type == "js":
            if value < 0.1:
                return "none"
            elif value < 0.3:
                return "moderate"
            else:
                return "severe"
        else:
            return "unknown"

    def get_drift_report(self) -> pd.DataFrame:
        """
        Generate a drift monitoring report.

        Returns:
            DataFrame with drift history
        """
        if not self.drift_history:
            return pd.DataFrame()

        report_data = []
        for entry in self.drift_history:
            for feature, result in entry["features"].items():
                if isinstance(result, dict) and "ks" in result:
                    # Continuous feature
                    report_data.append(
                        {
                            "timestamp": entry["timestamp"],
                            "feature": feature,
                            "test": "KS",
                            "statistic": result["ks"]["statistic"],
                            "p_value": result["ks"]["p_value"],
                            "drift_detected": result["ks"]["drift_detected"],
                            "severity": result["ks"]["severity"],
                        }
                    )
                    report_data.append(
                        {
                            "timestamp": entry["timestamp"],
                            "feature": feature,
                            "test": "PSI",
                            "statistic": result["psi"]["psi"],
                            "p_value": None,
                            "drift_detected": result["psi"]["drift_detected"],
                            "severity": result["psi"]["severity"],
                        }
                    )
                else:
                    # Categorical feature
                    report_data.append(
                        {
                            "timestamp": entry["timestamp"],
                            "feature": feature,
                            "test": result["test"],
                            "statistic": result.get("statistic"),
                            "p_value": result.get("p_value"),
                            "drift_detected": result["drift_detected"],
                            "severity": result["severity"],
                        }
                    )

        return pd.DataFrame(report_data)

    def should_retrain(self, drift_percentage_threshold: float = 20.0) -> bool:
        """
        Determine if model should be retrained based on drift.

        Args:
            drift_percentage_threshold: Percentage of drifted features to trigger retrain

        Returns:
            True if retraining is recommended
        """
        if not self.drift_history:
            return False

        latest = self.drift_history[-1]
        drift_pct = latest["drift_percentage"]

        should_retrain = drift_pct >= drift_percentage_threshold

        if should_retrain:
            logger.warning(
                "Model retraining recommended",
                extra={
                    "drift_percentage": drift_pct,
                    "threshold": drift_percentage_threshold,
                    "drifted_features": latest["drifted_features"],
                },
            )

        return should_retrain


# Made with Bob
